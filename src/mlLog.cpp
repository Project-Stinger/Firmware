#include "global.h"

#if HW_VERSION == 2
#ifdef USE_ML_LOG

#include <LittleFS.h>

// 1600 Hz gyro rate / 16 = 100 Hz output.
constexpr u32 ML_LOG_DECIMATION = 16;
constexpr const char *ML_LOG_FILENAME = "/ml_log.bin";

// RAM buffer: core 1 produces, core 0 consumes.
// Power-of-two sized SPSC ring buffer with acquire/release index publishing.
constexpr u32 SPSC_CAPACITY = 4096; // 40.96s at 100 Hz
static_assert((SPSC_CAPACITY & (SPSC_CAPACITY - 1)) == 0, "SPSC_CAPACITY must be power-of-two");
constexpr u32 SPSC_MASK = SPSC_CAPACITY - 1;

static MlSample spscBuf[SPSC_CAPACITY];
static volatile u32 spscWr = 0; // written by core 1
static volatile u32 spscRd = 0; // written by core 0

static inline bool spscPush(const MlSample &s) {
	const u32 wr = __atomic_load_n(&spscWr, __ATOMIC_RELAXED);
	const u32 next = (wr + 1) & SPSC_MASK;
	const u32 rd = __atomic_load_n(&spscRd, __ATOMIC_ACQUIRE);
	if (next == rd) return false; // full
	spscBuf[wr] = s;
	__atomic_store_n(&spscWr, next, __ATOMIC_RELEASE);
	return true;
}

static inline bool spscPop(MlSample &s) {
	const u32 rd = __atomic_load_n(&spscRd, __ATOMIC_RELAXED);
	const u32 wr = __atomic_load_n(&spscWr, __ATOMIC_ACQUIRE);
	if (rd == wr) return false; // empty
	s = spscBuf[rd];
	__atomic_store_n(&spscRd, (rd + 1) & SPSC_MASK, __ATOMIC_RELEASE);
	return true;
}

static inline u32 spscCount() {
	const u32 wr = __atomic_load_n(&spscWr, __ATOMIC_ACQUIRE);
	const u32 rd = __atomic_load_n(&spscRd, __ATOMIC_ACQUIRE);
	return (wr - rd) & SPSC_MASK;
}

static bool fsReady = false;
static File logFile;
static volatile bool logActive = false; // set in init, cleared on full/error
static bool logFullWarned = false;
static u32 sampleCountWritten = 0;
static u8 decim = 0;
static bool flushRequested = false;

void mlLogInit() {
	// Keep UX identical unless compiled in.
	if (!LittleFS.begin()) {
		if (!LittleFS.format() || !LittleFS.begin()) {
			Serial.println("[ml] ERROR: LittleFS mount failed");
			return;
		}
	}
	fsReady = true;

	// Start recording immediately; overwrite previous session.
	logFile = LittleFS.open(ML_LOG_FILENAME, "w");
	if (!logFile) {
		Serial.println("[ml] ERROR: failed to open log file");
		fsReady = false;
		return;
	}

	__atomic_store_n(&spscWr, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&spscRd, 0, __ATOMIC_RELAXED);
	decim = 0;
	sampleCountWritten = 0;
	logFullWarned = false;
	logActive = true;
}

void mlLogLoop() {
	if (!logActive) return;
	if (++decim < ML_LOG_DECIMATION) return;
	decim = 0;

	MlSample s;
	s.timestamp_ms = millis();
	s.ax = (i16)accelDataRaw[0];
	s.ay = (i16)accelDataRaw[1];
	s.az = (i16)accelDataRaw[2];
	s.gx = (i16)gyroDataRaw[0];
	s.gy = (i16)gyroDataRaw[1];
	s.gz = (i16)gyroDataRaw[2];
	s.trigger = triggerState ? 1 : 0;

	// Best effort: drop if buffer full.
	spscPush(s);
}

static inline bool isFiringCritical() {
	// Avoid flash writes while actually firing / ramping down.
	return operationState == STATE_RAMPUP || operationState == STATE_PUSH || operationState == STATE_RETRACT || operationState == STATE_RAMPDOWN;
}

void mlLogSlowLoop() {
	if (!logActive || !fsReady || !logFile) return;
	const bool firing = isFiringCritical();
	static bool wasFiring = false;
	if (wasFiring && !firing) {
		// A shot (or firing sequence) just ended; flush soon.
		flushRequested = true;
	}
	wasFiring = firing;
	if (firing) return;

	const u32 pending = spscCount();
	if (!pending) return;

	// Default behavior: buffer during normal use, then flush after a shot.
	// Also flush if buffer gets close to full, to avoid dropping long idle segments.
	constexpr u32 HIGH_WATERMARK = SPSC_CAPACITY * 3 / 4;
	if (!flushRequested && pending < HIGH_WATERMARK) {
		return;
	}

	// Drain in bounded chunks to keep core 0 responsive.
	constexpr u32 MAX_SAMPLES_PER_DRAIN = 256;
	const u32 toDrain = pending > MAX_SAMPLES_PER_DRAIN ? MAX_SAMPLES_PER_DRAIN : pending;

	MlSample batch[MAX_SAMPLES_PER_DRAIN];
	u32 n = 0;
	MlSample s;
	while (n < toDrain && spscPop(s)) {
		batch[n++] = s;
	}
	if (!n) return;

	const size_t want = n * sizeof(MlSample);
	const size_t got = logFile.write((const uint8_t *)batch, want);
	sampleCountWritten += (u32)(got / sizeof(MlSample));

	if (got != want) {
		logActive = false;
		logFile.flush();
		logFile.close();
		if (!logFullWarned) {
			logFullWarned = true;
			Serial.println("[ml] WARNING: log full (short write); logging disabled");
			makeSound(3000, 200);
		}
		return;
	}

	// Once we've started flushing post-shot, keep flushing until the buffer is mostly empty.
	// This reduces flash activity between shots and keeps more contiguous pre-shot windows in RAM.
	if (flushRequested) {
		constexpr u32 LOW_WATERMARK = SPSC_CAPACITY / 8;
		if (spscCount() <= LOW_WATERMARK) {
			flushRequested = false;
		}
	}
}

#endif // USE_ML_LOG
#endif // HW_VERSION == 2
