#include "global.h"

#if HW_VERSION == 2

#include <LittleFS.h>
#include <cstring>

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

static inline bool isFiringCritical() {
	// Avoid flash writes / export while actually firing / ramping down.
	return operationState == STATE_RAMPUP || operationState == STATE_PUSH || operationState == STATE_RETRACT || operationState == STATE_RAMPDOWN;
}

static bool mlLogFlushSome(u32 maxSamples) {
	if (!fsReady || !logFile) return false;
	if (!maxSamples) return true;

	constexpr u32 CHUNK = 32;
	static MlSample batch[CHUNK];
	u32 total = 0;
	while (total < maxSamples) {
		u32 n = 0;
		MlSample s;
		while (n < CHUNK && total < maxSamples && spscPop(s)) {
			batch[n++] = s;
			total++;
		}
		if (!n) break;
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
			return false;
		}
	}
	return true;
}

static void mlLogFlushAll() {
	// Used for export. Only called on core 0 from mlLogSlowLoop().
	(void)mlLogFlushSome(0xFFFFFFFFu);
	if (logFile) logFile.flush();
}

static void mlLogExportBinary() {
	if (!fsReady) {
		Serial.println("[ml] ERROR: filesystem not ready");
		return;
	}
	if (isFiringCritical()) {
		Serial.println("[ml] ERROR: busy (firing)");
		return;
	}

	// Pause sampling and flush RAM buffer to flash so export is consistent.
	const bool wasActive = logActive;
	logActive = false;
	mlLogFlushAll();
	if (logFile) {
		logFile.flush();
		logFile.close();
	}

	File f = LittleFS.open(ML_LOG_FILENAME, "r");
	if (!f) {
		Serial.println("[ml] ERROR: no log file");
		// fall through to resume
	} else {
		const u32 size = f.size();
		Serial.printf("MLDUMP1 %lu\n", size);

		uint8_t buf[256];
		while (true) {
			const int n = f.read(buf, sizeof(buf));
			if (n <= 0) break;
			Serial.write(buf, (size_t)n);
		}
		f.close();
		Serial.println("\nMLDUMP_DONE");
	}

	// Resume logging (append) unless it was already stopped due to full/error.
	if (fsReady && !logFullWarned) {
		logFile = LittleFS.open(ML_LOG_FILENAME, "a");
		if (logFile) {
			logActive = wasActive;
		}
	}
}

void mlLogInit() {
	// Keep UX identical unless compiled in.
	if (!LittleFS.begin()) {
		if (!LittleFS.format() || !LittleFS.begin()) {
			Serial.println("[ml] ERROR: LittleFS mount failed");
			return;
		}
	}
	fsReady = true;

	Serial.println("[ml] filesystem ready");
}

void mlLogStartRecording() {
	if (!fsReady) {
		Serial.println("[ml] ERROR: filesystem not ready");
		return;
	}

	logFile = LittleFS.open(ML_LOG_FILENAME, "w");
	if (!logFile) {
		Serial.println("[ml] ERROR: failed to open log file");
		return;
	}

	__atomic_store_n(&spscWr, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&spscRd, 0, __ATOMIC_RELAXED);
	decim = 0;
	sampleCountWritten = 0;
	logFullWarned = false;
	logActive = true;
	Serial.println("[ml] recording started");
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

void mlLogSlowLoop() {
	const bool firing = isFiringCritical();
	static bool wasFiring = false;
	if (wasFiring && !firing) {
		// A shot (or firing sequence) just ended; flush soon.
		flushRequested = true;
	}
	wasFiring = firing;
	if (firing) return;

	// Export command (consumer-friendly): send "MLDUMP\n" over USB serial.
	// Use our python helper to save `ml_log.bin` without needing debug tools.
	static char cmdBuf[16];
	static u8 cmdLen = 0;
	while (Serial.available()) {
		const char c = (char)Serial.read();
		if (c == '\n' || c == '\r') {
			cmdBuf[cmdLen] = '\0';
			if (cmdLen && strcmp(cmdBuf, "MLDUMP") == 0) {
				mlLogExportBinary();
			} else if (cmdLen && strcmp(cmdBuf, "MLINFO") == 0) {
				Serial.printf("[ml] samples_written=%lu (~%.1fs)\n", sampleCountWritten, sampleCountWritten / 100.0f);
			}
			cmdLen = 0;
		} else if (cmdLen + 1 < sizeof(cmdBuf)) {
			cmdBuf[cmdLen++] = c;
		} else {
			cmdLen = 0;
		}
	}

	if (!logActive || !fsReady || !logFile) return;

	const u32 pending = spscCount();
	if (!pending) return;

	// Default behavior: buffer during normal use, then flush after a shot.
	// Also flush if buffer gets close to full, to avoid dropping long idle segments.
	constexpr u32 HIGH_WATERMARK = SPSC_CAPACITY * 3 / 4;
	if (!flushRequested && pending < HIGH_WATERMARK) {
		return;
	}

	// Drain in bounded chunks to keep core 0 responsive.
	(void)mlLogFlushSome(32);

	// Once we've started flushing post-shot, keep flushing until the buffer is mostly empty.
	// This reduces flash activity between shots and keeps more contiguous pre-shot windows in RAM.
	if (flushRequested) {
		constexpr u32 LOW_WATERMARK = SPSC_CAPACITY / 8;
		if (spscCount() <= LOW_WATERMARK) {
			flushRequested = false;
		}
	}
}

bool mlLogIsActive() {
	return logActive;
}

u8 mlLogFlashPercent() {
	if (!fsReady) return 0;
	// 1.5 MB filesystem, estimate usable capacity conservatively at 1.4 MB for LittleFS overhead.
	constexpr u32 USABLE_BYTES = 1400u * 1024u;
	constexpr u32 MAX_SAMPLES = USABLE_BYTES / sizeof(MlSample);
	const u32 total = sampleCountWritten + spscCount();
	if (total >= MAX_SAMPLES) return 100;
	return (u8)((total * 100u) / MAX_SAMPLES);
}

#endif // HW_VERSION == 2
