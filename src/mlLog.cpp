#include "global.h"

#if HW_VERSION == 2

#include <LittleFS.h>
#include <algorithm>
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
static volatile bool logPausedStill = false; // set on core 1
static volatile bool logPausedUsb = false; // set on core 0 when USB session active
static bool logFullWarned = false;
static u32 sampleCountWritten = 0;
static u8 decim = 0;
static bool flushRequested = false;
static bool stopRequested = false;
static bool sessionStartedThisBoot = false;

static inline bool usbSess() { return ::usbCdcActive(); }

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
		// Some FS/USB timing combinations can make f.size() disagree with the number of
		// bytes actually readable right now. Do a quick count pass, then stream exactly
		// that many bytes so the host never hangs waiting.
		u32 size = 0;
		{
			uint8_t buf[256];
			while (true) {
				const int n = f.read(buf, sizeof(buf));
				if (n <= 0) break;
				size += (u32)n;
			}
			f.close();
		}
		if (!size) {
			Serial.println("[ml] ERROR: no log data");
		} else {
			Serial.printf("MLDUMP1 %lu\n", size);
			File f2 = LittleFS.open(ML_LOG_FILENAME, "r");
			if (!f2) {
				Serial.println("[ml] ERROR: no log file");
			} else {
				uint8_t buf[256];
				u32 sent = 0;
				while (sent < size) {
					const int want = (size - sent) < sizeof(buf) ? (int)(size - sent) : (int)sizeof(buf);
					const int n = f2.read(buf, want);
					if (n <= 0) break;
					Serial.write(buf, (size_t)n);
					sent += (u32)n;
				}
				f2.close();
				Serial.println("\nMLDUMP_DONE");
			}
		}
	}

	// Resume logging (append) unless it was already stopped due to full/error.
	// While USB is connected (service mode), keep the log file closed so model upload is reliable.
	if (fsReady && !logFullWarned && !usbSess()) {
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
	sessionStartedThisBoot = false;
	logActive = false;
	logPausedStill = false;
	logPausedUsb = false;
	stopRequested = false;
	flushRequested = false;
	logFullWarned = false;
	sampleCountWritten = 0;
	decim = 0;

	// New session per boot (user asked: only reset to 0% after a full reboot).
	if (LittleFS.exists(ML_LOG_FILENAME)) LittleFS.remove(ML_LOG_FILENAME);

	Serial.println("[ml] filesystem ready");
}

void mlLogStartRecording() {
	if (!fsReady) {
		Serial.println("[ml] ERROR: filesystem not ready");
		return;
	}

	if (logFullWarned) {
		Serial.println("[ml] ERROR: log is full; reboot to start a new session");
		return;
	}

	// Idempotent: if already active, do nothing.
	if (logActive) return;
	logPausedStill = false;
	logPausedUsb = false;

	// If a stop was requested, cancel it and keep writing.
	stopRequested = false;
	flushRequested = false;

	if (logFile) {
		logFile.flush();
		logFile.close();
	}

	// First start of the boot truncates/creates a new file. Subsequent starts append.
	const char *mode = sessionStartedThisBoot ? "a" : "w";
	logFile = LittleFS.open(ML_LOG_FILENAME, mode);
	if (!logFile) {
		Serial.println("[ml] ERROR: failed to open log file");
		return;
	}

	__atomic_store_n(&spscWr, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&spscRd, 0, __ATOMIC_RELAXED);
	decim = 0;
	if (!sessionStartedThisBoot) {
		sampleCountWritten = 0;
		logFullWarned = false;
		sessionStartedThisBoot = true;
	} else {
		// Resume: reflect current file size for accurate % display.
		sampleCountWritten = (u32)(logFile.size() / sizeof(MlSample));
	}
	logActive = true;
	logPausedStill = false;
	logPausedUsb = false;
	stopRequested = false;
	Serial.println("[ml] recording started");
}

void mlLogStopRecording() {
	// Stop capturing immediately; flush the remaining buffered samples on core 0 when safe.
	if (!logActive && !stopRequested) return;
	logActive = false;
	logPausedStill = false;
	logPausedUsb = false;
	stopRequested = true;
	flushRequested = true;
	Serial.println("[ml] recording stopping");
}

void mlLogLoop() {
	if (!logActive) return;
	if (++decim < ML_LOG_DECIMATION) return;
	decim = 0;

	// While USB is connected, pause sampling to avoid growing the file while exporting /
	// uploading models over Serial.
	if (usbSess()) return;

	// Auto-pause logging while completely still to save flash and keep datasets cleaner.
	// Uses per-sample deltas (not absolute accel magnitude) so gravity doesn't matter.
	constexpr i32 STILL_DELTA = 30; // raw i16 delta threshold (noise tolerant)
	constexpr i32 UNPAUSE_DELTA = 140; // hysteresis (harder to resume)
	constexpr i32 STILL_GYRO_ABS = 45; // gyro near-zero threshold
	constexpr i32 UNPAUSE_GYRO_ABS = 240; // gyro threshold to resume even if deltas are small
	constexpr u32 STILL_SAMPLES_TO_PAUSE = 50; // ~0.50s @ 100Hz
	constexpr u32 UNPAUSE_SAMPLES = 10; // require sustained motion to resume
	static bool haveLast = false;
	static i16 lastAx = 0, lastAy = 0, lastAz = 0, lastGx = 0, lastGy = 0, lastGz = 0;
	static u32 stillCount = 0;
	static u32 wakeCount = 0;

	MlSample s;
	s.timestamp_ms = millis();
	s.ax = (i16)accelDataRaw[0];
	s.ay = (i16)accelDataRaw[1];
	s.az = (i16)accelDataRaw[2];
	s.gx = (i16)gyroDataRaw[0];
	s.gy = (i16)gyroDataRaw[1];
	s.gz = (i16)gyroDataRaw[2];
	s.trigger = triggerState ? 1 : 0;

	// Never pause while the trigger is down (we want to capture around shots),
	// and reset stillness tracking when logging starts.
	if (!haveLast) {
		haveLast = true;
		lastAx = s.ax;
		lastAy = s.ay;
		lastAz = s.az;
		lastGx = s.gx;
		lastGy = s.gy;
		lastGz = s.gz;
		stillCount = 0;
		logPausedStill = false;
	} else {
		auto iabs = [](i32 v) -> i32 { return (v < 0) ? -v : v; };
		i32 maxDelta = 0;
		maxDelta = std::max(maxDelta, iabs((i32)s.ax - (i32)lastAx));
		maxDelta = std::max(maxDelta, iabs((i32)s.ay - (i32)lastAy));
		maxDelta = std::max(maxDelta, iabs((i32)s.az - (i32)lastAz));
		maxDelta = std::max(maxDelta, iabs((i32)s.gx - (i32)lastGx));
		maxDelta = std::max(maxDelta, iabs((i32)s.gy - (i32)lastGy));
		maxDelta = std::max(maxDelta, iabs((i32)s.gz - (i32)lastGz));

		lastAx = s.ax;
		lastAy = s.ay;
		lastAz = s.az;
		lastGx = s.gx;
		lastGy = s.gy;
		lastGz = s.gz;

		const bool gyroStill =
			iabs((i32)s.gx) <= STILL_GYRO_ABS &&
			iabs((i32)s.gy) <= STILL_GYRO_ABS &&
			iabs((i32)s.gz) <= STILL_GYRO_ABS;

		if (s.trigger) {
			stillCount = 0;
			wakeCount = 0;
			logPausedStill = false;
		} else if (!logPausedStill) {
			if (gyroStill && maxDelta <= STILL_DELTA) {
				if (stillCount < 0xFFFFFFFFu) stillCount++;
				if (stillCount >= STILL_SAMPLES_TO_PAUSE) logPausedStill = true;
			} else {
				stillCount = 0;
			}
		} else {
			// paused: only unpause on a stronger movement
			const bool gyroWake =
				iabs((i32)s.gx) >= UNPAUSE_GYRO_ABS ||
				iabs((i32)s.gy) >= UNPAUSE_GYRO_ABS ||
				iabs((i32)s.gz) >= UNPAUSE_GYRO_ABS;
			const bool wantsWake = gyroWake || !gyroStill || maxDelta >= UNPAUSE_DELTA;
			if (wantsWake) {
				if (wakeCount < 0xFFFFFFFFu) wakeCount++;
			} else {
				wakeCount = 0;
			}
			if (wakeCount >= UNPAUSE_SAMPLES) {
				stillCount = 0;
				wakeCount = 0;
				logPausedStill = false;
			}
		}
	}

	if (logPausedStill) return;

	// Best effort: drop if buffer full.
	spscPush(s);
}

void mlLogSlowLoop() {
	// While connected over USB, pause logging.
	// IMPORTANT: avoid background flash writes while USB is active, otherwise USB CDC
	// can become unreliable during model upload (flash ops can stall interrupts).
	static bool usbWasActive = false;
	static bool resumeAfterUsb = false;
	const bool usbNow = usbSess();
	logPausedUsb = usbNow;

	// On USB entry: close the log file (and stop background flush) so LittleFS operations
	// for model upload don't contend with an open log file handle.
	if (usbNow && !usbWasActive) {
		usbWasActive = true;
		resumeAfterUsb = logActive;
		// Pause capturing without resetting session counters.
		flushRequested = false;
		stopRequested = false;
		if (logFile) {
			logFile.flush();
			logFile.close();
		}
	}
	// On USB exit: reopen the log file if it was previously recording.
	if (!usbNow && usbWasActive) {
		usbWasActive = false;
		if (resumeAfterUsb && fsReady && !logFullWarned) {
			logFile = LittleFS.open(ML_LOG_FILENAME, "a");
			// If reopen fails, leave logActive true but sampling will be best-effort; user can toggle recording.
			if (logFile) {
				sampleCountWritten = (u32)(logFile.size() / sizeof(MlSample));
			}
		}
		resumeAfterUsb = false;
	}

	const bool firing = isFiringCritical();
	static bool wasFiring = false;
	if (wasFiring && !firing) {
		// A shot (or firing sequence) just ended; flush soon.
		flushRequested = true;
	}
	wasFiring = firing;

	// Serial commands for consumer-friendly workflows:
	// - MLDUMP / MLINFO (logs)
	// - MLMODEL_* (upload + load a personalized model without reflashing UF2)
	static char cmdBuf[64];
	static u8 cmdLen = 0;
	static bool modelRxActive = false;
	static File modelRxFile;
	static u32 modelRxRemaining = 0;
	static u32 modelRxCrc = 0;
	static u32 modelRxExpectedCrc = 0;
	enum class ModelRxTarget : u8 {
		GENERIC,
		LR,
		MLP,
	};
	static ModelRxTarget modelRxTarget = ModelRxTarget::GENERIC;
	static elapsedMillis modelRxTimer = 0;

	auto modelRxAbort = [&]() {
		modelRxActive = false;
		modelRxRemaining = 0;
		modelRxCrc = 0;
		modelRxExpectedCrc = 0;
		if (modelRxFile) {
			modelRxFile.flush();
			modelRxFile.close();
		}
		if (LittleFS.exists("/ml_model.tmp")) LittleFS.remove("/ml_model.tmp");
		if (LittleFS.exists("/ml_model_lr.tmp")) LittleFS.remove("/ml_model_lr.tmp");
		if (LittleFS.exists("/ml_model_mlp.tmp")) LittleFS.remove("/ml_model_mlp.tmp");
	};

	// If we are currently receiving raw model bytes, drain them first.
	if (modelRxActive) {
		if (modelRxTimer > 10000) { // 10s timeout
			Serial.println("[ml] MLMODEL_ERR timeout");
			modelRxAbort();
		} else {
			while (Serial.available() && modelRxRemaining) {
				uint8_t buf[256];
				const size_t want = (modelRxRemaining < sizeof(buf)) ? (size_t)modelRxRemaining : sizeof(buf);
				const int n = Serial.readBytes((char *)buf, want);
				if (n <= 0) break;
				modelRxTimer = 0;
				modelRxFile.write(buf, (size_t)n);
				// CRC32 update (same poly as in mlInfer)
				u32 crc = ~modelRxCrc;
				for (int i = 0; i < n; i++) {
					u32 x = (crc ^ buf[i]) & 0xFFu;
					for (u32 j = 0; j < 8; j++)
						x = (x >> 1) ^ (0xEDB88320u & (-(i32)(x & 1u)));
					crc = (crc >> 8) ^ x;
				}
				modelRxCrc = ~crc;
				modelRxRemaining -= (u32)n;
			}
			if (modelRxActive && modelRxRemaining == 0) {
				modelRxFile.flush();
				modelRxFile.close();
				modelRxActive = false;

				if (modelRxCrc != modelRxExpectedCrc) {
					Serial.println("[ml] MLMODEL_ERR crc");
					modelRxAbort();
				} else {
					bool ok = false;
					if (modelRxTarget == ModelRxTarget::LR) {
						if (LittleFS.exists("/ml_model_lr.bin")) LittleFS.remove("/ml_model_lr.bin");
						ok = LittleFS.rename("/ml_model_lr.tmp", "/ml_model_lr.bin");
					} else if (modelRxTarget == ModelRxTarget::MLP) {
						if (LittleFS.exists("/ml_model_mlp.bin")) LittleFS.remove("/ml_model_mlp.bin");
						ok = LittleFS.rename("/ml_model_mlp.tmp", "/ml_model_mlp.bin");
					} else {
						if (LittleFS.exists("/ml_model.bin")) LittleFS.remove("/ml_model.bin");
						ok = LittleFS.rename("/ml_model.tmp", "/ml_model.bin");
					}
					if (!ok) {
						Serial.println("[ml] MLMODEL_ERR rename");
						modelRxAbort();
					} else {
						Serial.println("[ml] MLMODEL_OK");
					}
				}
			}
		}
		// Don't parse command text while receiving raw bytes.
		goto flush_logic;
	}

	// Important: do not unconditionally drain the serial RX buffer.
	// Other debug features (e.g. USE_BLACKBOX "press any key") may also be watching Serial.
	// We therefore only consume bytes once we detect an ML command prefix.
	while (Serial.available()) {
		if (cmdLen == 0) {
			const int pk = Serial.peek();
			if (pk < 0) break;
			if ((char)pk != 'M') break; // leave bytes for other consumers
		}
		const char c = (char)Serial.read();
		if (c == '\n' || c == '\r') {
			cmdBuf[cmdLen] = '\0';
			if (cmdLen && strcmp(cmdBuf, "MLDUMP") == 0) {
				mlLogExportBinary();
			} else if (cmdLen && strcmp(cmdBuf, "MLINFO") == 0) {
				Serial.printf("[ml] samples_written=%lu (~%.1fs)\n", sampleCountWritten, sampleCountWritten / 100.0f);
			} else if (cmdLen && strcmp(cmdBuf, "MLMODEL_INFO") == 0) {
				mlInferPrintInfo();
			} else if (cmdLen && strcmp(cmdBuf, "MLMODEL_LOAD") == 0) {
				if (mlInferLoadUserModel())
					Serial.println("[ml] MLMODEL_LOADED");
				else
					Serial.println("[ml] MLMODEL_ERR load");
			} else if (cmdLen && strcmp(cmdBuf, "MLMODEL_LOAD_LR") == 0) {
				if (mlInferLoadUserModelType(ML_MODEL_LOGREG))
					Serial.println("[ml] MLMODEL_LOADED");
				else
					Serial.println("[ml] MLMODEL_ERR load");
			} else if (cmdLen && strcmp(cmdBuf, "MLMODEL_LOAD_MLP") == 0) {
				if (mlInferLoadUserModelType(ML_MODEL_MLP))
					Serial.println("[ml] MLMODEL_LOADED");
				else
					Serial.println("[ml] MLMODEL_ERR load");
			} else if (cmdLen && strcmp(cmdBuf, "MLMODEL_DELETE") == 0) {
				if (mlInferDeleteUserModel())
					Serial.println("[ml] MLMODEL_DELETED");
				else
					Serial.println("[ml] MLMODEL_ERR delete");
			} else if (cmdLen && strncmp(cmdBuf, "MLMODEL_PUT ", 12) == 0) {
				// Format: MLMODEL_PUT <size> <crc32hex>
				// CRC is over the full file bytes.
				if (!fsReady) {
					Serial.println("[ml] MLMODEL_ERR fs");
				} else if (isFiringCritical()) {
					Serial.println("[ml] MLMODEL_ERR busy");
				} else {
					u32 size = 0;
					u32 crc = 0;
					if (sscanf(cmdBuf + 12, "%lu %lx", &size, &crc) == 2 && size > 0 && size < 200000) {
						if (LittleFS.exists("/ml_model.tmp")) LittleFS.remove("/ml_model.tmp");
						modelRxFile = LittleFS.open("/ml_model.tmp", "w");
						if (!modelRxFile) {
							Serial.println("[ml] MLMODEL_ERR open");
						} else {
							modelRxActive = true;
							modelRxTarget = ModelRxTarget::GENERIC;
							modelRxRemaining = size;
							modelRxExpectedCrc = crc;
							modelRxCrc = 0;
							modelRxTimer = 0;
							Serial.println("[ml] MLMODEL_READY");
						}
					} else {
						Serial.println("[ml] MLMODEL_ERR args");
					}
				}
			} else if (cmdLen && strncmp(cmdBuf, "MLMODEL_PUT_LR ", 15) == 0) {
				if (!fsReady) {
					Serial.println("[ml] MLMODEL_ERR fs");
				} else if (isFiringCritical()) {
					Serial.println("[ml] MLMODEL_ERR busy");
				} else {
					u32 size = 0;
					u32 crc = 0;
					if (sscanf(cmdBuf + 15, "%lu %lx", &size, &crc) == 2 && size > 0 && size < 200000) {
						if (LittleFS.exists("/ml_model_lr.tmp")) LittleFS.remove("/ml_model_lr.tmp");
						modelRxFile = LittleFS.open("/ml_model_lr.tmp", "w");
						if (!modelRxFile) {
							Serial.println("[ml] MLMODEL_ERR open");
						} else {
							modelRxActive = true;
							modelRxTarget = ModelRxTarget::LR;
							modelRxRemaining = size;
							modelRxExpectedCrc = crc;
							modelRxCrc = 0;
							modelRxTimer = 0;
							Serial.println("[ml] MLMODEL_READY");
						}
					} else {
						Serial.println("[ml] MLMODEL_ERR args");
					}
				}
			} else if (cmdLen && strncmp(cmdBuf, "MLMODEL_PUT_MLP ", 16) == 0) {
				if (!fsReady) {
					Serial.println("[ml] MLMODEL_ERR fs");
				} else if (isFiringCritical()) {
					Serial.println("[ml] MLMODEL_ERR busy");
				} else {
					u32 size = 0;
					u32 crc = 0;
					if (sscanf(cmdBuf + 16, "%lu %lx", &size, &crc) == 2 && size > 0 && size < 200000) {
						if (LittleFS.exists("/ml_model_mlp.tmp")) LittleFS.remove("/ml_model_mlp.tmp");
						modelRxFile = LittleFS.open("/ml_model_mlp.tmp", "w");
						if (!modelRxFile) {
							Serial.println("[ml] MLMODEL_ERR open");
						} else {
							modelRxActive = true;
							modelRxTarget = ModelRxTarget::MLP;
							modelRxRemaining = size;
							modelRxExpectedCrc = crc;
							modelRxCrc = 0;
							modelRxTimer = 0;
							Serial.println("[ml] MLMODEL_READY");
						}
					} else {
						Serial.println("[ml] MLMODEL_ERR args");
					}
				}
			}
			cmdLen = 0;
		} else if (cmdLen + 1 < sizeof(cmdBuf)) {
			cmdBuf[cmdLen++] = c;
		} else {
			cmdLen = 0;
		}
	}

flush_logic:
	// While USB is active, do not perform background flash flushes.
	// MLDUMP explicitly flushes/pauses for a consistent export.
	if (usbNow) return;
	// Never write to flash while firing / ramping, but still allow command parsing above so
	// the host gets a deterministic response (busy/ready) instead of silence.
	if (firing) return;
	if ((!logActive && !stopRequested) || !fsReady) return;
	if (!logFile) {
		// If we were asked to stop but the file is already closed (e.g. full/error), clear the request.
		if (stopRequested) {
			stopRequested = false;
			flushRequested = false;
		}
		return;
	}

	const u32 pending = spscCount();
	if (!pending) {
		if (stopRequested) {
			logFile.flush();
			logFile.close();
			stopRequested = false;
			flushRequested = false;
			Serial.println("[ml] recording stopped");
		}
		return;
	}

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
	if (flushRequested && !stopRequested) {
		constexpr u32 LOW_WATERMARK = SPSC_CAPACITY / 8;
		if (spscCount() <= LOW_WATERMARK) {
			flushRequested = false;
		}
	}
}

bool mlLogIsActive() {
	return logActive;
}

bool mlLogIsPaused() {
	const bool a = __atomic_load_n(&logPausedStill, __ATOMIC_ACQUIRE);
	const bool b = __atomic_load_n(&logPausedUsb, __ATOMIC_ACQUIRE);
	return a || b;
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
