#include "global.h"

#if HW_VERSION == 2

#include "mlWeights.h"
#include <cmath>
#include <cstring>
#include <LittleFS.h>

// Sliding window ring buffer for ML_WINDOW_SAMPLES (50) x 6 channels.
// Written by core 1, read by core 0. Access is safe because:
//   - core 1 writes one sample per tick, advancing writeIdx
//   - core 0 reads only when writeIdx has wrapped (windowFull)
//   - no simultaneous read/write of the same slot (50-sample lag)
static i16 windowBuf[ML_WINDOW_SAMPLES][ML_CHANNELS];
static volatile u8 writeIdx = 0;
static volatile bool windowFull = false;
// Sequence lock to prevent torn reads while core 1 writes a slot.
static volatile u32 inferSeq = 0;

static inline u32 packF32(float x) {
	u32 u;
	memcpy(&u, &x, sizeof(u));
	return u;
}
static inline float unpackF32(u32 u) {
	float x;
	memcpy(&x, &u, sizeof(x));
	return x;
}

static volatile u32 cachedProbLR_u32 = 0;
static volatile u32 cachedProbMLP_u32 = 0;
static volatile u32 cachedProbUpdatedMs = 0;

static constexpr const char *ML_MODEL_FILENAME = "/ml_model.bin";
static constexpr const char *ML_MODEL_LR_FILENAME = "/ml_model_lr.bin";
static constexpr const char *ML_MODEL_MLP_FILENAME = "/ml_model_mlp.bin";

// Runtime model parameters (start as defaults from mlWeights.h; can be overridden by user model).
static float gLrMean[ML_LR_FEATURES];
static float gLrScale[ML_LR_FEATURES];
static float gLrCoef[ML_LR_FEATURES];
static float gLrIntercept = 0.0f;

static float gMlpMean[ML_MLP_FEATURES];
static float gMlpScale[ML_MLP_FEATURES];
static float gMlpW1[ML_MLP_H1 * ML_MLP_FEATURES];
static float gMlpB1[ML_MLP_H1];
static float gMlpW2[ML_MLP_H2 * ML_MLP_H1];
static float gMlpB2[ML_MLP_H2];
static float gMlpW3[ML_MLP_H2];
static float gMlpB3 = 0.0f;

static bool gUsingUserModel = false;
static bool gUserHasLR = false;
static bool gUserHasMLP = false;

static u32 crc32_update(u32 crc, const uint8_t *data, size_t len) {
	// Standard CRC32 (poly 0xEDB88320)
	crc = ~crc;
	for (size_t i = 0; i < len; i++) {
		u32 x = (crc ^ data[i]) & 0xFFu;
		for (u32 j = 0; j < 8; j++) x = (x >> 1) ^ (0xEDB88320u & (-(i32)(x & 1u)));
		crc = (crc >> 8) ^ x;
	}
	return ~crc;
}

struct __attribute__((packed)) MlModelHeader {
	char magic[4];           // "MLMD"
	u16 version;             // 1
	u16 windowSamples;       // must equal ML_WINDOW_SAMPLES
	u8 modelType;            // 0=LR, 1=MLP
	u8 reserved;
	u16 features;            // 18 or 30
	u16 h1;                  // 64 for MLP
	u16 h2;                  // 32 for MLP
	u32 payloadBytes;        // bytes after header
	u32 payloadCrc32;        // CRC32 over payload bytes
};

static size_t expectedPayloadBytes(u8 modelType) {
	if (modelType == 0) {
		// mean + scale + coef + intercept
		const size_t floats = (size_t)ML_LR_FEATURES * 3 + 1;
		return floats * sizeof(float);
	}
	// MLP: scaler mean/scale + W1,B1,W2,B2,W3,B3
	const size_t floats =
		(size_t)ML_MLP_FEATURES * 2 +
		(size_t)ML_MLP_H1 * ML_MLP_FEATURES +
		(size_t)ML_MLP_H1 +
		(size_t)ML_MLP_H2 * ML_MLP_H1 +
		(size_t)ML_MLP_H2 +
		(size_t)ML_MLP_H2 +
		1;
	return floats * sizeof(float);
}

static bool loadUserModelFromPath(const char *path, bool expectType, u8 expectedType);

static void loadDefaultsIntoRam() {
	memcpy(gLrMean, mlLrScalerMean, sizeof(gLrMean));
	memcpy(gLrScale, mlLrScalerScale, sizeof(gLrScale));
	memcpy(gLrCoef, mlLrCoef, sizeof(gLrCoef));
	gLrIntercept = mlLrIntercept;

	memcpy(gMlpMean, mlMlpScalerMean, sizeof(gMlpMean));
	memcpy(gMlpScale, mlMlpScalerScale, sizeof(gMlpScale));
	memcpy(gMlpW1, mlMlpW1, sizeof(gMlpW1));
	memcpy(gMlpB1, mlMlpB1, sizeof(gMlpB1));
	memcpy(gMlpW2, mlMlpW2, sizeof(gMlpW2));
	memcpy(gMlpB2, mlMlpB2, sizeof(gMlpB2));
	memcpy(gMlpW3, mlMlpW3, sizeof(gMlpW3));
	gMlpB3 = mlMlpB3;
	gUsingUserModel = false;
}

static void resetLrToFactory() {
	memcpy(gLrMean, mlLrScalerMean, sizeof(gLrMean));
	memcpy(gLrScale, mlLrScalerScale, sizeof(gLrScale));
	memcpy(gLrCoef, mlLrCoef, sizeof(gLrCoef));
	gLrIntercept = mlLrIntercept;
	gUserHasLR = false;
	gUsingUserModel = gUserHasLR || gUserHasMLP;
}

static void resetMlpToFactory() {
	memcpy(gMlpMean, mlMlpScalerMean, sizeof(gMlpMean));
	memcpy(gMlpScale, mlMlpScalerScale, sizeof(gMlpScale));
	memcpy(gMlpW1, mlMlpW1, sizeof(gMlpW1));
	memcpy(gMlpB1, mlMlpB1, sizeof(gMlpB1));
	memcpy(gMlpW2, mlMlpW2, sizeof(gMlpW2));
	memcpy(gMlpB2, mlMlpB2, sizeof(gMlpB2));
	memcpy(gMlpW3, mlMlpW3, sizeof(gMlpW3));
	gMlpB3 = mlMlpB3;
	gUserHasMLP = false;
	gUsingUserModel = gUserHasLR || gUserHasMLP;
}

void mlInferInit() {
	for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++)
		for (u32 j = 0; j < ML_CHANNELS; j++)
			windowBuf[i][j] = 0;
	writeIdx = 0;
	windowFull = false;
	__atomic_store_n(&inferSeq, 0, __ATOMIC_RELAXED);
	__atomic_store_n(&cachedProbLR_u32, packF32(0.0f), __ATOMIC_RELAXED);
	__atomic_store_n(&cachedProbMLP_u32, packF32(0.0f), __ATOMIC_RELAXED);
	__atomic_store_n(&cachedProbUpdatedMs, 0, __ATOMIC_RELAXED);

	loadDefaultsIntoRam();

	// Auto-load user models from LittleFS if they were uploaded in a previous session.
	if (LittleFS.exists(ML_MODEL_LR_FILENAME))
		loadUserModelFromPath(ML_MODEL_LR_FILENAME, true, 0);
	if (LittleFS.exists(ML_MODEL_MLP_FILENAME))
		loadUserModelFromPath(ML_MODEL_MLP_FILENAME, true, 1);
	if (!gUserHasLR && !gUserHasMLP && LittleFS.exists(ML_MODEL_FILENAME))
		loadUserModelFromPath(ML_MODEL_FILENAME, false, 0);

	if (gUsingUserModel)
		Serial.println("[ml] user model loaded from flash");
}

void mlInferPushSample(i16 ax, i16 ay, i16 az, i16 gx, i16 gy, i16 gz) {
	__atomic_fetch_add(&inferSeq, 1, __ATOMIC_ACQ_REL); // odd = write in progress
	u8 idx = writeIdx;
	windowBuf[idx][0] = ax;
	windowBuf[idx][1] = ay;
	windowBuf[idx][2] = az;
	windowBuf[idx][3] = gx;
	windowBuf[idx][4] = gy;
	windowBuf[idx][5] = gz;
	idx++;
	if (idx >= ML_WINDOW_SAMPLES) {
		idx = 0;
		windowFull = true;
	}
	__atomic_store_n(&writeIdx, idx, __ATOMIC_RELEASE);
	__atomic_fetch_add(&inferSeq, 1, __ATOMIC_RELEASE); // even = stable
}

// Same decimation as mlLog: 1600 Hz gyro / 16 = 100 Hz
static constexpr u32 ML_INFER_DECIMATION = 16;
static u8 inferDecim = 0;

void mlInferLoop() {
	if (++inferDecim < ML_INFER_DECIMATION) return;
	inferDecim = 0;
	mlInferPushSample(
		(i16)accelDataRaw[0], (i16)accelDataRaw[1], (i16)accelDataRaw[2],
		(i16)gyroDataRaw[0], (i16)gyroDataRaw[1], (i16)gyroDataRaw[2]
	);
}

bool mlInferReady() {
	return windowFull;
}

// --- Featurization helpers (run on core 0) ---

// Copy the ring buffer into a contiguous ordered array.
// Oldest sample first, newest last.
static bool copyWindowI16(i16 out[ML_WINDOW_SAMPLES][ML_CHANNELS]) {
	// Seqlock snapshot. Retry if core 1 writes during the copy.
	for (u32 attempt = 0; attempt < 5; attempt++) {
		u32 s0 = __atomic_load_n(&inferSeq, __ATOMIC_ACQUIRE);
		if (s0 & 1u) continue;
		const u8 start = __atomic_load_n(&writeIdx, __ATOMIC_ACQUIRE);
		for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++) {
			const u32 src = (start + i) % ML_WINDOW_SAMPLES;
			// copy 6 i16s
			for (u32 c = 0; c < ML_CHANNELS; c++) out[i][c] = windowBuf[src][c];
		}
		u32 s1 = __atomic_load_n(&inferSeq, __ATOMIC_ACQUIRE);
		if (s0 == s1 && !(s1 & 1u)) return true;
	}
	return false;
}

static void copyWindowF32(float out[ML_WINDOW_SAMPLES][ML_CHANNELS]) {
	i16 w16[ML_WINDOW_SAMPLES][ML_CHANNELS];
	if (!copyWindowI16(w16)) {
		for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++)
			for (u32 c = 0; c < ML_CHANNELS; c++)
				out[i][c] = 0.0f;
		return;
	}
	for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++)
		for (u32 c = 0; c < ML_CHANNELS; c++)
			out[i][c] = (float)w16[i][c];
}

static void featurizeSummary(const float w[ML_WINDOW_SAMPLES][ML_CHANNELS], float out[ML_LR_FEATURES]) {
	// mean(6), std(6), absmax(6) = 18
	for (u32 c = 0; c < ML_CHANNELS; c++) {
		float sum = 0, sumSq = 0, mx = 0;
		for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++) {
			float v = w[i][c];
			sum += v;
			sumSq += v * v;
			float av = fabsf(v);
			if (av > mx) mx = av;
		}
		float mean = sum / ML_WINDOW_SAMPLES;
		float var = sumSq / ML_WINDOW_SAMPLES - mean * mean;
		if (var < 0) var = 0;
		out[c] = mean;                         // mean[0..5]
		out[c + ML_CHANNELS] = sqrtf(var);     // std[6..11]
		out[c + ML_CHANNELS * 2] = mx;         // absmax[12..17]
	}
}

static void featurizeRich(const float w[ML_WINDOW_SAMPLES][ML_CHANNELS], float out[ML_MLP_FEATURES]) {
	// mean(6), std(6), absmax(6), absmean(6), amag_stats(3), gmag_stats(3) = 30
	for (u32 c = 0; c < ML_CHANNELS; c++) {
		float sum = 0, sumSq = 0, mx = 0, absSum = 0;
		for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++) {
			float v = w[i][c];
			sum += v;
			sumSq += v * v;
			float av = fabsf(v);
			if (av > mx) mx = av;
			absSum += av;
		}
		float mean = sum / ML_WINDOW_SAMPLES;
		float var = sumSq / ML_WINDOW_SAMPLES - mean * mean;
		if (var < 0) var = 0;
		out[c] = mean;                              // mean[0..5]
		out[c + 6] = sqrtf(var);                    // std[6..11]
		out[c + 12] = mx;                           // absmax[12..17]
		out[c + 18] = absSum / ML_WINDOW_SAMPLES;   // absmean[18..23]
	}

	// Magnitude stats
	float aSumM = 0, aSumSq = 0, aMax = 0;
	float gSumM = 0, gSumSq = 0, gMax = 0;
	for (u32 i = 0; i < ML_WINDOW_SAMPLES; i++) {
		float a2 = w[i][0] * w[i][0] + w[i][1] * w[i][1] + w[i][2] * w[i][2];
		float g2 = w[i][3] * w[i][3] + w[i][4] * w[i][4] + w[i][5] * w[i][5];
		float am = sqrtf(a2);
		float gm = sqrtf(g2);
		aSumM += am;
		aSumSq += a2; // am^2 = a2
		if (am > aMax) aMax = am;
		gSumM += gm;
		gSumSq += g2;
		if (gm > gMax) gMax = gm;
	}
	float aMean = aSumM / ML_WINDOW_SAMPLES;
	float aVar = aSumSq / ML_WINDOW_SAMPLES - aMean * aMean;
	if (aVar < 0) aVar = 0;
	float gMean = gSumM / ML_WINDOW_SAMPLES;
	float gVar = gSumSq / ML_WINDOW_SAMPLES - gMean * gMean;
	if (gVar < 0) gVar = 0;

	out[24] = aMean;          // amag_mean
	out[25] = sqrtf(aVar);    // amag_std
	out[26] = aMax;           // amag_max
	out[27] = gMean;          // gmag_mean
	out[28] = sqrtf(gVar);    // gmag_std
	out[29] = gMax;           // gmag_max
}

static float sigmoid(float x) {
	if (x > 20.0f) return 1.0f;
	if (x < -20.0f) return 0.0f;
	return 1.0f / (1.0f + expf(-x));
}

static float predictLogreg(const float w[ML_WINDOW_SAMPLES][ML_CHANNELS]) {
	float feat[ML_LR_FEATURES];
	featurizeSummary(w, feat);

	// StandardScaler
	for (u32 i = 0; i < ML_LR_FEATURES; i++)
		feat[i] = (feat[i] - gLrMean[i]) / gLrScale[i];

	// dot product + intercept
	float z = gLrIntercept;
	for (u32 i = 0; i < ML_LR_FEATURES; i++)
		z += feat[i] * gLrCoef[i];

	return sigmoid(z);
}

static float predictMlp(const float w[ML_WINDOW_SAMPLES][ML_CHANNELS]) {
	float feat[ML_MLP_FEATURES];
	featurizeRich(w, feat);

	// StandardScaler
	for (u32 i = 0; i < ML_MLP_FEATURES; i++)
		feat[i] = (feat[i] - gMlpMean[i]) / gMlpScale[i];

	// Layer 1: 30 -> 64, ReLU
	float h1[ML_MLP_H1];
	for (u32 j = 0; j < ML_MLP_H1; j++) {
		float sum = gMlpB1[j];
		const float *wRow = &gMlpW1[j * ML_MLP_FEATURES];
		for (u32 i = 0; i < ML_MLP_FEATURES; i++)
			sum += wRow[i] * feat[i];
		h1[j] = sum > 0 ? sum : 0; // ReLU
	}

	// Layer 2: 64 -> 32, ReLU
	float h2[ML_MLP_H2];
	for (u32 j = 0; j < ML_MLP_H2; j++) {
		float sum = gMlpB2[j];
		const float *wRow = &gMlpW2[j * ML_MLP_H1];
		for (u32 i = 0; i < ML_MLP_H1; i++)
			sum += wRow[i] * h1[i];
		h2[j] = sum > 0 ? sum : 0; // ReLU
	}

	// Output: 32 -> 1, sigmoid
	float z = gMlpB3;
	for (u32 i = 0; i < ML_MLP_H2; i++)
		z += gMlpW3[i] * h2[i];

	return sigmoid(z);
}

float mlInferPredict(MlModel model) {
	if (!windowFull) return 0.0f;

	float w[ML_WINDOW_SAMPLES][ML_CHANNELS];
	copyWindowF32(w);

	if (model == ML_MODEL_LOGREG)
		return predictLogreg(w);
	else
		return predictMlp(w);
}

float mlInferGetCachedProb(MlModel model) {
	if (model == ML_MODEL_LOGREG) {
		const u32 u = __atomic_load_n(&cachedProbLR_u32, __ATOMIC_ACQUIRE);
		return unpackF32(u);
	}
	const u32 u = __atomic_load_n(&cachedProbMLP_u32, __ATOMIC_ACQUIRE);
	return unpackF32(u);
}

u32 mlInferCachedAgeMs() {
	const u32 last = __atomic_load_n(&cachedProbUpdatedMs, __ATOMIC_ACQUIRE);
	if (!last) return 0xFFFFFFFFu;
	const u32 now = millis();
	return (u32)(now - last);
}

void mlInferSlowLoop(bool enable, MlModel model) {
	// Best-effort: loop() frequency varies. We cap compute to ~100Hz.
	static u32 lastMs = 0;
	const u32 now = millis();
	if ((u32)(now - lastMs) < 10) return;
	lastMs = now;

	if (!windowFull) return;

	if (!enable) return;

	const float p = mlInferPredict(model);
	if (model == ML_MODEL_LOGREG) {
		__atomic_store_n(&cachedProbLR_u32, packF32(p), __ATOMIC_RELEASE);
	} else {
		__atomic_store_n(&cachedProbMLP_u32, packF32(p), __ATOMIC_RELEASE);
	}
	__atomic_store_n(&cachedProbUpdatedMs, now, __ATOMIC_RELEASE);
}

bool mlInferUserModelExists() {
	return LittleFS.exists(ML_MODEL_FILENAME) || LittleFS.exists(ML_MODEL_LR_FILENAME) || LittleFS.exists(ML_MODEL_MLP_FILENAME);
}

static bool readExact(File &f, void *buf, size_t n) {
	uint8_t *p = (uint8_t *)buf;
	size_t got = 0;
	while (got < n) {
		const int r = f.read(p + got, n - got);
		if (r <= 0) return false;
		got += (size_t)r;
	}
	return true;
}

static bool loadUserModelFromPath(const char *path, bool expectType, u8 expectedType) {
	if (!LittleFS.begin()) return false;
	File f = LittleFS.open(path, "r");
	if (!f) return false;

	MlModelHeader h{};
	if (!readExact(f, &h, sizeof(h))) {
		f.close();
		return false;
	}
	if (memcmp(h.magic, "MLMD", 4) != 0 || h.version != 1 || h.windowSamples != ML_WINDOW_SAMPLES) {
		f.close();
		return false;
	}
	if (h.modelType > 1) {
		f.close();
		return false;
	}
	if (expectType && h.modelType != expectedType) {
		f.close();
		return false;
	}
	if (h.payloadBytes != expectedPayloadBytes(h.modelType)) {
		f.close();
		return false;
	}
	if (h.modelType == 0) {
		if (h.features != ML_LR_FEATURES) {
			f.close();
			return false;
		}
	} else {
		if (h.features != ML_MLP_FEATURES || h.h1 != ML_MLP_H1 || h.h2 != ML_MLP_H2) {
			f.close();
			return false;
		}
	}

	// Read payload and CRC it as we go.
	u32 crc = 0;
	auto readFloats = [&](float *dst, size_t count) -> bool {
		const size_t bytes = count * sizeof(float);
		if (!readExact(f, dst, bytes)) return false;
		crc = crc32_update(crc, (const uint8_t *)dst, bytes);
		return true;
	};

	bool ok = true;
	if (h.modelType == 0) {
		ok = ok && readFloats(gLrMean, ML_LR_FEATURES);
		ok = ok && readFloats(gLrScale, ML_LR_FEATURES);
		ok = ok && readFloats(gLrCoef, ML_LR_FEATURES);
		ok = ok && readFloats(&gLrIntercept, 1);
	} else {
		ok = ok && readFloats(gMlpMean, ML_MLP_FEATURES);
		ok = ok && readFloats(gMlpScale, ML_MLP_FEATURES);
		ok = ok && readFloats(gMlpW1, ML_MLP_H1 * ML_MLP_FEATURES);
		ok = ok && readFloats(gMlpB1, ML_MLP_H1);
		ok = ok && readFloats(gMlpW2, ML_MLP_H2 * ML_MLP_H1);
		ok = ok && readFloats(gMlpB2, ML_MLP_H2);
		ok = ok && readFloats(gMlpW3, ML_MLP_H2);
		ok = ok && readFloats(&gMlpB3, 1);
	}
	f.close();
	if (!ok || crc != h.payloadCrc32) {
		// Restore the affected model on failure (avoids half-loaded state).
		if (h.modelType == 0)
			resetLrToFactory();
		else
			resetMlpToFactory();
		return false;
	}
	// Mark which model types are actually present in this user file.
	if (h.modelType == 0)
		gUserHasLR = true;
	else
		gUserHasMLP = true;
	gUsingUserModel = gUserHasLR || gUserHasMLP;
	return true;
}

bool mlInferLoadUserModel() {
	return loadUserModelFromPath(ML_MODEL_FILENAME, false, 0);
}

bool mlInferLoadUserModelType(MlModel model) {
	if (model == ML_MODEL_LOGREG) return loadUserModelFromPath(ML_MODEL_LR_FILENAME, true, 0);
	return loadUserModelFromPath(ML_MODEL_MLP_FILENAME, true, 1);
}

bool mlInferDeleteUserModel() {
	if (!LittleFS.begin()) return false;
	if (LittleFS.exists(ML_MODEL_FILENAME) && !LittleFS.remove(ML_MODEL_FILENAME)) return false;
	if (LittleFS.exists(ML_MODEL_LR_FILENAME) && !LittleFS.remove(ML_MODEL_LR_FILENAME)) return false;
	if (LittleFS.exists(ML_MODEL_MLP_FILENAME) && !LittleFS.remove(ML_MODEL_MLP_FILENAME)) return false;

	// Revert to factory weights immediately (so device behavior matches "no user model").
	loadDefaultsIntoRam();
	gUserHasLR = false;
	gUserHasMLP = false;
	gUsingUserModel = false;
	__atomic_store_n(&cachedProbLR_u32, packF32(0.0f), __ATOMIC_RELEASE);
	__atomic_store_n(&cachedProbMLP_u32, packF32(0.0f), __ATOMIC_RELEASE);
	__atomic_store_n(&cachedProbUpdatedMs, 0, __ATOMIC_RELEASE);
	return true;
}

void mlInferPrintInfo() {
	Serial.printf("[ml] model_source=%s\n", gUsingUserModel ? "user" : "factory");
	Serial.printf("[ml] user_model_present=%d\n", mlInferUserModelExists() ? 1 : 0);
	Serial.printf("[ml] user_has_lr=%d\n", gUserHasLR ? 1 : 0);
	Serial.printf("[ml] user_has_mlp=%d\n", gUserHasMLP ? 1 : 0);
	Serial.printf("[ml] cached_age_ms=%lu\n", (u32)mlInferCachedAgeMs());
	// Weight fingerprint for debugging (compare with Python export output).
	Serial.printf("[ml] lr_intercept=%.6f\n", (double)gLrIntercept);
	Serial.printf("[ml] lr_coef[0]=%.6f\n", (double)gLrCoef[0]);
	Serial.printf("[ml] mlp_b3=%.6f\n", (double)gMlpB3);
	Serial.printf("[ml] mlp_w1[0]=%.6f\n", (double)gMlpW1[0]);
}

#endif // HW_VERSION == 2
