#include "global.h"

#if HW_VERSION == 2

#include "mlWeights.h"
#include <cmath>
#include <cstring>

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
		feat[i] = (feat[i] - mlLrScalerMean[i]) / mlLrScalerScale[i];

	// dot product + intercept
	float z = mlLrIntercept;
	for (u32 i = 0; i < ML_LR_FEATURES; i++)
		z += feat[i] * mlLrCoef[i];

	return sigmoid(z);
}

static float predictMlp(const float w[ML_WINDOW_SAMPLES][ML_CHANNELS]) {
	float feat[ML_MLP_FEATURES];
	featurizeRich(w, feat);

	// StandardScaler
	for (u32 i = 0; i < ML_MLP_FEATURES; i++)
		feat[i] = (feat[i] - mlMlpScalerMean[i]) / mlMlpScalerScale[i];

	// Layer 1: 30 -> 64, ReLU
	float h1[ML_MLP_H1];
	for (u32 j = 0; j < ML_MLP_H1; j++) {
		float sum = mlMlpB1[j];
		const float *wRow = &mlMlpW1[j * ML_MLP_FEATURES];
		for (u32 i = 0; i < ML_MLP_FEATURES; i++)
			sum += wRow[i] * feat[i];
		h1[j] = sum > 0 ? sum : 0; // ReLU
	}

	// Layer 2: 64 -> 32, ReLU
	float h2[ML_MLP_H2];
	for (u32 j = 0; j < ML_MLP_H2; j++) {
		float sum = mlMlpB2[j];
		const float *wRow = &mlMlpW2[j * ML_MLP_H1];
		for (u32 i = 0; i < ML_MLP_H1; i++)
			sum += wRow[i] * h1[i];
		h2[j] = sum > 0 ? sum : 0; // ReLU
	}

	// Output: 32 -> 1, sigmoid
	float z = mlMlpB3;
	for (u32 i = 0; i < ML_MLP_H2; i++)
		z += mlMlpW3[i] * h2[i];

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

#endif // HW_VERSION == 2
