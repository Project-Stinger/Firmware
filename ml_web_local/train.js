// train.js — Client-side ML training pipeline for Stinger idle prediction.
// Ports the Python pipeline (ml_log_convert, ml_log_explore, ml_model_export) to pure JS.
// No dependencies — runs in any modern browser.

// ── Binary log parsing ──────────────────────────────────────────────────────

const SAMPLE_SIZE = 17; // u32 + 6×i16 + u8

export function parseBinaryLog(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = Math.floor(buf.length / SAMPLE_SIZE);
  const ts = new Uint32Array(n);
  const ax = new Int16Array(n);
  const ay = new Int16Array(n);
  const az = new Int16Array(n);
  const gx = new Int16Array(n);
  const gy = new Int16Array(n);
  const gz = new Int16Array(n);
  const trig = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const off = i * SAMPLE_SIZE;
    ts[i] = dv.getUint32(off, true);
    ax[i] = dv.getInt16(off + 4, true);
    ay[i] = dv.getInt16(off + 6, true);
    az[i] = dv.getInt16(off + 8, true);
    gx[i] = dv.getInt16(off + 10, true);
    gy[i] = dv.getInt16(off + 12, true);
    gz[i] = dv.getInt16(off + 14, true);
    trig[i] = buf[off + 16];
  }
  return { ts, ax, ay, az, gx, gy, gz, trig, n };
}

// ── Shot detection ──────────────────────────────────────────────────────────

export function detectShots(ts, trig, minGapMs = 1000) {
  const risingAll = [];
  for (let i = 1; i < trig.length; i++) {
    if (trig[i - 1] === 0 && trig[i] === 1) risingAll.push(i);
  }
  const accepted = [];
  let lastT = -1e18;
  for (const i of risingAll) {
    if (ts[i] - lastT >= minGapMs) {
      accepted.push(i);
      lastT = ts[i];
    }
  }
  return { risingAll, accepted };
}

// ── Dataset extraction ──────────────────────────────────────────────────────

function bisectLeft(arr, val) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (arr[mid] < val) lo = mid + 1; else hi = mid; }
  return lo;
}

function bisectRight(arr, val) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (arr[mid] <= val) lo = mid + 1; else hi = mid; }
  return lo;
}

// Simple seeded PRNG (xorshift32)
function makeRng(seed) {
  let s = seed | 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
  }
}

export function extractDataset(data, accepted, opts = {}) {
  const {
    windowSamples = 50, leadMs = 100, windowMs = 500,
    negAvoidPostMs = 1000, negMinSepMs = 250, seed = 1,
  } = opts;
  const { ts, ax, ay, az, gx, gy, gz, trig, n } = data;

  // Keep sample counts consistent with firmware:
  // - Firmware window is exactly 50 samples (500ms @ 100Hz)
  // - Lead is exactly 100ms (10 samples @ 100Hz)
  //
  // We still estimate dt from timestamps for sanity checks / logging, but we do not
  // let jitter change the sample counts (otherwise the MLMD header won't match
  // the on-device expectations).
  let dtSum = 0, dtCount = 0;
  for (let i = 1; i < Math.min(n, 500); i++) {
    const d = ts[i] - ts[i - 1];
    if (d > 0 && d < 100) { dtSum += d; dtCount++; }
  }
  const dtEstMs = dtCount > 0 ? dtSum / dtCount : 10;
  const sampleMs = windowMs / windowSamples; // nominal 10ms
  const leadSamples = Math.max(1, Math.round(leadMs / sampleMs));
  const ws = windowSamples;

  // Positive windows: [edgeI - lead - ws, edgeI - lead)
  const posWindows = [];
  for (let si = 0; si < accepted.length; si++) {
    const edgeI = accepted[si];
    const endI = edgeI - leadSamples;
    const startI = endI - ws;
    if (startI < 0 || endI <= startI) continue;
    posWindows.push({ shotNum: si, edgeI, startI, endI });
  }

  // Build avoid mask
  const avoid = new Uint8Array(n);
  const allEdges = [];
  for (let i = 1; i < n; i++) {
    if (trig[i - 1] === 0 && trig[i] === 1) allEdges.push(i);
  }
  for (const edgeI of allEdges) {
    const t0 = ts[edgeI];
    const loT = t0 - (leadMs + windowMs);
    const hiT = t0 + negAvoidPostMs;
    const lo = bisectLeft(ts, loT);
    const hi = bisectRight(ts, hiT);
    for (let j = lo; j < hi && j < n; j++) avoid[j] = 1;
  }

  // Prefix sum for "does this whole window overlap an avoided region?"
  const avoidPref = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) avoidPref[i + 1] = avoidPref[i] + avoid[i];
  function windowHasAvoid(startI, endI) {
    return (avoidPref[endI] - avoidPref[startI]) !== 0;
  }

  // Far negative candidates
  const rng = makeRng(seed);
  const negTarget = posWindows.length; // 1:1 ratio
  const candidates = [];
  for (let endI = ws; endI < n; endI++) {
    const startI = endI - ws;
    if (startI < 0) continue;
    if (windowHasAvoid(startI, endI)) continue;
    let hasTrig = false;
    for (let j = startI; j < endI; j++) { if (trig[j]) { hasTrig = true; break; } }
    if (hasTrig) continue;
    candidates.push(endI);
  }
  shuffle(candidates, rng);

  // Sample with min separation
  const negEnds = [];
  const chosenTs = [];
  for (const endI of candidates) {
    if (negEnds.length >= negTarget) break;
    const t = ts[endI];
    let tooClose = false;
    for (const ct of chosenTs) { if (Math.abs(t - ct) < negMinSepMs) { tooClose = true; break; } }
    if (tooClose) continue;
    negEnds.push(endI);
    chosenTs.push(t);
  }

  // Build X (windows) and y (labels)
  const totalWindows = posWindows.length + negEnds.length;
  const X = new Float32Array(totalWindows * ws * 6);
  const y = new Uint8Array(totalWindows);

  function writeWindow(wi, startI) {
    const base = wi * ws * 6;
    for (let i = 0; i < ws; i++) {
      const si = startI + i;
      const off = base + i * 6;
      X[off] = ax[si]; X[off + 1] = ay[si]; X[off + 2] = az[si];
      X[off + 3] = gx[si]; X[off + 4] = gy[si]; X[off + 5] = gz[si];
    }
  }

  let wi = 0;
  for (const pw of posWindows) { writeWindow(wi, pw.startI); y[wi] = 1; wi++; }
  for (const endI of negEnds) { writeWindow(wi, endI - ws); y[wi] = 0; wi++; }

  return {
    X, y,
    totalWindows,
    windowSamples: ws,
    leadSamples,
    dtEstMs,
    channels: 6,
    posCount: posWindows.length,
    negCount: negEnds.length,
  };
}

// ── Featurization ───────────────────────────────────────────────────────────

export function featurizeSummary(X, nWindows, ws, channels) {
  // mean(6) + std(6) + absmax(6) = 18
  const nFeat = 18;
  const F = new Float32Array(nWindows * nFeat);
  for (let w = 0; w < nWindows; w++) {
    const wBase = w * ws * channels;
    const fBase = w * nFeat;
    for (let c = 0; c < channels; c++) {
      let sum = 0, sumSq = 0, mx = 0;
      for (let i = 0; i < ws; i++) {
        const v = X[wBase + i * channels + c];
        sum += v; sumSq += v * v;
        const av = Math.abs(v); if (av > mx) mx = av;
      }
      const mean = sum / ws;
      let vr = sumSq / ws - mean * mean; if (vr < 0) vr = 0;
      F[fBase + c] = mean;
      F[fBase + c + 6] = Math.sqrt(vr);
      F[fBase + c + 12] = mx;
    }
  }
  return { F, nFeat };
}

export function featurizeRich(X, nWindows, ws, channels) {
  // mean(6) + std(6) + absmax(6) + absmean(6) + amag(3) + gmag(3) = 30
  const nFeat = 30;
  const F = new Float32Array(nWindows * nFeat);
  for (let w = 0; w < nWindows; w++) {
    const wBase = w * ws * channels;
    const fBase = w * nFeat;
    for (let c = 0; c < channels; c++) {
      let sum = 0, sumSq = 0, mx = 0, absSum = 0;
      for (let i = 0; i < ws; i++) {
        const v = X[wBase + i * channels + c];
        sum += v; sumSq += v * v;
        const av = Math.abs(v); if (av > mx) mx = av; absSum += av;
      }
      const mean = sum / ws;
      let vr = sumSq / ws - mean * mean; if (vr < 0) vr = 0;
      F[fBase + c] = mean;
      F[fBase + c + 6] = Math.sqrt(vr);
      F[fBase + c + 12] = mx;
      F[fBase + c + 18] = absSum / ws;
    }
    let aSumM = 0, aSumSq = 0, aMax = 0, gSumM = 0, gSumSq = 0, gMax = 0;
    for (let i = 0; i < ws; i++) {
      const off = wBase + i * channels;
      const a2 = X[off] * X[off] + X[off + 1] * X[off + 1] + X[off + 2] * X[off + 2];
      const g2 = X[off + 3] * X[off + 3] + X[off + 4] * X[off + 4] + X[off + 5] * X[off + 5];
      const am = Math.sqrt(a2), gm = Math.sqrt(g2);
      aSumM += am; aSumSq += a2; if (am > aMax) aMax = am;
      gSumM += gm; gSumSq += g2; if (gm > gMax) gMax = gm;
    }
    const aMean = aSumM / ws;
    let aVar = aSumSq / ws - aMean * aMean; if (aVar < 0) aVar = 0;
    const gMean = gSumM / ws;
    let gVar = gSumSq / ws - gMean * gMean; if (gVar < 0) gVar = 0;
    F[fBase + 24] = aMean;
    F[fBase + 25] = Math.sqrt(aVar);
    F[fBase + 26] = aMax;
    F[fBase + 27] = gMean;
    F[fBase + 28] = Math.sqrt(gVar);
    F[fBase + 29] = gMax;
  }
  return { F, nFeat };
}

// ── StandardScaler ──────────────────────────────────────────────────────────

export function fitScaler(F, nRows, nFeat) {
  const mean = new Float32Array(nFeat);
  const scale = new Float32Array(nFeat);
  for (let j = 0; j < nFeat; j++) {
    let s = 0, s2 = 0;
    for (let i = 0; i < nRows; i++) { const v = F[i * nFeat + j]; s += v; s2 += v * v; }
    const m = s / nRows;
    let v = s2 / nRows - m * m; if (v < 0) v = 0;
    mean[j] = m;
    scale[j] = Math.sqrt(v) || 1;
  }
  return { mean, scale };
}

export function applyScaler(F, nRows, nFeat, mean, scale) {
  const out = new Float32Array(F.length);
  for (let i = 0; i < nRows; i++)
    for (let j = 0; j < nFeat; j++)
      out[i * nFeat + j] = (F[i * nFeat + j] - mean[j]) / scale[j];
  return out;
}

// ── Logistic Regression (gradient descent with L2) ──────────────────────────

function sigmoid(x) {
  if (x > 20) return 1;
  if (x < -20) return 0;
  return 1 / (1 + Math.exp(-x));
}

export function trainLogReg(F, y, nRows, nFeat, { maxIter = 5000, lr = 1.0, l2 = 1.0 } = {}) {
  const coef = new Float32Array(nFeat);
  let intercept = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    const gCoef = new Float32Array(nFeat);
    let gIntercept = 0;

    for (let i = 0; i < nRows; i++) {
      let z = intercept;
      for (let j = 0; j < nFeat; j++) z += F[i * nFeat + j] * coef[j];
      const p = sigmoid(z);
      const err = p - y[i];
      for (let j = 0; j < nFeat; j++) gCoef[j] += err * F[i * nFeat + j];
      gIntercept += err;
    }

    for (let j = 0; j < nFeat; j++) gCoef[j] = gCoef[j] / nRows + l2 * coef[j];
    gIntercept /= nRows;

    const step = lr / (1 + iter * 0.001);
    for (let j = 0; j < nFeat; j++) coef[j] -= step * gCoef[j];
    intercept -= step * gIntercept;
  }

  return { coef, intercept };
}

// ── MLP (64→32→1) with Adam optimizer ───────────────────────────────────────

function heInit(rows, cols, rng) {
  const w = new Float32Array(rows * cols);
  const std = Math.sqrt(2 / cols);
  for (let i = 0; i < w.length; i++) {
    const u1 = rng() || 1e-10, u2 = rng();
    w[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std;
  }
  return w;
}

export function trainMLP(F, y, nRows, nFeat, {
  h1Size = 64, h2Size = 32, maxIter = 800, alpha = 1e-4, learningRate = 0.001, seed = 42,
} = {}) {
  const rng = makeRng(seed);

  let w1 = heInit(h1Size, nFeat, rng);
  let b1 = new Float32Array(h1Size);
  let w2 = heInit(h2Size, h1Size, rng);
  let b2 = new Float32Array(h2Size);
  let w3 = heInit(1, h2Size, rng);
  let b3 = 0;

  const adamState = (len) => ({ m: new Float32Array(len), v: new Float32Array(len) });
  const s_w1 = adamState(w1.length), s_b1 = adamState(b1.length);
  const s_w2 = adamState(w2.length), s_b2 = adamState(b2.length);
  const s_w3 = adamState(w3.length);
  let s_b3_m = 0, s_b3_v = 0;

  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

  function adamUpdate(param, grad, state, t) {
    for (let i = 0; i < param.length; i++) {
      state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i];
      state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i] * grad[i];
      const mHat = state.m[i] / (1 - Math.pow(beta1, t));
      const vHat = state.v[i] / (1 - Math.pow(beta2, t));
      param[i] -= learningRate * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  for (let iter = 0; iter < maxIter; iter++) {
    const t = iter + 1;
    const gw1 = new Float32Array(w1.length);
    const gb1 = new Float32Array(h1Size);
    const gw2 = new Float32Array(w2.length);
    const gb2 = new Float32Array(h2Size);
    const gw3 = new Float32Array(w3.length);
    let gb3 = 0;

    for (let s = 0; s < nRows; s++) {
      const x = F.subarray(s * nFeat, s * nFeat + nFeat);

      const h1 = new Float32Array(h1Size);
      for (let j = 0; j < h1Size; j++) {
        let sum = b1[j];
        const wOff = j * nFeat;
        for (let k = 0; k < nFeat; k++) sum += w1[wOff + k] * x[k];
        h1[j] = sum > 0 ? sum : 0;
      }

      const h2 = new Float32Array(h2Size);
      for (let j = 0; j < h2Size; j++) {
        let sum = b2[j];
        const wOff = j * h1Size;
        for (let k = 0; k < h1Size; k++) sum += w2[wOff + k] * h1[k];
        h2[j] = sum > 0 ? sum : 0;
      }

      let z = b3;
      for (let k = 0; k < h2Size; k++) z += w3[k] * h2[k];
      const p = sigmoid(z);

      const dz = p - y[s];
      for (let k = 0; k < h2Size; k++) gw3[k] += dz * h2[k];
      gb3 += dz;

      const dh2 = new Float32Array(h2Size);
      for (let j = 0; j < h2Size; j++) dh2[j] = dz * w3[j] * (h2[j] > 0 ? 1 : 0);
      for (let j = 0; j < h2Size; j++) {
        const wOff = j * h1Size;
        for (let k = 0; k < h1Size; k++) gw2[wOff + k] += dh2[j] * h1[k];
        gb2[j] += dh2[j];
      }

      const dh1 = new Float32Array(h1Size);
      for (let j = 0; j < h1Size; j++) {
        let sum = 0;
        for (let k = 0; k < h2Size; k++) sum += dh2[k] * w2[k * h1Size + j];
        dh1[j] = sum * (h1[j] > 0 ? 1 : 0);
      }
      for (let j = 0; j < h1Size; j++) {
        const wOff = j * nFeat;
        for (let k = 0; k < nFeat; k++) gw1[wOff + k] += dh1[j] * x[k];
        gb1[j] += dh1[j];
      }
    }

    const invN = 1 / nRows;
    for (let i = 0; i < gw1.length; i++) gw1[i] = gw1[i] * invN + alpha * w1[i];
    for (let i = 0; i < gb1.length; i++) gb1[i] *= invN;
    for (let i = 0; i < gw2.length; i++) gw2[i] = gw2[i] * invN + alpha * w2[i];
    for (let i = 0; i < gb2.length; i++) gb2[i] *= invN;
    for (let i = 0; i < gw3.length; i++) gw3[i] = gw3[i] * invN + alpha * w3[i];
    gb3 = gb3 * invN;

    adamUpdate(w1, gw1, s_w1, t);
    adamUpdate(b1, gb1, s_b1, t);
    adamUpdate(w2, gw2, s_w2, t);
    adamUpdate(b2, gb2, s_b2, t);
    adamUpdate(w3, gw3, s_w3, t);
    s_b3_m = beta1 * s_b3_m + (1 - beta1) * gb3;
    s_b3_v = beta2 * s_b3_v + (1 - beta2) * gb3 * gb3;
    const mHat3 = s_b3_m / (1 - Math.pow(beta1, t));
    const vHat3 = s_b3_v / (1 - Math.pow(beta2, t));
    b3 -= learningRate * mHat3 / (Math.sqrt(vHat3) + eps);
  }

  return { w1, b1, w2, b2, w3, b3 };
}

// ── CRC32 (IEEE, matches zlib.crc32) ────────────────────────────────────────

export function crc32(u8) {
  let crc = 0xffffffff;
  for (let i = 0; i < u8.length; i++) {
    crc ^= u8[i];
    for (let j = 0; j < 8; j++) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

// ── MLMD binary builder ─────────────────────────────────────────────────────

function f32LEBytes(arr) {
  const buf = new ArrayBuffer(arr.length * 4);
  const dv = new DataView(buf);
  for (let i = 0; i < arr.length; i++) dv.setFloat32(i * 4, arr[i], true);
  return new Uint8Array(buf);
}

function concatU8(arrays) {
  let total = 0;
  for (const a of arrays) total += a.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const a of arrays) { out.set(a, off); off += a.length; }
  return out;
}

export function buildMlmdLR(windowSamples, scalerMean, scalerScale, coef, intercept) {
  const nFeat = coef.length;
  if (windowSamples !== 50) throw new Error(`LR: windowSamples must be 50, got ${windowSamples}`);
  if (nFeat !== 18) throw new Error(`LR: features must be 18, got ${nFeat}`);
  const payload = concatU8([
    f32LEBytes(scalerMean), f32LEBytes(scalerScale),
    f32LEBytes(coef), f32LEBytes(new Float32Array([intercept])),
  ]);
  const payloadCrc = crc32(payload);
  const header = new ArrayBuffer(24);
  const hd = new DataView(header);
  hd.setUint8(0, 0x4d); hd.setUint8(1, 0x4c); hd.setUint8(2, 0x4d); hd.setUint8(3, 0x44);
  hd.setUint16(4, 1, true); hd.setUint16(6, windowSamples, true);
  hd.setUint8(8, 0); hd.setUint8(9, 0);
  hd.setUint16(10, nFeat, true); hd.setUint16(12, 0, true); hd.setUint16(14, 0, true);
  hd.setUint32(16, payload.length, true); hd.setUint32(20, payloadCrc, true);
  return concatU8([new Uint8Array(header), payload]);
}

export function buildMlmdMLP(windowSamples, scalerMean, scalerScale, w1, b1, w2, b2, w3, b3Val, nFeat, h1Size, h2Size) {
  if (windowSamples !== 50) throw new Error(`MLP: windowSamples must be 50, got ${windowSamples}`);
  if (nFeat !== 30) throw new Error(`MLP: features must be 30, got ${nFeat}`);
  if (h1Size !== 64 || h2Size !== 32) throw new Error(`MLP: hidden sizes must be 64,32; got ${h1Size},${h2Size}`);
  const payload = concatU8([
    f32LEBytes(scalerMean), f32LEBytes(scalerScale),
    f32LEBytes(w1), f32LEBytes(b1),
    f32LEBytes(w2), f32LEBytes(b2),
    f32LEBytes(w3), f32LEBytes(new Float32Array([b3Val])),
  ]);
  const payloadCrc = crc32(payload);
  const header = new ArrayBuffer(24);
  const hd = new DataView(header);
  hd.setUint8(0, 0x4d); hd.setUint8(1, 0x4c); hd.setUint8(2, 0x4d); hd.setUint8(3, 0x44);
  hd.setUint16(4, 1, true); hd.setUint16(6, windowSamples, true);
  hd.setUint8(8, 1); hd.setUint8(9, 0);
  hd.setUint16(10, nFeat, true); hd.setUint16(12, h1Size, true); hd.setUint16(14, h2Size, true);
  hd.setUint32(16, payload.length, true); hd.setUint32(20, payloadCrc, true);
  return concatU8([new Uint8Array(header), payload]);
}

// ── Prediction (for plotting) ───────────────────────────────────────────────

export function predictLRSingle(feat, scalerMean, scalerScale, coef, intercept) {
  let z = intercept;
  for (let j = 0; j < coef.length; j++)
    z += ((feat[j] - scalerMean[j]) / scalerScale[j]) * coef[j];
  return sigmoid(z);
}

export function predictMLPSingle(feat, scalerMean, scalerScale, w1, b1, w2, b2, w3, b3Val, nFeat, h1Size, h2Size) {
  const x = new Float32Array(nFeat);
  for (let j = 0; j < nFeat; j++) x[j] = (feat[j] - scalerMean[j]) / scalerScale[j];
  const h1 = new Float32Array(h1Size);
  for (let j = 0; j < h1Size; j++) {
    let sum = b1[j]; for (let k = 0; k < nFeat; k++) sum += w1[j * nFeat + k] * x[k];
    h1[j] = sum > 0 ? sum : 0;
  }
  const h2 = new Float32Array(h2Size);
  for (let j = 0; j < h2Size; j++) {
    let sum = b2[j]; for (let k = 0; k < h1Size; k++) sum += w2[j * h1Size + k] * h1[k];
    h2[j] = sum > 0 ? sum : 0;
  }
  let z = b3Val;
  for (let k = 0; k < h2Size; k++) z += w3[k] * h2[k];
  return sigmoid(z);
}

// ── Full training pipeline ──────────────────────────────────────────────────

export function trainPipeline(logBytes, onLog = () => {}) {
  onLog("Parsing binary log...");
  const data = parseBinaryLog(logBytes);
  onLog(`Parsed ${data.n} samples (${(data.n / 100).toFixed(1)}s at 100Hz)`);

  onLog("Detecting shots...");
  const { risingAll, accepted } = detectShots(data.ts, data.trig);
  onLog(`Shots: ${risingAll.length} total, ${accepted.length} accepted`);

  if (accepted.length < 2)
    throw new Error(`Need at least 2 valid shots to train, got ${accepted.length}. Record more shots with gaps > 1s.`);

  onLog("Extracting training windows...");
  const ds = extractDataset(data, accepted);
  onLog(`Dataset: ${ds.posCount} positive, ${ds.negCount} negative windows (${ds.windowSamples} samples each)`);
  if (Math.abs(ds.dtEstMs - 10) > 2) onLog(`WARNING: estimated dt is ${ds.dtEstMs.toFixed(2)}ms (expected ~10ms). Training still uses fixed 50-sample windows.`);

  if (ds.posCount === 0 || ds.negCount === 0)
    throw new Error("Not enough training data. Need both positive and negative windows.");

  // Featurize once (unscaled), then:
  // 1) compute quick held-out metrics (train/test split)
  // 2) train final models on ALL data for export/upload
  onLog("Featurizing...");
  const { F: F_lr_raw, nFeat: nfLR } = featurizeSummary(ds.X, ds.totalWindows, ds.windowSamples, ds.channels);
  const { F: F_mlp_raw, nFeat: nfMLP } = featurizeRich(ds.X, ds.totalWindows, ds.windowSamples, ds.channels);

  onLog("Evaluating (held-out split)...");
  const metrics = evalHeldOut(F_lr_raw, nfLR, F_mlp_raw, nfMLP, ds.y, ds.totalWindows, 0.25, 1337);
  onLog(`LR metrics (test): P=${(metrics.lr.precision * 100).toFixed(1)} R=${(metrics.lr.recall * 100).toFixed(1)} F1=${(metrics.lr.f1 * 100).toFixed(1)}`);
  onLog(`MLP metrics (test): P=${(metrics.mlp.precision * 100).toFixed(1)} R=${(metrics.mlp.recall * 100).toFixed(1)} F1=${(metrics.mlp.f1 * 100).toFixed(1)}`);

  onLog("Training LogReg (final, all data)...");
  const scalerLR = fitScaler(F_lr_raw, ds.totalWindows, nfLR);
  const F_lr_s = applyScaler(F_lr_raw, ds.totalWindows, nfLR, scalerLR.mean, scalerLR.scale);
  const lr = trainLogReg(F_lr_s, ds.y, ds.totalWindows, nfLR);

  onLog("Training MLP (final, all data)...");
  const scalerMLP = fitScaler(F_mlp_raw, ds.totalWindows, nfMLP);
  const F_mlp_s = applyScaler(F_mlp_raw, ds.totalWindows, nfMLP, scalerMLP.mean, scalerMLP.scale);
  const mlp = trainMLP(F_mlp_s, ds.y, ds.totalWindows, nfMLP);

  onLog("Building MLMD binaries...");
  const lrBlob = buildMlmdLR(ds.windowSamples, scalerLR.mean, scalerLR.scale, lr.coef, lr.intercept);
  const mlpBlob = buildMlmdMLP(
    ds.windowSamples, scalerMLP.mean, scalerMLP.scale,
    mlp.w1, mlp.b1, mlp.w2, mlp.b2, mlp.w3, mlp.b3, nfMLP, 64, 32,
  );
  onLog(`LR model: ${lrBlob.length} bytes, CRC=${crc32(lrBlob).toString(16)}`);
  onLog(`MLP model: ${mlpBlob.length} bytes, CRC=${crc32(mlpBlob).toString(16)}`);

  onLog("Computing per-shot predictions...");
  const shotPlots = computeShotPlots(data, accepted, ds.windowSamples, ds.leadSamples,
    scalerLR, lr, nfLR, scalerMLP, mlp, nfMLP);
  onLog(`Generated ${shotPlots.length} shot plots`);

  return { data, summary: {
    samples: data.n,
    durationMs: data.n > 0 ? data.ts[data.n - 1] - data.ts[0] : 0,
    shotsAll: risingAll.length, shotsAccepted: accepted.length,
    posWindows: ds.posCount, negWindows: ds.negCount,
  }, metrics, lrBlob, mlpBlob, shotPlots };
}

// ── Per-shot plot data ──────────────────────────────────────────────────────

function computeShotPlots(data, accepted, windowSamples, leadSamples, scalerLR, lr, nfLR, scalerMLP, mlp, nfMLP) {
  const { ts, ax, ay, az, gx, gy, gz } = data;
  const n = data.n;
  const maxShots = 2;

  const scored = [];
  for (const trigI of accepted) {
    const t0 = ts[trigI];
    const i0 = Math.max(0, bisectLeft(ts, t0 - 1000));
    const i1 = Math.min(n, bisectRight(ts, t0 + 200));
    if (i1 - i0 < windowSamples + 2) continue;

    const xMs = [], probsLR = [], probsMLP = [];
    const axW = [], ayW = [], azW = [], gxW = [], gyW = [], gzW = [];

    for (let i = i0; i < i1; i++) {
      xMs.push(ts[i] - t0);
      axW.push(ax[i]); ayW.push(ay[i]); azW.push(az[i]);
      gxW.push(gx[i]); gyW.push(gy[i]); gzW.push(gz[i]);

      // Align with training objective:
      // probability at time t uses the window ending leadSamples earlier (i - leadSamples).
      const endI = i - leadSamples;
      if (endI < windowSamples) { probsLR.push(0); probsMLP.push(0); continue; }
      const win = new Float32Array(windowSamples * 6);
      for (let j = 0; j < windowSamples; j++) {
        const si = endI - windowSamples + j;
        win[j * 6] = ax[si]; win[j * 6 + 1] = ay[si]; win[j * 6 + 2] = az[si];
        win[j * 6 + 3] = gx[si]; win[j * 6 + 4] = gy[si]; win[j * 6 + 5] = gz[si];
      }
      const { F: fLR } = featurizeSummary(win, 1, windowSamples, 6);
      const { F: fMLP } = featurizeRich(win, 1, windowSamples, 6);
      probsLR.push(predictLRSingle(fLR, scalerLR.mean, scalerLR.scale, lr.coef, lr.intercept));
      probsMLP.push(predictMLPSingle(fMLP, scalerMLP.mean, scalerMLP.scale,
        mlp.w1, mlp.b1, mlp.w2, mlp.b2, mlp.w3, mlp.b3, nfMLP, 64, 32));
    }

    const scoreLR = scoreSingle(xMs, probsLR);
    const scoreMLP = scoreSingle(xMs, probsMLP);
    const scoreBoth = Math.min(scoreLR, scoreMLP);
    const scoreEither = Math.max(scoreLR, scoreMLP);
    scored.push({ trigI, t0, xMs, probsLR, probsMLP, axW, ayW, azW, gxW, gyW, gzW, scoreLR, scoreMLP, scoreBoth, scoreEither });
  }

  const byTrig = new Map();
  function addUnique(list) {
    for (const s of list) {
      if (byTrig.has(s.trigI)) continue;
      byTrig.set(s.trigI, s);
      if (byTrig.size >= maxShots) break;
    }
  }

  // Prefer examples where BOTH models show a clear low→high transition.
  const bothGood = scored.filter((s) => s.scoreBoth > 0).sort((a, b) => b.scoreBoth - a.scoreBoth);
  addUnique(bothGood);

  // Fallback: if not enough, allow shots where at least one model is clear.
  if (byTrig.size < maxShots) {
    const eitherGood = scored.filter((s) => s.scoreEither > 0).sort((a, b) => b.scoreEither - a.scoreEither);
    addUnique(eitherGood);
  }

  return Array.from(byTrig.values());
}

function scoreSingle(xMs, probs) {
  const basePts = [], predPts = [];
  for (let i = 0; i < xMs.length; i++) {
    const x = xMs[i];
    if (x >= -1000 && x <= -700) basePts.push(probs[i]);
    if (x >= -600 && x <= -100) predPts.push(probs[i]);
  }
  if (basePts.length < 3 || predPts.length < 3) return 0;
  basePts.sort((a, b) => a - b);
  predPts.sort((a, b) => a - b);
  const pLow = pctile(basePts, 10);
  const pHigh = pctile(predPts, 90);
  const delta = pHigh - pLow;
  if (pLow > 0.25 || pHigh < 0.65 || delta < 0.40) return 0;
  return delta + 0.25 * pHigh - 0.10 * pLow;
}

function pctile(sorted, p) {
  if (!sorted.length) return 0;
  const k = (sorted.length - 1) * (p / 100);
  const f = Math.floor(k), c = Math.ceil(k);
  if (f === c) return sorted[f];
  return sorted[f] * (c - k) + sorted[c] * (k - f);
}
