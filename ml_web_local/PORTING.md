# Porting the ML Training Pipeline from Python to JavaScript

This document describes how the Python-based ML training pipeline (scikit-learn + numpy + FastAPI backend) was ported to run entirely in the browser as vanilla JavaScript, with no dependencies.

## Why

The original flow required users to install Python, pip packages (scikit-learn, numpy, fastapi, uvicorn, pyserial), and run a local server. For end consumers who just want to personalize their Stinger, that's a non-starter. The browser port eliminates all of that — open a web page, plug in, done.

## What was ported

The Python pipeline lived across several files:

| Python source | JS equivalent | What it does |
|---|---|---|
| `python/ml_log_convert.py` | `train.js: parseBinaryLog()` | Parse 17-byte binary samples from the device log |
| `python/ml_log_explore.py` | `train.js: detectShots(), extractDataset()` | Shot detection (trigger rising edges), positive/negative window extraction |
| `python/ml_model_export.py` | `train.js: featurizeSummary(), featurizeRich()` | Feature engineering (18 summary / 30 rich features) |
| `python/ml_model_export.py` | `train.js: fitScaler(), applyScaler()` | StandardScaler (mean/std normalization) |
| scikit-learn `LogisticRegression` | `train.js: trainLogReg()` | Logistic regression with L2 regularization |
| scikit-learn `MLPClassifier` | `train.js: trainMLP()` | 30→64→32→1 MLP with Adam optimizer |
| `python/ml_model_export.py` | `train.js: buildMlmdLR(), buildMlmdMLP()` | MLMD binary format builder |
| `zlib.crc32` | `train.js: crc32()` | CRC32 (IEEE polynomial) |
| `ml_web/server/app.py` | `app.js` | Orchestration, serial communication, UI |
| matplotlib (in server) | `app.js: drawShotPlot()` | Per-shot prediction + IMU canvas plots |

## Porting decisions

### No linear algebra library

numpy makes Python code concise, but pulling in a JS matrix library (like math.js or TensorFlow.js) would add hundreds of KB and complexity. Instead, all operations are written as explicit loops over `Float32Array`. This is fine because:

- The feature vectors are small (18 or 30 values)
- The MLP weight matrices are modest (largest is 64×30 = 1920 floats)
- Training datasets are tiny (typically 40–200 windows)
- Explicit loops are easy to verify against the C firmware implementation

### Logistic regression from scratch

scikit-learn's `LogisticRegression` uses liblinear or L-BFGS under the hood. For our case (18 features, <1000 samples, L2 regularization), simple gradient descent converges in well under 100ms:

```
for each iteration:
  z = X @ w + b
  p = sigmoid(z)
  error = p - y
  grad_w = (X^T @ error) / n + alpha * w
  grad_b = mean(error)
  w -= lr * grad_w
  b -= lr * grad_b
```

This implementation uses plain gradient descent (no line search / LBFGS). It won’t match scikit-learn exactly, but it produces compatible weights for the firmware and converges quickly on typical per-user datasets.

### MLP with Adam optimizer

scikit-learn's `MLPClassifier` uses Adam by default. The JS port implements the same:

- Forward pass: `h1 = ReLU(X @ W1 + b1)`, `h2 = ReLU(h1 @ W2 + b2)`, `out = sigmoid(h2 @ W3 + b3)`
- Loss: binary cross-entropy with L2 weight penalty
- Backprop: standard chain rule through each layer
- Adam update: per-parameter momentum (`m`) and RMS (`v`) with bias correction

The architecture is fixed at 30→64→32→1 to match the firmware. Xavier initialization seeds the weights. 800 iterations is enough for convergence on these small datasets.

One subtlety: scikit-learn stores weight matrices as `(input, output)` shape, but the firmware reads them as `(output, input)` — row-major with one row per output neuron. The JS `buildMlmdMLP()` writes weights in firmware order (output × input), matching the C code in `mlInfer.cpp` which indexes `gMlpW1[j * ML_MLP_FEATURES + i]`.

### StandardScaler

Trivial port — compute column-wise mean and std, then `(x - mean) / std`. The only gotcha is handling zero-variance columns (divide-by-zero). The JS version clamps std to a minimum of 1e-10, same behavior as scikit-learn.

Note: scikit-learn’s `StandardScaler` effectively uses a scale of `1` for zero-variance features; this JS port does the same.

### Binary log parsing

The Python code used `struct.unpack('<IhhhhhhB', ...)` to parse each 17-byte sample. The JS equivalent uses `DataView` on an `ArrayBuffer`:

```js
const ts = dv.getUint32(off, true);      // little-endian u32
const ax = dv.getInt16(off + 4, true);    // little-endian i16
// ... etc
const trig = dv.getUint8(off + 16);      // u8
```

### Shot detection

Direct port of the Python logic: scan for 0→1 transitions in the trigger byte, merge shots closer than `minGapMs` (default 1000ms). The Python version used pandas-style vectorized ops; the JS version uses a simple for-loop which is equally readable and fast for the data sizes involved.

### Dataset extraction

The most complex part of the port. Key pieces:

1. **Positive windows**: For each accepted shot, take the window `[edgeI - lead - ws, edgeI - lead)` where lead=10 samples (100ms) and ws=50 samples (500ms).

2. **Avoid mask**: Mark all samples within 600ms before or 1000ms after ANY trigger edge as "avoid" zones for negative sampling. This prevents negatives from overlapping with pre-shot or recoil motion.

3. **Negative sampling**: Scan for contiguous runs of non-avoided samples, generate candidate window positions, shuffle with a seeded PRNG, and pick up to N (matching the positive count for 1:1 ratio).

The Python version used numpy boolean indexing and `np.random.RandomState`. The JS version uses a simple xorshift32 PRNG seeded deterministically, and explicit index scanning. The results are equivalent but not identical (different PRNG sequences), which is fine — the training labels and features are the same.

### MLMD binary builder

Direct port using `DataView` for the 24-byte header and `Float32Array` for the payload. The CRC32 implementation uses the standard lookup-table approach with polynomial 0xEDB88320 (same as zlib). Verified by comparing output against Python's `zlib.crc32` on test vectors.

### Plots: matplotlib → Canvas 2D

The Python server generated matplotlib PNG images and sent them as base64. The JS version draws directly to `<canvas>` elements using the Canvas 2D API:

- Two subplots per shot (probability + IMU), same layout as the matplotlib version
- DPR-aware rendering for crisp output on Retina displays
- Color scheme matches the original plots

This was the least mechanical part of the port — matplotlib handles axes, legends, and scaling automatically, while the Canvas version computes all coordinates manually.

### Web Serial (replacing pyserial + FastAPI)

The Python backend used pyserial for device communication and FastAPI for the browser↔server bridge. The JS version uses the Web Serial API directly:

- `navigator.serial.requestPort()` for port selection
- `ReadableStream` with a custom chunked reader for binary data
- `WritableStream` for sending commands
- Line-based protocol parsing (same `\n`-delimited commands as before)

The serial protocol itself (MLDUMP, MLMODEL_PUT, MLMODEL_LOAD, MLMODEL_INFO) is unchanged — the device doesn't know or care whether it's talking to Python or JavaScript.

## What's different

- **No server process** — everything runs in the browser tab
- **No file I/O** — log data and model blobs stay in memory (with download buttons for the user)
- **Deterministic but different** — the seeded PRNG for negative sampling produces different shuffles than numpy, so the exact training set differs. Model weights will be slightly different but equivalent in quality.
- **No batch training** — the Python tools supported training across multiple log files for a generalized model. The browser version trains on a single session. Multi-session batch training is left for the Python tools (users can download raw logs for this).

## Verification

To verify the port produces correct output:

1. Record a session, pull the log via the browser tool
2. Also pull the same log via the Python tools
3. Compare: shot count, feature values (should match exactly), model predictions (should be very close but not identical due to different PRNG seeds and optimizer convergence)
4. Upload the JS-trained model, verify MLMODEL_LOADED and weight fingerprint via MLMODEL_INFO
5. The MLMD binary format is identical — the firmware accepts it without changes
