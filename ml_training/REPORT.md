# ML Training Report (latest)

Date: 2026-02-01

This file is the “what changed + do the results make sense?” overview, tying together:
- dataset export (`python/ml_log_explore.py`)
- offline benchmarks (`ml_training/benchmark.py`)
- online/streaming prediction plots (`ml_training/plot_predictions.py`)

For firmware + on-device inference + model upload format, see `ML.md`.

## What the model predicts (current objective)

The training set is a **window classification** problem:
- Input: a **500ms window** of IMU data (6 axes), ending **100ms before** a trigger pull.
- Output: `p(about_to_shoot)` for that window (binary label).

Positive window definition:
- For a trigger rising edge at time `t_shot`, a window is positive if it covers:
  - `[t_shot-600ms, t_shot-100ms]` (i.e., `window_ms=500`, `lead_ms=100`).

In `ml_training/plot_predictions.py`, the probability at timeline timestamp `t` is computed using the window that ends at `t-100ms`. So a high probability at `t` means “a shot is likely at `t`”.

## Dataset (this run)

From `ml_out/summary.json`:
- samples: 18050 (~182.9s @ ~100Hz)
- trigger rising edges: 88 total, 46 “accepted” (>=1000ms apart), 42 rejected

Export command used (mixed near+far negatives, motion-matched + de-duplicated):
```sh
./.venv/bin/python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset \
  --lead-ms 100 --window-ms 500 --min-shot-gap-ms 1000 \
  --neg-strategy mixed --neg-mixed-near-frac 0.6 --neg-mult 8 \
  --neg-min-motion-quantile 0.2 \
  --neg-match-pos-motion --neg-pos-motion-q 0.05 \
  --neg-min-sep-ms 200 --neg-with-replacement \
  --neg-avoid-edges all --neg-avoid-post-ms 2000 \
  --shot-pre-ms 1200 --shot-post-ms 200 --dpi 200
```

From `ml_out/train/meta.json`:
- rows: 243 (pos=46, neg=197)
- PR AUC baseline (pos rate): 0.189
- neg_strategy: `mixed` (near+far)

Note: `python/ml_log_explore.py` now writes `shot_num` into `ml_out/train/index.csv` so we can evaluate using a **group split by shot** (no mixing windows from the same shot across train/test).

## Offline benchmark results

See also: `ml_training/report_latest.md` (raw output from the benchmark script).

| model | features | split | runs | roc_auc | pr_auc | best_f1 |
|---|---:|---:|---:|---:|---:|---:|
| logreg | summary | random | 30 | 0.975±0.015 | 0.901±0.054 | 0.870±0.051 |
| logreg | summary | group | 30 | 0.969±0.022 | 0.884±0.090 | 0.868±0.085 |
| logreg | summary | time | 30 | 0.990±0.000 | 0.954±0.000 | 0.909±0.000 |
| logreg | rich | random | 30 | 0.971±0.014 | 0.892±0.053 | 0.857±0.051 |
| logreg | rich | group | 30 | 0.968±0.020 | 0.887±0.085 | 0.860±0.080 |
| logreg | rich | time | 30 | 0.986±0.000 | 0.906±0.000 | 0.952±0.000 |
| ridge | summary | random | 30 | 0.968±0.018 | 0.856±0.078 | 0.843±0.063 |
| ridge | summary | group | 30 | 0.958±0.027 | 0.836±0.102 | 0.840±0.089 |
| ridge | summary | time | 30 | 0.988±0.000 | 0.926±0.000 | 0.952±0.000 |
| mlp_flat | flatten | random | 30 | 0.940±0.042 | 0.837±0.069 | 0.838±0.062 |
| mlp_flat | flatten | group | 30 | 0.933±0.041 | 0.820±0.098 | 0.821±0.084 |
| mlp_flat | flatten | time | 30 | 0.987±0.007 | 0.943±0.031 | 0.896±0.032 |
| mlp_rich | rich | random | 30 | 0.934±0.053 | 0.869±0.066 | 0.835±0.065 |
| mlp_rich | rich | group | 30 | 0.951±0.039 | 0.878±0.090 | 0.844±0.093 |
| mlp_rich | rich | time | 30 | 0.966±0.009 | 0.818±0.047 | 0.837±0.041 |
| rf | rich | random | 30 | 0.969±0.020 | 0.894±0.062 | 0.859±0.066 |
| rf | rich | group | 30 | 0.967±0.026 | 0.893±0.094 | 0.862±0.085 |
| rf | rich | time | 30 | 0.985±0.003 | 0.920±0.018 | 0.902±0.016 |
| et | rich | random | 30 | 0.972±0.018 | 0.906±0.057 | 0.866±0.070 |
| et | rich | group | 30 | 0.973±0.020 | 0.915±0.080 | 0.879±0.081 |
| et | rich | time | 30 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| hgb | rich | random | 30 | 0.974±0.015 | 0.908±0.053 | 0.853±0.073 |
| hgb | rich | group | 30 | 0.971±0.028 | 0.901±0.090 | 0.852±0.105 |
| hgb | rich | time | 30 | 0.988±0.000 | 0.926±0.000 | 0.952±0.000 |

Sanity check (shuffled labels):
- logreg+summary (random split): ROC AUC=0.524, PR AUC=0.285 (near chance, good sign)

Interpretation:
- Offline metrics are strong across several models.
- `group` split is the one to look at most (it prevents mixing windows from the same shot across train/test).
- `time` split can still look “too good” for some models (especially trees); treat it as a weak signal.

## Online/streaming behavior (why “always high” can happen)

Even with good offline metrics, online plots can look bad when negatives don’t represent “aiming but no shot”.

This run’s multi-model comparison plot folder:
- `ml_out/predictions_compare_latest/overview.png`
- `ml_out/predictions_compare_latest/shots/shot_*.png` (10 shots; top: probabilities for each model, bottom: 6-axis IMU)
- `ml_out/predictions_compare_latest/negatives/neg_*.png`
- `ml_out/predictions_compare_latest/pos_vs_neg_hist.png`

Quick “neutral time” check (timestamps >2.0s before and >1.2s after any accepted shot; threshold=0.5):
- `logreg:summary`: ~2.0% of neutral points >=0.5 (median p~0.0008)
- `et:rich`: ~1.7% of neutral points >=0.5 (median p~0.077)
- `mlp:rich`: ~1.9% of neutral points >=0.5 (median p~0.0000)

Timing inside the labeled pre-shot window `[t_shot-600ms, t_shot-100ms]` (threshold=0.5):
- `logreg:summary`: missed 7/46 windows; among hits, first>=0.5 at median ~468ms before shot
- `et:rich`: missed 1/46; first>=0.5 at median ~531ms before shot
- `mlp:rich`: missed 1/46; first>=0.5 at median ~449ms before shot

## Recommended next steps

1) Collect more **hard negatives**: “aiming/handling but intentionally not shooting” (this is the main fix for plateaus).
2) Add an online post-process (for your “idle mode” control): EMA smoothing + hysteresis (on/off thresholds) + minimum-on duration.
3) Consider changing the objective later:
   - either predict a continuous “time-to-shot” proxy (regression),
   - or a streaming label (binary per timestep) with explicit false-positive penalties.
