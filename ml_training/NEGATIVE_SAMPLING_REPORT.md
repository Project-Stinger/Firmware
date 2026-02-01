# Negative Sampling Upgrade Report

This report documents changes made to address “unstable / always-positive” model behavior that is often caused by unrealistic negative sampling, plus the updated benchmark/prediction outputs on the current `ml_log.csv`.

Date: 2026-02-01 (updated with mixed negatives + shot grouping)

For firmware behavior, inference integration, and model upload format, see `ML.md`.

## What was wrong (symptoms)

- Prediction plots showed long plateaus where models output high probabilities outside the labeled pre-shot interval.
- “Time split” results could appear unrealistically strong (even perfect), which is not a trustworthy generalization test when negatives are sampled relative to shot timing.
- Negative windows could be overly repetitive (overlapping heavily or duplicated), especially with `near` sampling + `neg_mult` + replacement.

## What changed (code)

### 1) Negative sampling improvements (`python/ml_log_explore.py`)

Added new knobs:
- `--neg-match-pos-motion` + `--neg-pos-motion-q`  
  Filters negative candidates to match the positive window motion distribution (keeps negatives within `[q, 1-q]` quantiles of positive motion score).
- `--neg-min-sep-ms`  
  Enforces a minimum timestamp separation between selected negative windows (reduces heavy overlap / “near duplicates”).
- Improved `--neg-with-replacement` behavior  
  Now oversamples, then de-dups + enforces separation, instead of blindly creating duplicates.

Added a new strategy:
- `--neg-strategy mixed` + `--neg-mixed-near-frac`
  Builds a dataset that mixes:
  - **near negatives** (close in time before each shot; similar posture/context)
  - **far negatives** (anywhere away from shots; more variety)

Motion is measured by a lightweight “motion score” computed over the window:
- mean squared gyro magnitude + a smaller weighted accel term

Also added dataset grouping metadata:
- `shot_num` is written into `ml_out/train/index.csv` so benchmarks can use a **group split by shot**.

### 2) Plotting uses consistent MLP config (`ml_training/plot_predictions.py`)

The compare plots now use:
- MLP hidden sizes `(64,32)` for `summary/rich` features (matching `ml_training/train.py` defaults)

## What was run (commands)

Dataset export (mixed near+far negatives, motion-matched, separated):
```sh
./.venv/bin/python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset \
  --lead-ms 100 --window-ms 500 --min-shot-gap-ms 1000 \
  --neg-strategy mixed --neg-mixed-near-frac 0.6 --neg-mult 8 \
  --neg-min-motion-quantile 0.2 \
  --neg-match-pos-motion --neg-pos-motion-q 0.05 \
  --neg-min-sep-ms 200 --neg-with-replacement \
  --neg-avoid-edges all --neg-avoid-post-ms 2000
```

Benchmark:
```sh
./.venv/bin/python ml_training/benchmark.py --train-dir ml_out/train --seeds 30 --out ml_training/report_latest.md
```

Prediction comparison plots:
```sh
./.venv/bin/python ml_training/plot_predictions.py --csv ml_log.csv --train-dir ml_out/train --outdir ml_out/predictions_compare_latest \
  --compare --max-shots 10 --max-negs 12 --dpi 300 \
  --model-spec logreg:summary --model-spec et:rich --model-spec mlp:rich
```

## Dataset summary (after change)

From `ml_out/train/meta.json`:
- Positives: 46
- Negatives: 197
- `neg_mult` requested: 8
- `neg_strategy`: mixed (`neg_mixed_near_frac`: 0.6)
- `neg_min_motion_quantile`: 0.2
- `neg_match_pos_motion`: true (`neg_pos_motion_q`: 0.05)
- `neg_min_sep_ms`: 200
- `neg_with_replacement`: true

De-dup note:
- Mixed sampling is de-duped and separation-enforced as a final pass across near+far windows.

## Updated benchmark results (offline window classification)

See: `ml_training/report_latest.md`

Key points:
- Group-split metrics are now available (`split=group`), using `shot_num` from the exported `index.csv`.
- Offline PR AUC is high across several models even under `group` split.

Important caveat:
- The “time split” columns can still look perfect because negatives are constructed relative to shot timing; treat random/shot-based/session-based splits as the primary evaluation.

## Online-behavior check (why plateaus still happen)

Even with better negatives, models can still show high plateaus on the full timeline because:
- The offline dataset is still a *window classification* task, not an explicit *streaming controller* objective.
- “Aiming/no-shot” hard negatives are still underrepresented unless you explicitly record them.
- A fixed threshold like 0.5 is not calibrated to your control cost function; use hysteresis/smoothing.

Artifacts to inspect:
- `ml_out/predictions_compare_latest/pos_vs_neg_hist.png`
- `ml_out/predictions_compare_latest/negatives/neg_*.png`
- `ml_out/predictions_compare_latest/shots/shot_*.png`

## Recommended next steps

1) Collect more “aiming but no shot” data (true hard negatives).
2) Add an online post-process for control:
   - EMA smoothing (50–100ms)
   - hysteresis thresholds (e.g. on=0.7, off=0.4)
   - minimum-on duration
3) Move to a streaming objective:
   - predict a continuous target `y(t)` (e.g. ramp from -600ms to -100ms) rather than a binary interval label.
