# ML Training (local)

This folder contains simple, repeatable training scripts for the Stinger ML dataset exported by `python/ml_log_explore.py`.

Full write-up: `ml_training/REPORT.md`

## 1) Export a dataset

From repo root:

```sh
source .venv/bin/activate
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset
```

Recommended (harder, more realistic negatives; helps reduce “always high” probability plateaus):

```sh
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset \
  --lead-ms 100 --window-ms 500 --min-shot-gap-ms 1000 \
  --neg-strategy mixed --neg-mixed-near-frac 0.6 --neg-mult 8 \
  --neg-min-motion-quantile 0.2 \
  --neg-match-pos-motion --neg-pos-motion-q 0.05 \
  --neg-min-sep-ms 200 --neg-with-replacement \
  --neg-avoid-edges all --neg-avoid-post-ms 2000
```

This creates `ml_out/train/` with:
- `windows_i16le.bin` (int16 little-endian), shape `[N, window_samples, 6]` with channels `[ax,ay,az,gx,gy,gz]`
- `labels_u8.bin` (0/1)
- `meta.json`, `index.csv`

## 2) Train a baseline model

```sh
source .venv/bin/activate
python ml_training/train.py --train-dir ml_out/train --model logreg --features summary
```

Other options:

```sh
python ml_training/train.py --train-dir ml_out/train --model ridge --features summary
python ml_training/train.py --train-dir ml_out/train --model mlp --features flatten
python ml_training/train.py --train-dir ml_out/train --model mlp --features rich
python ml_training/train.py --train-dir ml_out/train --model rf --features rich
python ml_training/train.py --train-dir ml_out/train --model et --features rich
python ml_training/train.py --train-dir ml_out/train --model hgb --features rich
```

You can also test a time-based split (train on early timestamps, test on later):

```sh
python ml_training/train.py --train-dir ml_out/train --model logreg --features summary --split time
```

Or split by shot groups (requires `shot_num` in `ml_out/train/index.csv`, which `ml_log_explore.py` now writes):

```sh
python ml_training/train.py --train-dir ml_out/train --model logreg --features summary --split group
```

## 3) Benchmark (iterative testing)

First, export a dataset. For more realistic negatives, use `near` sampling:

```sh
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset --neg-strategy near --neg-mult 1
```

Then run the benchmark:

```sh
python ml_training/benchmark.py --train-dir ml_out/train --seeds 10 --out ml_training/report.md
```

Note: the benchmark now includes `split=group` (shot-based group split) in addition to `random` and `time`.

## 4) Visualize predictions (graphs)

This trains a model on `ml_out/train` and plots its predicted probability over the entire `ml_log.csv`, plus per-shot zoomed plots:

```sh
python ml_training/plot_predictions.py --csv ml_log.csv --train-dir ml_out/train --outdir ml_out/predictions --model mlp --features rich
```

To compare multiple models on the same 10-shot plots (top = probabilities, bottom = IMU axes):

```sh
python ml_training/plot_predictions.py --csv ml_log.csv --train-dir ml_out/train --outdir ml_out/predictions_compare \
  --compare --max-shots 10 \
  --model-spec logreg:summary \
  --model-spec et:rich \
  --model-spec mlp:rich
```

To also visualize how models behave on negative windows (pos/neg histogram + `negatives/neg_*.png`):

```sh
python ml_training/plot_predictions.py --csv ml_log.csv --train-dir ml_out/train --outdir ml_out/predictions_compare \
  --compare --max-shots 10 --max-negs 10 --dpi 250 \
  --model-spec logreg:summary --model-spec et:rich --model-spec mlp:rich
```

Outputs:
- `ml_out/predictions/overview.png`
- `ml_out/predictions/shots/shot_XXX.png` (includes p + all 6 IMU axes)
- `ml_out/predictions/predictions.csv`

## Notes

- Results from a single session can look optimistic. The real test is generalization across sessions/users/hold styles.
- Negatives are sampled away from shots and (by default) not within `--neg-avoid-post-ms` after shots. Tune those settings during dataset export.
- For end-to-end personalization (record → train → upload without UF2), see `ML.md` and `ml_web/README.md`.
