#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_dataset(train_dir: Path):
    meta = json.loads((train_dir / "meta.json").read_text())
    window_samples = int(meta["window_samples"])
    channels = meta["channels"]
    if channels != ["ax", "ay", "az", "gx", "gy", "gz"]:
        raise SystemExit(f"Unexpected channels: {channels}")

    import numpy as np

    X = np.fromfile(train_dir / "windows_i16le.bin", dtype="<i2").reshape(-1, window_samples, 6).astype(np.float32)
    y = np.fromfile(train_dir / "labels_u8.bin", dtype=np.uint8)
    if X.shape[0] != y.shape[0]:
        raise SystemExit(f"Mismatched rows: X={X.shape[0]} y={y.shape[0]}")
    # Load timestamps if index.csv exists (used for time-split)
    ts = None
    groups = None
    index_path = train_dir / "index.csv"
    if index_path.exists():
        import csv

        ts_list = []
        group_list = []
        with index_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                ts_list.append(int(row.get("event_timestamp_ms") or 0))
                # shot_num is added by python/ml_log_explore.py; used for group splits
                group_list.append(int(row.get("shot_num") or 0))
        ts = np.array(ts_list, dtype=np.int64) if ts_list else None
        groups = np.array(group_list, dtype=np.int64) if group_list else None
    return meta, X, y, ts, groups


def featurize(X, kind: str):
    import numpy as np

    if kind == "flatten":
        return X.reshape(X.shape[0], -1)
    if kind == "summary":
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mx = np.abs(X).max(axis=1)
        return np.concatenate([mean, std, mx], axis=1)
    if kind == "rich":
        # Per-channel stats (6): mean, std, absmax, absmean.
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mx = np.abs(X).max(axis=1)
        absmean = np.abs(X).mean(axis=1)

        # Magnitude stats (orientation-invariant-ish) for accel and gyro.
        a = X[:, :, 0:3]
        g = X[:, :, 3:6]
        amag = np.sqrt((a * a).sum(axis=2))
        gmag = np.sqrt((g * g).sum(axis=2))
        amag_mean = amag.mean(axis=1, keepdims=True)
        amag_std = amag.std(axis=1, keepdims=True)
        amag_max = amag.max(axis=1, keepdims=True)
        gmag_mean = gmag.mean(axis=1, keepdims=True)
        gmag_std = gmag.std(axis=1, keepdims=True)
        gmag_max = gmag.max(axis=1, keepdims=True)

        return np.concatenate(
            [
                mean,
                std,
                mx,
                absmean,
                amag_mean,
                amag_std,
                amag_max,
                gmag_mean,
                gmag_std,
                gmag_max,
            ],
            axis=1,
        )
    if kind == "invariant":
        # Orientation-invariant features (16): must match firmware featurizeInvariant.
        return _featurize_invariant(X, rich=False)
    if kind == "invariant_rich":
        # Invariant + per-axis extras (22): must match firmware featurizeInvariantRich.
        return _featurize_invariant(X, rich=True)
    raise SystemExit(f"Unknown features: {kind}")


def _featurize_invariant(X, rich=False):
    """Orientation-invariant features matching firmware mlInfer.cpp."""
    import numpy as np

    N = X.shape[1]  # window samples (50)
    NJ = N - 1
    HALF = N // 2
    QUARTER = N // 4

    a = X[:, :, 0:3].astype(np.float64)
    g = X[:, :, 3:6].astype(np.float64)

    # Per-sample magnitudes
    amag = np.sqrt((a * a).sum(axis=2))  # (B, N)
    gmag = np.sqrt((g * g).sum(axis=2))  # (B, N)

    # Jerk = accel[i+1] - accel[i]
    jerk = a[:, 1:, :] - a[:, :-1, :]  # (B, NJ, 3)
    jmag = np.sqrt((jerk * jerk).sum(axis=2))  # (B, NJ)

    # [0] jerk_mag_mean
    jmean = jmag.mean(axis=1, keepdims=True)
    # [1] jerk_mag_std
    jstd = jmag.std(axis=1, keepdims=True)
    # [2] jerk_mag_max
    jmax = jmag.max(axis=1, keepdims=True)
    # [3] jerk_mag_rms
    jrms = np.sqrt((jmag * jmag).mean(axis=1, keepdims=True))

    # HP accel magnitude (subtract window mean)
    amag_mean = amag.mean(axis=1, keepdims=True)
    hp_amag = amag - amag_mean
    # [4] hp_amag_std
    hp_std = hp_amag.std(axis=1, keepdims=True)
    # [5] hp_amag_max
    hp_absmax = np.maximum(np.abs(hp_amag.max(axis=1, keepdims=True)), np.abs(hp_amag.min(axis=1, keepdims=True)))
    # [6] hp_amag_range
    hp_range = (hp_amag.max(axis=1, keepdims=True) - hp_amag.min(axis=1, keepdims=True))

    # Gyro magnitude stats
    gmean = gmag.mean(axis=1, keepdims=True)
    gstd = gmag.std(axis=1, keepdims=True)
    gmax = gmag.max(axis=1, keepdims=True)
    gmin = gmag.min(axis=1, keepdims=True)
    grange = gmax - gmin

    # [11] gyro_slope: 2nd half mean - 1st half mean
    g1mean = gmag[:, :HALF].mean(axis=1, keepdims=True)
    g2mean = gmag[:, HALF:].mean(axis=1, keepdims=True)
    gyro_slope = g2mean - g1mean

    # [12] jerk_mean_ratio: 2nd half / 1st half
    j1count = max(HALF - 1, 1)
    j1mean = jmag[:, :j1count].mean(axis=1, keepdims=True)
    j2mean = jmag[:, j1count:].mean(axis=1, keepdims=True)
    jerk_ratio = np.where(j1mean > 1e-6, j2mean / j1mean, 1.0)

    # [13] gyro_std_ratio: 2nd half std / 1st half std
    g1std = gmag[:, :HALF].std(axis=1, keepdims=True)
    g2std = gmag[:, HALF:].std(axis=1, keepdims=True)
    gyro_std_ratio = np.where(g1std > 1e-6, g2std / g1std, 1.0)

    # [14] energy_ratio_last_q
    lastQ = max(NJ - QUARTER, 0)
    jmag_sq = jmag * jmag
    j_energy_total = jmag_sq.sum(axis=1, keepdims=True)
    j_energy_lastq = jmag_sq[:, lastQ:].sum(axis=1, keepdims=True)
    energy_ratio = np.where(j_energy_total > 1e-6, j_energy_lastq / j_energy_total, 0.25)

    # [15] zcr_jerk_mag
    jmean_bc = jmag.mean(axis=1, keepdims=True)
    above = jmag >= jmean_bc
    crossings = (above[:, 1:] != above[:, :-1]).sum(axis=1, keepdims=True).astype(np.float64)
    zcr = crossings / max(NJ - 1, 1)

    feats = [jmean, jstd, jmax, jrms,
             hp_std, hp_absmax, hp_range,
             gmean, gstd, gmax, grange,
             gyro_slope, jerk_ratio, gyro_std_ratio, energy_ratio, zcr]

    if rich:
        # Per-axis gyro std (3)
        for c in range(3):
            gyro_c = g[:, :, c]
            feats.append(gyro_c.std(axis=1, keepdims=True))
        # Per-axis jerk std (3)
        for c in range(3):
            jerk_c = jerk[:, :, c]
            feats.append(jerk_c.std(axis=1, keepdims=True))

    return np.concatenate(feats, axis=1).astype(np.float32)


def pr_auc_score(y_true, y_score):
    # sklearn's average_precision_score is PR-AUC (area under precision-recall curve).
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def best_f1_threshold(y_true, y_score):
    # Scan thresholds for best F1 (coarse but good enough for baseline).
    from sklearn.metrics import f1_score

    best_t = 0.5
    best_f1 = -1.0
    for t in [i / 100.0 for i in range(1, 100)]:
        f1 = float(f1_score(y_true, (y_score >= t).astype(int)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline models on exported Stinger ML windows")
    ap.add_argument("--train-dir", type=Path, default=Path("ml_out/train"), help="Dataset directory (default: ml_out/train)")
    ap.add_argument(
        "--model",
        choices=["logreg", "ridge", "mlp", "rf", "et", "hgb", "xgb"],
        default="logreg",
        help="Model type (default: logreg)",
    )
    ap.add_argument("--features", choices=["summary", "rich", "flatten", "invariant", "invariant_rich"], default="summary", help="Feature type (default: summary)")
    ap.add_argument("--test-size", type=float, default=0.25, help="Test split fraction (default: 0.25)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    ap.add_argument("--max-rows", type=int, default=0, help="Limit dataset rows (0=all)")
    ap.add_argument(
        "--split",
        choices=["random", "time", "group"],
        default="random",
        help="Split strategy: random=stratified random split, time=train early timestamps test late, group=split by shot_num (default: random)",
    )
    ap.add_argument("--mlp-hidden", default="", help='MLP hidden sizes, e.g. "64,32" (default: auto)')
    ap.add_argument("--mlp-alpha", type=float, default=1e-4, help="MLP L2 alpha (default: 1e-4)")
    ap.add_argument("--trees", type=int, default=300, help="Number of trees for rf/et (default: 300)")
    args = ap.parse_args()

    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, classification_report
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        raise SystemExit("Missing deps. Install with:\n  ./.venv/bin/python -m pip install numpy scikit-learn")

    meta, X, y, ts, groups = load_dataset(args.train_dir)
    if args.max_rows and args.max_rows > 0:
        X = X[: args.max_rows]
        y = y[: args.max_rows]
        if ts is not None:
            ts = ts[: args.max_rows]

    F = featurize(X, args.features)

    if args.split == "random":
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    elif args.split == "time":
        if ts is None:
            raise SystemExit("time split requested but index.csv timestamps are missing")
        order = np.argsort(ts)
        cut = int(math.floor((1.0 - args.test_size) * len(order)))
        tr_idx = order[:cut]
        te_idx = order[cut:]
        Xtr, ytr = F[tr_idx], y[tr_idx]
        Xte, yte = F[te_idx], y[te_idx]
    else:  # group
        if groups is None:
            raise SystemExit("group split requested but index.csv shot_num is missing")
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=50, test_size=args.test_size, random_state=args.seed)
        chosen = None
        for tr_idx, te_idx in gss.split(F, y, groups=groups):
            # Ensure both classes appear in test split (can fail on tiny datasets)
            if len(set(y[te_idx].tolist())) < 2:
                continue
            chosen = (tr_idx, te_idx)
            break
        if chosen is None:
            raise SystemExit("Could not find a group split with both classes in test set; try different seed/test_size")
        tr_idx, te_idx = chosen
        Xtr, ytr = F[tr_idx], y[tr_idx]
        Xte, yte = F[te_idx], y[te_idx]

    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
        name = "LogisticRegression"
    elif args.model == "ridge":
        # Regression baseline for probability-like output.
        from sklearn.linear_model import Ridge

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(Xtr, ytr.astype(np.float32))
        score = reg.predict(Xte)
        score = 1.0 / (1.0 + np.exp(-score))  # squash to (0,1) for comparable metrics
        name = "Ridge (sigmoid)"
    elif args.model == "mlp":
        from sklearn.neural_network import MLPClassifier

        if args.mlp_hidden:
            hidden = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip())
        else:
            # Defaults that tend to work better on small datasets.
            hidden = (64, 32) if args.features in ("summary", "rich") else (32,)

        # early_stopping can fail on very small datasets due to stratified val split;
        # disable it in that case.
        unique, counts = np.unique(ytr, return_counts=True)
        min_class = int(counts.min()) if len(counts) else 0
        use_early = min_class >= 10 and len(ytr) >= 80

        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=args.mlp_alpha,
                max_iter=800,
                random_state=args.seed,
                early_stopping=use_early,
                n_iter_no_change=10,
            ),
        )
        mlp.fit(Xtr, ytr)
        score = mlp.predict_proba(Xte)[:, 1]
        name = f"MLPClassifier{hidden}"
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=args.trees,
            random_state=args.seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf.fit(Xtr, ytr)
        score = rf.predict_proba(Xte)[:, 1]
        name = f"RandomForest(n={args.trees})"
    elif args.model == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        et = ExtraTreesClassifier(
            n_estimators=args.trees,
            random_state=args.seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        et.fit(Xtr, ytr)
        score = et.predict_proba(Xte)[:, 1]
        name = f"ExtraTrees(n={args.trees})"
    elif args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.1,
            max_iter=300,
            random_state=args.seed,
        )
        hgb.fit(Xtr, ytr)
        score = hgb.predict_proba(Xte)[:, 1]
        name = "HistGradientBoosting(depth=3)"
    else:  # xgb
        try:
            import xgboost as xgb  # type: ignore
        except Exception:
            raise SystemExit("xgboost not installed. Install with:\n  ./.venv/bin/python -m pip install xgboost")

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=0,
            eval_metric="logloss",
        )
        model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        name = "XGBoost(depth=3)"

    auc = float(roc_auc_score(yte, score)) if len(set(yte.tolist())) > 1 else float("nan")
    prauc = pr_auc_score(yte, score)
    t, f1 = best_f1_threshold(yte, score)

    print("Dataset")
    print(f"- rows: {len(y)} (pos={int(y.sum())}, neg={int((y==0).sum())})")
    print(f"- window_samples: {meta.get('window_samples')}  (~{meta.get('window_ms')}ms at {meta.get('estimated_hz')}Hz)")
    print(f"- lead_ms/window_ms: {meta.get('lead_ms')}/{meta.get('window_ms')}")
    print()
    print("Model")
    print(f"- type: {name}")
    print(f"- features: {args.features} (dim={F.shape[1]})")
    print(f"- split: {args.split} (test_size={args.test_size})")
    print()
    print("Metrics (test split)")
    print(f"- ROC AUC: {auc:.3f}")
    print(f"- PR AUC:  {prauc:.3f}")
    print(f"- best F1: {f1:.3f} @ threshold={t:.2f}")
    print()
    print("Report @ 0.50 threshold")
    print(classification_report(yte, (score >= 0.5).astype(int), digits=3))


if __name__ == "__main__":
    main()
