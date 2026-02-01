#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    model: str
    features: str
    split: str
    seed: int
    roc_auc: float
    pr_auc: float
    best_f1: float
    best_thr: float


def load_dataset(train_dir: Path):
    meta = json.loads((train_dir / "meta.json").read_text())

    import numpy as np
    import csv

    ws = int(meta["window_samples"])
    X = np.fromfile(train_dir / "windows_i16le.bin", dtype="<i2").reshape(-1, ws, 6).astype(np.float32)
    y = np.fromfile(train_dir / "labels_u8.bin", dtype=np.uint8)

    ts = None
    groups = None
    idx = train_dir / "index.csv"
    if idx.exists():
        ts_list = []
        group_list = []
        with idx.open() as f:
            r = csv.DictReader(f)
            for row in r:
                ts_list.append(int(row.get("event_timestamp_ms") or 0))
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
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        mx = np.abs(X).max(axis=1)
        absmean = np.abs(X).mean(axis=1)

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
    raise ValueError(kind)


def best_f1_threshold(y_true, y_score):
    from sklearn.metrics import f1_score

    best_t = 0.5
    best_f1 = -1.0
    for t in [i / 100.0 for i in range(1, 100)]:
        f1 = float(f1_score(y_true, (y_score >= t).astype(int)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def train_eval(F, y, ts, groups, model: str, split: str, seed: int, test_size: float) -> RunResult:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    if split == "random":
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=test_size, random_state=seed, stratify=y)
    elif split == "time":
        if ts is None:
            raise RuntimeError("time split requested but timestamps missing (index.csv)")
        order = np.argsort(ts)
        cut = int(np.floor((1.0 - test_size) * len(order)))
        tr = order[:cut]
        te = order[cut:]
        Xtr, ytr = F[tr], y[tr]
        Xte, yte = F[te], y[te]
    elif split == "group":
        if groups is None:
            raise RuntimeError("group split requested but shot_num missing (index.csv)")
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=seed)
        chosen = None
        for tr, te in gss.split(F, y, groups=groups):
            if len(set(y[te].tolist())) < 2:
                continue
            chosen = (tr, te)
            break
        if chosen is None:
            raise RuntimeError("could not find group split with both classes in test set")
        tr, te = chosen
        Xtr, ytr = F[tr], y[tr]
        Xte, yte = F[te], y[te]
    else:
        raise ValueError(split)

    if model == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    elif model == "ridge":
        from sklearn.linear_model import Ridge

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(Xtr, ytr.astype(np.float32))
        score = reg.predict(Xte)
        score = 1.0 / (1.0 + np.exp(-score))
    elif model == "mlp_flat":
        from sklearn.neural_network import MLPClassifier

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=600,
                random_state=seed,
                early_stopping=False,
            ),
        )
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    elif model == "mlp_rich":
        from sklearn.neural_network import MLPClassifier

        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-4,
                max_iter=800,
                random_state=seed,
                early_stopping=False,
            ),
        )
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    elif model == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        clf = ExtraTreesClassifier(
            n_estimators=600,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    elif model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.1,
            max_iter=300,
            random_state=seed,
        )
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
    else:
        raise ValueError(model)

    roc = float(roc_auc_score(yte, score)) if len(set(yte.tolist())) > 1 else float("nan")
    pr = float(average_precision_score(yte, score))
    thr, f1 = best_f1_threshold(yte, score)
    return RunResult(model=model, features="n/a", split=split, seed=seed, roc_auc=roc, pr_auc=pr, best_f1=f1, best_thr=thr)


def summarize(results: list[RunResult], model: str, features: str, split: str) -> str:
    import statistics

    rs = [r for r in results if r.model == model and r.split == split and r.features == features]
    roc = [r.roc_auc for r in rs]
    pr = [r.pr_auc for r in rs]
    f1 = [r.best_f1 for r in rs]
    if not rs:
        return ""
    return (
        f"{statistics.mean(roc):.3f}±{statistics.pstdev(roc):.3f} | "
        f"{statistics.mean(pr):.3f}±{statistics.pstdev(pr):.3f} | "
        f"{statistics.mean(f1):.3f}±{statistics.pstdev(f1):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark baseline models across seeds/splits")
    ap.add_argument("--train-dir", type=Path, default=Path("ml_out/train"))
    ap.add_argument("--seeds", type=int, default=10, help="How many seeds to run (default: 10)")
    ap.add_argument("--test-size", type=float, default=0.25, help="Test split fraction (default: 0.25)")
    ap.add_argument("--out", type=Path, default=Path("ml_training/report.md"), help="Write markdown report here")
    args = ap.parse_args()

    try:
        import numpy as np  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:
        raise SystemExit("Missing deps. Install with:\n  ./.venv/bin/python -m pip install numpy scikit-learn")

    meta, X, y, ts, groups = load_dataset(args.train_dir)

    configs = [
        ("logreg", "summary"),
        ("logreg", "rich"),
        ("ridge", "summary"),
        ("mlp_flat", "flatten"),
        ("mlp_rich", "rich"),
        ("rf", "rich"),
        ("et", "rich"),
        ("hgb", "rich"),
    ]
    splits = ["random", "group", "time"]

    results: list[RunResult] = []
    for model, feat in configs:
        F = featurize(X, feat)
        for split in splits:
            for seed in range(1, args.seeds + 1):
                try:
                    r = train_eval(F, y, ts, groups, model=model, split=split, seed=seed, test_size=args.test_size)
                except Exception:
                    # On very small datasets, some seeds can't produce a valid group split; skip those.
                    continue
                results.append(RunResult(**{**r.__dict__, "features": feat}))

    # Shuffle-label sanity check
    import numpy as np

    rng = np.random.default_rng(0)
    ysh = y.copy()
    rng.shuffle(ysh)
    F = featurize(X, "summary")
    sh = train_eval(F, ysh, ts, groups, model="logreg", split="random", seed=1, test_size=args.test_size)

    report: list[str] = []
    report.append("# ML Baseline Benchmark\n")
    report.append("## Dataset\n")
    report.append(f"- train_dir: `{args.train_dir}`")
    report.append(f"- rows: {len(y)} (pos={int(y.sum())}, neg={int((y==0).sum())})")
    report.append(f"- PR AUC baseline (pos rate): {float(y.mean()):.3f}")
    report.append(f"- window_samples: {meta.get('window_samples')} (~{meta.get('window_ms')}ms @ {meta.get('estimated_hz')}Hz)")
    report.append(f"- lead_ms/window_ms: {meta.get('lead_ms')}/{meta.get('window_ms')}")
    report.append(f"- neg_strategy: {meta.get('neg_strategy')}")
    report.append("")

    report.append("## Results (mean±std across seeds)\n")
    report.append("| model | features | split | runs | roc_auc | pr_auc | best_f1 |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, feat in configs:
        for split in splits:
            rs = [r for r in results if r.model == model and r.features == feat and r.split == split]
            if not rs:
                continue
            summary = summarize(results, model, feat, split)
            roc, pr, f1 = [s.strip() for s in summary.split("|")]
            report.append(f"| {model} | {feat} | {split} | {len(rs)} | {roc} | {pr} | {f1} |")
    report.append("")

    report.append("## Sanity check (shuffled labels)\n")
    report.append(f"- logreg+summary (random split): ROC AUC={sh.roc_auc:.3f}, PR AUC={sh.pr_auc:.3f}")

    args.out.write_text("\n".join(report) + "\n")
    print(f"Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
