#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


CSV_FIELDS = ("timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "trigger")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def read_ml_csv(path: Path):
    ts: list[int] = []
    cols = {k: [] for k in CHANNELS}
    trig: list[int] = []

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        missing = [k for k in CSV_FIELDS if k not in (r.fieldnames or ())]
        if missing:
            raise SystemExit(f"Missing columns in CSV: {missing}. Expected: {CSV_FIELDS}")
        for row in r:
            ts.append(int(row["timestamp_ms"]))
            for k in CHANNELS:
                cols[k].append(int(row[k]))
            trig.append(int(row["trigger"]))

    return ts, cols, trig


def find_trigger_rising_edges(trig: list[int]) -> list[int]:
    edges: list[int] = []
    for i in range(1, len(trig)):
        if trig[i] == 1 and trig[i - 1] == 0:
            edges.append(i)
    return edges


def filter_edges_by_gap(ts: list[int], edges: list[int], min_gap_ms: int) -> tuple[list[int], list[int]]:
    if min_gap_ms <= 0:
        return edges[:], []
    accepted: list[int] = []
    rejected: list[int] = []
    last_t: int | None = None
    for idx in edges:
        t = ts[idx]
        if last_t is None or (t - last_t) >= min_gap_ms:
            accepted.append(idx)
            last_t = t
        else:
            rejected.append(idx)
    return accepted, rejected


def robust_ylim(values):
    import numpy as np

    if values.size < 20:
        return None
    ys = np.sort(values.astype(float))
    lo = ys[int(math.floor(0.01 * (len(ys) - 1)))]
    hi = ys[int(math.ceil(0.99 * (len(ys) - 1)))]
    pad = (hi - lo) * 0.15 + 1.0
    return lo - pad, hi + pad


def featurize_windows(windows, kind: str):
    import numpy as np

    if kind == "summary":
        mean = windows.mean(axis=1)
        std = windows.std(axis=1)
        mx = np.abs(windows).max(axis=1)
        return np.concatenate([mean, std, mx], axis=1)

    if kind == "rich":
        mean = windows.mean(axis=1)
        std = windows.std(axis=1)
        mx = np.abs(windows).max(axis=1)
        absmean = np.abs(windows).mean(axis=1)

        a = windows[:, :, 0:3]
        g = windows[:, :, 3:6]
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

    raise SystemExit(f"Unknown features: {kind}")


def train_model(train_dir: Path, model: str, features: str, seed: int):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta = json.loads((train_dir / "meta.json").read_text())
    ws = int(meta["window_samples"])
    X = np.fromfile(train_dir / "windows_i16le.bin", dtype="<i2").reshape(-1, ws, 6).astype(np.float32)
    y = np.fromfile(train_dir / "labels_u8.bin", dtype=np.uint8)

    F = featurize_windows(X, features)

    if model == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    elif model == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        clf = ExtraTreesClassifier(
            n_estimators=600,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
    elif model == "mlp":
        from sklearn.neural_network import MLPClassifier

        # Match ml_training/train.py defaults: (64,32) for low-dim features, (32,) for high-dim.
        hidden = (64, 32) if features in ("summary", "rich") else (32,)
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=800,
                random_state=seed,
                early_stopping=False,
            ),
        )
    else:
        raise SystemExit("Unknown model. Use: logreg, et, mlp")

    clf.fit(F, y)
    return meta, clf


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"Invalid --model-spec '{s}'. Expected format like 'mlp:rich'.")
        model, feats = s.split(":", 1)
        model = model.strip()
        feats = feats.strip()
        if model not in ("logreg", "et", "mlp"):
            raise SystemExit(f"Unknown model in --model-spec: {model}")
        if feats not in ("summary", "rich"):
            raise SystemExit(f"Unknown features in --model-spec: {feats}")
        out.append((model, feats))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a model and plot its predictions over time on ml_log.csv")
    ap.add_argument("--csv", type=Path, default=Path("ml_log.csv"), help="Input ml_log.csv (default: ml_log.csv)")
    ap.add_argument("--train-dir", type=Path, default=Path("ml_out/train"), help="Training dataset dir (default: ml_out/train)")
    ap.add_argument("--outdir", type=Path, default=Path("ml_out/predictions"), help="Output dir (default: ml_out/predictions)")
    ap.add_argument("--model", choices=["logreg", "et", "mlp"], default="mlp", help="Model to train (default: mlp)")
    ap.add_argument("--features", choices=["summary", "rich"], default="rich", help="Feature set (default: rich)")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Overlay multiple model outputs in the per-shot plots (uses --model-spec).",
    )
    ap.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Model+features spec for compare mode, e.g. --model-spec logreg:summary --model-spec et:rich --model-spec mlp:rich",
    )
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    ap.add_argument("--lead-ms", type=int, default=100, help="Lead time (default: 100)")
    ap.add_argument("--window-ms", type=int, default=500, help="Window length (default: 500)")
    ap.add_argument("--min-shot-gap-ms", type=int, default=1000, help="Reject shots closer than this (default: 1000)")
    ap.add_argument(
        "--include-rejected-shots",
        action="store_true",
        help="Also plot shots that were rejected by --min-shot-gap-ms (default: false)",
    )
    ap.add_argument("--stride", type=int, default=1, help="Compute prediction every N samples (default: 1)")
    ap.add_argument("--overview-max-points", type=int, default=25000, help="Downsample overview plot to this many points (default: 25000)")
    ap.add_argument("--shot-pre-ms", type=int, default=1500, help="Per-shot plot pre context (default: 1500)")
    ap.add_argument("--shot-post-ms", type=int, default=300, help="Per-shot plot post context (default: 300)")
    ap.add_argument("--max-shots", type=int, default=30, help="How many per-shot plots to write (default: 30)")
    ap.add_argument("--threshold", type=float, default=-1.0, help="Overlay threshold line (default: auto from 0.5)")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI (default: 200)")
    ap.add_argument("--max-negs", type=int, default=10, help="How many negative-example plots to write (default: 10)")
    args = ap.parse_args()

    try:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise SystemExit("Missing deps. Ensure matplotlib + numpy + scikit-learn installed in .venv.")

    if not (args.train_dir / "meta.json").exists():
        raise SystemExit(f"Missing training dataset: {args.train_dir}/meta.json (run ml_log_explore.py --export-dataset first)")

    ts, cols, trig = read_ml_csv(args.csv)
    n = len(ts)
    if n < 100:
        raise SystemExit("CSV too small")

    # Estimate dt
    dt = [ts[i] - ts[i - 1] for i in range(1, n) if (ts[i] - ts[i - 1]) > 0]
    dt_med = float(np.median(np.array(dt, dtype=np.float32))) if dt else 10.0
    lead_samples = max(1, int(round(args.lead_ms / dt_med)))
    window_samples = max(1, int(round(args.window_ms / dt_med)))

    # Prepare [n,6] float32
    X = np.stack([np.array(cols[k], dtype=np.float32) for k in CHANNELS], axis=1)

    # Train model(s) on exported dataset (for visualization)
    if args.compare:
        model_specs = parse_model_specs(args.model_spec) if args.model_spec else [("logreg", "summary"), ("et", "rich"), ("mlp", "rich")]
        trained: list[tuple[str, str, dict[str, object], object]] = []
        for m, feats in model_specs:
            train_meta, clf = train_model(args.train_dir, m, feats, args.seed)
            trained.append((m, feats, train_meta, clf))
        # Use first meta for mismatch warning
        if trained:
            ws_train = int(trained[0][2].get("window_samples", window_samples))  # type: ignore[call-arg]
            if ws_train != window_samples:
                print(
                    f"[warn] train window_samples={ws_train} but plot window_samples={window_samples} "
                    f"(consider matching --lead-ms/--window-ms to your export)."
                )
    else:
        train_meta, clf = train_model(args.train_dir, args.model, args.features, args.seed)
        if int(train_meta.get("window_samples", window_samples)) != window_samples:
            print(
                f"[warn] train window_samples={train_meta.get('window_samples')} but plot window_samples={window_samples} "
                f"(consider matching --lead-ms/--window-ms to your export)."
            )

    # Sliding windows view: [n-window+1, window, 6]
    if n < window_samples + lead_samples + 1:
        raise SystemExit("Not enough samples for chosen lead/window.")
    wv = sliding_window_view(X, window_shape=window_samples, axis=0)  # (n-window+1, 6, window) on our numpy
    if wv.ndim != 3:
        raise SystemExit(f"Unexpected sliding_window_view shape: {wv.shape}")
    # Normalize to (n-window+1, window, 6)
    if wv.shape[1] == 6 and wv.shape[2] == window_samples:
        wv = wv.swapaxes(1, 2)

    # Prediction indices: compute at i, using window ending at i-lead_samples => start = i-lead-window
    i_start = lead_samples + window_samples
    pred_i = np.arange(i_start, n, max(1, args.stride), dtype=np.int64)
    start_i = pred_i - lead_samples - window_samples
    # Select windows once (used for all models)
    windows = wv[start_i]  # (m, window, 6)

    if args.compare:
        probs: dict[str, np.ndarray] = {}
        for m, feats, _meta, clf in trained:
            Fm = featurize_windows(windows, feats)
            pm = clf.predict_proba(Fm)[:, 1]  # type: ignore[attr-defined]
            probs[f"{m}:{feats}"] = np.asarray(pm, dtype=np.float32)
        # primary p for overview is the last model
        key_last = list(probs.keys())[-1]
        p = probs[key_last]
    else:
        F = featurize_windows(windows, args.features)
        p = clf.predict_proba(F)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(F)
        p = np.asarray(p, dtype=np.float32)

    pred_ts = np.array([ts[i] for i in pred_i], dtype=np.int64)

    # Trigger edges
    edges_all = find_trigger_rising_edges(trig)
    edges_ok, edges_rej = filter_edges_by_gap(ts, edges_all, args.min_shot_gap_ms)
    print(f"Trigger rising edges: all={len(edges_all)} accepted={len(edges_ok)} rejected={len(edges_rej)} (min_gap_ms={args.min_shot_gap_ms})")

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "shots").mkdir(parents=True, exist_ok=True)
    (args.outdir / "negatives").mkdir(parents=True, exist_ok=True)

    # Write predictions CSV
    out_csv = args.outdir / "predictions.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        if args.compare:
            header = ["timestamp_ms", "trigger", *[f"p_{k.replace(':', '_')}" for k in probs.keys()]]
            w.writerow(header)
            for j, idx in enumerate(pred_i.tolist()):
                row = [int(pred_ts[j]), int(trig[idx])]
                for k in probs.keys():
                    row.append(float(probs[k][j]))
                w.writerow(row)
        else:
            w.writerow(["timestamp_ms", "p", "trigger"])
            for t, prob, idx in zip(pred_ts.tolist(), p.tolist(), pred_i.tolist()):
                w.writerow([t, float(prob), int(trig[idx])])

    thr = args.threshold if args.threshold >= 0 else 0.5

    # Consistent palette for multi-model plots
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # Overview plot: p(t) and triggers
    # Downsample p curve if needed
    m = len(pred_ts)
    step = max(1, int(math.ceil(m / max(1, args.overview_max_points))))
    t_s = (pred_ts[::step] - pred_ts[0]).astype(np.float32) / 1000.0
    p_ds = p[::step]

    fig = plt.figure(figsize=(14, 5))
    axp = fig.add_subplot(1, 1, 1)
    if args.compare:
        for i, (name, pm_all) in enumerate(probs.items()):
            axp.plot(
                t_s,
                pm_all[::step],
                linewidth=0.8,
                alpha=0.9,
                label=name,
                color=color_cycle[i % len(color_cycle)],
            )
    else:
        axp.plot(t_s, p_ds, linewidth=0.8, label="p(pre-shot)")
    axp.axhline(thr, color="black", linewidth=1.0, alpha=0.6, linestyle="--", label=f"threshold={thr:.2f}")

    t0 = ts[0]
    # Mark triggers (accepted vs rejected)
    for idx in edges_rej:
        x = (ts[idx] - t0) / 1000.0
        axp.axvline(x, color="gray", linewidth=0.6, alpha=0.25, linestyle=":")
    for idx in edges_ok:
        x = (ts[idx] - t0) / 1000.0
        axp.axvline(x, color="red", linewidth=0.8, alpha=0.25)

    title_model = (
        "compare"
        if args.compare
        else f"{args.model}+{args.features}"
    )
    axp.set_title(f"Model predictions over time ({title_model})")
    axp.set_xlabel("time (s)")
    axp.set_ylabel("probability")
    axp.set_ylim(-0.05, 1.05)
    axp.grid(True, alpha=0.25)
    axp.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.outdir / "overview.png", dpi=args.dpi)
    plt.close(fig)

    # Per-shot plots
    lead = args.lead_ms
    window = args.window_ms
    pred_start_ms = -(lead + window)
    pred_end_ms = -lead

    # helper: for a shot time, slice predictions in a time range
    pred_ts_list = pred_ts.tolist()
    shot_list = edges_ok + (edges_rej if args.include_rejected_shots else [])
    if args.max_shots > len(shot_list):
        print(f"[note] requested max_shots={args.max_shots} but only {len(shot_list)} available from this log.")
    for shot_num, idx in enumerate(shot_list[: max(0, args.max_shots)]):
        t_shot = ts[idx]
        t_a = t_shot - args.shot_pre_ms
        t_b = t_shot + args.shot_post_ms
        a = bisect.bisect_left(pred_ts_list, t_a)
        b = bisect.bisect_right(pred_ts_list, t_b)
        if b - a < 5:
            continue

        t_rel_p = (pred_ts[a:b] - t_shot).astype(np.float32) / 1000.0
        p_win = p[a:b]

        # Slice raw IMU signals in the same window for context.
        i0 = bisect.bisect_left(ts, t_a)
        i1 = bisect.bisect_right(ts, t_b)
        if i1 - i0 < 5:
            continue
        t_rel_imu = (np.array(ts[i0:i1], dtype=np.int64) - t_shot).astype(np.float32) / 1000.0

        ax_w = np.array(cols["ax"][i0:i1], dtype=np.float32)
        ay_w = np.array(cols["ay"][i0:i1], dtype=np.float32)
        az_w = np.array(cols["az"][i0:i1], dtype=np.float32)
        gx_w = np.array(cols["gx"][i0:i1], dtype=np.float32)
        gy_w = np.array(cols["gy"][i0:i1], dtype=np.float32)
        gz_w = np.array(cols["gz"][i0:i1], dtype=np.float32)

        fig = plt.figure(figsize=(12.5, 7.6))
        axp = fig.add_subplot(2, 1, 1)
        aximu = fig.add_subplot(2, 1, 2, sharex=axp)

        # p(t) (overlay multiple models if requested)
        axp.axvspan(pred_start_ms / 1000.0, pred_end_ms / 1000.0, color="gold", alpha=0.22, label="label window")
        axp.axvline(0.0, color="red", linewidth=1.0, alpha=0.9, label="trigger")
        axp.axhline(thr, color="black", linewidth=1.0, alpha=0.6, linestyle="--", label=f"thr={thr:.2f}")
        if args.compare:
            for i, (name, pm_all) in enumerate(probs.items()):
                axp.plot(t_rel_p, pm_all[a:b], linewidth=1.2, color=color_cycle[i % len(color_cycle)], label=name)
        else:
            axp.plot(t_rel_p, p_win, linewidth=1.2, label="p(pre-shot)")
        axp.set_ylabel("probability")
        axp.set_ylim(-0.05, 1.05)
        axp.grid(True, alpha=0.25)
        axp.legend(loc="upper left", frameon=False, ncols=4, fontsize=9)

        # IMU (accel left, gyro right)
        aximu.axvspan(pred_start_ms / 1000.0, pred_end_ms / 1000.0, color="gold", alpha=0.18)
        aximu.axvline(0.0, color="red", linewidth=1.0, alpha=0.9)

        # accel solid
        aximu.plot(t_rel_imu, ax_w, color="#1f77b4", linewidth=0.8, alpha=0.8, label="ax")
        aximu.plot(t_rel_imu, ay_w, color="#2ca02c", linewidth=0.8, alpha=0.8, label="ay")
        aximu.plot(t_rel_imu, az_w, color="#ff7f0e", linewidth=0.8, alpha=0.8, label="az")
        aximu.set_ylabel("accel (raw)")
        yl = robust_ylim(np.concatenate([ax_w, ay_w, az_w], axis=0))
        if yl is not None:
            aximu.set_ylim(*yl)

        axg = aximu.twinx()
        axg.plot(t_rel_imu, gx_w, color="#d62728", linewidth=0.8, alpha=0.75, linestyle="--", label="gx")
        axg.plot(t_rel_imu, gy_w, color="#9467bd", linewidth=0.8, alpha=0.75, linestyle="--", label="gy")
        axg.plot(t_rel_imu, gz_w, color="#8c564b", linewidth=0.8, alpha=0.75, linestyle="--", label="gz")
        axg.set_ylabel("gyro (raw)")
        yr = robust_ylim(np.concatenate([gx_w, gy_w, gz_w], axis=0))
        if yr is not None:
            axg.set_ylim(*yr)

        aximu.grid(True, alpha=0.25)
        aximu.set_xlabel("time relative to trigger (s)")

        # combined legend for imu plot
        h1, l1 = aximu.get_legend_handles_labels()
        h2, l2 = axg.get_legend_handles_labels()
        aximu.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, ncols=6, fontsize=8)

        fig.suptitle(f"Shot {shot_num} @ {t_shot}ms (p + IMU)", fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fig.savefig(args.outdir / "shots" / f"shot_{shot_num:03d}.png", dpi=args.dpi)
        plt.close(fig)

    # Negative visualization: show where models fire on negative examples too.
    # We use the exported training index.csv (window_start/end) to locate example windows on the prediction timeline.
    prob_series = probs if args.compare else {"p": p}

    index_path = args.train_dir / "index.csv"
    if index_path.exists():
        rows = []
        with index_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    label = int(row.get("label") or 0)
                    window_end_ms = int(row.get("window_end_ms") or 0)
                except Exception:
                    continue
                # Align each exported window to a "current time" where our online predictor would use it:
                # predictor at time t uses window ending at t-lead => t_align = window_end + lead_ms
                t_align = window_end_ms + args.lead_ms
                rows.append((label, t_align))

        pred_ts_list = pred_ts.tolist()

        def nearest_pred_index(t_ms: int) -> int:
            j = bisect.bisect_left(pred_ts_list, t_ms)
            if j <= 0:
                return 0
            if j >= len(pred_ts_list):
                return len(pred_ts_list) - 1
            if abs(pred_ts_list[j] - t_ms) < abs(pred_ts_list[j - 1] - t_ms):
                return j
            return j - 1

        pos_idx = [nearest_pred_index(t) for (lab, t) in rows if lab == 1]
        neg_idx = [nearest_pred_index(t) for (lab, t) in rows if lab == 0]

        # Histogram plot: p distributions for pos vs neg examples
        import numpy as np

        fig = plt.figure(figsize=(12.5, 5.8))
        axh = fig.add_subplot(1, 1, 1)
        bins = 40
        for i, (name, pm) in enumerate(prob_series.items()):
            c = color_cycle[i % len(color_cycle)]
            axh.hist(np.asarray(pm)[pos_idx], bins=bins, density=True, alpha=0.25, color=c, label=f"{name} pos")
            axh.hist(
                np.asarray(pm)[neg_idx],
                bins=bins,
                density=True,
                alpha=0.25,
                color=c,
                histtype="step",
                linewidth=2.0,
                label=f"{name} neg",
            )
        axh.axvline(thr, color="black", linewidth=1.0, alpha=0.6, linestyle="--", label=f"thr={thr:.2f}")
        axh.set_title("Predicted probability distribution (training pos vs neg examples)")
        axh.set_xlabel("p(pre-shot)")
        axh.set_ylabel("density")
        axh.grid(True, alpha=0.25)
        axh.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.outdir / "pos_vs_neg_hist.png", dpi=args.dpi)
        plt.close(fig)

        # Negative example plots (same layout as shot plots, but anchored at a negative window end+lead).
        for neg_num, j in enumerate(neg_idx[: max(0, args.max_negs)]):
            t_anchor = pred_ts_list[j]
            t_a = t_anchor - args.shot_pre_ms
            t_b = t_anchor + args.shot_post_ms
            a = bisect.bisect_left(pred_ts_list, t_a)
            b = bisect.bisect_right(pred_ts_list, t_b)
            if b - a < 5:
                continue
            t_rel_p = (pred_ts[a:b] - t_anchor).astype(np.float32) / 1000.0

            i0 = bisect.bisect_left(ts, t_a)
            i1 = bisect.bisect_right(ts, t_b)
            if i1 - i0 < 5:
                continue
            t_rel_imu = (np.array(ts[i0:i1], dtype=np.int64) - t_anchor).astype(np.float32) / 1000.0

            ax_w = np.array(cols["ax"][i0:i1], dtype=np.float32)
            ay_w = np.array(cols["ay"][i0:i1], dtype=np.float32)
            az_w = np.array(cols["az"][i0:i1], dtype=np.float32)
            gx_w = np.array(cols["gx"][i0:i1], dtype=np.float32)
            gy_w = np.array(cols["gy"][i0:i1], dtype=np.float32)
            gz_w = np.array(cols["gz"][i0:i1], dtype=np.float32)

            fig = plt.figure(figsize=(12.5, 7.6))
            axp = fig.add_subplot(2, 1, 1)
            aximu = fig.add_subplot(2, 1, 2, sharex=axp)

            axp.axvspan(pred_start_ms / 1000.0, pred_end_ms / 1000.0, color="gold", alpha=0.22, label="window")
            axp.axvline(0.0, color="gray", linewidth=1.0, alpha=0.9, label="neg anchor")
            axp.axhline(thr, color="black", linewidth=1.0, alpha=0.6, linestyle="--", label=f"thr={thr:.2f}")
            for i, (name, pm_all) in enumerate(prob_series.items()):
                axp.plot(t_rel_p, np.asarray(pm_all)[a:b], linewidth=1.2, color=color_cycle[i % len(color_cycle)], label=name)
            axp.set_ylabel("probability")
            axp.set_ylim(-0.05, 1.05)
            axp.grid(True, alpha=0.25)
            axp.legend(loc="upper left", frameon=False, ncols=3, fontsize=9)

            aximu.axvspan(pred_start_ms / 1000.0, pred_end_ms / 1000.0, color="gold", alpha=0.18)
            aximu.axvline(0.0, color="gray", linewidth=1.0, alpha=0.9)
            aximu.plot(t_rel_imu, ax_w, color="#1f77b4", linewidth=0.8, alpha=0.8, label="ax")
            aximu.plot(t_rel_imu, ay_w, color="#2ca02c", linewidth=0.8, alpha=0.8, label="ay")
            aximu.plot(t_rel_imu, az_w, color="#ff7f0e", linewidth=0.8, alpha=0.8, label="az")
            aximu.set_ylabel("accel (raw)")
            yl = robust_ylim(np.concatenate([ax_w, ay_w, az_w], axis=0))
            if yl is not None:
                aximu.set_ylim(*yl)
            axg = aximu.twinx()
            axg.plot(t_rel_imu, gx_w, color="#d62728", linewidth=0.8, alpha=0.75, linestyle="--", label="gx")
            axg.plot(t_rel_imu, gy_w, color="#9467bd", linewidth=0.8, alpha=0.75, linestyle="--", label="gy")
            axg.plot(t_rel_imu, gz_w, color="#8c564b", linewidth=0.8, alpha=0.75, linestyle="--", label="gz")
            axg.set_ylabel("gyro (raw)")
            yr = robust_ylim(np.concatenate([gx_w, gy_w, gz_w], axis=0))
            if yr is not None:
                axg.set_ylim(*yr)
            aximu.grid(True, alpha=0.25)
            aximu.set_xlabel("time relative to anchor (s)")
            h1, l1 = aximu.get_legend_handles_labels()
            h2, l2 = axg.get_legend_handles_labels()
            aximu.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, ncols=6, fontsize=8)

            fig.suptitle(f"Negative example {neg_num} @ {t_anchor}ms (p + IMU)", fontsize=12)
            fig.tight_layout(rect=(0, 0.02, 1, 0.95))
            fig.savefig(args.outdir / "negatives" / f"neg_{neg_num:03d}.png", dpi=args.dpi)
            plt.close(fig)

    print(f"Wrote: {args.outdir / 'overview.png'}")
    print(f"Wrote: {args.outdir / 'predictions.csv'}")
    print(f"Wrote: {args.outdir / 'shots'}")
    if (args.outdir / 'pos_vs_neg_hist.png').exists():
        print(f"Wrote: {args.outdir / 'pos_vs_neg_hist.png'}")
    if (args.outdir / 'negatives').exists():
        print(f"Wrote: {args.outdir / 'negatives'}")


if __name__ == "__main__":
    main()
