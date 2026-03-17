#!/usr/bin/env python3
"""
Explore Stinger ml_log.csv (basic stats + plots).

Outputs:
  - dataset plot with all 6 axes + trigger pulls
  - per-shot cropped plots with all 6 axes (more context pre-shot, only 200ms post-shot)

Typical flow:
  python python/ml_log_pull.py -o ml_log.bin
  python python/ml_log_convert.py ml_log.bin -o ml_log.csv
  python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


CSV_FIELDS = ("timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz", "trigger")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def read_ml_csv(path: Path) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]:
    ts: list[int] = []
    ax: list[int] = []
    ay: list[int] = []
    az: list[int] = []
    gx: list[int] = []
    gy: list[int] = []
    gz: list[int] = []
    trig: list[int] = []

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        missing = [k for k in CSV_FIELDS if k not in (r.fieldnames or ())]
        if missing:
            raise SystemExit(f"Missing columns in CSV: {missing}. Expected: {CSV_FIELDS}")
        for row in r:
            ts.append(int(row["timestamp_ms"]))
            ax.append(int(row["ax"]))
            ay.append(int(row["ay"]))
            az.append(int(row["az"]))
            gx.append(int(row["gx"]))
            gy.append(int(row["gy"]))
            gz.append(int(row["gz"]))
            trig.append(int(row["trigger"]))

    return ts, ax, ay, az, gx, gy, gz, trig


def motion_score_window(
    ax: list[int],
    ay: list[int],
    az: list[int],
    gx: list[int],
    gy: list[int],
    gz: list[int],
    start_i: int,
    end_i: int,
) -> float:
    """Heuristic motion score (higher = more motion).

    Used to filter out near-stationary negative windows.
    Uses mean squared magnitude to avoid sqrt per-sample.
    """
    n = end_i - start_i
    if n <= 0:
        return 0.0
    acc = 0.0
    gyr = 0.0
    for i in range(start_i, end_i):
        acc += float(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i])
        gyr += float(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i])
    # Gyro usually matters more for "gesture/aiming"; keep accel as a smaller term.
    return (gyr / n) + 0.05 * (acc / n)


def sample_unique_with_min_separation(
    rng,
    candidates: list[int],
    ts: list[int],
    target: int,
    min_sep_ms: int,
) -> list[int]:
    """Sample up to target unique indices with a minimum timestamp separation."""
    if target <= 0 or not candidates:
        return []
    min_sep_ms = max(0, int(min_sep_ms))

    pool = list(dict.fromkeys(candidates))  # de-dup while preserving order
    rng.shuffle(pool)

    chosen: list[int] = []
    chosen_ts: list[int] = []
    for idx in pool:
        if len(chosen) >= target:
            break
        t = ts[idx]
        if min_sep_ms and any(abs(t - ct) < min_sep_ms for ct in chosen_ts):
            continue
        chosen.append(idx)
        chosen_ts.append(t)
    return chosen


def find_trigger_rising_edges(trig: list[int]) -> list[int]:
    edges: list[int] = []
    for i in range(1, len(trig)):
        if trig[i] == 1 and trig[i - 1] == 0:
            edges.append(i)
    return edges


def robust_ylim(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 20:
        return None
    ys = sorted(values)
    lo = ys[int(math.floor(0.01 * (len(ys) - 1)))]
    hi = ys[int(math.ceil(0.99 * (len(ys) - 1)))]
    pad = (hi - lo) * 0.15 + 1.0
    return (lo - pad, hi + pad)


def filter_shots_by_gap(ts: list[int], edges: list[int], min_gap_ms: int) -> tuple[list[int], list[int]]:
    """Return (accepted, rejected) by min timestamp gap between accepted shots."""
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


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        # Best-effort cleanup: already removed (or never existed).
        pass


def rm_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            rm_tree(child)
        else:
            safe_unlink(child)
    try:
        path.rmdir()
    except OSError:
        # Best-effort cleanup: ignore errors if the directory cannot be removed.
        pass


def clamp_i16(x: int) -> int:
    if x < -32768:
        return -32768
    if x > 32767:
        return 32767
    return x


def write_i16le(f, v: int) -> None:
    vv = clamp_i16(v) & 0xFFFF
    f.write(bytes((vv & 0xFF, (vv >> 8) & 0xFF)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore Stinger ml_log.csv (plots + basic stats)")
    ap.add_argument("csv", type=Path, help="Path to ml_log.csv")
    ap.add_argument("--outdir", type=Path, default=Path("ml_out"), help="Output directory (default: ml_out)")
    ap.add_argument("--clean", action="store_true", help="Delete old ml_out plots before writing new ones")

    ap.add_argument("--shot-pre-ms", type=int, default=1000, help="Milliseconds before trigger in per-shot crops (default: 1000)")
    ap.add_argument("--shot-post-ms", type=int, default=200, help="Milliseconds after trigger in per-shot crops (default: 200)")
    ap.add_argument("--max-shots", type=int, default=0, help="Limit plotted shots (0=all)")
    ap.add_argument("--start-shot", type=int, default=0, help="Skip N accepted shots before plotting (default: 0)")

    ap.add_argument("--lead-ms", type=int, default=100, help="Prediction lead time before trigger (default: 100)")
    ap.add_argument("--window-ms", type=int, default=500, help="Prediction window length ending at lead time (default: 500)")
    ap.add_argument("--min-shot-gap-ms", type=int, default=1000, help="Exclude shots closer than this (default: 1000)")

    ap.add_argument("--overview-max-points", type=int, default=20000, help="Max points in dataset plot (default: 20000)")
    ap.add_argument("--dpi", type=int, default=150, help="PNG DPI (default: 150)")

    ap.add_argument(
        "--export-dataset",
        action="store_true",
        help="Export a labeled window dataset for training into <outdir>/train/",
    )
    ap.add_argument(
        "--neg-mult",
        type=int,
        default=1,
        help="Number of negative windows per positive window (default: 1)",
    )
    ap.add_argument(
        "--neg-min-motion-quantile",
        type=float,
        default=0.0,
        help="Reject the lowest-motion negatives. 0 disables; 0.2 rejects lowest 20%% (default: 0)",
    )
    ap.add_argument(
        "--neg-match-pos-motion",
        action="store_true",
        help="Try to match negative motion to positive motion (filters candidates by pos motion quantiles).",
    )
    ap.add_argument(
        "--neg-pos-motion-q",
        type=float,
        default=0.1,
        help="If --neg-match-pos-motion: keep negatives within [q,1-q] quantiles of positive motion (default: 0.1)",
    )
    ap.add_argument(
        "--neg-min-sep-ms",
        type=int,
        default=250,
        help="Minimum time separation between sampled negative window ends (default: 250ms)",
    )
    ap.add_argument(
        "--neg-with-replacement",
        action="store_true",
        help="Allow sampling negative windows with replacement if you request more than available (default: false)",
    )
    ap.add_argument(
        "--neg-strategy",
        choices=["far", "near", "mixed"],
        default="far",
        help="How to sample negatives: far=anywhere away from shots; near=close-in-time before each shot; mixed=near+far (default: far)",
    )
    ap.add_argument(
        "--neg-mixed-near-frac",
        type=float,
        default=0.5,
        help="(mixed strategy) Fraction of negatives to draw from near strategy; remainder from far (default: 0.5)",
    )
    ap.add_argument(
        "--neg-near-min-ms",
        type=int,
        default=200,
        help="(near strategy) Min gap (ms) between a negative window end and the positive window start (default: 200)",
    )
    ap.add_argument(
        "--neg-near-max-ms",
        type=int,
        default=2000,
        help="(near strategy) Max gap (ms) between a negative window end and the positive window start (default: 2000)",
    )
    ap.add_argument(
        "--neg-avoid-post-ms",
        type=int,
        default=1000,
        help="Avoid sampling negative windows within this many ms after shots (see --neg-avoid-edges) (default: 1000)",
    )
    ap.add_argument(
        "--neg-avoid-edges",
        choices=["accepted", "all"],
        default="all",
        help="Which trigger edges to avoid around when sampling negatives (default: all)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for negative window sampling (default: 1)",
    )
    args = ap.parse_args()

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise SystemExit(
            "Missing dependency: matplotlib\n"
            "Install with:\n"
            "  ./.venv/bin/python -m pip install matplotlib\n"
            "Then re-run."
        )

    ts, ax, ay, az, gx, gy, gz, trig = read_ml_csv(args.csv)
    n = len(ts)
    if n < 10:
        raise SystemExit("CSV too small")

    # Timestamp / sample rate stats
    dt_ms: list[float] = []
    last = ts[0]
    for t in ts[1:]:
        dt = t - last
        if dt > 0:
            dt_ms.append(float(dt))
        last = t
    dt_sorted = sorted(dt_ms)
    dt_med = statistics.median(dt_sorted) if dt_sorted else 0.0
    hz_est = (1000.0 / dt_med) if dt_med > 0 else 0.0

    duration_ms = ts[-1] - ts[0]
    duration_s = duration_ms / 1000.0 if duration_ms > 0 else 0.0

    all_edges = find_trigger_rising_edges(trig)
    accepted_edges, rejected_edges = filter_shots_by_gap(ts, all_edges, args.min_shot_gap_ms)

    # i16 clipping check (often indicates shock saturating the accel range)
    clip_count = 0
    for i in range(n):
        for v in (ax[i], ay[i], az[i], gx[i], gy[i], gz[i]):
            if v == -32768 or v == 32767:
                clip_count += 1
                break

    # Output dirs
    outdir: Path = args.outdir
    shots_dir = outdir / "shots"
    train_dir = outdir / "train"
    if args.clean:
        rm_tree(shots_dir)
        rm_tree(train_dir)
        for old in (
            outdir / "overview.png",
            outdir / "gyro_mag_mean_hist.png",
            outdir / "features.csv",
            outdir / "profile.csv",
            outdir / "profile.png",
            outdir / "dataset.png",
        ):
            safe_unlink(old)
        for old in outdir.glob("shot*.png"):
            safe_unlink(old)
        for old in outdir.glob("shots_grid*.png"):
            safe_unlink(old)
    outdir.mkdir(parents=True, exist_ok=True)
    shots_dir.mkdir(parents=True, exist_ok=True)
    if args.export_dataset:
        train_dir.mkdir(parents=True, exist_ok=True)

    # Write summary JSON
    summary = {
        "input_csv": str(args.csv),
        "samples": n,
        "duration_ms": duration_ms,
        "duration_s": duration_s,
        "i16_clipping_samples": clip_count,
        "timestamp_dt_ms": {
            "count": len(dt_sorted),
            "median": dt_med,
            "p5": percentile(dt_sorted, 5),
            "p95": percentile(dt_sorted, 95),
            "min": dt_sorted[0] if dt_sorted else None,
            "max": dt_sorted[-1] if dt_sorted else None,
            "estimated_hz": hz_est,
        },
        "trigger": {
            "rising_edges_all": len(all_edges),
            "rising_edges_accepted": len(accepted_edges),
            "rising_edges_rejected": len(rejected_edges),
            "min_shot_gap_ms": args.min_shot_gap_ms,
            "rejected_reason": "too close to previous accepted shot",
            "shot_pre_ms": args.shot_pre_ms,
            "shot_post_ms": args.shot_post_ms,
            "lead_ms": args.lead_ms,
            "window_ms": args.window_ms,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Console output
    print(f"Loaded {n} samples from {args.csv}")
    print(f"Duration: {duration_s:.1f}s ({duration_ms} ms)")
    print(f"Estimated sample rate: {hz_est:.2f} Hz (median dt={dt_med:.2f} ms)")
    print(f"Trigger rising edges: {len(all_edges)} (accepted: {len(accepted_edges)}, rejected: {len(rejected_edges)})")
    print(f"Wrote: {outdir / 'summary.json'}")

    # Dataset plot (downsampled) with all 6 axes + trigger pulls
    step = max(1, int(math.ceil(n / max(1, args.overview_max_points))))
    idxs = list(range(0, n, step))
    t_s = [(ts[i] - ts[0]) / 1000.0 for i in idxs]

    fig = plt.figure(figsize=(14, 6))
    ax_left = fig.add_subplot(1, 1, 1)
    ax_right = ax_left.twinx()

    # Accel on left
    ax_left.plot(t_s, [ax[i] for i in idxs], color="#1f77b4", linewidth=0.6, alpha=0.35, label="ax")
    ax_left.plot(t_s, [ay[i] for i in idxs], color="#2ca02c", linewidth=0.6, alpha=0.35, label="ay")
    ax_left.plot(t_s, [az[i] for i in idxs], color="#ff7f0e", linewidth=0.6, alpha=0.35, label="az")

    # Gyro on right (dashed)
    ax_right.plot(t_s, [gx[i] for i in idxs], color="#d62728", linewidth=0.6, alpha=0.35, linestyle="--", label="gx")
    ax_right.plot(t_s, [gy[i] for i in idxs], color="#9467bd", linewidth=0.6, alpha=0.35, linestyle="--", label="gy")
    ax_right.plot(t_s, [gz[i] for i in idxs], color="#8c564b", linewidth=0.6, alpha=0.35, linestyle="--", label="gz")

    # Trigger pull markers (rising edges)
    t0 = ts[0]
    for idx in rejected_edges:
        x = (ts[idx] - t0) / 1000.0
        ax_left.axvline(x, color="gray", linewidth=0.6, alpha=0.35, linestyle=":")
    for idx in accepted_edges:
        x = (ts[idx] - t0) / 1000.0
        ax_left.axvline(x, color="red", linewidth=0.8, alpha=0.35)

    ax_left.set_title("Dataset overview (all 6 axes, trigger pulls marked)")
    ax_left.set_xlabel("time (s)")
    ax_left.set_ylabel("accel (raw)")
    ax_right.set_ylabel("gyro (raw)")
    ax_left.grid(True, alpha=0.2)

    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper right",
        frameon=False,
        ncols=6,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outdir / "dataset.png", dpi=args.dpi)
    plt.close(fig)

    # Per-shot plots: more context before, only 200ms after.
    lead = args.lead_ms
    window = args.window_ms
    pred_start_ms = -(lead + window)
    pred_end_ms = -lead

    start_shot = max(0, args.start_shot)
    accepted_for_plot = accepted_edges[start_shot:]
    if args.max_shots and args.max_shots > 0:
        accepted_for_plot = accepted_for_plot[: args.max_shots]

    # Index CSV so you can map plots back to timestamps.
    index_rows: list[dict[str, Any]] = []

    for shot_num, idx in enumerate(accepted_for_plot, start=start_shot):
        t_ms = ts[idx]
        w_start = t_ms - args.shot_pre_ms
        w_end = t_ms + args.shot_post_ms
        i0 = bisect.bisect_left(ts, w_start)
        i1 = bisect.bisect_right(ts, w_end)
        if i1 - i0 < 5:
            continue

        t_rel = [(ts[i] - t_ms) / 1000.0 for i in range(i0, i1)]
        ax_w = [ax[i] for i in range(i0, i1)]
        ay_w = [ay[i] for i in range(i0, i1)]
        az_w = [az[i] for i in range(i0, i1)]
        gx_w = [gx[i] for i in range(i0, i1)]
        gy_w = [gy[i] for i in range(i0, i1)]
        gz_w = [gz[i] for i in range(i0, i1)]

        fig = plt.figure(figsize=(12, 5.2))
        ax_left = fig.add_subplot(1, 1, 1)
        ax_right = ax_left.twinx()

        ax_left.axvspan(pred_start_ms / 1000.0, pred_end_ms / 1000.0, color="gold", alpha=0.22, label="prediction window")
        ax_left.axvline(0.0, color="red", linewidth=1.0, alpha=0.9, label="trigger")

        # Accel (left)
        l1 = ax_left.plot(t_rel, ax_w, color="#1f77b4", linewidth=0.9, label="ax")
        l2 = ax_left.plot(t_rel, ay_w, color="#2ca02c", linewidth=0.9, label="ay")
        l3 = ax_left.plot(t_rel, az_w, color="#ff7f0e", linewidth=0.9, label="az")

        # Gyro (right)
        r1 = ax_right.plot(t_rel, gx_w, color="#d62728", linewidth=0.9, linestyle="--", label="gx")
        r2 = ax_right.plot(t_rel, gy_w, color="#9467bd", linewidth=0.9, linestyle="--", label="gy")
        r3 = ax_right.plot(t_rel, gz_w, color="#8c564b", linewidth=0.9, linestyle="--", label="gz")

        ax_left.grid(True, alpha=0.25)
        ax_left.set_xlabel("time relative to trigger (s)")
        ax_left.set_ylabel("accel (raw)")
        ax_right.set_ylabel("gyro (raw)")

        yl = robust_ylim([float(v) for v in ax_w + ay_w + az_w])
        if yl is not None:
            ax_left.set_ylim(*yl)
        yr = robust_ylim([float(v) for v in gx_w + gy_w + gz_w])
        if yr is not None:
            ax_right.set_ylim(*yr)

        handles = [*l1, *l2, *l3, *r1, *r2, *r3]
        labels = [h.get_label() for h in handles]
        ax_left.legend(handles, labels, loc="upper right", frameon=False, ncols=6, fontsize=9)

        fig.suptitle(
            f"Shot {shot_num}: trigger@{t_ms} ms | crop [{-args.shot_pre_ms},{args.shot_post_ms}] ms | pred [{pred_start_ms},{pred_end_ms}] ms",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        out = shots_dir / f"shot_{shot_num:03d}_t{t_ms}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)

        index_rows.append(
            {
                "shot_num": shot_num,
                "trigger_timestamp_ms": t_ms,
                "crop_start_ms": w_start,
                "crop_end_ms": w_end,
                "prediction_window_start_ms": t_ms + pred_start_ms,
                "prediction_window_end_ms": t_ms + pred_end_ms,
                "png": str(out.relative_to(outdir)),
            }
        )

    # Write shot index
    index_path = shots_dir / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["shot_num"])
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    print(f"Wrote: {outdir / 'dataset.png'}")
    print(f"Wrote: {shots_dir} ({len(index_rows)} shots)")

    # Training dataset export (raw windows)
    if args.export_dataset:
        import random

        if dt_med <= 0:
            raise SystemExit("Invalid timestamps (no positive dt)")
        lead_samples = max(1, int(round(args.lead_ms / dt_med)))
        window_samples = max(1, int(round(args.window_ms / dt_med)))

        # Positives: [idx - lead - window, idx - lead)
        pos_windows: list[tuple[int, int, int, int]] = []  # (shot_num, edge_i, start_i, end_i)
        shot_ts = [ts[idx] for idx in accepted_edges]
        for shot_num, edge_i in enumerate(accepted_edges):
            end_i = edge_i - lead_samples
            start_i = end_i - window_samples
            if start_i < 0 or end_i <= start_i:
                continue
            pos_windows.append((shot_num, edge_i, start_i, end_i))

        # Motion distribution of positives (used for optional motion-matching of negatives)
        pos_motion: list[float] = []
        for _shot_num, _edge_i, start_i, end_i in pos_windows:
            pos_motion.append(motion_score_window(ax, ay, az, gx, gy, gz, start_i, end_i))
        pos_motion_sorted = sorted(pos_motion)
        pos_lo = None
        pos_hi = None
        if args.neg_match_pos_motion and pos_motion_sorted:
            q = max(0.0, min(0.49, float(args.neg_pos_motion_q)))
            lo_i = int(math.floor(q * (len(pos_motion_sorted) - 1)))
            hi_i = int(math.ceil((1.0 - q) * (len(pos_motion_sorted) - 1)))
            pos_lo = pos_motion_sorted[lo_i]
            pos_hi = pos_motion_sorted[hi_i]

        # Build a mask of indices we don't want to sample negatives from (shot-adjacent and post-shot).
        avoid = bytearray(n)
        avoid_post = max(0, args.neg_avoid_post_ms)
        avoid_edges = accepted_edges if args.neg_avoid_edges == "accepted" else all_edges
        for edge_i in avoid_edges:
            t0_ms = ts[edge_i]
            lo_t = t0_ms - (args.lead_ms + args.window_ms)  # avoid the actual positive window region
            hi_t = t0_ms + avoid_post
            lo_i = bisect.bisect_left(ts, lo_t)
            hi_i = bisect.bisect_right(ts, hi_t)
            for j in range(lo_i, hi_i):
                avoid[j] = 1

        rng = random.Random(args.seed)
        neg_ends: list[int] = []

        def window_has_trigger(start_i: int, end_i: int) -> bool:
            # Ensure windows are from trigger=0 time (no partial trigger pulls).
            return any(trig[i] != 0 for i in range(start_i, end_i))

        def far_candidates() -> list[int]:
            # Candidate window end indices for negatives. Window is [end-window, end)
            candidates: list[tuple[float, int]] = []  # (motion_score, end_i)
            for end_i in range(window_samples, n):
                if avoid[end_i]:
                    continue
                start_i = end_i - window_samples
                if start_i < 0:
                    continue
                if window_has_trigger(start_i, end_i):
                    continue
                mscore = motion_score_window(ax, ay, az, gx, gy, gz, start_i, end_i)
                candidates.append((mscore, end_i))

            if args.neg_min_motion_quantile and candidates:
                q = max(0.0, min(1.0, float(args.neg_min_motion_quantile)))
                scores = sorted(m for m, _ in candidates)
                kq = int(math.floor(q * (len(scores) - 1)))
                thr = scores[kq]
                candidates = [(m, e) for (m, e) in candidates if m >= thr]

            if pos_lo is not None and pos_hi is not None:
                candidates = [(m, e) for (m, e) in candidates if pos_lo <= m <= pos_hi]

            return [e for _m, e in candidates]

        def sample_far(k_total: int) -> list[int]:
            cand_ends = far_candidates()
            if k_total <= 0 or not cand_ends:
                return []
            if args.neg_with_replacement:
                overs = rng.choices(cand_ends, k=k_total * 3)
                return sample_unique_with_min_separation(rng, overs, ts, k_total, args.neg_min_sep_ms)
            k_total = min(len(cand_ends), k_total)
            return sample_unique_with_min_separation(rng, cand_ends, ts, k_total, args.neg_min_sep_ms)

        def sample_near(k_per_shot: int) -> list[int]:
            # Near negatives: for each positive window, sample negative windows that end shortly
            # BEFORE the positive window starts, to keep posture/context similar.
            # positive window: [pos_start, pos_end)
            # negative end is in [pos_start - neg_near_max_ms, pos_start - neg_near_min_ms]
            if k_per_shot <= 0:
                return []
            out: list[int] = []
            min_ms = max(0, args.neg_near_min_ms)
            max_ms = max(min_ms, args.neg_near_max_ms)

            for _shot_num, _edge_i, pos_start_i, _pos_end_i in pos_windows:
                pos_start_t = ts[pos_start_i]
                lo_t = pos_start_t - max_ms
                hi_t = pos_start_t - min_ms
                if hi_t <= lo_t:
                    continue
                lo_i = bisect.bisect_left(ts, lo_t)
                hi_i = bisect.bisect_right(ts, hi_t)
                if hi_i - lo_i <= 0:
                    continue
                # Candidate ends in this band.
                band: list[tuple[float, int]] = []
                for end_i in range(max(window_samples, lo_i), hi_i):
                    if avoid[end_i]:
                        continue
                    start_i = end_i - window_samples
                    if start_i < 0:
                        continue
                    if window_has_trigger(start_i, end_i):
                        continue
                    mscore = motion_score_window(ax, ay, az, gx, gy, gz, start_i, end_i)
                    band.append((mscore, end_i))
                if not band:
                    continue
                if args.neg_min_motion_quantile:
                    q = max(0.0, min(1.0, float(args.neg_min_motion_quantile)))
                    scores = sorted(m for m, _ in band)
                    kq = int(math.floor(q * (len(scores) - 1)))
                    thr = scores[kq]
                    band = [(m, e) for (m, e) in band if m >= thr]
                    if not band:
                        continue
                if pos_lo is not None and pos_hi is not None:
                    band = [(m, e) for (m, e) in band if pos_lo <= m <= pos_hi]
                    if not band:
                        continue

                ends = [e for _m, e in band]
                if args.neg_with_replacement:
                    overs = rng.choices(ends, k=k_per_shot * 3)
                    out.extend(sample_unique_with_min_separation(rng, overs, ts, k_per_shot, args.neg_min_sep_ms))
                elif k_per_shot >= len(ends):
                    out.extend(ends)
                else:
                    out.extend(sample_unique_with_min_separation(rng, ends, ts, k_per_shot, args.neg_min_sep_ms))
            return out

        if args.neg_strategy == "far":
            # Candidate window end indices for negatives. Window is [end-window, end)
            neg_target = len(pos_windows) * max(0, args.neg_mult)
            neg_ends = sample_far(neg_target)
        elif args.neg_strategy == "near":
            neg_ends = sample_near(max(0, args.neg_mult))
        else:  # mixed
            frac = max(0.0, min(1.0, float(args.neg_mixed_near_frac)))
            near_k = int(round(max(0, args.neg_mult) * frac))
            far_total = len(pos_windows) * max(0, args.neg_mult)
            neg_ends = sample_near(near_k)
            remaining = max(0, far_total - len(neg_ends))
            neg_ends.extend(sample_far(remaining))
            # Final de-dup + separation pass across combined strategies.
            neg_ends = sample_unique_with_min_separation(rng, neg_ends, ts, far_total, args.neg_min_sep_ms)

        def nearest_shot_num(t_ms: int) -> int:
            if not shot_ts:
                return -1
            j = bisect.bisect_left(shot_ts, t_ms)
            if j <= 0:
                return 0
            if j >= len(shot_ts):
                return len(shot_ts) - 1
            before = shot_ts[j - 1]
            after = shot_ts[j]
            return j - 1 if (t_ms - before) <= (after - t_ms) else j

        # Write binary windows + labels + index.
        windows_path = train_dir / "windows_i16le.bin"
        labels_path = train_dir / "labels_u8.bin"
        index_path = train_dir / "index.csv"
        meta_path = train_dir / "meta.json"

        def write_window(f, start_i: int, end_i: int) -> None:
            # Always write exactly window_samples rows.
            # If the CSV has gaps, this is still index-based (best effort).
            if end_i - start_i != window_samples:
                raise RuntimeError("internal: non-fixed window length")
            for i in range(start_i, end_i):
                write_i16le(f, ax[i])
                write_i16le(f, ay[i])
                write_i16le(f, az[i])
                write_i16le(f, gx[i])
                write_i16le(f, gy[i])
                write_i16le(f, gz[i])

        index_rows2: list[dict[str, Any]] = []
        with windows_path.open("wb") as wf, labels_path.open("wb") as lf, index_path.open("w", newline="") as cf:
            wcsv = csv.DictWriter(
                cf,
                fieldnames=[
                    "row",
                    "label",
                    "shot_num",
                    "shot_timestamp_ms",
                    "distance_to_shot_ms",
                    "event_timestamp_ms",
                    "window_start_ms",
                    "window_end_ms",
                    "source",
                ],
            )
            wcsv.writeheader()

            row_num = 0
            # Positives first
            for shot_num, edge_i, start_i, end_i in pos_windows:
                write_window(wf, start_i, end_i)
                lf.write(bytes((1,)))
                t_event = ts[edge_i]
                rec = {
                    "row": row_num,
                    "label": 1,
                    "shot_num": shot_num,
                    "shot_timestamp_ms": t_event,
                    "distance_to_shot_ms": 0,
                    "event_timestamp_ms": t_event,
                    "window_start_ms": ts[start_i],
                    "window_end_ms": ts[end_i - 1],
                    "source": "pos",
                }
                wcsv.writerow(rec)
                index_rows2.append(rec)
                row_num += 1

            # Negatives
            for end_i in neg_ends:
                start_i = end_i - window_samples
                write_window(wf, start_i, end_i)
                lf.write(bytes((0,)))
                t_event = ts[end_i]
                s_num = nearest_shot_num(t_event)
                s_ts = shot_ts[s_num] if 0 <= s_num < len(shot_ts) else 0
                rec = {
                    "row": row_num,
                    "label": 0,
                    "shot_num": s_num,
                    "shot_timestamp_ms": s_ts,
                    "distance_to_shot_ms": int(t_event - s_ts) if s_ts else 0,
                    "event_timestamp_ms": t_event,
                    "window_start_ms": ts[start_i],
                    "window_end_ms": ts[end_i - 1],
                    "source": "neg",
                }
                wcsv.writerow(rec)
                index_rows2.append(rec)
                row_num += 1

        meta = {
            "input_csv": str(args.csv),
            "channels": list(CHANNELS),
            "dtype": "int16",
            "endianness": "little",
            "window_ms": args.window_ms,
            "lead_ms": args.lead_ms,
            "window_samples": window_samples,
            "estimated_dt_ms": dt_med,
            "estimated_hz": hz_est,
            "positives": len(pos_windows),
            "negatives": len(neg_ends),
            "neg_mult": args.neg_mult,
            "neg_strategy": args.neg_strategy,
            "neg_mixed_near_frac": args.neg_mixed_near_frac,
            "neg_min_motion_quantile": args.neg_min_motion_quantile,
            "neg_match_pos_motion": args.neg_match_pos_motion,
            "neg_pos_motion_q": args.neg_pos_motion_q,
            "neg_min_sep_ms": args.neg_min_sep_ms,
            "neg_with_replacement": args.neg_with_replacement,
            "neg_near_min_ms": args.neg_near_min_ms,
            "neg_near_max_ms": args.neg_near_max_ms,
            "neg_avoid_post_ms": args.neg_avoid_post_ms,
            "neg_avoid_edges": args.neg_avoid_edges,
            "format": {
                "windows_i16le.bin": "concatenated windows; each window is window_samples rows of 6 int16 (ax,ay,az,gx,gy,gz)",
                "labels_u8.bin": "one byte per window (0/1), in the same order as windows",
                "index.csv": "row metadata matching the binary order",
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        print(f"Wrote training dataset: {train_dir}")
        print(f"- {windows_path.name}  ({(train_dir / windows_path.name).stat().st_size} bytes)")
        print(f"- {labels_path.name}   ({(train_dir / labels_path.name).stat().st_size} bytes)")
        print(f"- {index_path.name}")
        print(f"- {meta_path.name}")


if __name__ == "__main__":
    main()
