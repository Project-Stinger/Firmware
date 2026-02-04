#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO_ROOT / "ml_web" / "_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stinger ML Personalization (MVP)")


def _import_repo_modules():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "python"))

    import ml_log_convert  # type: ignore
    import ml_log_explore  # type: ignore
    import ml_model_export  # type: ignore

    return ml_log_convert, ml_log_explore, ml_model_export


def _detect_shots(ts: list[int], trig: list[int], *, min_gap_ms: int = 1000) -> tuple[list[int], list[int]]:
    """Return (rising_edges_all_idx, accepted_idx)."""
    rising = [i for i in range(1, len(trig)) if trig[i - 1] == 0 and trig[i] == 1]
    accepted: list[int] = []
    last_t = -10**18
    for i in rising:
        if ts[i] - last_t >= min_gap_ms:
            accepted.append(i)
            last_t = ts[i]
    return rising, accepted


def _plot_shot(
    *,
    ts: list[int],
    ax: list[int],
    ay: list[int],
    az: list[int],
    gx: list[int],
    gy: list[int],
    gz: list[int],
    trig: list[int],
    i0: int,
    i1: int,
    trigger_i: int,
    prob_lr: list[float],
    prob_mlp: list[float],
    lead_ms: int = 100,
    window_ms: int = 500,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = ts[trigger_i]
    pre0 = t0 - (lead_ms + window_ms)
    pre1 = t0 - lead_ms

    x_ms = [ts[i] - t0 for i in range(i0, i1)]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.5, 6.5), dpi=160, sharex=True, constrained_layout=True)

    # prediction window shading
    ax0.axvspan(pre0 - t0, pre1 - t0, color="0.85", alpha=0.6, zorder=0)
    ax0.axvline(0, color="red", lw=1.2, alpha=0.9)
    ax0.plot(x_ms, prob_lr, label="LR", color="#38bdf8", lw=1.6)
    ax0.plot(x_ms, prob_mlp, label="MLP", color="#22c55e", lw=1.6)
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_ylabel("p(about to shoot)")
    ax0.grid(True, alpha=0.15)
    ax0.legend(loc="upper left", ncol=2, frameon=False)

    # IMU axes
    ax1.axvspan(pre0 - t0, pre1 - t0, color="0.85", alpha=0.6, zorder=0)
    ax1.axvline(0, color="red", lw=1.2, alpha=0.9, label="trigger")
    ax1.plot(x_ms, [ax[i] for i in range(i0, i1)], label="ax", lw=1.0)
    ax1.plot(x_ms, [ay[i] for i in range(i0, i1)], label="ay", lw=1.0)
    ax1.plot(x_ms, [az[i] for i in range(i0, i1)], label="az", lw=1.0)
    ax1.plot(x_ms, [gx[i] for i in range(i0, i1)], label="gx", lw=1.0)
    ax1.plot(x_ms, [gy[i] for i in range(i0, i1)], label="gy", lw=1.0)
    ax1.plot(x_ms, [gz[i] for i in range(i0, i1)], label="gz", lw=1.0)
    ax1.set_ylabel("raw i16")
    ax1.set_xlabel("time relative to trigger (ms)")
    ax1.grid(True, alpha=0.15)
    ax1.legend(loc="upper left", ncol=6, frameon=False, fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()

def _score_good_prediction(x_ms: list[int], probs: list[float]) -> tuple[bool, float, float, float]:
    """Return (is_good, score, p_low, p_high).

    Heuristic: a good pre-shot predictor should be low earlier, then rise high
    during the prediction window before trigger.
    """
    import numpy as np

    if not x_ms or not probs or len(x_ms) != len(probs):
        return False, 0.0, 0.0, 0.0

    x = np.asarray(x_ms, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float32)

    # Baseline: earlier part of the crop window (far from trigger)
    base_mask = (x >= -1000) & (x <= -700)
    if base_mask.sum() < 5:
        base_mask = x <= -600
    if base_mask.sum() < 5:
        base_mask = np.ones_like(x, dtype=bool)

    # Prediction window: 100–600ms before trigger
    pred_mask = (x >= -(100 + 500)) & (x <= -100)
    if pred_mask.sum() < 5:
        pred_mask = (x >= -600) & (x <= -50)

    if pred_mask.sum() < 5:
        return False, 0.0, 0.0, 0.0

    p_low = float(np.percentile(p[base_mask], 10))
    p_high = float(np.percentile(p[pred_mask], 90))
    delta = p_high - p_low

    # "Near zero to high"
    is_good = (p_low <= 0.25) and (p_high >= 0.65) and (delta >= 0.40)
    score = float(delta + 0.25 * p_high - 0.10 * p_low)
    return is_good, score, p_low, p_high


@app.get("/", response_class=HTMLResponse)
def index():
    return (REPO_ROOT / "ml_web" / "index.html").read_text(encoding="utf-8")

@app.get("/api/ping")
def ping():
    return {"ok": True, "time": time.time()}


@app.post("/api/train")
async def train(req: Request):
    # Persist job artifacts for debugging
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    raw = await req.body()
    if not raw or len(raw) < 100:
        return Response("Empty/too small log", status_code=400)

    bin_path = job_dir / "ml_log.bin"
    csv_path = job_dir / "ml_log.csv"
    bin_path.write_bytes(raw)

    ml_log_convert, ml_log_explore, ml_model_export = _import_repo_modules()

    # Convert
    ml_log_convert.convert(str(bin_path), str(csv_path))

    # Read arrays
    ts, ax, ay, az, gx, gy, gz, trig = ml_log_explore.read_ml_csv(csv_path)

    # Use same defaults as the training pipeline
    lead_ms = 100
    window_ms = 500
    window_samples = 50
    shot_pre_ms = 1000
    shot_post_ms = 200
    min_gap_ms = 1000
    max_shots = 6

    rising_all, accepted = _detect_shots(ts, trig, min_gap_ms=min_gap_ms)
    rejected = len(rising_all) - len(accepted)

    summary = {
        "input_csv": str(csv_path),
        "samples": len(ts),
        "duration_ms": (ts[-1] - ts[0]) if ts else 0,
        "trigger": {
            "rising_edges_all": len(rising_all),
            "rising_edges_accepted": len(accepted),
            "rising_edges_rejected": rejected,
            "min_shot_gap_ms": min_gap_ms,
            "rejected_reason": "too close to previous accepted shot",
            "shot_pre_ms": shot_pre_ms,
            "shot_post_ms": shot_post_ms,
            "lead_ms": lead_ms,
            "window_ms": window_ms,
        },
        "viz": {
            "max_shots": max_shots,
            "selected_policy": "Only show shots where either LR or MLP goes from near-zero baseline to high probability in the 100–600ms pre-trigger window.",
        },
    }

    # Export dataset using the existing tool (keeps behavior consistent)
    outdir = job_dir / "ml_out"
    cmd = [
        os.fspath(REPO_ROOT / ".venv" / "bin" / "python"),
        os.fspath(REPO_ROOT / "python" / "ml_log_explore.py"),
        os.fspath(csv_path),
        "--outdir",
        os.fspath(outdir),
        "--clean",
        "--export-dataset",
    ]
    # Fall back to current interpreter if venv path is missing
    if not (REPO_ROOT / ".venv" / "bin" / "python").exists():
        import sys

        cmd[0] = sys.executable
    import subprocess

    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        return Response("ml_log_explore failed:\n" + p.stderr + "\n" + p.stdout, status_code=500)

    train_dir = outdir / "train"
    meta, X, y = ml_model_export.load_dataset(train_dir)

    # Train both models (same as exporter)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    F_lr = ml_model_export.featurize(X, "summary")
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=1))
    pipe_lr.fit(F_lr, y)

    F_mlp = ml_model_export.featurize(X, "rich")
    pipe_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=800,
            random_state=1,
            early_stopping=False,
        ),
    )
    pipe_mlp.fit(F_mlp, y)

    scaler_lr: StandardScaler = pipe_lr.steps[0][1]  # type: ignore[assignment]
    clf_lr: LogisticRegression = pipe_lr.steps[1][1]  # type: ignore[assignment]
    lr_blob = ml_model_export.build_mlmd_file(
        window_samples=int(meta["window_samples"]),
        model_type="lr",
        scaler_mean=scaler_lr.mean_.astype(np.float32),
        scaler_scale=scaler_lr.scale_.astype(np.float32),
        lr_coef=clf_lr.coef_.reshape(-1).astype(np.float32),
        lr_intercept=float(clf_lr.intercept_.reshape(-1)[0]),
    )

    scaler_mlp: StandardScaler = pipe_mlp.steps[0][1]  # type: ignore[assignment]
    clf_mlp: MLPClassifier = pipe_mlp.steps[1][1]  # type: ignore[assignment]
    w1 = clf_mlp.coefs_[0].T.astype(np.float32).reshape(-1)
    b1 = clf_mlp.intercepts_[0].astype(np.float32)
    w2 = clf_mlp.coefs_[1].T.astype(np.float32).reshape(-1)
    b2 = clf_mlp.intercepts_[1].astype(np.float32)
    w3 = clf_mlp.coefs_[2].astype(np.float32).reshape(-1)
    b3 = float(clf_mlp.intercepts_[2].reshape(-1)[0])
    mlp_blob = ml_model_export.build_mlmd_file(
        window_samples=int(meta["window_samples"]),
        model_type="mlp",
        scaler_mean=scaler_mlp.mean_.astype(np.float32),
        scaler_scale=scaler_mlp.scale_.astype(np.float32),
        mlp_w1=w1,
        mlp_b1=b1,
        mlp_w2=w2,
        mlp_b2=b2,
        mlp_w3=w3,
        mlp_b3=b3,
    )

    # Build per-shot plots with predictions
    shots_out = []
    # Map timestamps to indices for crop window
    ts_np = np.asarray(ts, dtype=np.int64)
    scored: list[tuple[float, int, bytes]] = []
    for trig_i in accepted:
        t0 = ts[trig_i]
        t_start = t0 - shot_pre_ms
        t_end = t0 + shot_post_ms
        i0 = int(np.searchsorted(ts_np, t_start, side="left"))
        i1 = int(np.searchsorted(ts_np, t_end, side="right"))
        i0 = max(0, i0)
        i1 = min(len(ts), i1)
        if i1 - i0 < window_samples + 2:
            continue

        # Compute probs aligned to samples in [i0, i1)
        probs_lr: list[float] = []
        probs_mlp: list[float] = []
        for i in range(i0, i1):
            if i < window_samples:
                probs_lr.append(0.0)
                probs_mlp.append(0.0)
                continue
            w = np.stack(
                [
                    np.asarray(ax[i - window_samples : i], dtype=np.float32),
                    np.asarray(ay[i - window_samples : i], dtype=np.float32),
                    np.asarray(az[i - window_samples : i], dtype=np.float32),
                    np.asarray(gx[i - window_samples : i], dtype=np.float32),
                    np.asarray(gy[i - window_samples : i], dtype=np.float32),
                    np.asarray(gz[i - window_samples : i], dtype=np.float32),
                ],
                axis=1,
            )  # (50,6)
            w = w.reshape(1, window_samples, 6)
            f1 = ml_model_export.featurize(w, "summary")
            f2 = ml_model_export.featurize(w, "rich")
            probs_lr.append(float(pipe_lr.predict_proba(f1)[0, 1]))
            probs_mlp.append(float(pipe_mlp.predict_proba(f2)[0, 1]))

        x_ms = [ts[i] - t0 for i in range(i0, i1)]
        good_lr, score_lr, _, _ = _score_good_prediction(x_ms, probs_lr)
        good_mlp, score_mlp, _, _ = _score_good_prediction(x_ms, probs_mlp)
        good = good_lr or good_mlp
        score = max(score_lr, score_mlp)

        if not good:
            continue

        png = _plot_shot(
            ts=ts,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            trig=trig,
            i0=i0,
            i1=i1,
            trigger_i=trig_i,
            prob_lr=probs_lr,
            prob_mlp=probs_mlp,
            lead_ms=lead_ms,
            window_ms=window_ms,
        )

        scored.append((score, trig_i, png))

    scored.sort(key=lambda t: t[0], reverse=True)
    for score, trig_i, png in scored[:max_shots]:
        shots_out.append({"png_b64": base64.b64encode(png).decode("ascii")})

    return {
        "job_id": job_id,
        "summary": summary,
        "shots": shots_out,
        "lr_b64": base64.b64encode(lr_blob).decode("ascii"),
        "mlp_b64": base64.b64encode(mlp_blob).decode("ascii"),
    }

app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "ml_web")), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
