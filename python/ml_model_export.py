#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import zlib
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
    return meta, X, y


def featurize(X, kind: str):
    import numpy as np

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
    raise SystemExit(f"Unknown features: {kind}")


def build_mlmd_file(
    *,
    window_samples: int,
    model_type: str,
    scaler_mean,
    scaler_scale,
    lr_coef=None,
    lr_intercept=None,
    mlp_w1=None,
    mlp_b1=None,
    mlp_w2=None,
    mlp_b2=None,
    mlp_w3=None,
    mlp_b3=None,
) -> bytes:
    import numpy as np

    def f32_bytes(arr) -> bytes:
        a = np.asarray(arr, dtype="<f4")
        return a.tobytes(order="C")

    if model_type == "lr":
        if lr_coef is None or lr_intercept is None:
            raise ValueError("missing lr params")
        features = int(np.asarray(lr_coef).shape[0])
        payload = b"".join(
            [
                f32_bytes(scaler_mean),
                f32_bytes(scaler_scale),
                f32_bytes(lr_coef),
                f32_bytes([float(lr_intercept)]),
            ]
        )
        payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(
            "<4sHHBBHHHII",
            b"MLMD",
            1,  # version
            int(window_samples),
            0,  # modelType LR
            0,  # reserved
            int(features),
            0,  # h1
            0,  # h2
            len(payload),
            payload_crc,
        )
        return header + payload

    if model_type == "mlp":
        for name, v in [
            ("mlp_w1", mlp_w1),
            ("mlp_b1", mlp_b1),
            ("mlp_w2", mlp_w2),
            ("mlp_b2", mlp_b2),
            ("mlp_w3", mlp_w3),
            ("mlp_b3", mlp_b3),
        ]:
            if v is None:
                raise ValueError(f"missing {name}")
        features = int(np.asarray(scaler_mean).shape[0])
        h1 = int(np.asarray(mlp_b1).shape[0])
        h2 = int(np.asarray(mlp_b2).shape[0])
        payload = b"".join(
            [
                f32_bytes(scaler_mean),
                f32_bytes(scaler_scale),
                f32_bytes(mlp_w1),
                f32_bytes(mlp_b1),
                f32_bytes(mlp_w2),
                f32_bytes(mlp_b2),
                f32_bytes(mlp_w3),
                f32_bytes([float(mlp_b3)]),
            ]
        )
        payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack(
            "<4sHHBBHHHII",
            b"MLMD",
            1,  # version
            int(window_samples),
            1,  # modelType MLP
            0,  # reserved
            int(features),
            int(h1),
            int(h2),
            len(payload),
            payload_crc,
        )
        return header + payload

    raise ValueError(model_type)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and export an MLMD model.bin for on-device upload (no UF2).")
    ap.add_argument("--train-dir", type=Path, default=Path("ml_out/train"), help="Dataset directory (default: ml_out/train)")
    ap.add_argument("--out", type=Path, default=Path("ml_model.bin"), help="Output MLMD file (single-model mode)")
    ap.add_argument("--out-lr", type=Path, default=Path("ml_model_lr.bin"), help="Output LR MLMD file (default: ml_model_lr.bin)")
    ap.add_argument("--out-mlp", type=Path, default=Path("ml_model_mlp.bin"), help="Output MLP MLMD file (default: ml_model_mlp.bin)")
    ap.add_argument("--model", choices=["both", "lr", "mlp"], default="both", help="What to export (default: both)")
    ap.add_argument("--seed", type=int, default=1, help="Seed (default: 1)")
    ap.add_argument("--mlp-hidden", default="64,32", help='MLP hidden sizes, e.g. "64,32" (default: 64,32)')
    ap.add_argument("--mlp-alpha", type=float, default=1e-4, help="MLP L2 alpha (default: 1e-4)")
    args = ap.parse_args()

    try:
        import numpy as np
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        raise SystemExit("Missing deps. Install with:\n  ./.venv/bin/python -m pip install numpy scikit-learn")

    meta, X, y = load_dataset(args.train_dir)
    window_samples = int(meta["window_samples"])

    def train_export_lr(out_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression

        F = featurize(X, "summary")
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=args.seed))
        pipe.fit(F, y)

        scaler: StandardScaler = pipe.steps[0][1]  # type: ignore[assignment]
        clf: LogisticRegression = pipe.steps[1][1]  # type: ignore[assignment]

        coef = clf.coef_.reshape(-1).astype(np.float32)
        intercept = float(clf.intercept_.reshape(-1)[0])
        blob = build_mlmd_file(
            window_samples=window_samples,
            model_type="lr",
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
            lr_coef=coef,
            lr_intercept=intercept,
        )
        out_path.write_bytes(blob)
        file_crc = zlib.crc32(blob) & 0xFFFFFFFF
        print("Wrote:", out_path)
        print("MLMODEL_PUT_LR args:")
        print(f"- size: {len(blob)}")
        print(f"- crc32: {file_crc:08x}")

    def train_export_mlp(out_path: Path) -> None:
        from sklearn.neural_network import MLPClassifier

        hidden = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip())
        if hidden != (64, 32):
            raise SystemExit("Firmware currently expects MLP hidden sizes (64,32).")

        F = featurize(X, "rich")
        pipe = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=args.mlp_alpha,
                max_iter=800,
                random_state=args.seed,
                early_stopping=False,
            ),
        )
        pipe.fit(F, y)

        scaler: StandardScaler = pipe.steps[0][1]  # type: ignore[assignment]
        clf: MLPClassifier = pipe.steps[1][1]  # type: ignore[assignment]

        # sklearn stores coefs as [in,out] per layer. Firmware expects row-major [out][in].
        w1 = clf.coefs_[0].T.astype(np.float32).reshape(-1)
        b1 = clf.intercepts_[0].astype(np.float32)
        w2 = clf.coefs_[1].T.astype(np.float32).reshape(-1)
        b2 = clf.intercepts_[1].astype(np.float32)
        w3 = clf.coefs_[2].astype(np.float32).reshape(-1)  # (32,1) -> 32
        b3 = float(clf.intercepts_[2].reshape(-1)[0])

        blob2 = build_mlmd_file(
            window_samples=window_samples,
            model_type="mlp",
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
            mlp_w1=w1,
            mlp_b1=b1,
            mlp_w2=w2,
            mlp_b2=b2,
            mlp_w3=w3,
            mlp_b3=b3,
        )
        out_path.write_bytes(blob2)
        file_crc = zlib.crc32(blob2) & 0xFFFFFFFF
        print("Wrote:", out_path)
        print("MLMODEL_PUT_MLP args:")
        print(f"- size: {len(blob2)}")
        print(f"- crc32: {file_crc:08x}")

    if args.model == "lr":
        train_export_lr(args.out)
    elif args.model == "mlp":
        train_export_mlp(args.out)
    else:
        train_export_lr(args.out_lr)
        train_export_mlp(args.out_mlp)


if __name__ == "__main__":
    main()
