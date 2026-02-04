#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path


def read_until(ser, needles: list[bytes], timeout_s: float = 5.0) -> bytes:
    end = time.time() + timeout_s
    buf = b""
    while time.time() < end:
        buf += ser.read(ser.in_waiting or 1)
        for n in needles:
            if n in buf:
                return buf
    return buf


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload an MLMD model.bin to the device over serial and load it.")
    ap.add_argument("--port", default="", help="Serial port (e.g. /dev/cu.usbmodemXXXX). Auto-detect if omitted.")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=None,
        help="Single-model upload (legacy): send MLMODEL_PUT + MLMODEL_LOAD for one file (e.g. ml_model.bin)",
    )
    ap.add_argument("--lr", type=Path, default=Path("ml_model_lr.bin"), help="LR model file (default: ml_model_lr.bin)")
    ap.add_argument("--mlp", type=Path, default=Path("ml_model_mlp.bin"), help="MLP model file (default: ml_model_mlp.bin)")
    ap.add_argument("--chunk", type=int, default=512, help="Write chunk size (default: 512)")
    ap.add_argument("--no-load", action="store_true", help="Only upload; do not send MLMODEL_LOAD")
    ap.add_argument("--delete", action="store_true", help="Delete user model instead of uploading")
    ap.add_argument("--both", action="store_true", help="Upload both LR+MLP (default if --in not set)")
    ap.add_argument("--info", action="store_true", help="Only query MLMODEL_INFO; do not upload/delete/load")
    args = ap.parse_args()

    try:
        import serial
        from serial.tools import list_ports
    except Exception:
        raise SystemExit("Missing dependency: pyserial. Install with:\n  ./.venv/bin/python -m pip install pyserial")

    port = args.port
    if not port:
        ports = [p.device for p in list_ports.comports() if p.device and "usbmodem" in p.device]
        if len(ports) == 1:
            port = ports[0]
        else:
            raise SystemExit("Could not auto-detect port. Pass --port.")

    with serial.Serial(port, args.baud, timeout=1) as ser:
        ser.reset_input_buffer()

        if args.info:
            ser.write(b"MLMODEL_INFO\n")
            ser.flush()
            time.sleep(0.3)
            print(ser.read(ser.in_waiting or 1).decode(errors="replace"), end="")
            return

        if args.delete:
            ser.write(b"MLMODEL_DELETE\n")
            ser.flush()
            out = read_until(ser, [b"MLMODEL_DELETED", b"MLMODEL_ERR"], timeout_s=2.0)
            print(out.decode(errors="replace"))
            ser.write(b"MLMODEL_INFO\n")
            ser.flush()
            time.sleep(0.2)
            print(ser.read(ser.in_waiting or 1).decode(errors="replace"))
            return

        def upload_one(cmd_prefix: str, path: Path) -> None:
            if not path.is_file():
                raise SystemExit(f"Model file not found: {path}")
            blob = path.read_bytes()
            crc = zlib.crc32(blob) & 0xFFFFFFFF
            ser.write(f"{cmd_prefix} {len(blob)} {crc:08x}\n".encode())
            ser.flush()
            out = read_until(ser, [b"MLMODEL_READY", b"MLMODEL_ERR"], timeout_s=2.0)
            print(out.decode(errors="replace"), end="")
            if b"MLMODEL_READY" not in out:
                raise SystemExit(f"Device did not respond with MLMODEL_READY for {cmd_prefix}")
            for i in range(0, len(blob), args.chunk):
                ser.write(blob[i : i + args.chunk])
            ser.flush()
            out = read_until(ser, [b"MLMODEL_OK", b"MLMODEL_ERR"], timeout_s=10.0)
            print(out.decode(errors="replace"), end="")
            if b"MLMODEL_OK" not in out:
                raise SystemExit(f"Upload failed for {cmd_prefix}")

        if args.inp is not None and args.both:
            raise SystemExit("Use either --in (single legacy file) OR --both (LR+MLP), not both.")

        # Legacy single-file mode (writes /ml_model.bin)
        if args.inp is not None:
            upload_one("MLMODEL_PUT", args.inp)
            if not args.no_load:
                ser.write(b"MLMODEL_LOAD\n")
                ser.flush()
                out = read_until(ser, [b"MLMODEL_LOADED", b"MLMODEL_ERR"], timeout_s=5.0)
                print(out.decode(errors="replace"), end="")
        else:
            # Default: upload both (writes /ml_model_lr.bin and /ml_model_mlp.bin)
            upload_one("MLMODEL_PUT_LR", args.lr)
            upload_one("MLMODEL_PUT_MLP", args.mlp)
            if not args.no_load:
                ser.write(b"MLMODEL_LOAD_LR\n")
                ser.flush()
                out = read_until(ser, [b"MLMODEL_LOADED", b"MLMODEL_ERR"], timeout_s=5.0)
                print(out.decode(errors="replace"), end="")
                ser.write(b"MLMODEL_LOAD_MLP\n")
                ser.flush()
                out = read_until(ser, [b"MLMODEL_LOADED", b"MLMODEL_ERR"], timeout_s=5.0)
                print(out.decode(errors="replace"), end="")

        ser.write(b"MLMODEL_INFO\n")
        ser.flush()
        time.sleep(0.3)
        print(ser.read(ser.in_waiting or 1).decode(errors="replace"), end="")


if __name__ == "__main__":
    main()
