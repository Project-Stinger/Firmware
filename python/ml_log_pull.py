#!/usr/bin/env python3
"""
Download `ml_log.bin` from Stinger over USB serial (CDC).

Firmware command: send "MLDUMP\\n", firmware replies:
  "MLDUMP1 <size>\\n" + <size raw bytes> + "\\nMLDUMP_DONE\\n"

Usage:
  python ml_log_pull.py --port /dev/ttyACM0
  python ml_log_pull.py  # auto-picks the first matching port
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except ModuleNotFoundError as e:
    if e.name == "serial":
        exe = sys.executable or "python3"
        print("Missing dependency: pyserial (provides the `serial` module).", file=sys.stderr)
        print("", file=sys.stderr)
        print("Recommended (works on Homebrew/macOS + avoids PEP 668 issues):", file=sys.stderr)
        print(f"  {exe} -m venv .venv", file=sys.stderr)
        print("  ./.venv/bin/python -m pip install pyserial", file=sys.stderr)
        print("  ./.venv/bin/python python/ml_log_pull.py", file=sys.stderr)
        print("", file=sys.stderr)
        print("If you use conda, install `pyserial` into that environment and run the script with that python.", file=sys.stderr)
    raise
except Exception:
    print("Failed importing pyserial (`serial`). Try installing `pyserial` and re-run.", file=sys.stderr)
    raise


HEADER_RE = re.compile(rb"^MLDUMP1\s+(\d+)\s*$")


def pick_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise SystemExit("No serial ports found")

    preferred = []
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if "pico" in desc or "rp2040" in desc or "tinyusb" in desc or "2e8a" in hwid:
            preferred.append(p.device)
    if preferred:
        return preferred[0]
    return ports[0].device


def read_line(ser: serial.Serial, timeout_s: float) -> bytes:
    deadline = time.time() + timeout_s
    buf = bytearray()
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            return bytes(buf).strip()
        if b != b"\r":
            buf += b
    raise TimeoutError("Timed out waiting for line")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Serial port (e.g. COM5, /dev/ttyACM0). Auto-detect if omitted.")
    ap.add_argument("-o", "--out", default="ml_log.bin", help="Output path (default: ml_log.bin)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = ap.parse_args()

    port = args.port or pick_port()
    out_path = Path(args.out)

    with serial.Serial(port, args.baud, timeout=0.1) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.write(b"MLDUMP\n")
        ser.flush()

        # Scan for header (ignore any other log lines).
        size = None
        for _ in range(200):
            line = read_line(ser, timeout_s=2.0)
            m = HEADER_RE.match(line)
            if m:
                size = int(m.group(1))
                break
        if size is None:
            raise SystemExit("Did not receive MLDUMP1 header; is the ml logger firmware running?")

        remaining = size
        with out_path.open("wb") as out:
            while remaining:
                chunk = ser.read(min(4096, remaining))
                if not chunk:
                    continue
                out.write(chunk)
                remaining -= len(chunk)

        # Best-effort: read trailer line.
        try:
            trailer = read_line(ser, timeout_s=2.0)
            if trailer:
                print(trailer.decode(errors="replace"))
        except Exception:
            pass

    print(f"Wrote {size} bytes -> {out_path}")


if __name__ == "__main__":
    main()
