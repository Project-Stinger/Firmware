#!/usr/bin/env python3
"""Convert Stinger ml_log.bin to CSV for ML training.

Usage:
    python ml_log_convert.py ml_log.bin              # outputs ml_log.csv
    python ml_log_convert.py ml_log.bin -o data.csv  # custom output name
"""

import argparse
import struct
import sys

SAMPLE_FORMAT = "<IhhhhhhB"  # u32 + 6x i16 + u8 = 17 bytes
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)
HEADER = "timestamp_ms,ax,ay,az,gx,gy,gz,trigger"


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        data = f.read()

    n_samples = len(data) // SAMPLE_SIZE
    if len(data) % SAMPLE_SIZE != 0:
        print(f"Warning: {len(data) % SAMPLE_SIZE} trailing bytes ignored", file=sys.stderr)

    with open(output_path, "w") as out:
        out.write(HEADER + "\n")
        for i in range(n_samples):
            sample = struct.unpack_from(SAMPLE_FORMAT, data, i * SAMPLE_SIZE)
            out.write(",".join(str(v) for v in sample) + "\n")

    print(f"Converted {n_samples} samples ({n_samples / 100:.1f}s at 100Hz) -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Stinger ML log binary to CSV")
    parser.add_argument("input", help="Path to ml_log.bin")
    parser.add_argument("-o", "--output", help="Output CSV path (default: <input>.csv)")
    args = parser.parse_args()

    output = args.output or args.input.rsplit(".", 1)[0] + ".csv"
    convert(args.input, output)


if __name__ == "__main__":
    main()
