#!/usr/bin/env python3
"""Generate waveform PNG from a video file using ffmpeg.

Usage:
  python generate_waveform.py /path/to/video.mp4 /path/to/output.png

Requires:
  ffmpeg available in PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate waveform PNG from a video file.")
    parser.add_argument("input", type=Path, help="Input video file (e.g., .mp4)")
    parser.add_argument("output", type=Path, help="Output PNG file")
    parser.add_argument("--size", default="1280x300", help="Waveform image size, e.g. 1280x300")
    parser.add_argument("--color", default="DodgerBlue", help="Waveform color (ffmpeg color name)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH. Please install ffmpeg and try again.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    filter_expr = f"showwavespic=s={args.size}:colors={args.color}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(args.input),
        "-filter_complex",
        filter_expr,
        "-frames:v",
        "1",
        str(args.output),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode("utf-8", errors="ignore"), file=sys.stderr)
        return exc.returncode

    print(f"Saved waveform to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
