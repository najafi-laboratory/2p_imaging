"""Define the command-line interface."""

import argparse
from pathlib import Path

from tiff_io import find_ch2
from .video import process

ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Turn sequential Ch2 two-photon TIFF stacks into an MP4.")
    parser.add_argument("data", type=Path, help="raw-data directory containing Ch2 TIFF files")
    parser.add_argument("--duration", type=float, required=True, help="duration to process, in seconds")
    parser.add_argument("--output", type=Path, default=Path("output"), help="output directory")
    parser.add_argument("--rolling-average", type=int, default=1, metavar="FRAMES",
                        help="number of frames in the rolling average (default: 1)")
    parser.add_argument("--speed", type=float, default=1,
                        help="playback speed, e.g. 0.5 or 2 (default: 1)")
    parser.add_argument("--gpu", default="all",
                        help="CUDA devices, e.g. 0,1, or all (default: all)")
    parser.add_argument("--model", type=Path, default=ROOT / "trained_model",
                        help="folder containing SRDTrans .pth checkpoints")
    parser.add_argument("--patch", type=int, default=128,
                        help="SRDTrans spatial and temporal patch size (default: 128)")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.rolling_average < 1:
        parser.error("--rolling-average must be at least 1")
    if args.patch < 8 or args.patch % 8:
        parser.error("--patch must be at least 8 and divisible by 8")
    if not args.data.is_dir():
        parser.error(f"data directory does not exist: {args.data}")
    return args


def main() -> None:
    """Run the command-line conversion."""
    args = _arguments()
    video, fps, frames = process(
        find_ch2(args.data), args.duration, args.output, args.rolling_average, args.speed,
        args.gpu, args.patch, args.model)
    print(f"Processed {frames} source frames at {fps:.4f} fps; video is 60 fps at {args.speed:g}x"
          f"\nVideo: {video}")
