"""Denoise, resample, and encode two-photon fluorescence video."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .srdtrans import denoised_stack
from .tiff import frame_counts, frame_rate, pages


def _denoise(frame: np.ndarray, history: deque[np.ndarray]) -> np.ndarray:
    """Apply the requested rolling average and mild spatial smoothing."""
    history.append(frame)
    signal = np.mean(history, axis=0) if history.maxlen > 1 else frame
    smooth = gaussian_filter(signal.astype(np.float32), sigma=.65)
    return np.clip(np.rint(smooth), 0, np.iinfo(frame.dtype).max).astype(frame.dtype)


def _contrast(files: list[Path], count: int) -> tuple[float, float]:
    """Estimate robust display limits from about 50 sparse frames."""
    stride = max(1, count // 50)
    scan = tqdm(pages(files, count), total=count, desc="Contrast", unit="frame")
    # Spatial subsampling makes percentile estimation faster.
    samples = [f[::4, ::4] for i, f in enumerate(scan) if i % stride == 0]
    low, high = np.percentile(np.concatenate([x.ravel() for x in samples]), (1, 99.8))
    return float(low), float(max(high, low + 1))


def _render(files: list[Path], count: int, fps: float, output: Path,
            rolling: int, speed: float) -> Path:
    """Render an already denoised stack as a green 60 fps video."""
    mp4_path = output / "fov_ch2.mp4"
    print("Analyzing fluorescence contrast...")
    low, high = _contrast(files, count)
    print(f"Display range: {low:.1f} to {high:.1f}.")

    # Map fixed 60 fps output times to fractional source positions.
    video_frames = max(1, round(count / fps / speed * 60))
    positions = np.minimum(np.arange(video_frames) * fps * speed / 60, count - 1)
    print(f"Rendering {video_frames} output frames at 60 fps...")
    writer = imageio.get_writer(mp4_path, fps=60, codec="libx264", quality=8,
                                macro_block_size=None, pixelformat="yuv420p")
    progress = tqdm(total=video_frames, desc="Video", unit="frame")
    history: deque[np.ndarray] = deque(maxlen=rolling)

    def write(frame: np.ndarray) -> None:
        """Map fluorescence to the green channel and encode one frame."""
        gray = np.clip((frame - low) * 255 / (high - low), 0, 255).astype("uint8")
        writer.append_data(np.dstack((np.zeros_like(gray), gray, np.zeros_like(gray))))
        progress.update()

    try:
        previous = None
        out = 0
        for i, raw in enumerate(pages(files, count)):
            current = _denoise(raw, history).astype(np.float32)
            while out < video_frames and positions[out] <= i:
                if previous is None:
                    write(current)
                else:
                    weight = positions[out] - (i - 1)
                    write(previous + weight * (current - previous))
                out += 1
            previous = current
        while out < video_frames:
            write(previous)
            out += 1
    finally:
        progress.close()
        writer.close()
    return mp4_path


def process(files: list[Path], duration: float, output: Path, rolling: int,
            speed: float, gpu: str, patch: int, srd_root: Path,
            model: Path) -> tuple[Path, float, int]:
    """Read, denoise, resample, and encode the selected recording interval."""
    print(f"Found {len(files)} Ch2 TIFF file(s).")
    print("Reading acquisition timing and stack sizes...")
    fps = frame_rate(files)
    available = sum(frame_counts(files))
    # Cover the duration without exceeding the available recording.
    count = min(math.ceil(duration * fps), available)
    if count < 1:
        raise ValueError("The input contains no frames")

    output.mkdir(parents=True, exist_ok=True)
    print(f"Source: {fps:.4f} fps, {available} available frames.")
    print(f"Using {count} frames ({count / fps:.2f} s).")
    print(f"Rolling average: {rolling} frame(s). Playback speed: {speed:g}x.")

    with denoised_stack(files, count, output, gpu, patch, srd_root, model) as denoised:
        mp4_path = _render([denoised], count, fps, output, rolling, speed)

    print(f"Finished: {mp4_path.resolve()}")
    return mp4_path, fps, count
