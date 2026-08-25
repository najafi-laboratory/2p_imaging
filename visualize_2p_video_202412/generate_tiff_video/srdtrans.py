"""Direct, GPU-batched inference with the bundled SRDTrans network."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile
from tqdm import tqdm

from tiff_io import pages
from train_model.srdtrans_runtime import network, runtime


def _starts(size: int, patch: int) -> list[int]:
    """Return half-overlapping patch starts that include the final edge."""
    if size < patch:
        raise ValueError(f"SRDTrans patch {patch} exceeds image dimension {size}")
    starts = list(range(0, size - patch + 1, patch // 2))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _regions(starts: list[int], patch: int) -> list[tuple[slice, slice]]:
    """Map each overlapping patch to one non-overlapping output region."""
    regions = []
    for i, start in enumerate(starts):
        left = start if i == 0 else (starts[i - 1] + patch + start) // 2
        right = start + patch if i == len(starts) - 1 else (start + patch + starts[i + 1]) // 2
        regions.append((slice(left, right), slice(left - start, right - start)))
    return regions


def _stage(files: list[Path], count: int, patch: int, path: Path) -> tuple[np.memmap, float]:
    """Stage source frames for random patch access and return their mean."""
    source = iter(pages(files, count))
    first = next(source)
    total = max(count, patch)
    stack = np.memmap(path, mode="w+", shape=(total, *first.shape), dtype=first.dtype)
    stack[0] = first
    pixel_sum = float(first.sum(dtype=np.float64))
    last = first
    for i, frame in enumerate(source, 1):
        stack[i] = frame
        pixel_sum += float(frame.sum(dtype=np.float64))
        last = frame
    # Short clips repeat their final frame to fill one temporal patch.
    if total > count:
        stack[count:] = last
    stack.flush()
    return stack, pixel_sum / (count * first.size)


def _model(model_dir: Path, patch: int, gpu: str):
    """Load the newest checkpoint on the requested CUDA devices."""
    torch, devices = runtime(gpu)
    model = network(patch)
    checkpoints = sorted(model_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No SRDTrans checkpoint found in {model_dir}")
    try:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        state = torch.load(checkpoints[-1], map_location="cpu")
    model.load_state_dict(state)
    model.cuda().eval()

    if devices > 1:
        model = torch.nn.DataParallel(model, device_ids=range(devices))
    print(f"Loaded {checkpoints[-1].name} on {devices} CUDA GPU(s).")
    return torch, model, devices


def _infer(raw: np.memmap, mean: float, output: Path, patch: int,
           torch, network, batch_size: int) -> None:
    """Run overlapping patches and assemble their center regions."""
    total, height, width = raw.shape
    starts = [_starts(size, patch) for size in (total, height, width)]
    regions = [_regions(axis, patch) for axis in starts]
    coordinates = [(t, y, x) for t in range(len(starts[0]))
                   for y in range(len(starts[1])) for x in range(len(starts[2]))]
    result = tifffile.memmap(output, shape=raw.shape, dtype=raw.dtype, bigtiff=True)
    maximum = np.iinfo(raw.dtype).max

    progress = tqdm(total=len(coordinates), desc="SRDTrans", unit="patch")
    with torch.inference_mode():
        for offset in range(0, len(coordinates), batch_size):
            batch = coordinates[offset:offset + batch_size]
            patches = np.stack([
                raw[starts[0][t]:starts[0][t] + patch,
                    starts[1][y]:starts[1][y] + patch,
                    starts[2][x]:starts[2][x] + patch]
                for t, y, x in batch
            ]).astype(np.float32)
            tensor = torch.from_numpy(patches[:, None] - mean).pin_memory().cuda(non_blocking=True)
            predicted = network(tensor).float().cpu().numpy()[:, 0]

            for source, denoised, (t, y, x) in zip(patches, predicted, batch):
                destination = tuple(regions[a][index][0] for a, index in enumerate((t, y, x)))
                crop = tuple(regions[a][index][1] for a, index in enumerate((t, y, x)))
                clean, noisy = denoised[crop] + mean, source[crop]
                # Match local intensity after mean-centered neural inference.
                scale = np.sqrt(max(noisy.sum(), 0) / max(clean.sum(), 1e-6))
                result[destination] = np.clip(clean * scale, 0, maximum).astype(raw.dtype)
            progress.update(len(batch))
    progress.close()
    result.flush()


@contextmanager
def denoised_stack(files: list[Path], count: int, output: Path, gpu: str,
                   patch: int, model: Path) -> Iterator[Path]:
    """Yield a temporary direct-inference result and remove it afterward."""
    model, output = model.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="srdtrans_", dir=output) as temporary:
        work = Path(temporary)
        print("Staging frames for direct SRDTrans inference...")
        raw, mean = _stage(files, count, patch, work / "raw.dat")
        torch, network, devices = _model(model, patch, gpu)
        denoised = work / "denoised.tif"
        _infer(raw, mean, denoised, patch, torch, network, max(1, devices))
        del raw
        yield denoised
