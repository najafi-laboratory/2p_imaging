"""Train SRDTrans from TIFF stacks with its self-supervised neighbor loss."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .srdtrans import network, runtime
from .tiff import find_tiffs, frame_counts, pages

ROOT = Path(__file__).resolve().parents[1]


def _stage(files: list[Path], frames: int, path: Path) -> tuple[np.memmap, float]:
    """Read the requested number of frames into a temporary training array."""
    count = min(frames, sum(frame_counts(files)))
    source = iter(pages(files, count))
    first = next(source)
    stack = np.memmap(path, mode="w+", shape=(count, *first.shape), dtype=first.dtype)
    stack[0] = first
    total = float(first.sum(dtype=np.float64))
    for i, frame in enumerate(tqdm(source, total=count - 1, desc="Loading", unit="frame"), 1):
        stack[i] = frame
        total += float(frame.sum(dtype=np.float64))
    stack.flush()
    return stack, total / stack.size


def _patches(stack: np.memmap, mean: float, patch: int, batch: int) -> np.ndarray:
    """Draw a random batch of square spatiotemporal patches."""
    limits = [size - patch + 1 for size in stack.shape]
    if min(limits) < 1:
        raise ValueError(f"Training data dimensions {stack.shape} must each be at least {patch}")
    samples = []
    for _ in range(batch):
        t, y, x = [np.random.randint(limit) for limit in limits]
        sample = stack[t:t + patch, y:y + patch, x:x + patch].astype(np.float32) - mean
        # Random rotations and flips add spatial variety.
        sample = np.rot90(sample, np.random.randint(4), axes=(1, 2))
        if np.random.random() < .5:
            sample = sample[:, :, ::-1]
        samples.append(sample)
    return np.stack(samples)


def _neighbors(torch, images):
    """Choose three distinct pixels from every spatial 2-by-2 block."""
    batch, _, time, height, width = images.shape
    blocks = torch.nn.functional.pixel_unshuffle(
        images.permute(0, 2, 1, 3, 4).reshape(batch * time, 1, height, width), 2)
    blocks = blocks.reshape(batch, time, 4, height // 2, width // 2)
    choices = torch.tensor(
        [[0, 1, 2], [0, 2, 1], [1, 0, 3], [1, 3, 0],
         [2, 0, 3], [2, 3, 0], [3, 2, 1], [3, 1, 2]], device=images.device)
    selected = choices[torch.randint(8, (batch, time, height // 2, width // 2),
                                     device=images.device)].permute(0, 1, 4, 2, 3)
    values = torch.gather(blocks, 2, selected).permute(2, 0, 1, 3, 4).unsqueeze(2)
    return values[0], values[1], values[2]


def train(data: Path, output: Path, frames: int, epochs: int, patches: int,
          patch: int, batch: int, gpu: str, root: Path, learning_rate: float) -> Path:
    """Train SRDTrans and save one inference-compatible checkpoint."""
    files = find_tiffs(data)
    print(f"Found {len(files)} TIFF file(s). Other file types are ignored.")
    torch, devices = runtime(gpu)
    batch = batch or devices
    model = network(root.resolve(), patch).cuda()
    if devices > 1:
        model = torch.nn.DataParallel(model, device_ids=range(devices))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(.5, .999))
    l1, l2 = torch.nn.L1Loss(), torch.nn.MSELoss()
    # TF32 keeps float32's range while accelerating supported CUDA hardware.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Training on {devices} GPU(s), batch size {batch}, patch {patch}³.")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training_", dir=output.resolve()) as temporary:
        stack, mean = _stage(files, frames, Path(temporary) / "frames.dat")
        if not np.isfinite(mean):
            raise ValueError("Training TIFFs contain NaN or infinite pixel values")
        print(f"Using {stack.shape[0]} frames. Mean intensity: {mean:.2f}.")
        steps = math.ceil(patches / batch)
        model.train()
        for epoch in range(epochs):
            progress = tqdm(range(steps), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for _ in progress:
                images = torch.from_numpy(_patches(stack, mean, patch, batch)[:, None]).cuda()
                source, target1, target2 = _neighbors(torch, images)
                # Match the original SRDTrans float32 loss. FP16 MSE overflows on uint16 data.
                prediction = model(source)
                loss = .25 * (l1(prediction, target1) + l2(prediction, target1)
                              + l1(prediction, target2) + l2(prediction, target2))
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite SRDTrans loss. Check TIFF values or lower --learning-rate.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clipping prevents a rare large patch from corrupting model weights.
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                if not torch.isfinite(gradient):
                    raise FloatingPointError("Non-finite SRDTrans gradient; checkpoint not saved")
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.3f}")
        del stack

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Model contains non-finite weights; checkpoint not saved")
    checkpoint = output / "srdtrans_trained.pth"
    temporary_checkpoint = output / "srdtrans_trained.tmp"
    torch.save(state, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint)
    print(f"Checkpoint saved: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    """Parse training options and start training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="folder containing training TIFFs and other files")
    parser.add_argument("--output", type=Path, default=Path("trained_model"))
    parser.add_argument("--frames", type=int, default=1000, help="maximum frames to read")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patches-per-epoch", type=int, default=6000)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses one patch per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gpu", default="all")
    parser.add_argument("--srdtrans-root", type=Path, default=ROOT / "SRDTrans")
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"data directory does not exist: {args.data}")
    if min(args.frames, args.epochs, args.patches_per_epoch) < 1:
        parser.error("--frames, --epochs, and --patches-per-epoch must be positive")
    if args.patch < 8 or args.patch % 8:
        parser.error("--patch must be at least 8 and divisible by 8")
    if args.batch_size < 0 or args.learning_rate <= 0:
        parser.error("--batch-size cannot be negative and --learning-rate must be positive")
    train(args.data, args.output, args.frames, args.epochs, args.patches_per_epoch,
          args.patch, args.batch_size, args.gpu, args.srdtrans_root, args.learning_rate)
