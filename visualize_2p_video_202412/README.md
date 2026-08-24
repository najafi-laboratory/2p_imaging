# Two-photon Ch2 video visualizer

This project converts sequential two-photon `Ch2` TIFF stacks into a denoised,
green fluorescence MP4. It can use the bundled SRDTrans checkpoint or train a
new self-supervised checkpoint from raw TIFF data.

## Requirements

- Python 3.9 or newer.
- An NVIDIA GPU and CUDA-enabled PyTorch.
- TIFF stacks with one grayscale frame per page.

Install the packages and verify CUDA:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Quick start

```powershell
python visualize_2p.py data --duration 30 --rolling-average 8 --speed 1
```

The command creates `output/fov_ch2.mp4`. The MP4 always uses 60 frames per
second. Temporary SRDTrans arrays are removed after processing.

## How video generation works

### 1. Discover and order the stacks

The program searches the input directory recursively. It keeps TIFF files whose
names contain `Ch2`, ignores other files, and applies natural sorting so
`..._9.tif`, `..._10.tif`, and `..._11.tif` remain in acquisition order.

### 2. Read acquisition timing

The source frame rate is found in this order:

1. OME `TimeIncrement`.
2. Differences between OME `Plane DeltaT` values.
3. Text fields named `framePeriod`, `frameRate`, or `fps`.
4. For split Prairie exports without companion metadata, consecutive stack
   sizes and preserved file-completion times.

The source-frame count is `ceil(duration × source fps)`. If fewer frames exist,
all available frames are used.

### 3. Denoise with SRDTrans

Selected frames are staged in a temporary memory-mapped array. SRDTrans reads
half-overlapping 3D patches with dimensions `patch × patch × patch` in time,
height, and width.

The project calls the network directly instead of the original SRDTrans
`train.py` and `test.py` wrappers. It owns patch extraction, CUDA batching,
edge coverage, and result assembly. Overlapping patches contribute their center
regions, covering every source pixel without double-writing it. A clip shorter
than one temporal patch temporarily repeats its final frame; padded frames are
not included in the video.

By default, all CUDA GPUs visible to PyTorch are used. Multiple GPUs use
`DataParallel`.

### 4. Apply rolling averaging

`--rolling-average N` averages the current denoised frame with up to `N - 1`
preceding frames. The default `1` disables temporal averaging. A mild spatial
Gaussian filter is then applied. Larger windows suppress more noise but smooth
fast calcium events and motion.

### 5. Adjust contrast

Approximately 50 spatially downsampled frames are sampled. Their 1st and 99.8th
percentiles become black and maximum green. This reduces the effect of isolated
dark or saturated pixels. Contrast mapping affects only the eight-bit MP4.

### 6. Produce a 60 fps video

Every output position is calculated as:

```text
source position = output frame index × source fps × speed / 60
```

Frames are dropped when the source supplies more samples than needed. Adjacent
frames are linearly interpolated when more output samples are needed.
`--speed 2` therefore plays twice as fast while retaining a 60 fps file.

## Video command and options

```powershell
python visualize_2p.py data `
  --duration 30 `
  --rolling-average 8 `
  --speed 1 `
  --output output
```

Use a checkpoint trained by this project:

```powershell
python visualize_2p.py data `
  --duration 30 `
  --rolling-average 8 `
  --speed 1 `
  --model trained_model
```

| Option | Default | Meaning |
|---|---:|---|
| `data` | required | Directory searched recursively for `Ch2` TIFFs. |
| `--duration` | required | Source duration to process, in seconds. |
| `--output` | `output` | Directory for the MP4 and temporary data. |
| `--rolling-average` | `1` | Number of recent denoised frames to average. |
| `--speed` | `1` | Playback-speed multiplier. |
| `--gpu` | `all` | CUDA devices, such as `0` or `0,1`. |
| `--model` | `SRDTrans/pth` | Folder containing checkpoint files. |
| `--patch` | `128` | SRDTrans temporal and spatial patch size. |
| `--srdtrans-root` | `SRDTrans` | Minimal network package directory. |

The patch size must be at least 8 and divisible by 8. If a model folder contains
multiple checkpoints, the alphabetically last checkpoint is loaded.

## Training a checkpoint

Training recursively reads `.tif` and `.tiff` files and ignores unrelated
files. Training filenames do not need to contain `Ch2`.

```powershell
python train_srdtrans.py data `
  --frames 1000 `
  --epochs 20 `
  --patches-per-epoch 6000 `
  --patch 128 `
  --output trained_model
```

Training creates only `trained_model/srdtrans_trained.pth`. It does not create
a video or retain a denoised TIFF.

### How self-supervised training works

1. Up to `--frames` frames are loaded in TIFF order into a temporary array.
2. The global mean is subtracted, matching original SRDTrans preprocessing.
3. Random `patch³` spatiotemporal crops are selected.
4. Random 90-degree rotations and horizontal flips augment each crop.
5. Every spatial 2-by-2 block supplies three different neighboring pixels.
6. One neighbor image passes through SRDTrans. The other two are noisy targets.
7. The objective averages L1 and mean-squared errors against both targets.
8. Adam updates the network, and the final state dictionary is saved.

Clean target images are not required. `--patches-per-epoch` controls random
examples per epoch rather than source frames read.

The loss runs in float32. Float16 MSE can overflow when mean-centered 16-bit
fluorescence values are squared. TF32 remains enabled on supported GPUs.
Gradients are clipped to norm 1. Non-finite inputs, losses, gradients, or weights
stop training before an invalid checkpoint is saved. Saving is atomic.

### Training options

| Option | Default | Meaning |
|---|---:|---|
| `data` | required | Directory searched recursively for training TIFFs. |
| `--output` | `trained_model` | Checkpoint directory. |
| `--frames` | `1000` | Maximum total frames to load. |
| `--epochs` | `20` | Number of training epochs. |
| `--patches-per-epoch` | `6000` | Random patches drawn per epoch. |
| `--patch` | `128` | Time, height, and width of each raw patch. |
| `--batch-size` | `0` | Patches per update; 0 uses one per visible GPU. |
| `--learning-rate` | `1e-4` | Adam learning rate. |
| `--gpu` | `all` | CUDA devices used for training. |
| `--srdtrans-root` | `SRDTrans` | Minimal SRDTrans network directory. |

Training data must contain at least `patch` frames, rows, and columns. Reduce
`--patch` or `--batch-size` if CUDA memory is insufficient.

## Progress and temporary data

Commands print discovered files, source timing, frame counts, GPUs, checkpoint,
contrast range, and output path. Progress bars cover TIFF loading, SRDTrans
patches, training batches, contrast analysis, and video encoding.

Temporary raw and denoised arrays live under the selected output directory.
They are removed when a stage completes or raises an exception.

## Project structure

```text
visualize_2p.py             Video command entry point
train_srdtrans.py           Training command entry point
two_photon/
  cli.py                    Video arguments and validation
  tiff.py                   TIFF discovery, metadata, and streaming
  srdtrans.py               Network loading and tiled CUDA inference
  train.py                  Self-supervised training
  video.py                  Contrast, smoothing, timing, and MP4 encoding
SRDTrans/
  SRDTrans/                 Minimal upstream network architecture
  pth/                      Bundled default checkpoint
```

## Common problems

- **CUDA unavailable:** install CUDA-enabled PyTorch and verify
  `torch.cuda.is_available()` is `True`.
- **No Ch2 TIFFs:** video input filenames must contain `Ch2`.
- **Missing frame rate:** provide OME timing, recognized text timing, or
  consecutive Prairie chunks with preserved timestamps.
- **Patch exceeds a dimension:** select a smaller divisible-by-8 `--patch`.
- **CUDA out of memory:** lower `--patch` or training `--batch-size`.
- **Non-finite loss:** confirm TIFF pixels are finite and try a lower
  `--learning-rate`.
- **Encoder failure:** reinstall `imageio-ffmpeg` from `requirements.txt`.
