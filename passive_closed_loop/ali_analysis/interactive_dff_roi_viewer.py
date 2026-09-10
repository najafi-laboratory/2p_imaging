"""
Generate a standalone interactive ROI dF/F viewer.

The output is a single HTML file that can be shared without a notebook kernel.
It embeds:

* functional-channel mean image
* anatomical-channel mean image
* ROI mask image
* ROI center/radius overlays
* QC dF/F matrix

Interactions in the browser:

* click an ROI circle on any top image to select that ROI
* view raw dF/F and z-scored dF/F in the two trace rows
* mouse wheel over a trace to zoom in/out in time
* click-drag a trace to pan
* double-click a trace or press Reset zoom to show the full trace

The dF/F matrix is embedded as base64-encoded float32 data. For this session
that creates a large but still practical standalone HTML file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")


DEFAULT_OUTPUT_NAME = "interactive_dff_roi_viewer.html"


def robust_normalize(image: np.ndarray, lower: float = 1, upper: float = 99.7) -> np.ndarray:
    """Normalize an image to uint8 using robust percentiles."""
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)

    norm = np.clip((image - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def grayscale_png_data_uri(image: np.ndarray) -> str:
    """Convert a 2D image to a base64 PNG data URI."""
    uint8 = robust_normalize(image)
    rgb = np.repeat(uint8[..., None], 3, axis=2)
    return array_to_png_data_uri(rgb)


def mask_png_data_uri(mask: np.ndarray) -> str:
    """Convert an integer ROI mask image to a colored PNG data URI."""
    mask = np.asarray(mask)
    labels = np.unique(mask[mask > 0]).astype(int)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    if labels.size:
        cmap = matplotlib.colormaps["turbo"].resampled(int(labels.max()) + 1)
        for label in labels:
            color = np.asarray(cmap(label), dtype=float)
            rgba[mask == label, :3] = (color[:3] * 255).astype(np.uint8)
            rgba[mask == label, 3] = 210
    return array_to_png_data_uri(rgba)


def array_to_png_data_uri(array: np.ndarray) -> str:
    """Encode an RGB/RGBA uint8 array as a PNG data URI."""
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_ops(data_dir: Path) -> dict[str, Any]:
    """Load Suite2p ops from the session root or suite2p/plane0."""
    for rel in ["ops.npy", "suite2p/plane0/ops.npy"]:
        path = data_dir / rel
        if path.exists():
            return np.load(path, allow_pickle=True).item()
    raise FileNotFoundError(f"Could not find ops.npy in {data_dir}")


def load_stat(data_dir: Path) -> np.ndarray:
    """Load QC stat first, falling back to Suite2p stat."""
    for rel in ["qc_results/stat.npy", "suite2p/plane0/stat.npy"]:
        path = data_dir / rel
        if path.exists():
            return np.load(path, allow_pickle=True)
    raise FileNotFoundError(f"Could not find stat.npy in {data_dir}")


def load_masks(data_dir: Path) -> np.ndarray:
    """Load ROI label mask image."""
    for rel in ["qc_results/masks.npy", "masks.npy"]:
        path = data_dir / rel
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"Could not find masks.npy in {data_dir}")


def load_dff(data_dir: Path) -> np.ndarray:
    """Load QC dF/F traces as float32, shape: ROI x frame."""
    path = data_dir / "qc_results/dff.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing dF/F file: {path}")
    with h5py.File(path, "r") as h5:
        if "dff" not in h5:
            raise KeyError(f"{path} does not contain dataset 'dff'")
        return np.asarray(h5["dff"], dtype=np.float32)


def stat_to_roi_table(stat: np.ndarray, n_rois: int) -> list[dict[str, float | int]]:
    """Convert Suite2p stat entries to compact ROI metadata."""
    rois: list[dict[str, float | int]] = []
    for idx in range(n_rois):
        entry = stat[idx]
        med = entry.get("med", [np.nan, np.nan])
        ypix = np.asarray(entry.get("ypix", []), dtype=float)
        xpix = np.asarray(entry.get("xpix", []), dtype=float)
        radius = entry.get("radius", np.nan)
        if not np.isfinite(radius):
            radius = np.sqrt(max(len(xpix), 1) / np.pi)
        rois.append(
            {
                "roi": idx,
                "label": idx + 1,
                "y": float(med[0]) if len(med) else float(np.nanmean(ypix)),
                "x": float(med[1]) if len(med) > 1 else float(np.nanmean(xpix)),
                "radius": float(max(radius, 4.0)),
                "npix": int(entry.get("npix", len(xpix))),
            }
        )
    return rois


def float32_to_base64(array: np.ndarray) -> str:
    """Encode a float32 array as base64 bytes."""
    contiguous = np.ascontiguousarray(array.astype("<f4", copy=False))
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def build_html(
    *,
    title: str,
    functional_img_uri: str,
    anatomical_img_uri: str,
    mask_img_uri: str,
    rois: list[dict[str, float | int]],
    dff: np.ndarray,
    frame_rate: float,
) -> str:
    """Build the standalone HTML document."""
    n_rois, n_frames = dff.shape
    means = dff.mean(axis=1, dtype=np.float64).astype(np.float32)
    stds = dff.std(axis=1, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1

    payload = {
        "title": title,
        "nRois": int(n_rois),
        "nFrames": int(n_frames),
        "frameRate": float(frame_rate),
        "rois": rois,
        "dffBase64": float32_to_base64(dff),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "functionalImage": functional_img_uri,
        "anatomicalImage": anatomical_img_uri,
        "maskImage": mask_img_uri,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #f7f7f5;
    --panel: #ffffff;
    --ink: #1f2933;
    --muted: #667085;
    --accent: #c2410c;
    --line: #d0d5dd;
  }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Arial, Helvetica, sans-serif;
  }}
  .page {{
    width: min(1680px, calc(100vw - 32px));
    margin: 18px auto 28px;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: end;
    margin-bottom: 14px;
  }}
  h1 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: 0;
  }}
  .meta {{
    color: var(--muted);
    font-size: 13px;
    text-align: right;
    line-height: 1.4;
  }}
  .top-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 10px;
    box-sizing: border-box;
  }}
  .panel-title {{
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .image-wrap {{
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #111;
    overflow: hidden;
  }}
  .image-wrap img {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: fill;
    image-rendering: pixelated;
  }}
  .roi-overlay {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }}
  .roi-circle {{
    fill: transparent;
    stroke: rgba(255,255,255,0.9);
    stroke-width: 1.15;
    cursor: pointer;
    vector-effect: non-scaling-stroke;
  }}
  .roi-circle:hover {{
    stroke: #22d3ee;
    stroke-width: 2;
  }}
  .roi-circle.selected {{
    stroke: var(--accent);
    stroke-width: 2.4;
  }}
  .controls {{
    margin-top: 12px;
    display: grid;
    grid-template-columns: 1fr auto auto auto;
    gap: 10px;
    align-items: center;
  }}
  .roi-readout {{
    font-size: 14px;
    color: var(--ink);
  }}
  button, input {{
    font: inherit;
  }}
  button {{
    border: 1px solid var(--line);
    background: #fff;
    border-radius: 6px;
    padding: 7px 10px;
    cursor: pointer;
  }}
  button:hover {{
    border-color: #98a2b3;
  }}
  input[type="number"] {{
    width: 92px;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 7px 8px;
  }}
  .trace-grid {{
    display: grid;
    grid-template-rows: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
  }}
  .trace-panel {{
    height: 285px;
  }}
  canvas {{
    width: 100%;
    height: 232px;
    display: block;
    background: #fff;
    border: 1px solid var(--line);
    cursor: grab;
    box-sizing: border-box;
  }}
  canvas.dragging {{
    cursor: grabbing;
  }}
  .trace-caption {{
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }}
  @media (max-width: 980px) {{
    .top-grid {{
      grid-template-columns: 1fr;
    }}
    .controls {{
      grid-template-columns: 1fr;
    }}
    .meta {{
      text-align: left;
    }}
    .header {{
      display: block;
    }}
  }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <h1>{title}</h1>
    </div>
    <div class="meta" id="sessionMeta"></div>
  </div>

  <div class="top-grid">
    <div class="panel">
      <div class="panel-title">Functional channel mean</div>
      <div class="image-wrap">
        <img id="functionalImg" alt="Functional channel mean">
        <svg class="roi-overlay" viewBox="0 0 512 512" preserveAspectRatio="none" data-panel="functional"></svg>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Anatomical channel mean</div>
      <div class="image-wrap">
        <img id="anatomicalImg" alt="Anatomical channel mean">
        <svg class="roi-overlay" viewBox="0 0 512 512" preserveAspectRatio="none" data-panel="anatomical"></svg>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">ROI masks</div>
      <div class="image-wrap">
        <img id="maskImg" alt="ROI masks">
        <svg class="roi-overlay" viewBox="0 0 512 512" preserveAspectRatio="none" data-panel="masks"></svg>
      </div>
    </div>
  </div>

  <div class="controls panel">
    <div class="roi-readout" id="roiReadout"></div>
    <label>ROI <input id="roiInput" type="number" min="0" max="{n_rois - 1}" value="0"></label>
    <button id="selectRoiBtn">Select ROI</button>
    <button id="resetZoomBtn">Reset zoom</button>
  </div>

  <div class="trace-grid">
    <div class="panel trace-panel">
      <div class="panel-title">Raw dF/F</div>
      <canvas id="rawCanvas"></canvas>
      <div class="trace-caption">Wheel to zoom time. Drag to pan. Double-click to reset.</div>
    </div>
    <div class="panel trace-panel">
      <div class="panel-title">Z-scored dF/F</div>
      <canvas id="zCanvas"></canvas>
      <div class="trace-caption">Z score is computed per ROI: (trace - ROI mean) / ROI std.</div>
    </div>
  </div>
</div>

<script id="payload" type="application/json">{payload_json}</script>
<script>
"use strict";

const data = JSON.parse(document.getElementById("payload").textContent);
document.getElementById("functionalImg").src = data.functionalImage;
document.getElementById("anatomicalImg").src = data.anatomicalImage;
document.getElementById("maskImg").src = data.maskImage;
document.getElementById("sessionMeta").textContent =
  `${{data.nRois}} ROIs | ${{data.nFrames.toLocaleString()}} frames | ${{data.frameRate.toFixed(3)}} Hz`;

function base64ToFloat32Array(base64) {{
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  const chunk = 1024 * 1024;
  for (let start = 0; start < len; start += chunk) {{
    const end = Math.min(start + chunk, len);
    for (let i = start; i < end; i++) {{
      bytes[i] = binary.charCodeAt(i);
    }}
  }}
  return new Float32Array(bytes.buffer);
}}

const dff = base64ToFloat32Array(data.dffBase64);
const nFrames = data.nFrames;
const frameRate = data.frameRate;
let selectedRoi = 0;
let rawTrace = null;
let zTrace = null;
let viewStart = 0;
let viewEnd = nFrames - 1;

function makeRoiOverlays() {{
  const overlays = document.querySelectorAll(".roi-overlay");
  overlays.forEach(svg => {{
    data.rois.forEach(roi => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", roi.x);
      circle.setAttribute("cy", roi.y);
      circle.setAttribute("r", Math.max(roi.radius, 4));
      circle.classList.add("roi-circle");
      circle.dataset.roi = roi.roi;
      circle.addEventListener("click", () => selectRoi(Number(roi.roi)));
      svg.appendChild(circle);
    }});
  }});
}}

function updateSelectedCircles() {{
  document.querySelectorAll(".roi-circle").forEach(circle => {{
    circle.classList.toggle("selected", Number(circle.dataset.roi) === selectedRoi);
  }});
}}

function getRawTrace(roi) {{
  const start = roi * nFrames;
  return dff.subarray(start, start + nFrames);
}}

function computeZTrace(roi, raw) {{
  const out = new Float32Array(nFrames);
  const mean = data.means[roi];
  const std = data.stds[roi] || 1;
  for (let i = 0; i < nFrames; i++) {{
    out[i] = (raw[i] - mean) / std;
  }}
  return out;
}}

function selectRoi(roi) {{
  roi = Math.max(0, Math.min(data.nRois - 1, Math.round(roi)));
  selectedRoi = roi;
  rawTrace = getRawTrace(roi);
  zTrace = computeZTrace(roi, rawTrace);
  const meta = data.rois[roi];
  document.getElementById("roiInput").value = roi;
  document.getElementById("roiReadout").textContent =
    `Selected ROI ${{roi}} | center x=${{meta.x.toFixed(1)}}, y=${{meta.y.toFixed(1)}} | pixels=${{meta.npix}}`;
  updateSelectedCircles();
  drawAll();
}}

function setupCanvas(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
}}

function drawTrace(canvas, values, label, color) {{
  setupCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const padL = 62;
  const padR = 16;
  const padT = 14;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const start = Math.max(0, Math.floor(viewStart));
  const end = Math.min(nFrames - 1, Math.ceil(viewEnd));
  const span = Math.max(2, end - start + 1);
  const visible = values.subarray(start, end + 1);

  let ymin = Infinity;
  let ymax = -Infinity;
  for (let i = 0; i < visible.length; i++) {{
    const v = visible[i];
    if (Number.isFinite(v)) {{
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }}
  }}
  if (!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymin === ymax) {{
    ymin = -1;
    ymax = 1;
  }}
  const padY = (ymax - ymin) * 0.08 || 1;
  ymin -= padY;
  ymax += padY;

  function xOf(frame) {{
    return padL + ((frame - viewStart) / (viewEnd - viewStart)) * plotW;
  }}
  function yOf(value) {{
    return padT + (1 - ((value - ymin) / (ymax - ymin))) * plotH;
  }}

  ctx.strokeStyle = "#d0d5dd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.fillStyle = "#475467";
  ctx.font = `${{12 * (window.devicePixelRatio || 1)}}px Arial`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let t = 0; t <= 4; t++) {{
    const val = ymin + (t / 4) * (ymax - ymin);
    const y = yOf(val);
    ctx.strokeStyle = "#eef0f2";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), padL - 8, y);
  }}

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let t = 0; t <= 5; t++) {{
    const frame = viewStart + (t / 5) * (viewEnd - viewStart);
    const x = xOf(frame);
    ctx.fillText((frame / frameRate).toFixed(1) + " s", x, padT + plotH + 10);
  }}

  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  ctx.beginPath();

  const columns = Math.max(1, Math.floor(plotW));
  const framesPerPixel = span / columns;
  if (framesPerPixel <= 1.5) {{
    let first = true;
    for (let frame = start; frame <= end; frame++) {{
      const x = xOf(frame);
      const y = yOf(values[frame]);
      if (first) {{
        ctx.moveTo(x, y);
        first = false;
      }} else {{
        ctx.lineTo(x, y);
      }}
    }}
  }} else {{
    for (let col = 0; col < columns; col++) {{
      const f0 = Math.floor(start + col * framesPerPixel);
      const f1 = Math.min(end, Math.floor(start + (col + 1) * framesPerPixel));
      let minV = Infinity;
      let maxV = -Infinity;
      for (let frame = f0; frame <= f1; frame++) {{
        const v = values[frame];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }}
      const x = padL + col;
      ctx.moveTo(x, yOf(minV));
      ctx.lineTo(x, yOf(maxV));
    }}
  }}
  ctx.stroke();

  ctx.fillStyle = "#101828";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, padL + 4, 6);
}}

function drawAll() {{
  drawTrace(document.getElementById("rawCanvas"), rawTrace, "Raw dF/F", "#1d4ed8");
  drawTrace(document.getElementById("zCanvas"), zTrace, "Z-scored dF/F", "#b42318");
}}

function resetZoom() {{
  viewStart = 0;
  viewEnd = nFrames - 1;
  drawAll();
}}

function installTraceInteractions(canvas) {{
  let dragging = false;
  let dragStartX = 0;
  let dragViewStart = 0;
  let dragViewEnd = 0;

  canvas.addEventListener("wheel", (event) => {{
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const xFrac = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const center = viewStart + xFrac * (viewEnd - viewStart);
    const scale = event.deltaY < 0 ? 0.78 : 1.28;
    const newSpan = Math.max(20, Math.min(nFrames - 1, (viewEnd - viewStart) * scale));
    viewStart = center - xFrac * newSpan;
    viewEnd = viewStart + newSpan;
    if (viewStart < 0) {{
      viewEnd -= viewStart;
      viewStart = 0;
    }}
    if (viewEnd > nFrames - 1) {{
      viewStart -= viewEnd - (nFrames - 1);
      viewEnd = nFrames - 1;
    }}
    viewStart = Math.max(0, viewStart);
    drawAll();
  }}, {{ passive: false }});

  canvas.addEventListener("mousedown", (event) => {{
    dragging = true;
    dragStartX = event.clientX;
    dragViewStart = viewStart;
    dragViewEnd = viewEnd;
    canvas.classList.add("dragging");
  }});

  window.addEventListener("mousemove", (event) => {{
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    const dx = event.clientX - dragStartX;
    const frameShift = -dx / rect.width * (dragViewEnd - dragViewStart);
    viewStart = dragViewStart + frameShift;
    viewEnd = dragViewEnd + frameShift;
    if (viewStart < 0) {{
      viewEnd -= viewStart;
      viewStart = 0;
    }}
    if (viewEnd > nFrames - 1) {{
      viewStart -= viewEnd - (nFrames - 1);
      viewEnd = nFrames - 1;
    }}
    viewStart = Math.max(0, viewStart);
    drawAll();
  }});

  window.addEventListener("mouseup", () => {{
    dragging = false;
    canvas.classList.remove("dragging");
  }});

  canvas.addEventListener("dblclick", resetZoom);
}}

document.getElementById("selectRoiBtn").addEventListener("click", () => {{
  selectRoi(Number(document.getElementById("roiInput").value));
}});
document.getElementById("roiInput").addEventListener("keydown", (event) => {{
  if (event.key === "Enter") selectRoi(Number(event.target.value));
}});
document.getElementById("resetZoomBtn").addEventListener("click", resetZoom);
window.addEventListener("resize", drawAll);

makeRoiOverlays();
installTraceInteractions(document.getElementById("rawCanvas"));
installTraceInteractions(document.getElementById("zCanvas"));
selectRoi(0);
</script>
</body>
</html>
"""


def create_interactive_dff_viewer(
    data_dir: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Create the interactive ROI dF/F viewer and return the HTML path."""
    data_dir = Path(data_dir).expanduser().resolve()
    if output_path is None:
        output_path = data_dir / DEFAULT_OUTPUT_NAME
    else:
        output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ops = load_ops(data_dir)
    stat = load_stat(data_dir)
    masks = load_masks(data_dir)
    dff = load_dff(data_dir)

    mean_img = ops.get("meanImg")
    mean_img_chan2 = ops.get("meanImg_chan2")
    if mean_img is None:
        raise KeyError("ops.npy does not contain meanImg")
    if mean_img_chan2 is None:
        mean_img_chan2 = mean_img

    frame_rate = float(ops.get("fs", 30.0))
    n_rois = int(dff.shape[0])
    if len(stat) < n_rois:
        raise ValueError(f"stat.npy has {len(stat)} ROIs but dff has {n_rois} traces")

    rois = stat_to_roi_table(stat, n_rois=n_rois)
    html = build_html(
        title=title or f"Interactive dF/F ROI Viewer: {data_dir.name}",
        functional_img_uri=grayscale_png_data_uri(mean_img),
        anatomical_img_uri=grayscale_png_data_uri(mean_img_chan2),
        mask_img_uri=mask_png_data_uri(masks),
        rois=rois,
        dff=dff,
        frame_rate=frame_rate,
    )
    output_path.write_text(html)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="Generate a standalone interactive ROI dF/F HTML viewer.")
    parser.add_argument("--data-dir", required=True, help="Session directory containing ops.npy, masks.npy, stat.npy, and qc_results/dff.h5.")
    parser.add_argument("--output", default=None, help=f"Output HTML path. Default: data-dir/{DEFAULT_OUTPUT_NAME}")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    path = create_interactive_dff_viewer(args.data_dir, output_path=args.output, title=args.title)
    print(f"Saved interactive viewer: {path}")


if __name__ == "__main__":
    main()
