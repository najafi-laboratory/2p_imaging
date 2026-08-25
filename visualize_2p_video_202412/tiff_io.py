"""TIFF discovery, metadata, and streaming utilities shared by both workflows."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile


def _natural_key(path: Path) -> list[str | int]:
    """Split a path into text and numbers for acquisition-order sorting."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def find_ch2(folder: Path) -> list[Path]:
    """Find and naturally sort Ch2 TIFF files below a directory."""
    files = [path for path in find_tiffs(folder) if "ch2" in path.name.lower()]
    if not files:
        raise FileNotFoundError(f"No Ch2 TIFF files found below {folder}")
    return files


def find_tiffs(folder: Path) -> list[Path]:
    """Find TIFF files while ignoring unrelated files in the same folder."""
    files = sorted((p for p in folder.rglob("*") if p.is_file()
                    and p.suffix.lower() in {".tif", ".tiff"}), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No TIFF files found below {folder}")
    return files


def frame_rate(files: list[Path]) -> float:
    """Read FPS from OME, text metadata, or consecutive stack times."""
    path = files[0]
    with tifffile.TiffFile(path) as tif:
        # OME metadata may store a direct time interval.
        xml = tif.ome_metadata or ""
        if xml:
            root = ET.fromstring(xml)
            pixels = next((x for x in root.iter() if x.tag.endswith("Pixels")), None)
            if pixels is not None and float(pixels.attrib.get("TimeIncrement", 0)) > 0:
                return 1 / float(pixels.attrib["TimeIncrement"])
            times = [float(x.attrib["DeltaT"]) for x in root.iter()
                     if x.tag.endswith("Plane") and "DeltaT" in x.attrib]
            gaps = np.diff(sorted(set(times)))
            if np.any(gaps > 0):
                return 1 / float(np.median(gaps[gaps > 0]))
        text = "\n".join(filter(None, (tif.pages[0].description, xml)))

    patterns = (
        (r"framePeriod\s*[=:]\s*([\d.eE+-]+)", True),
        (r"(?:frameRate|fps)\s*[=:]\s*([\d.eE+-]+)", False),
    )
    for pattern, is_period in patterns:
        # Some acquisition tools store timing as plain text.
        match = re.search(pattern, text, re.I)
        if match and float(match.group(1)) > 0:
            value = float(match.group(1))
            return 1 / value if is_period else value

    # Split Prairie exports may omit their companion metadata.
    rates = []
    for left, right in zip(files, files[1:]):
        elapsed = right.stat().st_mtime - left.stat().st_mtime
        if elapsed > 0:
            with tifffile.TiffFile(left) as tif:
                rates.append(len(tif.pages) / elapsed)
    if rates:
        return float(np.median(rates))
    raise ValueError(f"Frame rate is absent from TIFF metadata: {path.name}")


def frame_counts(files: list[Path]) -> list[int]:
    """Count pages in each TIFF stack without loading image data."""
    counts = []
    for path in files:
        with tifffile.TiffFile(path) as tif:
            counts.append(len(tif.pages))
    return counts


def pages(files: list[Path], limit: int) -> Iterator[np.ndarray]:
    """Yield TIFF pages across files until the frame limit is reached."""
    emitted = 0
    for path in files:
        # Opening one file at a time keeps memory use low.
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                if emitted >= limit:
                    return
                yield page.asarray()
                emitted += 1
