"""Recover the ECG signal from an AliveCor Kardia PDF export.

Kardia draws the trace as vector graphics: thousands of short black strokes, one per sample,
laid out in strips on a 1 mm grid at 25 mm/s and 10 mm/mV. This module reads those strokes
directly, so the signal is exactly what Kardia plotted, not a traced picture.

Privacy: the PDF's text (which can include the person's name and date of birth) is removed
before anything is parsed and is never read or returned. Map files to coded participant IDs
with a separate CSV; see `import_kardia_folder`.

Scope: single-lead Kardia exports. Each strip is treated as a continuation of the previous
one, and each strip's baseline is set to its median (Kardia positions strips on the page;
the offset carries no signal and the band-pass filter removes any residual step).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PT_PER_MM = 72 / 25.4
TRACE_RGB = ("0", "0", "0")
GRID_RGB = (".8196", ".8196", ".9608")
NUMBER = re.compile(r"-?\d*\.?\d+")


class KardiaPDFError(ValueError):
    """The file does not look like a single-lead Kardia vector export."""


@dataclass
class KardiaTrace:
    signal_mv: np.ndarray
    fs_hz: float
    n_strips: int
    grid_mm_pt: float | None  # measured grid spacing in points (1 mm = 2.835 pt)


def _content_streams(data: bytes) -> list[str]:
    streams = []
    for m in re.finditer(rb"<<(.*?)>>\s*stream\r?\n(.*?)\r?\nendstream", data, re.S):
        if b"/Image" in m.group(1):
            continue
        raw = m.group(2)
        if b"/FlateDecode" in m.group(1):
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                continue
        if not raw.isascii():
            continue
        # Drop every text block before parsing, so personal details are never read.
        streams.append(re.sub(r"BT.*?ET", " ", raw.decode("ascii"), flags=re.S))
    return streams


def _strokes(stream: str) -> dict[tuple, list[list[float]]]:
    """Single 'x y m x y l' strokes, grouped by stroke colour."""
    tok = stream.split()
    colour: tuple = ()
    out: dict[tuple, list[list[float]]] = {}
    i = 0
    while i < len(tok):
        t = tok[i]
        if t == "RG" and i >= 3:
            colour = tuple(tok[i - 3 : i])
        elif (
            t == "m"
            and i >= 2
            and i + 3 < len(tok)
            and tok[i + 3] == "l"
            and all(NUMBER.fullmatch(x) for x in (tok[i - 2], tok[i - 1], tok[i + 1], tok[i + 2]))
        ):
            seg = [float(tok[i - 2]), float(tok[i - 1]), float(tok[i + 1]), float(tok[i + 2])]
            out.setdefault(colour, []).append(seg)
            i += 3
        i += 1
    return out


def _grid_spacing(segments: list[list[float]]) -> float | None:
    g = np.asarray(segments)
    if len(g) < 10:
        return None
    xs = np.unique(np.round(g[np.abs(g[:, 0] - g[:, 2]) < 1e-6][:, 0], 3))
    return float(np.median(np.diff(xs))) if len(xs) > 2 else None


def read_kardia_pdf(
    path: str | Path,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    min_strip_points: int = 100,
) -> KardiaTrace:
    """Signal in mV and its sampling rate, from one Kardia PDF."""
    data = Path(path).read_bytes()
    trace, grid = [], []
    for stream in _content_streams(data):
        strokes = _strokes(stream)
        trace += strokes.get(TRACE_RGB, [])
        grid += strokes.get(GRID_RGB, [])
    if len(trace) < min_strip_points:
        raise KardiaPDFError(f"{Path(path).name}: no ECG trace found (is this a Kardia PDF?)")

    seg = np.asarray(trace)
    # A new strip starts where the pen jumps back to the left.
    starts = np.where(seg[1:, 0] < seg[:-1, 2] - 1.0)[0] + 1
    strips = [s for s in np.split(seg, starts) if len(s) >= min_strip_points]
    if not strips:
        raise KardiaPDFError(f"{Path(path).name}: no strip has {min_strip_points}+ points")

    step = float(np.median(np.concatenate([s[:, 2] - s[:, 0] for s in strips])))
    if step <= 0:
        raise KardiaPDFError(f"{Path(path).name}: trace has no horizontal progression")
    fs = mm_per_s * PT_PER_MM / step
    pt_per_mv = mm_per_mv * PT_PER_MM

    pieces = []
    for s in strips:
        y = np.concatenate([[s[0, 1]], s[:, 3]])
        # Page y grows downwards after Kardia's flip transform, so up on paper is smaller y.
        pieces.append(-(y - np.median(y)) / pt_per_mv)
    grid_pt = _grid_spacing(grid)
    if grid_pt is not None and abs(grid_pt - PT_PER_MM) > 0.1:
        raise KardiaPDFError(
            f"{Path(path).name}: grid spacing {grid_pt:.2f} pt is not 1 mm; "
            "check mm_per_s and mm_per_mv"
        )
    return KardiaTrace(np.concatenate(pieces), round(fs, 3), len(strips), grid_pt)


def import_kardia_folder(mapping_csv: str | Path, out_csv: str | Path, lead: str = "I") -> dict:
    """Convert every PDF listed in a mapping CSV into the pipeline's long-form waveform file.

    The mapping CSV has columns file, subject_id, record_id; paths are relative to the CSV.
    Returns counts only.
    """
    mapping_csv = Path(mapping_csv)
    mapping = pd.read_csv(mapping_csv)
    missing = {"file", "subject_id", "record_id"} - set(mapping.columns)
    if missing:
        raise KardiaPDFError(f"mapping CSV lacks columns {sorted(missing)}")
    frames, failed = [], []
    for row in mapping.itertuples(index=False):
        try:
            tr = read_kardia_pdf(mapping_csv.parent / row.file)
        except (KardiaPDFError, OSError) as exc:
            failed.append(f"{row.record_id}: {exc.__class__.__name__}")
            continue
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": str(row.subject_id),
                    "record_id": str(row.record_id),
                    "lead": lead,
                    "sample_index": np.arange(tr.signal_mv.size),
                    "sample_rate_hz": tr.fs_hz,
                    "amplitude": np.round(tr.signal_mv, 5),
                }
            )
        )
    if not frames:
        raise KardiaPDFError("no PDF could be converted")
    out = pd.concat(frames, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return {
        "records_converted": len(frames),
        "records_failed": failed,
        "sample_rates_hz": sorted(set(out["sample_rate_hz"])),
    }
