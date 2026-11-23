"""Kardia PDF import, tested on hand-built PDFs that mimic Kardia's vector layout."""

import zlib

import numpy as np
import pandas as pd
import pytest

from pcos_ecg.ingestion.kardia_pdf import (
    PT_PER_MM,
    KardiaPDFError,
    import_kardia_folder,
    read_kardia_pdf,
)

FS = 300.0
STRIP_SAMPLES = 2400  # 8 s per strip at 25 mm/s on a ~200 mm wide page


def _fake_kardia_pdf(path, signal_mv, grid_mm=1.0, name="Jane Doe"):
    step = 25.0 * PT_PER_MM / FS  # points per sample at 25 mm/s
    pt_per_mv = 10.0 * PT_PER_MM
    ops = ["q 1 0 0 -1 0 792 cm"]
    ops.append(".8196 .8196 .9608 RG")
    for k in range(60):  # light grid, vertical lines
        x = 26 + k * grid_mm * PT_PER_MM
        ops.append(f"{x:.4f} 100 m {x:.4f} 700 l S")
    ops.append(f"BT /F1 9 Tf 30 60 Td ({name}, DOB 01/01/2000) Tj ET")
    ops.append("0 0 0 RG")
    for s, start in enumerate(range(0, len(signal_mv), STRIP_SAMPLES)):
        chunk = signal_mv[start : start + STRIP_SAMPLES + 1]
        base = 170 + 140 * s
        xs = 26 + np.arange(len(chunk)) * step
        ys = base - chunk * pt_per_mv
        for i in range(len(chunk) - 1):
            ops.append(f"{xs[i]:.4f} {ys[i]:.4f} m {xs[i + 1]:.4f} {ys[i + 1]:.4f} l S")
    ops.append("Q")
    stream = zlib.compress("\n".join(ops).encode("ascii"))
    body = (
        b"%PDF-1.4\n14 0 obj\n<< /Length "
        + str(len(stream)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    path.write_bytes(body)


@pytest.fixture
def signal():
    t = np.arange(int(30 * FS)) / FS
    return 0.8 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 7 * t)


def test_recovers_sampling_rate_and_signal_shape(tmp_path, signal):
    pdf = tmp_path / "rec.pdf"
    _fake_kardia_pdf(pdf, signal)
    tr = read_kardia_pdf(pdf)
    assert tr.fs_hz == pytest.approx(300, abs=0.5)
    assert tr.n_strips == 4
    assert tr.grid_mm_pt == pytest.approx(PT_PER_MM, abs=0.01)
    # Each strip is re-centred on its median; compare shapes strip by strip.
    n = STRIP_SAMPLES
    for s in range(4):
        got = tr.signal_mv[s * (n + 1) : s * (n + 1) + 100]
        ref = signal[s * n : s * n + 100]
        assert np.corrcoef(got, ref)[0, 1] > 0.999
        assert np.ptp(got) == pytest.approx(np.ptp(ref), rel=0.01)  # mV scale is right


def test_text_with_personal_details_is_never_parsed(tmp_path, signal, monkeypatch):
    import pcos_ecg.ingestion.kardia_pdf as kp

    pdf = tmp_path / "rec.pdf"
    _fake_kardia_pdf(pdf, signal, name="Jane Doe")
    seen = []
    original = kp._strokes
    monkeypatch.setattr(kp, "_strokes", lambda s: seen.append(s) or original(s))
    kp.read_kardia_pdf(pdf)
    assert seen and all("Jane" not in s and "DOB" not in s for s in seen)


def test_non_kardia_pdf_is_rejected(tmp_path):
    pdf = tmp_path / "other.pdf"
    stream = zlib.compress(b"BT (hello) Tj ET")
    pdf.write_bytes(b"1 0 obj << /Filter /FlateDecode >>\nstream\n" + stream + b"\nendstream\n")
    with pytest.raises(KardiaPDFError, match="no ECG trace"):
        read_kardia_pdf(pdf)


def test_unexpected_grid_scale_is_an_error(tmp_path, signal):
    pdf = tmp_path / "rec.pdf"
    _fake_kardia_pdf(pdf, signal, grid_mm=2.0)
    with pytest.raises(KardiaPDFError, match="grid spacing"):
        read_kardia_pdf(pdf)


def test_folder_import_uses_coded_ids_from_the_mapping(tmp_path, signal):
    _fake_kardia_pdf(tmp_path / "Jane Doe 2026-05-19.pdf", signal)
    pd.DataFrame(
        {"file": ["Jane Doe 2026-05-19.pdf"], "subject_id": ["P07"], "record_id": ["P07-R1"]}
    ).to_csv(tmp_path / "map.csv", index=False)
    out = tmp_path / "ecg_long.csv"
    summary = import_kardia_folder(tmp_path / "map.csv", out)
    df = pd.read_csv(out)
    assert summary["records_converted"] == 1 and summary["records_failed"] == []
    assert set(df["subject_id"]) == {"P07"} and set(df["lead"]) == {"I"}
    assert "Jane" not in out.read_text()
    assert df["sample_rate_hz"].iloc[0] == pytest.approx(300, abs=0.5)
