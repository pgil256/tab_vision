from pathlib import Path
from unittest.mock import patch

import numpy as np

from tabvision.audio.review import analyze_take


def test_analyze_take_reports_waveform_trim_and_clipping(tmp_path: Path):
    source = tmp_path / "take.wav"
    source.write_bytes(b"fake")
    wav = np.zeros(22_050, dtype=np.float32)
    wav[4_000:18_000] = 0.2 * np.sin(np.linspace(0, 200, 14_000))
    wav[9_000:9_010] = 1.0

    with patch("tabvision.audio.review._decode", return_value=wav):
        result = analyze_take(source, bins=64)

    assert result["duration"] == 1.0
    assert result["clipped_samples"] == 10
    assert result["clipped_runs"] == 1
    assert 0 < result["auto_trim_start"] < result["auto_trim_end"] < 1
    assert len(result["waveform_min"]) == len(result["waveform_max"]) == 64
