"""Smoke tests for Phase 7 augmentation/training scaffold CLIs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_phase7_scaffold_scripts_have_deterministic_dry_run_outputs(tmp_path):
    scripts = [
        REPO_ROOT / "tabvision" / "scripts" / "augment" / "audio.py",
        REPO_ROOT / "tabvision" / "scripts" / "augment" / "video.py",
        REPO_ROOT / "tabvision" / "scripts" / "train" / "audio_finetune.py",
        REPO_ROOT / "tabvision" / "scripts" / "train" / "hand_finetune.py",
        REPO_ROOT / "tabvision" / "scripts" / "train" / "self_label.py",
    ]

    for script in scripts:
        out = tmp_path / f"{script.stem}.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--seed",
                "123",
                "--output",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["script"] == script.stem
        assert payload["dry_run"] is True
        assert payload["seed"] == 123
        assert payload["status"] == "ready"
