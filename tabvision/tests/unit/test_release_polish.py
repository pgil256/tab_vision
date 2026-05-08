from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_fresh_install_script_documents_fresh_clone_flow() -> None:
    script = Path("scripts/test_fresh_install.sh")
    text = script.read_text()
    assert '"$PYTHON_BIN" -m venv' in text
    assert "command -v python3" in text
    assert "pip install -e" in text
    assert "tabvision --version" in text
    assert "pytest -m render" in text


def test_default_license_check_script_passes_current_pyproject() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_default_licenses.py",
            "--pyproject",
            "pyproject.toml",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "default dependency policy: PASS" in result.stdout
