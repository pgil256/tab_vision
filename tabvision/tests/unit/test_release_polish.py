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


def test_default_license_check_verifies_loaded_artifacts() -> None:
    """Phase 9 / LICENSES.md action item: the check also verifies the default
    pipeline's resolved model artifacts are on the permissive (✅) list."""
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
    assert "default artifact policy: PASS" in result.stdout


def test_default_artifact_resolver_tracks_shipped_cli_defaults() -> None:
    """The artifact allowlist is only meaningful if it tracks the real CLI
    defaults — guard the resolver so a future default flip is caught here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_default_licenses", Path("scripts/check_default_licenses.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resolved = dict(module._resolve_default_artifacts())
    # Clean-acoustic default flipped to the ensemble 2026-07-20 (DECISIONS.md).
    assert resolved["audio-backend"] == "highres-ensemble"
    assert resolved["position-prior"] == "guitarset-v1"
    assert resolved["sequence-prior"] == "guitarset-seq-v1"
    # Every resolved default must be on the permissive allowlist.
    for key in resolved.values():
        assert key in module.PERMISSIVE_DEFAULT_ARTIFACTS


def test_windows_studio_defaults_to_promoted_ensemble() -> None:
    """The browser Studio must use the same best clean-acoustic backend as
    the shipped automatic CLI route unless the user explicitly overrides it."""
    repo_root = Path(__file__).resolve().parents[3]
    script = (repo_root / "studio.ps1").read_text()

    assert "[string]$AudioBackend = 'highres-ensemble'" in script
    assert "[ValidateSet('highres-ensemble', 'highres', 'highres-fl', 'basicpitch')]" in script
    assert "$env:TABVISION_AUDIO_BACKEND  = $AudioBackend" in script
