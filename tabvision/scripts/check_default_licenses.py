#!/usr/bin/env python3
"""Check that the shipping default pipeline stays license-clean.

Two policies, both enforced (CI fails if either fails):

1. **Default dependencies** — ``[project].dependencies`` in ``pyproject.toml``
   must not contain a copyleft/opt-in package. Optional extras (``vision``,
   ``render``, ``audio-*``) carry their own documented trade-offs in
   LICENSES.md and must not silently move into the shipping default.

2. **Loaded model artifacts** (SPEC §7 Phase 9 / LICENSES.md action item) —
   the model checkpoints and prior artifacts the *default* pipeline actually
   resolves to must all be on the LICENSES.md permissive (✅) list. This is the
   "NC weights leaked into the default" guard from the SPEC §11 risk table.
   Read from the real CLI defaults (``tabvision.cli`` is import-light — no
   torch), so the check tracks the shipped config and runs in the ``[dev]`` CI
   env without the heavy audio extras installed.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

# --- Policy 1: default dependencies ------------------------------------------

BLOCKED_DEFAULT_PACKAGES = {
    "ultralytics": "AGPL-3.0 detector is optional/accepted, not a default dependency",
    "pyguitarpro": "LGPL render dependency must remain in the render extra",
    "basic-pitch": "audio baseline extra pulls model/runtime dependencies",
    "hf-midi-transcription": (
        "high-res backend must remain opt-in until Phase 2 license gate closes"
    ),
}

# --- Policy 2: loaded model artifacts ----------------------------------------

# Every artifact the DEFAULT pipeline may load, keyed by the CLI's resolved
# backend/prior name. Each must map to a permissively licensed artifact per the
# LICENSES.md ✅ rows. Keep this in sync with LICENSES.md.
PERMISSIVE_DEFAULT_ARTIFACTS = {
    "highres": ("xavriley/midi-transcription-models:guitar-gaps.pth", "MIT"),
    # Promoted to the clean-acoustic auto default 2026-07-20 (personal-use
    # posture; DECISIONS.md). Loads both MIT checkpoints plus the in-repo
    # calibration artifact.
    "highres-ensemble": (
        "xavriley/midi-transcription-models:{guitar-gaps.pth,guitar-fl.pth}"
        " + tabvision/audio/ensemble_v1.json",
        "MIT (checkpoints) + repo license (calibration artifact)",
    ),
    "guitarset-v1": (
        "tabvision/fusion/priors/guitarset_v1.json",
        "CC-BY-4.0 (derived count statistics, attribution in LICENSES.md)",
    ),
    "guitarset-seq-v1": (
        "tabvision/fusion/priors/guitarset_seq_v1.json",
        "CC-BY-4.0 (derived count statistics, attribution in LICENSES.md)",
    ),
    # Classical-route priors derived from the GAPS train split. NC-SA under the
    # amended 2026-07-20 posture (LICENSES.md labels them; personal
    # non-commercial use only).
    "gaps-v1": (
        "tabvision/fusion/priors/gaps_v1.json",
        "CC-BY-NC-SA-4.0 (derived count statistics; NC-labeled in LICENSES.md)",
    ),
    "gaps-seq-v1": (
        "tabvision/fusion/priors/gaps_seq_v1.json",
        "CC-BY-NC-SA-4.0 (derived count statistics; NC-labeled in LICENSES.md)",
    ),
    "none": ("(no artifact)", "n/a"),
}

# Artifact keys that must NEVER be the default (they exist for opt-in / v2 use).
# 2026-07-20: guitar-fl.pth removed — the ensemble default loads it (MIT; the
# old block was a scope rule from before the ensemble promotion).
BLOCKED_DEFAULT_ARTIFACTS = {
    "highres-electric": "electric checkpoint is v2 / opt-in, not the acoustic default",
    "ultralytics": "AGPL-3.0 YOLO detector — vision extra only, never default",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="path to pyproject.toml",
    )
    args = parser.parse_args(argv)

    dep_failures = _check_dependencies(args.pyproject)
    artifact_failures = _check_default_artifacts()

    _report("default dependency policy", dep_failures, extra="checked default dependencies")
    _report(
        "default artifact policy", artifact_failures, extra="checked default pipeline artifacts"
    )

    return 1 if (dep_failures or artifact_failures) else 0


def _report(label: str, failures: list[str], *, extra: str) -> None:
    if failures:
        print(f"{label}: FAIL")
        for failure in failures:
            print(f" - {failure}")
    else:
        print(f"{label}: PASS")
        print(f"{extra}")


# --- Policy 1 implementation -------------------------------------------------


def _check_dependencies(pyproject: Path) -> list[str]:
    deps = _default_dependencies(pyproject)
    failures = []
    for dep in deps:
        name = _dependency_name(dep)
        if name in BLOCKED_DEFAULT_PACKAGES:
            failures.append(f"{name}: {BLOCKED_DEFAULT_PACKAGES[name]}")
    return failures


def _default_dependencies(pyproject: Path) -> list[str]:
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("project", {}).get("dependencies", []))


def _dependency_name(requirement: str) -> str:
    head = requirement.split(";", 1)[0].split("@", 1)[0]
    for separator in ("<", ">", "=", "!", "~", "["):
        head = head.split(separator, 1)[0]
    return head.strip().lower().replace("_", "-")


# --- Policy 2 implementation -------------------------------------------------


def _resolve_default_artifacts() -> list[tuple[str, str]]:
    """(component, resolved-key) for the DEFAULT pipeline, torch-free.

    Reads the real ``tabvision transcribe`` defaults so the check tracks the
    shipped config, then calls the same session-aware resolver used by the
    runtime. This remains torch-free.
    """
    from tabvision.cli import _build_parser
    from tabvision.fusion.inference_policy import resolve_inference_policy
    from tabvision.types import GuitarConfig, SessionConfig

    ns = _build_parser().parse_args(["transcribe", "clip.mov"])

    backend = ns.audio_backend
    if backend == "auto":
        # Mirror of tabvision.pipeline.audio_backend_for_session (kept inline
        # so this check stays torch-free): electric → electric checkpoint;
        # clean acoustic → the promoted GAPS+FL ensemble (2026-07-20);
        # else → single-checkpoint highres.
        if ns.instrument == "electric":
            backend = "highres-electric"
        elif ns.instrument == "acoustic" and ns.tone == "clean":
            backend = "highres-ensemble"
        else:
            backend = "highres"

    policy = resolve_inference_policy(
        requested_position_prior=ns.position_prior,
        requested_sequence_prior=ns.sequence_prior,
        requested_string_evidence=ns.string_evidence,
        cfg=GuitarConfig(capo=ns.capo),
        session=SessionConfig(
            instrument=ns.instrument,
            tone=ns.tone,
            style=ns.style,
        ),
        audio_backend_name=backend,
    )

    return [
        ("audio-backend", backend),
        ("position-prior", policy.resolved_position_prior),
        ("sequence-prior", policy.resolved_sequence_prior),
        ("string-evidence", policy.resolved_string_evidence),
    ]


def _check_default_artifacts() -> list[str]:
    failures = []
    try:
        resolved = _resolve_default_artifacts()
    except Exception as exc:  # noqa: BLE001 - surface as a policy failure, not a crash.
        return [f"could not resolve default artifacts from tabvision.cli: {exc}"]

    for component, key in resolved:
        if key in BLOCKED_DEFAULT_ARTIFACTS:
            failures.append(f"{component}={key}: {BLOCKED_DEFAULT_ARTIFACTS[key]}")
        elif key not in PERMISSIVE_DEFAULT_ARTIFACTS:
            failures.append(
                f"{component}={key}: not on the permissive default-artifact allowlist "
                "(LICENSES.md ✅). Clear its license or move it behind an opt-in extra."
            )
    return failures


if __name__ == "__main__":
    sys.exit(main())
