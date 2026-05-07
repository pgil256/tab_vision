"""Deterministic Phase 8 eval runner and debt report generator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabvision.eval.manifest import REQUIRED_TIERS, ManifestValidation, validate_manifest

Scope = Literal["full", "smoke"]

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "eval" / "manifest.toml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "EVAL_REPORTS"
DEFAULT_EVAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "eval"
SMOKE_BUDGET_S = 180.0
ABLATION_VARIANTS: tuple[str, ...] = ("audio_only", "audio_vision", "audio_vision_prior")


@dataclass(frozen=True)
class EvalRunResult:
    json_path: Path
    markdown_path: Path
    json_bytes: bytes
    markdown: str
    passed: bool


def run_eval(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scope: Scope = "full",
    seed: int = 0,
    timestamp: str | None = None,
    eval_root: str | Path = DEFAULT_EVAL_ROOT,
) -> EvalRunResult:
    """Run the deterministic eval/debt report.

    Heavy model execution is intentionally not embedded here yet. The runner
    validates data readiness, emits stable hook rows for the supported
    ablations, and uses a tiny synthetic scope for CI smoke determinism.
    """
    ts = timestamp or _utc_timestamp()
    manifest = validate_manifest(manifest_path)
    phase_debt = _phase_debt(eval_root)
    tier_breakdown = _tier_breakdown(manifest)
    ablations = _ablation_rows(scope)
    calibration = _confidence_calibration(scope, manifest)
    smoke = _smoke_summary(scope)

    payload: dict[str, object] = {
        "schema_version": 1,
        "timestamp": ts,
        "seed": seed,
        "scope": scope,
        "smoke_budget_s": SMOKE_BUDGET_S,
        "manifest": manifest.to_dict(),
        "tier_breakdown": tier_breakdown,
        "phase_debt": phase_debt,
        "ablations": ablations,
        "confidence_calibration": calibration,
        "smoke": smoke,
    }
    json_bytes = _json_bytes(payload)
    markdown = _markdown_report(payload)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{scope}_{_timestamp_slug(ts)}"
    json_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    json_path.write_bytes(json_bytes)
    markdown_path.write_text(markdown, encoding="utf-8")

    passed = bool(smoke["passed"]) if scope == "smoke" else bool(manifest.passed)
    return EvalRunResult(
        json_path=json_path,
        markdown_path=markdown_path,
        json_bytes=json_bytes,
        markdown=markdown,
        passed=passed,
    )


def _phase_debt(eval_root: str | Path) -> dict[str, object]:
    root = Path(eval_root)
    framing_count = _json_count(root / "framing")
    fretboard_count = _json_count(root / "fretboard")
    hand_fretting_labels = _hand_fretting_label_count(root / "fingering")
    return {
        "phase_1_5": {
            "gate": "manifest completeness and all required tiers represented",
            "command": "tabvision-eval --manifest tabvision/data/eval/manifest.toml --check",
        },
        "phase_3": {
            "guitar_detector": {
                "status": "passed_documented",
                "evidence": "docs/DECISIONS.md#2026-05-05-phase-3-detector-acceptance",
                "metric": "neck mAP50=0.995",
            },
            "preflight": {
                "status": "ready" if framing_count >= 10 else "blocked",
                "usable_labels": framing_count,
                "required_labels": 10,
                "command": "pytest -m preflight_eval tests/eval/test_phase3_eval.py",
            },
            "fretboard": {
                "status": "ready" if fretboard_count >= 5 else "blocked",
                "usable_labels": fretboard_count,
                "required_labels": 5,
                "command": "pytest -m fretboard_eval tests/eval/test_phase3_eval.py",
            },
        },
        "phase_4": {
            "hand": {
                "status": "ready" if hand_fretting_labels >= 100 else "blocked",
                "usable_fretting_labels": hand_fretting_labels,
                "required_fretting_labels": 100,
                "command": "pytest -m hand_eval tests/eval/test_phase4_eval.py",
            }
        },
    }


def _tier_breakdown(manifest: ManifestValidation) -> list[dict[str, object]]:
    counts = {tier: 0 for tier in REQUIRED_TIERS}
    if Path(manifest.manifest_path).exists():
        try:
            import tomllib

            payload = tomllib.loads(Path(manifest.manifest_path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            payload = {}
        for clip in payload.get("clips", []):
            if isinstance(clip, dict):
                tier = clip.get("tier")
                if isinstance(tier, str) and tier in counts:
                    counts[tier] += 1
    return [
        {
            "tier": tier,
            "clip_count": counts[tier],
            "status": "blocked" if counts[tier] == 0 else "pending_metrics",
            "tab_f1_target": _tier_target(tier),
            "tab_f1": None,
        }
        for tier in REQUIRED_TIERS
    ]


def _ablation_rows(scope: Scope) -> list[dict[str, object]]:
    if scope == "smoke":
        return [
            {
                "variant": variant,
                "status": "synthetic_smoke",
                "onset_f1": 1.0,
                "pitch_f1": 1.0,
                "tab_f1": 1.0,
                "chord_accuracy": 1.0,
                "blocker": None,
            }
            for variant in ABLATION_VARIANTS
        ]

    return [
        {
            "variant": variant,
            "status": "blocked",
            "onset_f1": None,
            "pitch_f1": None,
            "tab_f1": None,
            "chord_accuracy": None,
            "blocker": (
                "No complete Phase 1.5 manifest plus local media/annotations are available "
                "for model-backed eval."
            ),
        }
        for variant in ABLATION_VARIANTS
    ]


def _confidence_calibration(scope: Scope, manifest: ManifestValidation) -> dict[str, object]:
    if scope == "smoke":
        return {
            "status": "synthetic_smoke",
            "metric": "ece",
            "bins": 10,
            "ece": 0.0,
            "blocker": None,
        }
    return {
        "status": "blocked",
        "metric": "ece",
        "bins": 10,
        "ece": None,
        "blocker": (
            "confidence calibration is blocked until scored predictions with per-event "
            f"confidence exist for manifest clips (currently {manifest.clip_count} clips)."
        ),
    }


def _smoke_summary(scope: Scope) -> dict[str, object]:
    if scope != "smoke":
        return {"enabled": False, "passed": False, "subset": []}
    return {
        "enabled": True,
        "passed": True,
        "subset": [
            {
                "id": "synthetic-smoke-001",
                "tier": "clean_acoustic_single_line",
                "notes": 2,
            }
        ],
    }


def _markdown_report(payload: dict[str, object]) -> str:
    manifest = payload["manifest"]
    assert isinstance(manifest, dict)
    calibration = payload["confidence_calibration"]
    assert isinstance(calibration, dict)
    lines = [
        f"# Eval Debt And Harness Report ({payload['scope']})",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        f"Seed: `{payload['seed']}`",
        f"Smoke budget target: < {int(SMOKE_BUDGET_S)} s",
        "",
        "## Phase 1.5 Manifest",
        "",
        f"- Passed: `{manifest['passed']}`",
        f"- Clips: `{manifest['clip_count']}`",
        f"- Missing tiers: `{', '.join(manifest['missing_tiers']) or 'none'}`",
        "",
        "## Per-Tier Breakdown",
        "",
        "| Tier | Clips | Target Tab F1 | Status | Current Tab F1 |",
        "|---|---:|---:|---|---:|",
    ]
    for row in _list_of_dicts(payload["tier_breakdown"]):
        lines.append(
            f"| {row['tier']} | {row['clip_count']} | {row['tab_f1_target']:.2f} | "
            f"{row['status']} | {_metric_value(row['tab_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Phase 3/4 Acceptance Debt",
            "",
            "| Gate | Status | Evidence / Blocker | Command |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(_phase_debt_rows(payload["phase_debt"]))
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Variant | Status | Onset F1 | Pitch F1 | Tab F1 | Chord Acc | Blocker |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in _list_of_dicts(payload["ablations"]):
        lines.append(
            f"| {row['variant']} | {row['status']} | {_metric_value(row['onset_f1'])} | "
            f"{_metric_value(row['pitch_f1'])} | {_metric_value(row['tab_f1'])} | "
            f"{_metric_value(row['chord_accuracy'])} | {row['blocker'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Confidence Calibration",
            "",
            f"- Status: `{calibration['status']}`",
            f"- Metric: `{calibration['metric']}` with `{calibration['bins']}` bins",
            f"- ECE: `{_metric_value(calibration['ece'])}`",
        ]
    )
    if calibration.get("blocker"):
        lines.append(f"- Blocker: {calibration['blocker']}")
    return "\n".join(lines) + "\n"


def _phase_debt_rows(phase_debt: object) -> list[str]:
    assert isinstance(phase_debt, dict)
    phase3 = phase_debt["phase_3"]
    phase4 = phase_debt["phase_4"]
    assert isinstance(phase3, dict)
    assert isinstance(phase4, dict)
    rows: list[str] = []
    guitar = phase3["guitar_detector"]
    preflight = phase3["preflight"]
    fretboard = phase3["fretboard"]
    hand = phase4["hand"]
    assert isinstance(guitar, dict)
    assert isinstance(preflight, dict)
    assert isinstance(fretboard, dict)
    assert isinstance(hand, dict)
    rows.append(
        "| Phase 3 guitar detector | "
        f"{guitar['status']} | {guitar['metric']} ({guitar['evidence']}) | current report |"
    )
    rows.append(
        "| Phase 3 preflight | "
        f"{preflight['status']} | {preflight['usable_labels']}/"
        f"{preflight['required_labels']} labels | `{preflight['command']}` |"
    )
    rows.append(
        "| Phase 3 fretboard | "
        f"{fretboard['status']} | {fretboard['usable_labels']}/"
        f"{fretboard['required_labels']} labels | `{fretboard['command']}` |"
    )
    rows.append(
        "| Phase 4 hand | "
        f"{hand['status']} | {hand['usable_fretting_labels']}/"
        f"{hand['required_fretting_labels']} fretting labels | `{hand['command']}` |"
    )
    return rows


def _json_count(path: Path) -> int:
    return len(sorted(path.glob("*.json"))) if path.is_dir() else 0


def _hand_fretting_label_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for label_path in sorted(path.glob("*.json")):
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            fingers = frame.get("fingers", [])
            if not isinstance(fingers, list):
                continue
            for finger in fingers:
                if isinstance(finger, dict) and finger.get("is_fretting") is True:
                    total += 1
    return total


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    out: list[dict[str, object]] = []
    for item in value:
        assert isinstance(item, dict)
        out.append(item)
    return out


def _metric_value(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return ""


def _tier_target(tier: str) -> float:
    return {
        "clean_acoustic_single_line": 0.94,
        "clean_acoustic_strummed": 0.86,
        "clean_electric": 0.90,
        "distorted_electric": 0.82,
    }[tier]


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_slug(timestamp: str) -> str:
    return (
        timestamp.replace(":", "")
        .replace("-", "")
        .replace(".", "")
        .replace("+", "")
        .replace("Z", "Z")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scope", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--check", action="store_true", help="validate manifest and exit")
    parser.add_argument(
        "--twice-and-diff",
        action="store_true",
        help="run deterministic smoke twice and fail if report bytes differ",
    )
    args = parser.parse_args(argv)

    if args.check:
        manifest = validate_manifest(args.manifest)
        sys.stdout.buffer.write(manifest.to_json_bytes())
        return 0 if manifest.passed else 1

    if args.twice_and_diff:
        return _main_twice_and_diff(args)

    result = run_eval(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        scope=args.scope,
        seed=args.seed,
        timestamp=args.timestamp,
        eval_root=args.eval_root,
    )
    print(f"json={result.json_path}")
    print(f"markdown={result.markdown_path}")
    print(f"passed={str(result.passed).lower()}")
    return 0 if result.passed or args.scope == "full" else 1


def _main_twice_and_diff(args: argparse.Namespace) -> int:
    timestamp = args.timestamp or "2026-05-07T00:00:00Z"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        first = run_eval(
            manifest_path=args.manifest,
            output_dir=tmp / "a",
            scope="smoke",
            seed=args.seed,
            timestamp=timestamp,
            eval_root=args.eval_root,
        )
        second = run_eval(
            manifest_path=args.manifest,
            output_dir=tmp / "b",
            scope="smoke",
            seed=args.seed,
            timestamp=timestamp,
            eval_root=args.eval_root,
        )
        identical = first.json_bytes == second.json_bytes and first.markdown == second.markdown
        print(f"deterministic={str(identical).lower()}")
        print(f"smoke_budget_s={int(SMOKE_BUDGET_S)}")
        if not identical:
            return 1
    return 0


__all__ = [
    "ABLATION_VARIANTS",
    "DEFAULT_EVAL_ROOT",
    "DEFAULT_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "EvalRunResult",
    "SMOKE_BUDGET_S",
    "main",
    "run_eval",
]
