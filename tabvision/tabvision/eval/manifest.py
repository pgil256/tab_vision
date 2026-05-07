"""Eval manifest validation for optional public/full eval reports."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

REQUIRED_TIERS: tuple[str, ...] = (
    "clean_acoustic_single_line",
    "clean_acoustic_strummed",
    "clean_electric",
    "distorted_electric",
)

OPTIONAL_TIERS: tuple[str, ...] = ("iphone_ood",)
ALLOWED_TIERS = REQUIRED_TIERS + OPTIONAL_TIERS
REQUIRED_CLIP_FIELDS: tuple[str, ...] = (
    "id",
    "tier",
    "source",
    "split",
    "media_path",
    "annotation_path",
)
ALLOWED_SPLITS: tuple[str, ...] = ("train", "validation", "test")
MIN_PHASE15_CLIPS = 15

Severity = Literal["info", "warn", "fail"]


@dataclass(frozen=True)
class ManifestIssue:
    severity: Severity
    code: str
    message: str
    clip_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestValidation:
    manifest_path: str
    passed: bool
    clip_count: int
    clip_ids: list[str]
    present_tiers: list[str]
    missing_tiers: list[str]
    items: list[ManifestIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "passed": self.passed,
            "clip_count": self.clip_count,
            "clip_ids": self.clip_ids,
            "present_tiers": self.present_tiers,
            "missing_tiers": self.missing_tiers,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")


def validate_manifest(path: str | Path) -> ManifestValidation:
    manifest_path = Path(path)
    if not manifest_path.exists():
        missing_items = [
            ManifestIssue(
                severity="warn",
                code="MANIFEST_MISSING",
                message=(
                    f"Optional eval manifest is absent at {manifest_path}; "
                    "v1 release gates use deterministic smoke/public evidence instead."
                ),
            ),
            ManifestIssue(
                severity="info",
                code="TOO_FEW_CLIPS",
                message=(
                    f"Optional full eval target is >= {MIN_PHASE15_CLIPS} clips; found 0."
                ),
            ),
        ]
        missing_items.extend(_missing_tier_issues(REQUIRED_TIERS))
        return ManifestValidation(
            manifest_path=str(manifest_path),
            passed=True,
            clip_count=0,
            clip_ids=[],
            present_tiers=[],
            missing_tiers=list(REQUIRED_TIERS),
            items=missing_items,
        )

    try:
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return ManifestValidation(
            manifest_path=str(manifest_path),
            passed=False,
            clip_count=0,
            clip_ids=[],
            present_tiers=[],
            missing_tiers=list(REQUIRED_TIERS),
            items=[
                ManifestIssue(
                    severity="fail",
                    code="MANIFEST_PARSE_ERROR",
                    message=str(exc),
                )
            ],
        )

    raw_clips = payload.get("clips", [])
    clips = raw_clips if isinstance(raw_clips, list) else []
    items: list[ManifestIssue] = []
    if not isinstance(raw_clips, list):
        items.append(
            ManifestIssue(
                severity="fail",
                code="CLIPS_NOT_LIST",
                message="Manifest must define [[clips]] entries.",
            )
        )

    ids: list[str] = []
    tiers: set[str] = set()
    seen_ids: set[str] = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            items.append(
                ManifestIssue(
                    severity="fail",
                    code="CLIP_NOT_TABLE",
                    message=f"Clip entry {index} must be a TOML table.",
                )
            )
            continue

        clip_id = _string_field(clip, "id") or f"<clip[{index}]>"
        if clip_id in seen_ids:
            items.append(
                ManifestIssue(
                    severity="fail",
                    code="DUPLICATE_ID",
                    message=f"Duplicate clip id {clip_id!r}.",
                    clip_id=clip_id,
                )
            )
        seen_ids.add(clip_id)
        ids.append(clip_id)

        for field in REQUIRED_CLIP_FIELDS:
            if not _string_field(clip, field):
                items.append(
                    ManifestIssue(
                        severity="fail",
                        code=f"MISSING_{field.upper()}",
                        message=f"Clip {clip_id!r} is missing required field {field!r}.",
                        clip_id=clip_id,
                    )
                )

        tier = _string_field(clip, "tier")
        if tier:
            if tier not in ALLOWED_TIERS:
                items.append(
                    ManifestIssue(
                        severity="fail",
                        code="UNKNOWN_TIER",
                        message=(
                            f"Clip {clip_id!r} has tier {tier!r}; expected one of "
                            f"{', '.join(ALLOWED_TIERS)}."
                        ),
                        clip_id=clip_id,
                    )
                )
            else:
                tiers.add(tier)

        split = _string_field(clip, "split")
        if split and split not in ALLOWED_SPLITS:
            items.append(
                ManifestIssue(
                    severity="fail",
                    code="UNKNOWN_SPLIT",
                    message=(
                        f"Clip {clip_id!r} has split {split!r}; expected one of "
                        f"{', '.join(ALLOWED_SPLITS)}."
                    ),
                    clip_id=clip_id,
                )
            )

    if len(clips) < MIN_PHASE15_CLIPS:
        items.append(
            ManifestIssue(
                severity="info",
                code="TOO_FEW_CLIPS",
                message=(
                    f"Optional full eval target is >= {MIN_PHASE15_CLIPS} clips; "
                    f"found {len(clips)}."
                ),
            )
        )
    missing_tiers = [tier for tier in REQUIRED_TIERS if tier not in tiers]
    items.extend(_missing_tier_issues(missing_tiers))
    items.sort(key=lambda item: (item.severity, item.code, item.clip_id or "", item.message))

    return ManifestValidation(
        manifest_path=str(manifest_path),
        passed=not any(item.severity == "fail" for item in items),
        clip_count=len(clips),
        clip_ids=sorted(ids),
        present_tiers=sorted(tiers),
        missing_tiers=missing_tiers,
        items=items,
    )


def _string_field(clip: dict[object, object], field: str) -> str | None:
    value = clip.get(field)
    return value if isinstance(value, str) and value.strip() else None


def _missing_tier_issues(missing_tiers: tuple[str, ...] | list[str]) -> list[ManifestIssue]:
    return [
        ManifestIssue(
            severity="info",
            code="MISSING_TIER",
            message=(
                f"Optional full eval tier {tier!r} has no clip; this is not a v1 "
                "release blocker."
            ),
        )
        for tier in missing_tiers
    ]


__all__ = [
    "ALLOWED_SPLITS",
    "ALLOWED_TIERS",
    "MIN_PHASE15_CLIPS",
    "ManifestIssue",
    "ManifestValidation",
    "OPTIONAL_TIERS",
    "REQUIRED_CLIP_FIELDS",
    "REQUIRED_TIERS",
    "validate_manifest",
]
