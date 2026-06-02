"""Composite-eval manifest builder.

Scans known dataset roots on disk and emits a TOML manifest suitable
for ``tabvision-composite-eval``. Designed to be deterministic so
re-runs on the same data produce byte-identical output: clips are
emitted in sorted-id order, and per-tier caps + total limits are
applied after that sort.

Currently supports:

- **GuitarSet** (CC-BY-4.0) — clean acoustic single-line + strummed
  tiers. Default split = player 05 → validation, others → train.
- **Guitar-TECHS** (CC-BY-4.0) — stubbed; Phase 0 returns ``[]`` until
  the dataset is acquired locally and the on-disk layout is verified.

EGDB is intentionally not yet wired up (license-pending per the
2026-05-13 design plan).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tabvision.eval.manifest import (
    SYNTHETIC_SOURCE_PREFIXES,
    ManifestValidation,
    validate_manifest,
)

GUITARSET_VALIDATION_PLAYER = "05"


@dataclass(frozen=True)
class ClipEntry:
    """Minimal clip-row representation, one per manifest ``[[clips]]``."""

    id: str
    tier: str
    source: str
    split: str
    media_path: str
    annotation_path: str
    annotation_format: str


def _guitarset_tier(track_id: str) -> str | None:
    """Map a GuitarSet track id suffix to a SPEC §1.4 tier name.

    Returns ``None`` for unrecognised suffixes (track is skipped).
    """
    if track_id.endswith("_comp"):
        return "clean_acoustic_strummed"
    if track_id.endswith("_solo"):
        return "clean_acoustic_single_line"
    return None


def _guitarset_split(track_id: str, validation_player: str) -> str:
    """``validation`` for the held-out player, ``train`` otherwise."""
    if track_id.split("_", 1)[0] == validation_player:
        return "validation"
    return "train"


def scan_guitarset(
    root: Path,
    *,
    validation_player: str = GUITARSET_VALIDATION_PLAYER,
) -> list[ClipEntry]:
    """Scan a GuitarSet directory tree and return discovered clips.

    Expected layout::

        <root>/annotation/<track>.jams
        <root>/audio_mono-mic/<track>_mic.wav

    Tracks missing either file are skipped. Tracks whose suffix is
    neither ``_comp`` nor ``_solo`` are skipped.
    """
    annotation_dir = root / "annotation"
    audio_dir = root / "audio_mono-mic"
    if not annotation_dir.is_dir() or not audio_dir.is_dir():
        return []

    entries: list[ClipEntry] = []
    for jams_path in sorted(annotation_dir.glob("*.jams")):
        track_id = jams_path.stem
        media_path = audio_dir / f"{track_id}_mic.wav"
        if not media_path.is_file():
            continue
        tier = _guitarset_tier(track_id)
        if tier is None:
            continue
        entries.append(
            ClipEntry(
                id=f"guitarset/{track_id}",
                tier=tier,
                source="GuitarSet",
                split=_guitarset_split(track_id, validation_player),
                media_path=str(media_path.resolve()),
                annotation_path=str(jams_path.resolve()),
                annotation_format="guitarset_jams",
            )
        )
    return entries


GUITAR_TECHS_VALIDATION_PLAYER = "03"

# Stretch-goal articulations (SPEC §1.4 → v1.1). Skipped so the clean_electric
# tier scores clean transcription, not expression. Matched case-insensitively
# anywhere in a clip's path.
_GT_SKIP_KEYWORDS: tuple[str, ...] = (
    "bend", "vibrato", "pinch", "harmonic", "palm", "slide", "hammer", "pull", "trill",
)
_GT_AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".aiff", ".aif")
# Audio-capture preference for the clean_electric tier: direct input (clean DI)
# before mic'd amp. Ranked by first hit in the path (lower index = preferred).
_GT_AUDIO_PREF: tuple[str, ...] = (
    "directinput", "direct", "di", "clean", "micamp", "mic",
)
# Performer id from a path component: 'P1_chords', 'player01', 'guitarist3', 'p02'.
# Anchored at the component start with a trailing separator/end so substrings like
# 'tmp12' don't false-match. Unmatched -> split='train' (safe; fine for #2 where
# all of Guitar-TECHS is held out w.r.t. the GuitarSet-trained prior anyway).
_GT_PLAYER_RE = re.compile(r"^(?:player|guitarist|p)[_\-]?(\d{1,2})(?:[_\-]|$)", re.IGNORECASE)


def _guitar_techs_player(path_parts: tuple[str, ...]) -> str | None:
    """Best-effort performer id from a path *component* (e.g. 'P1_chords' -> '01')."""
    for part in path_parts:
        match = _GT_PLAYER_RE.match(part)
        if match:
            return match.group(1).zfill(2)
    return None


def _gt_content(stem: str) -> str:
    """Content id shared by a clip's MIDI and audio files.

    Guitar-TECHS names files ``<capture>_<content>`` -- MIDI ``midi_Drop3_7``
    and audio ``directinput_Drop3_7`` share ``Drop3_7``. Returns everything
    after the first underscore (or the whole stem if there is none).
    """
    return stem.split("_", 1)[1] if "_" in stem else stem


def _gt_group_dir(path: Path, root: Path) -> Path:
    """The performer/category group dir (e.g. ``P1_chords``) -- first part under root."""
    rel = path.relative_to(root)
    return root / rel.parts[0] if rel.parts else path.parent


def _gt_audio_rank(path: Path) -> int:
    """Lower = preferred capture (direct input before mic'd amp)."""
    low = str(path).lower()
    for i, hint in enumerate(_GT_AUDIO_PREF):
        if hint in low:
            return i
    return len(_GT_AUDIO_PREF)


def scan_guitar_techs(
    root: Path,
    *,
    validation_player: str = GUITAR_TECHS_VALIDATION_PLAYER,
) -> list[ClipEntry]:
    """Scan a Guitar-TECHS tree into ``clean_electric`` clip entries.

    Layout (verified 2026-06-02 against Zenodo record 14963133)::

        <root>/<Pn_category>/midi/midi_<content>.mid
        <root>/<Pn_category>/audio/directinput/directinput_<content>.wav
        <root>/<Pn_category>/audio/micamp/micamp_<content>.wav

    All electric -> the single ``clean_electric`` tier (SPEC 1.4 has no electric
    single-line/strummed split). MIDI<->audio are paired by the shared
    ``<content>`` token (the part after the first underscore), scoped to the same
    Pn_category group -- NOT by a common prefix. Direct-input audio is preferred
    over mic'd amp. Stretch-goal technique clips are skipped; ``__MACOSX`` zip
    cruft is ignored. Split by performer (``P3`` -> validation by default).
    Returns ``[]`` gracefully if no pairable MIDI is found.
    """
    if not root.is_dir():
        return []

    audio_by_group: dict[Path, list[Path]] = {}
    for ext in _GT_AUDIO_EXTS:
        for path in root.rglob(f"*{ext}"):
            if "__macosx" in str(path).lower():
                continue
            audio_by_group.setdefault(_gt_group_dir(path, root), []).append(path)

    entries: list[ClipEntry] = []
    seen: set[str] = set()
    for midi_path in sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi")):
        path_low = str(midi_path).lower()
        if "__macosx" in path_low or any(kw in path_low for kw in _GT_SKIP_KEYWORDS):
            continue
        content = _gt_content(midi_path.stem)
        candidates = [
            p
            for p in audio_by_group.get(_gt_group_dir(midi_path, root), [])
            if _gt_content(p.stem) == content
        ]
        if not candidates:
            continue
        audio_path = sorted(candidates, key=lambda p: (_gt_audio_rank(p), str(p)))[0]
        rel = midi_path.relative_to(root)
        clip_id = f"guitar-techs/{rel.with_suffix('').as_posix()}"
        if clip_id in seen:
            continue
        seen.add(clip_id)
        player = _guitar_techs_player(midi_path.parts)
        split = "validation" if player == validation_player else "train"
        entries.append(
            ClipEntry(
                id=clip_id,
                tier="clean_electric",
                source="GuitarTECHS",
                split=split,
                media_path=str(audio_path.resolve()),
                annotation_path=str(midi_path.resolve()),
                annotation_format="guitar_techs_midi",
            )
        )
    return entries


def apply_limits(
    entries: Iterable[ClipEntry],
    *,
    max_clips_per_tier: int | None = None,
    total_limit: int | None = None,
) -> list[ClipEntry]:
    """Apply per-tier and total limits deterministically.

    Entries are first sorted by ``id`` (so the same data produces the
    same output regardless of input scan order), then per-tier capped,
    then total-limited.
    """
    sorted_entries = sorted(entries, key=lambda entry: entry.id)

    if max_clips_per_tier is not None and max_clips_per_tier >= 0:
        by_tier: dict[str, int] = {}
        capped: list[ClipEntry] = []
        for entry in sorted_entries:
            count = by_tier.get(entry.tier, 0)
            if count >= max_clips_per_tier:
                continue
            capped.append(entry)
            by_tier[entry.tier] = count + 1
        sorted_entries = capped

    if total_limit is not None and 0 <= total_limit < len(sorted_entries):
        sorted_entries = sorted_entries[:total_limit]

    return sorted_entries


def _toml_escape(value: str) -> str:
    """Escape a TOML basic-string value (backslashes + double quotes)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _relativize_to_data_root(path_str: str, data_root: Path | None) -> str:
    """Rewrite ``path_str`` as ``$TABVISION_DATA_ROOT/<rest>`` when it lives
    under ``data_root``. Returns the original string when ``data_root`` is
    ``None`` or the path isn't under it.

    The composite-eval CLI expands ``$TABVISION_DATA_ROOT`` at eval time
    via the env var or its ``--media-root`` / ``--annotation-root`` args
    (see :func:`tabvision.eval.composite._resolve_path`), so this keeps
    checked-in manifests portable across developer machines.
    """
    if data_root is None:
        return path_str
    abs_root = str(data_root.expanduser().resolve())
    if path_str == abs_root:
        return "$TABVISION_DATA_ROOT"
    if path_str.startswith(abs_root + "/"):
        rest = path_str[len(abs_root) + 1 :]
        return f"$TABVISION_DATA_ROOT/{rest}"
    return path_str


def render_toml(
    entries: Iterable[ClipEntry],
    *,
    header_comment: str = "",
    data_root: Path | None = None,
) -> str:
    """Render entries as a TOML composite manifest.

    Output is sorted by clip id for byte-stable re-generation. When
    ``data_root`` is provided, ``media_path`` and ``annotation_path``
    values that fall under that root are rewritten as
    ``$TABVISION_DATA_ROOT/<rest>`` — the composite-eval CLI expands
    that token at eval time. Use this for checked-in manifests.
    """
    sorted_entries = sorted(entries, key=lambda entry: entry.id)
    lines: list[str] = []
    if header_comment:
        for raw_line in header_comment.splitlines():
            lines.append(f"# {raw_line}" if raw_line else "#")
        lines.append("")
    fields = (
        "id",
        "tier",
        "source",
        "split",
        "media_path",
        "annotation_path",
        "annotation_format",
    )
    for entry in sorted_entries:
        lines.append("[[clips]]")
        for field in fields:
            raw = getattr(entry, field)
            if field in ("media_path", "annotation_path"):
                raw = _relativize_to_data_root(raw, data_root)
            value = _toml_escape(raw)
            lines.append(f'{field} = "{value}"')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarise_coverage(entries: Iterable[ClipEntry]) -> str:
    """Human-readable coverage summary."""
    entries_list = list(entries)
    by_tier: dict[str, dict[str, int]] = {}
    by_split: dict[str, int] = {}
    for entry in entries_list:
        by_tier.setdefault(entry.tier, {}).setdefault(entry.source, 0)
        by_tier[entry.tier][entry.source] += 1
        by_split[entry.split] = by_split.get(entry.split, 0) + 1

    lines: list[str] = []
    lines.append(f"Total clips: {len(entries_list)}")
    lines.append("Per-tier × source:")
    for tier in sorted(by_tier):
        per_source = ", ".join(
            f"{source}={count}" for source, count in sorted(by_tier[tier].items())
        )
        total = sum(by_tier[tier].values())
        lines.append(f"  {tier}: {total} clips ({per_source})")
    if by_split:
        split_summary = ", ".join(
            f"{split}={count}" for split, count in sorted(by_split.items())
        )
        lines.append(f"Splits: {split_summary}")
    return "\n".join(lines)


def _refuse_synthetic_in_eval_splits(entries: Iterable[ClipEntry]) -> None:
    """Pre-write guard: bail loudly on bad synthetic-source manifests."""
    for entry in entries:
        if entry.split == "train":
            continue
        source = entry.source.lower()
        if any(source.startswith(prefix) for prefix in SYNTHETIC_SOURCE_PREFIXES):
            raise ValueError(
                f"Clip {entry.id!r} has synthetic source {entry.source!r} but "
                f"split={entry.split!r}; the manifest validator (and design "
                f"plan §5 R8) forbid synthetic-source clips in eval splits. "
                f"Either move to split='train' or remove."
            )


def build_manifest(
    *,
    guitarset_root: Path | None = None,
    guitar_techs_root: Path | None = None,
    splits: tuple[str, ...] | None = None,
    max_clips_per_tier: int | None = None,
    total_limit: int | None = None,
    validation_player: str = GUITARSET_VALIDATION_PLAYER,
) -> list[ClipEntry]:
    """Scan all configured roots and apply filters + limits.

    Sources whose root is ``None`` or doesn't exist are silently skipped.
    Optional ``splits`` restricts to the named splits (e.g.
    ``("validation",)`` for a smoke pre-flight). Limits are applied
    after the split filter, sorted by clip id for determinism.
    """
    entries: list[ClipEntry] = []
    if guitarset_root is not None:
        entries.extend(
            scan_guitarset(guitarset_root, validation_player=validation_player)
        )
    if guitar_techs_root is not None:
        entries.extend(scan_guitar_techs(guitar_techs_root))

    _refuse_synthetic_in_eval_splits(entries)

    if splits is not None:
        allowed = set(splits)
        entries = [entry for entry in entries if entry.split in allowed]

    return apply_limits(
        entries,
        max_clips_per_tier=max_clips_per_tier,
        total_limit=total_limit,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``tabvision-build-composite-manifest``."""
    parser = argparse.ArgumentParser(
        prog="build_composite_manifest",
        description=(
            "Scan dataset roots on disk and emit a composite-eval TOML manifest."
        ),
    )
    parser.add_argument(
        "--guitarset",
        type=Path,
        default=None,
        help="GuitarSet root directory (with annotation/ and audio_mono-mic/)",
    )
    parser.add_argument(
        "--guitar-techs",
        type=Path,
        default=None,
        help="Guitar-TECHS root directory (scanner is currently a stub)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-clips-per-tier",
        type=int,
        default=None,
        help="cap clips per tier; useful for smoke runs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap total clips after per-tier cap; useful for smoke runs",
    )
    parser.add_argument(
        "--guitarset-validation-player",
        default=GUITARSET_VALIDATION_PLAYER,
        help="GuitarSet player id whose tracks go into the validation split",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "comma-separated splits to include (e.g. 'validation' for a "
            "smoke pre-flight). Default: include all splits."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "rewrite media/annotation paths that fall under this root as "
            "$TABVISION_DATA_ROOT/<rest> for portable checked-in manifests"
        ),
    )

    args = parser.parse_args(argv)

    if args.guitarset is None and args.guitar_techs is None:
        parser.error("specify at least one of --guitarset or --guitar-techs")

    splits_filter: tuple[str, ...] | None = None
    if args.splits:
        splits_filter = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    try:
        entries = build_manifest(
            guitarset_root=args.guitarset,
            guitar_techs_root=args.guitar_techs,
            splits=splits_filter,
            max_clips_per_tier=args.max_clips_per_tier,
            total_limit=args.limit,
            validation_player=args.guitarset_validation_player,
        )
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2

    if not entries:
        print(
            "No clips discovered. Check --guitarset / --guitar-techs paths.",
            flush=True,
        )
        return 1

    header = (
        "Composite-eval manifest generated by "
        "tabvision/scripts/eval/build_composite_manifest.py."
        "\nRe-generate with the same args to refresh; this file is "
        "intended to be auto-managed."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_toml(entries, header_comment=header, data_root=args.data_root),
        encoding="utf-8",
    )

    print(f"Wrote {len(entries)} clips to {args.output}", flush=True)
    print(summarise_coverage(entries), flush=True)

    validation: ManifestValidation = validate_manifest(args.output)
    fail_items = [item for item in validation.items if item.severity == "fail"]
    if fail_items:
        print(f"\nValidation FAILED with {len(fail_items)} issue(s):", flush=True)
        for item in fail_items:
            print(f"  [{item.code}] {item.message}", flush=True)
        return 2

    print("\nManifest validation passed.", flush=True)
    return 0


__all__ = [
    "ClipEntry",
    "GUITARSET_VALIDATION_PLAYER",
    "apply_limits",
    "build_manifest",
    "main",
    "render_toml",
    "scan_guitar_techs",
    "scan_guitarset",
    "summarise_coverage",
]


if __name__ == "__main__":
    raise SystemExit(main())
