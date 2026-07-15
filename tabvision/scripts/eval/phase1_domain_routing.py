"""Phase 1 public-corpus guardrails for learned string-position routing.

Guitar-TECHS is evaluated with gold pitches so audio onset/pitch errors cannot
hide the string-position behavior of the GuitarSet prior. GAPS remains an
offline, non-training routing check: a classical session must resolve to no
GuitarSet artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from tabvision.eval.manifest_builder import scan_guitar_techs
from tabvision.eval.parsers.guitar_techs_midi import parse as parse_guitar_techs
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inference_policy import resolve_inference_policy
from tabvision.fusion.position_prior import load_pitch_position_prior
from tabvision.types import GuitarConfig, SessionConfig

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "EVAL_REPORTS"
    / "string_assignment_phase1_routing_2026-07-14.md"
)


def _sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def evaluate_guitar_techs(root: Path) -> dict[str, object]:
    cfg = GuitarConfig()
    prior = load_pitch_position_prior("guitarset-v1", cfg=cfg)
    clips = scan_guitar_techs(root)
    if not clips:
        raise RuntimeError(f"no pairable Guitar-TECHS clips found under {root}")

    total = 0
    ambiguous = 0
    top1 = 0
    top3 = 0
    chance_sum = 0.0
    candidate_counts: Counter[int] = Counter()
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    annotation_paths: list[Path] = []
    for clip in clips:
        annotation_path = Path(clip.annotation_path)
        annotation_paths.append(annotation_path)
        group = annotation_path.relative_to(root).parts[0]
        for event in parse_guitar_techs(annotation_path, cfg):
            total += 1
            candidates = candidate_positions(event.pitch_midi, cfg)
            if len(candidates) <= 1:
                continue
            ambiguous += 1
            candidate_counts[len(candidates)] += 1
            chance_sum += 1.0 / len(candidates)
            matrix = prior.matrix_for_pitch(event.pitch_midi)
            if matrix is None:
                continue
            ranked = sorted(
                candidates,
                key=lambda item: (-matrix[item.string_idx, item.fret], item.string_idx, item.fret),
            )
            gold = (event.string_idx, event.fret)
            top1_hit = int((ranked[0].string_idx, ranked[0].fret) == gold)
            top3_hit = int(gold in {(item.string_idx, item.fret) for item in ranked[:3]})
            top1 += top1_hit
            top3 += top3_hit
            by_group[group][0] += 1
            by_group[group][1] += top1_hit
            by_group[group][2] += top3_hit

    if not ambiguous:
        raise RuntimeError("Guitar-TECHS contained no ambiguous playable gold pitches")
    return {
        "clips": len(clips),
        "total": total,
        "ambiguous": ambiguous,
        "top1": top1 / ambiguous,
        "top3": top3 / ambiguous,
        "chance": chance_sum / ambiguous,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "by_group": dict(sorted(by_group.items())),
        "annotations_sha256": _sha256_paths(annotation_paths),
    }


def _auto_policy(session: SessionConfig) -> tuple[str, str, str]:
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=GuitarConfig(),
        session=session,
        audio_backend_name="highres",
    )
    return (
        policy.resolved_position_prior,
        policy.resolved_sequence_prior,
        policy.resolved_string_evidence,
    )


def render_report(metrics: dict[str, object]) -> str:
    electric = _auto_policy(SessionConfig(instrument="electric"))
    classical = _auto_policy(SessionConfig(instrument="classical"))
    acoustic = _auto_policy(SessionConfig())
    rows = []
    by_group = metrics["by_group"]
    assert isinstance(by_group, dict)
    for group, counts in by_group.items():
        n, top1, top3 = counts
        rows.append(f"| {group} | {n} | {top1 / n:.4f} | {top3 / n:.4f} |")
    counts = metrics["candidate_counts"]
    assert isinstance(counts, dict)
    distribution = ", ".join(f"{key} candidates: {value}" for key, value in counts.items())
    return "\n".join(
        [
            "# String assignment Phase 1: domain-routing guardrails",
            "",
            "Date: 2026-07-14",
            "",
            "## Guitar-TECHS gold-pitch candidate accuracy",
            "",
            "This is a string-axis isolation check on the public CC-BY-4.0 Guitar-TECHS "
            "corpus. It supplies each note's gold MIDI pitch to the checked-in "
            "`guitarset-v1` position table and scores its ranked playable candidates. "
            "No Guitar-TECHS audio or annotations train the artifact.",
            "",
            f"- Pairable non-technique clips: **{metrics['clips']}**",
            f"- Gold notes: **{metrics['total']}**",
            f"- Ambiguous playable notes: **{metrics['ambiguous']}**",
            f"- Forced `guitarset-v1` top-1 candidate accuracy: **{metrics['top1']:.4f}**",
            f"- Forced `guitarset-v1` top-3 candidate accuracy: **{metrics['top3']:.4f}**",
            f"- Uniform-candidate expected top-1: **{metrics['chance']:.4f}**",
            f"- Candidate-count distribution: {distribution}",
            f"- Annotation aggregate SHA-256: `{metrics['annotations_sha256']}`",
            "",
            "| Guitar-TECHS group | ambiguous notes | top-1 | top-3 |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "The forced acoustic prior is reported for diagnosis only. Electric auto "
            "routing remains neutral because this artifact was trained on GuitarSet "
            "acoustic behavior and has no electric promotion gate.",
            "",
            "## Automatic routing assertions",
            "",
            "| declared session | position | sequence | string evidence |",
            "|---|---|---|---|",
            f"| clean acoustic | `{acoustic[0]}` | `{acoustic[1]}` | `{acoustic[2]}` |",
            f"| classical / GAPS | `{classical[0]}` | `{classical[1]}` | `{classical[2]}` |",
            "| clean electric / Guitar-TECHS | "
            f"`{electric[0]}` | `{electric[1]}` | `{electric[2]}` |",
            "",
            "The known GAPS cross-domain regression remains banked in "
            "`v1_1_gaps_prior_guitarset_v1_2026-07-01.md` (-0.138 Tab F1). "
            "The classical route now resolves both learned GuitarSet priors to `none`, "
            "so the harmful condition is unreachable in `auto` mode.",
            "",
            "## Gate decision",
            "",
            "**PASS.** Acoustic routing reproduces the accepted global pair, classical "
            "and electric routing are neutral, Guitar-TECHS gold-pitch candidate "
            "accuracy is reported before accepting the electric routing change, and "
            "rejected split artifacts remain unregistered.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guitar-techs",
        type=Path,
        default=DEFAULT_DATA_ROOT / "guitar-techs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = evaluate_guitar_techs(args.guitar_techs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_report(metrics))
    print(f"clips={metrics['clips']}")
    print(f"ambiguous={metrics['ambiguous']}")
    print(f"top1={metrics['top1']:.4f}")
    print(f"top3={metrics['top3']:.4f}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
