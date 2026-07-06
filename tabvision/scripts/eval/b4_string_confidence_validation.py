"""B4 validation — is the Viterbi-margin string confidence enriched for wrong_position?

``fuse`` now writes ``TabEvent.confidence`` as a transform of the string-flip
margin (see ``playability.string_margin_to_confidence``). For the editor's red
= "check the string" UX to be honest, **low confidence must concentrate the
actual wrong-string errors**. This script measures that on val24 at the
accepted config (highres + guitarset-v1), the same set the B4 default change
must clear.

Per predicted note it assigns the six-bucket decomposition label (matcher
ported bit-for-bit from ``tabvision.eval.error_decomposition.decompose_errors``
so aggregate counts reconcile with the banked decomposition), then over the
**pitch-correct population** (``correct`` ∪ ``wrong_position_same_pitch`` — the
notes where the string is the open question) reports:

- threshold-free **AUC** = P(a wrong_position note is less confident than a
  correct note) — enrichment iff > 0.5;
- **wrong_position rate by confidence quartile** — should fall as confidence
  rises;
- **flagging precision/recall** if the UI reds the lowest-confidence X%.

**Gate (PASS to ship the default):** AUC ≥ 0.60 *and* bottom-quartile
wrong_position rate > the population base rate.

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.b4_string_confidence_validation \
        --output ../docs/EVAL_REPORTS/b4_string_confidence_val24_2026-07-06.md
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

from scripts.eval.v1_1_second_corpus_probe import CachingPredictor
from tabvision.eval.composite import (
    _resolve_path,
    _session_from_clip,
    make_run_pipeline_predictor,
)
from tabvision.eval.error_decomposition import DEFAULT_ONSET_TOLERANCE_S
from tabvision.eval.parsers import get_parser
from tabvision.types import GuitarConfig, TabEvent

_DEFAULT_CACHE_DIR = Path.home() / ".tabvision/cache/b4_string_confidence"


def label_predictions(
    predicted: list[TabEvent],
    gold: list[TabEvent],
    *,
    onset_tolerance_s: float = DEFAULT_ONSET_TOLERANCE_S,
    timing_extended_tolerance_s: float = 0.15,
) -> list[str]:
    """Per-predicted-note decomposition label, aligned to ``predicted`` order.

    Ports the priority matcher of ``decompose_errors`` (gold claims the best
    unused pred: same position → correct, else same pitch →
    wrong_position_same_pitch, else → pitch_off; extended pass → timing_only),
    recording which pred each gold claimed and with what label. Preds never
    claimed → ``extra_detection``. Aggregate label counts therefore equal the
    ``decompose_errors`` bucket counts on the same inputs.
    """
    labels = ["extra_detection"] * len(predicted)
    used = [False] * len(predicted)
    for g in sorted(gold, key=lambda e: e.onset_s):
        best_pos_idx = best_pitch_idx = best_any_idx = -1
        best_pos_dt = best_pitch_dt = best_any_dt = onset_tolerance_s + 1e-9
        for pi, p in enumerate(predicted):
            if used[pi]:
                continue
            dt = abs(p.onset_s - g.onset_s)
            if dt > onset_tolerance_s:
                continue
            same_pos = p.string_idx == g.string_idx and p.fret == g.fret
            if same_pos:
                if dt < best_pos_dt:
                    best_pos_idx, best_pos_dt = pi, dt
            elif p.pitch_midi == g.pitch_midi:
                if dt < best_pitch_dt:
                    best_pitch_idx, best_pitch_dt = pi, dt
            elif dt < best_any_dt:
                best_any_idx, best_any_dt = pi, dt

        if best_pos_idx >= 0:
            used[best_pos_idx] = True
            labels[best_pos_idx] = "correct"
            continue
        if best_pitch_idx >= 0:
            used[best_pitch_idx] = True
            labels[best_pitch_idx] = "wrong_position_same_pitch"
            continue
        if best_any_idx >= 0:
            used[best_any_idx] = True
            labels[best_any_idx] = "pitch_off"
            continue

        timing_idx = -1
        timing_dt = timing_extended_tolerance_s + 1e-9
        for pi, p in enumerate(predicted):
            if used[pi]:
                continue
            dt = abs(p.onset_s - g.onset_s)
            if dt > timing_extended_tolerance_s:
                continue
            same_pos = p.string_idx == g.string_idx and p.fret == g.fret
            if (same_pos or p.pitch_midi == g.pitch_midi) and dt < timing_dt:
                timing_idx, timing_dt = pi, dt
        if timing_idx >= 0:
            used[timing_idx] = True
            labels[timing_idx] = "timing_only"

    return labels


@dataclass
class Scored:
    confidence: float
    is_wrong_position: bool  # within the pitch-correct population


def auc_wrong_below_correct(scored: list[Scored]) -> float:
    """P(random wrong_position note less confident than random correct note).

    Rank-based Mann–Whitney statistic; 0.5 = no enrichment, ties count 0.5.
    """
    wrong = sorted(s.confidence for s in scored if s.is_wrong_position)
    correct = sorted(s.confidence for s in scored if not s.is_wrong_position)
    if not wrong or not correct:
        return float("nan")
    # For each wrong note, count correct notes with higher confidence (+ 0.5 ties).
    total = 0.0
    for w in wrong:
        lo = sum(1 for c in correct if c < w)
        eq = sum(1 for c in correct if c == w)
        total += (len(correct) - lo - eq) + 0.5 * eq
    return total / (len(wrong) * len(correct))


def _quartile_table(scored: list[Scored]) -> list[tuple[str, int, float, float, float]]:
    """(label, n, conf_lo, conf_hi, wrong_rate) per confidence quartile."""
    ordered = sorted(scored, key=lambda s: s.confidence)
    n = len(ordered)
    rows: list[tuple[str, int, float, float, float]] = []
    for q in range(4):
        lo_i = q * n // 4
        hi_i = (q + 1) * n // 4
        bucket = ordered[lo_i:hi_i]
        if not bucket:
            continue
        wrong = sum(s.is_wrong_position for s in bucket)
        rows.append(
            (
                f"Q{q + 1}",
                len(bucket),
                bucket[0].confidence,
                bucket[-1].confidence,
                wrong / len(bucket),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/eval/local_gs_val24.toml"))
    ap.add_argument("--backend", default="highres")
    ap.add_argument("--position-prior", default="guitarset-v1")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--media-root", type=Path, default=None)
    ap.add_argument("--annotation-root", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    position_prior: str | None = args.position_prior
    if position_prior and position_prior.lower() == "none":
        position_prior = None

    base_predictor = make_run_pipeline_predictor(
        audio_backend_name=args.backend, position_prior=position_prior
    )
    predictor = CachingPredictor(
        base_predictor,
        cache_dir=args.cache_dir,
        key_fields={
            "backend": args.backend,
            "position_prior": position_prior or "none",
            "b4_confidence": "margin",  # invalidate any pre-B4 cache
        },
        refresh_cache=args.refresh_cache,
    )

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    cfg = GuitarConfig()
    payload = tomllib.loads(args.manifest.read_text(encoding="utf-8"))

    scored: list[Scored] = []
    label_counts: dict[str, int] = {}
    for clip in payload.get("clips") or []:
        if clip["split"] not in splits:
            continue
        media = _resolve_path(clip["media_path"], args.media_root)
        annotation = _resolve_path(clip["annotation_path"], args.annotation_root)
        gold = get_parser(clip["annotation_format"])(annotation, cfg)
        pred = predictor(media, _session_from_clip(clip))
        labels = label_predictions(pred, gold)
        for note, label in zip(pred, labels, strict=True):
            label_counts[label] = label_counts.get(label, 0) + 1
            if label in ("correct", "wrong_position_same_pitch"):
                scored.append(Scored(note.confidence, label == "wrong_position_same_pitch"))

    report = _format_report(scored, label_counts, args, position_prior)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    return 0


def _format_report(
    scored: list[Scored],
    label_counts: dict[str, int],
    args: argparse.Namespace,
    position_prior: str | None,
) -> str:
    lines = ["# B4 — string-confidence enrichment validation (val24)", ""]
    lines.append(
        f"Config: `{args.backend}` + `{position_prior or 'none'}`, "
        f"manifest `{args.manifest}`, splits `{args.splits}`."
    )
    lines.append("")
    if not scored:
        lines.append("No pitch-correct predictions to score.")
        return "\n".join(lines) + "\n"

    base = sum(s.is_wrong_position for s in scored) / len(scored)
    auc = auc_wrong_below_correct(scored)
    quartiles = _quartile_table(scored)
    q1_rate = quartiles[0][4] if quartiles else float("nan")

    passed = auc >= 0.60 and q1_rate > base
    lines.append(f"**Gate: {'PASS' if passed else 'FAIL'}** — AUC ≥ 0.60 and Q1 rate > base rate.")
    lines.append("")
    lines.append(
        f"- Pitch-correct population (correct ∪ wrong_position): **{len(scored)}** notes; "
        f"wrong_position base rate **{base:.3f}**."
    )
    lines.append(
        f"- **AUC (wrong_position less confident than correct) = {auc:.3f}** "
        f"({'enriched' if auc > 0.5 else 'no enrichment'}; 0.5 = chance)."
    )
    lines.append("")

    lines.append("## Wrong_position rate by confidence quartile")
    lines.append("")
    lines.append("| quartile | notes | conf range | wrong_position rate | vs base |")
    lines.append("|---|---:|---|---:|---:|")
    for label, n, lo, hi, rate in quartiles:
        lines.append(f"| {label} | {n} | [{lo:.3f}, {hi:.3f}] | {rate:.3f} | {rate / base:.2f}× |")
    lines.append("")

    lines.append("## Flagging utility (red = lowest-confidence X%)")
    lines.append("")
    lines.append(
        "| flag fraction | conf threshold | precision (flagged are wrong) | recall (wrong caught) |"
    )
    lines.append("|---:|---:|---:|---:|")
    ordered = sorted(scored, key=lambda s: s.confidence)
    total_wrong = sum(s.is_wrong_position for s in scored)
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        k = max(1, int(frac * len(ordered)))
        flagged = ordered[:k]
        tp = sum(s.is_wrong_position for s in flagged)
        thr = flagged[-1].confidence
        precision = tp / k
        recall = tp / total_wrong if total_wrong else float("nan")
        lines.append(f"| {frac:.0%} | ≤ {thr:.3f} | {precision:.3f} | {recall:.3f} |")
    lines.append("")

    lines.append("## Predicted-note label counts (reconcile with decomposition)")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---:|")
    for label in (
        "correct",
        "wrong_position_same_pitch",
        "pitch_off",
        "timing_only",
        "extra_detection",
    ):
        lines.append(f"| {label} | {label_counts.get(label, 0)} |")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
