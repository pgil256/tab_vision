"""A14: cache-only video complementarity probe on the WS0 GAPS rich cache.

The chunk-6 capstone measured *aggregate* string resolution on the clean-12
ambiguous notes: audio playability prior 0.778 > WS1 video 0.574. What it left
unmeasured is the **per-note joint distribution** — whether the notes video
gets right are the *same* notes audio gets right (video redundant) or a
complementary set (a routed hybrid could lift). This probe closes that
question, cache-only, in seconds:

1. **Audio side** — decode gold-pitch ``AudioEvent``s with the exact ``fuse``
   cluster-Viterbi (mirrored in-script and self-checked against
   ``fuse(events, [], cfg)``), extracting per note the chosen string *and a
   string-flip local margin*: the cost gap (nats) between the chosen state and
   the cheapest alternative state that puts this note on a different string,
   with neighbouring cluster choices held fixed. This is the same
   best-vs-next-best trellis quantity the B4 confidence work will surface.
2. **Video side** — per-ambiguous-note predicted string from the rich v2 cache
   exactly as ``v1_1_gaps_string_diag`` computes it (gold pitch, best fixed
   orientation per clip, optional WS1 ``--calibrate`` fret-map).
3. **Join** — per-note 2×2 confusion (audio right/wrong × video right/wrong)
   on ambiguous notes with CV evidence, plus **margin-keyed routing**: route a
   note to video when the audio margin is below a threshold; sweep the
   threshold and compare against audio-only and the oracle router
   (audio-right OR video-right — the ceiling any router could reach).

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.a14_video_complementarity_probe --calibrate \
        --output ../docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import (
    CalibrateFn,
    load_frame_fingerings,
    make_fret_xs_calibrator,
    needed_frames,
)
from scripts.eval.v1_1_kaggle_oracle_probe import _events_from_gold
from tabvision.demux import _probe_metadata
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion import chord, playability
from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.fusion.vision_evidence import ORIENTATIONS, combine_fingerings, orient_fingering
from tabvision.fusion.viterbi import fuse
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, TabEvent


@dataclass(frozen=True)
class NoteJoin:
    """One ambiguous gold note with both evidence sources attached."""

    clip: str
    gold_string: int
    audio_string: int
    audio_margin: float  # string-flip local margin, nats; inf = no alternative
    video_string: int | None  # None = no CV evidence near onset
    cluster_size: int  # decode cluster size; >= 2 means chord member

    @property
    def audio_right(self) -> bool:
        return self.audio_string == self.gold_string

    @property
    def video_right(self) -> bool:
        return self.video_string == self.gold_string


# --------------------------------------------------------------------------- #
# Audio side: fuse-mirroring decode with per-note string-flip margins
# --------------------------------------------------------------------------- #
def decode_with_margins(
    events: list[AudioEvent],
    cfg: GuitarConfig,
) -> dict[int, tuple[Candidate, float, int]]:
    """Mirror ``fuse(events, [], cfg)``; per event: pick, margin, cluster size.

    Keyed by index into ``events``. The margin for event ``e`` in cluster
    ``i`` is ``min cost(s) - cost(ŝ)`` over states ``s`` of cluster ``i``
    that put ``e`` on a different string, holding the decoded states of
    clusters ``i−1`` / ``i+1`` fixed (local one-cluster perturbation —
    the trellis best-vs-next-best restricted to a string flip). ``inf``
    when no reachable state flips the string.
    """
    index_of = {id(ev): i for i, ev in enumerate(events)}
    valid_events = [ev for ev in events if candidate_positions(ev.pitch_midi, cfg)]
    if not valid_events:
        return {}

    clusters = chord.cluster_events(valid_events)
    cluster_data: list[tuple[list[AudioEvent], list[tuple[Candidate, ...]]]] = []
    for cluster in clusters:
        states = chord.enumerate_chord_states(cluster, cfg)
        if states:
            cluster_data.append((cluster, states))
    if not cluster_data:
        return {}

    def state_emission(cluster: list[AudioEvent], state: tuple[Candidate, ...]) -> float:
        return sum(
            playability.emission_cost(c, ev, None, cfg)
            for ev, c in zip(cluster, state, strict=True)
        )

    n = len(cluster_data)
    emissions: list[list[float]] = [
        [state_emission(cluster, st) for st in states] for cluster, states in cluster_data
    ]
    cost: list[list[float]] = [[] for _ in range(n)]
    backptr: list[list[int]] = [[] for _ in range(n)]
    cost[0] = list(emissions[0])
    backptr[0] = [-1] * len(cluster_data[0][1])

    def trans(i: int, prev_state: tuple[Candidate, ...], state: tuple[Candidate, ...]) -> float:
        """Transition cost into cluster ``i`` — mirrors ``_viterbi_clusters``."""
        single_line = len(cluster_data[i - 1][0]) == 1 and len(cluster_data[i][0]) == 1
        return playability.transition_cost(
            chord.chord_anchor(prev_state),
            chord.chord_anchor(state),
            cfg,
            use_sequence_prior=single_line,
        )

    for i in range(1, n):
        _, states_i = cluster_data[i]
        _, prev_states = cluster_data[i - 1]
        cost[i] = [math.inf] * len(states_i)
        backptr[i] = [-1] * len(states_i)
        for si, state in enumerate(states_i):
            emit = emissions[i][si]
            for pi, prev_state in enumerate(prev_states):
                total = cost[i - 1][pi] + trans(i, prev_state, state) + emit
                if total < cost[i][si]:
                    cost[i][si] = total
                    backptr[i][si] = pi

    final = cost[n - 1]
    picks_idx = [0] * n
    picks_idx[n - 1] = min(range(len(final)), key=lambda j: final[j])
    for i in range(n - 1, 0, -1):
        picks_idx[i - 1] = backptr[i][picks_idx[i]]

    out: dict[int, tuple[Candidate, float, int]] = {}
    for i, (cluster, states) in enumerate(cluster_data):
        chosen = states[picks_idx[i]]

        def local_cost(state: tuple[Candidate, ...], i: int = i) -> float:
            c = emissions[i][cluster_data[i][1].index(state)]
            if i > 0:
                c += trans(i, cluster_data[i - 1][1][picks_idx[i - 1]], state)
            if i < n - 1:
                c += trans(i + 1, state, cluster_data[i + 1][1][picks_idx[i + 1]])
            return c

        base = local_cost(chosen)
        for j, (ev, cand) in enumerate(zip(cluster, chosen, strict=True)):
            flip = min(
                (local_cost(s) for s in states if s[j].string_idx != cand.string_idx),
                default=math.inf,
            )
            out[index_of[id(ev)]] = (cand, flip - base, len(cluster))
    return out


# --------------------------------------------------------------------------- #
# Video side: per-note predicted string, diag conventions (best orientation)
# --------------------------------------------------------------------------- #
def video_note_strings(
    gold: list[TabEvent],
    ambiguous: list[int],
    per_frame: dict[int, FrameFingering | None],
    offset_s: float,
    fps: float,
    cfg: GuitarConfig,
    *,
    window_s: float,
    max_frames: int,
) -> tuple[dict[int, int], str]:
    """Per-ambiguous-note predicted string under the best fixed orientation.

    Identical selection rule to ``v1_1_gaps_string_diag.diagnose_clip_strings``
    (orientation maximizing correct-string count, gold pitch restriction),
    but returns the per-note predictions instead of aggregate counts.
    """
    events = _events_from_gold(gold)
    _, per_onset = needed_frames(
        [e.onset_s for e in events], offset_s, fps, window_s=window_s, max_frames=max_frames
    )
    raw_by_note = {
        i: [f for fi in per_onset[i] if (f := per_frame.get(fi)) is not None] for i in ambiguous
    }

    best_preds: dict[int, int] = {}
    best_correct = -1
    best_name = ORIENTATIONS[0].name
    for orientation in ORIENTATIONS:
        preds: dict[int, int] = {}
        correct = 0
        for i in ambiguous:
            g = gold[i]
            oriented = [orient_fingering(f, orientation) for f in raw_by_note[i]]
            voted = combine_fingerings(oriented, cfg, t=g.onset_s)
            if voted.homography_confidence <= 0.0:
                continue
            marginal = voted.marginal_string_fret()
            cands = candidate_positions(g.pitch_midi, cfg)
            pred = max(cands, key=lambda c: marginal[c.string_idx, c.fret])
            preds[i] = pred.string_idx
            if pred.string_idx == g.string_idx:
                correct += 1
        if correct > best_correct:
            best_correct = correct
            best_preds = preds
            best_name = orientation.name
    return best_preds, best_name


# --------------------------------------------------------------------------- #
# Join + analysis
# --------------------------------------------------------------------------- #
def probe_clip(
    stem: str,
    data_root: Path,
    video_cache: Path,
    cache_dir: Path,
    cfg: GuitarConfig,
    *,
    conf: float,
    window_s: float,
    max_frames: int,
    calibrate: CalibrateFn | None,
) -> tuple[list[NoteJoin], str] | None:
    xml = data_root / "gaps" / "musicxml" / f"{stem}.xml"
    vid = video_cache / f"{stem}.mp4"
    offset_pkl = cache_dir / f"{stem}.offset.pkl"
    for path, what in ((xml, "musicxml"), (vid, "video"), (offset_pkl, "cached offset")):
        if not path.exists():
            print(f"  [skip] {stem}: no {what}")
            return None
    gold = parse_gaps(xml)
    if not gold:
        print(f"  [skip] {stem}: empty gold")
        return None
    with open(offset_pkl, "rb") as fh:
        offset_s = float(pickle.load(fh).offset_s)
    _dur, fps = _probe_metadata(vid)
    try:
        per_frame = load_frame_fingerings(
            cache_dir, stem, conf=conf, cfg=cfg, fps=fps, calibrate=calibrate
        )
    except FileNotFoundError as exc:
        print(f"  [skip] {stem}: {exc}")
        return None

    events = _events_from_gold(gold)
    decoded = decode_with_margins(events, cfg)

    # Self-check: the mirrored DP must reproduce fuse() exactly.
    ours = sorted(
        (events[i].onset_s, events[i].pitch_midi, c.string_idx, c.fret)
        for i, (c, _m, _sz) in decoded.items()
    )
    ref = sorted((t.onset_s, t.pitch_midi, t.string_idx, t.fret) for t in fuse(events, [], cfg))
    if ours != ref:
        raise AssertionError(f"{stem}: mirrored decode diverges from fuse()")

    ambiguous = [i for i, g in enumerate(gold) if len(candidate_positions(g.pitch_midi, cfg)) >= 2]
    video_preds, best_orient = video_note_strings(
        gold, ambiguous, per_frame, offset_s, fps, cfg, window_s=window_s, max_frames=max_frames
    )

    joins = [
        NoteJoin(
            clip=stem,
            gold_string=gold[i].string_idx,
            audio_string=decoded[i][0].string_idx,
            audio_margin=decoded[i][1],
            video_string=video_preds.get(i),
            cluster_size=decoded[i][2],
        )
        for i in ambiguous
        if i in decoded
    ]
    return joins, best_orient


def routing_accuracy(joined: list[NoteJoin], threshold: float) -> float:
    """Accuracy when notes with margin < threshold are routed to video."""
    right = sum((n.video_right if n.audio_margin < threshold else n.audio_right) for n in joined)
    return right / len(joined) if joined else 0.0


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def format_report(
    all_notes: list[NoteJoin],
    per_clip_orient: dict[str, str],
    *,
    calibrated: bool,
    args: argparse.Namespace,
) -> str:
    joined = [n for n in all_notes if n.video_string is not None]
    lines: list[str] = ["# A14: cache-only video complementarity probe (GAPS clean-12)", ""]
    lines.append(
        f"Video mode: **{'WS1 calibrated' if calibrated else 'baseline uniform'}**, "
        f"best fixed orientation per clip (diag convention — video's ceiling); "
        f"audio: `fuse(events, [], cfg)` on gold-pitch events (capstone convention). "
        f"vote-frames={args.vote_frames}, window={args.vote_window_s}s, conf={args.conf}."
    )
    lines.append("")

    # Aggregate parity numbers
    audio_acc_all = sum(n.audio_right for n in all_notes) / len(all_notes)
    lines.append(f"- Ambiguous notes: **{len(all_notes)}**; with CV evidence: **{len(joined)}**")
    lines.append(
        f"- Audio prior string accuracy, all ambiguous: "
        f"**{sum(n.audio_right for n in all_notes)}/{len(all_notes)} = {audio_acc_all:.3f}** "
        f"(capstone parity: 0.778)"
    )
    if joined:
        audio_acc_j = sum(n.audio_right for n in joined) / len(joined)
        video_acc_j = sum(n.video_right for n in joined) / len(joined)
        lines.append(
            f"- On the joined subset (ambiguous ∩ CV): audio **{audio_acc_j:.3f}**, "
            f"video **{video_acc_j:.3f}** (capstone parity: 0.574 calibrated / 0.544 baseline)"
        )
    lines.append("")

    # 2x2 confusion
    both = sum(n.audio_right and n.video_right for n in joined)
    audio_only = sum(n.audio_right and not n.video_right for n in joined)
    video_only = sum((not n.audio_right) and n.video_right for n in joined)
    neither = sum((not n.audio_right) and (not n.video_right) for n in joined)
    lines.append("## Per-note confusion (ambiguous ∩ CV evidence)")
    lines.append("")
    lines.append("| | video right | video wrong | total |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| **audio right** | {both} | {audio_only} | {both + audio_only} |")
    lines.append(f"| **audio wrong** | {video_only} | {neither} | {video_only + neither} |")
    lines.append(f"| **total** | {both + video_only} | {audio_only + neither} | {len(joined)} |")
    lines.append("")
    if joined and (video_only + neither):
        p_video_right = (both + video_only) / len(joined)
        p_video_right_given_audio_wrong = video_only / (video_only + neither)
        enriched = p_video_right_given_audio_wrong > p_video_right + 0.02
        lines.append(
            f"- P(video right) = **{p_video_right:.3f}**; "
            f"P(video right | audio wrong) = **{p_video_right_given_audio_wrong:.3f}** — "
            f"{'ENRICHED' if enriched else 'no enrichment'} "
            f"(complementarity requires the conditional to exceed the marginal)"
        )
        oracle = (both + audio_only + video_only) / len(joined)
        audio_acc_j = (both + audio_only) / len(joined)
        lines.append(
            f"- Oracle-router ceiling (audio right OR video right): **{oracle:.3f}** "
            f"(+{oracle - audio_acc_j:.3f} over audio-only — the max ANY router could add)"
        )
    lines.append("")

    # Chord-axis split (D1: the 0.85 chord-instance video reference)
    lines.append("## Singleton vs chord-member split (the D1 chord axis)")
    lines.append("")
    lines.append(
        "Chord member = note decoded in a cluster of ≥ 2 simultaneous events. "
        "If video were to beat audio anywhere, the chord-frame hypothesis says "
        "it would be here (a chord shape is one static frame)."
    )
    lines.append("")
    lines.append(
        "| subset | notes | audio acc | video acc | audio-wrong ∩ video-right | oracle ceiling |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, subset in (
        ("singleton", [n for n in joined if n.cluster_size == 1]),
        ("chord member", [n for n in joined if n.cluster_size >= 2]),
    ):
        if not subset:
            continue
        a = sum(n.audio_right for n in subset) / len(subset)
        v = sum(n.video_right for n in subset) / len(subset)
        vo = sum((not n.audio_right) and n.video_right for n in subset)
        oracle_s = sum(n.audio_right or n.video_right for n in subset) / len(subset)
        lines.append(
            f"| {label} | {len(subset)} | {a:.3f} | {v:.3f} | {vo} ({vo / len(subset):.1%}) | "
            f"{oracle_s:.3f} |"
        )
    lines.append("")

    # Margin-keyed analysis
    lines.append("## Audio-uncertainty-keyed routing (string-flip margin)")
    lines.append("")
    lines.append(
        "Margin = cost gap (nats) between the decoded state and the cheapest "
        "state putting the note on a different string (neighbours fixed) — "
        "the B4 trellis confidence. Routing rule: margin < τ → take video."
    )
    lines.append("")
    if joined:
        margins = sorted(n.audio_margin for n in joined if math.isfinite(n.audio_margin))
        # Quartile enrichment table
        lines.append("| margin quartile | notes | audio acc | video acc | video − audio |")
        lines.append("|---|---:|---:|---:|---:|")
        qs = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        finite = [n for n in joined if math.isfinite(n.audio_margin)]
        finite.sort(key=lambda n: n.audio_margin)
        for lo_q, hi_q in qs:
            lo_i = int(lo_q * len(finite))
            hi_i = int(hi_q * len(finite))
            bucket = finite[lo_i:hi_i]
            if not bucket:
                continue
            a = sum(n.audio_right for n in bucket) / len(bucket)
            v = sum(n.video_right for n in bucket) / len(bucket)
            lines.append(
                f"| Q{int(lo_q * 4) + 1} [{bucket[0].audio_margin:.2f}, "
                f"{bucket[-1].audio_margin:.2f}] | {len(bucket)} | {a:.3f} | {v:.3f} | "
                f"{v - a:+.3f} |"
            )
        inf_bucket = [n for n in joined if not math.isfinite(n.audio_margin)]
        if inf_bucket:
            a = sum(n.audio_right for n in inf_bucket) / len(inf_bucket)
            v = sum(n.video_right for n in inf_bucket) / len(inf_bucket)
            lines.append(
                f"| ∞ (no string alternative) | {len(inf_bucket)} | {a:.3f} | {v:.3f} | "
                f"{v - a:+.3f} |"
            )
        lines.append("")

        # Threshold sweep
        audio_acc_j = sum(n.audio_right for n in joined) / len(joined)
        thresholds = sorted({0.0, *margins, math.inf})
        best_t, best_acc = 0.0, audio_acc_j
        for t in thresholds:
            acc = routing_accuracy(joined, t)
            if acc > best_acc:
                best_t, best_acc = t, acc
        lines.append("| routing threshold τ (nats) | routed→video | accuracy | Δ vs audio-only |")
        lines.append("|---:|---:|---:|---:|")
        sweep_points = sorted(
            {
                0.0,
                quantile(margins, 0.1),
                quantile(margins, 0.25),
                quantile(margins, 0.5),
                quantile(margins, 0.75),
                quantile(margins, 0.9),
                best_t,
            }
        )
        for t in sweep_points:
            routed = sum(n.audio_margin < t for n in joined)
            acc = routing_accuracy(joined, t)
            marker = " ← best" if t == best_t and best_acc > audio_acc_j else ""
            lines.append(f"| {t:.3f} | {routed} | {acc:.4f} | {acc - audio_acc_j:+.4f}{marker} |")
        lines.append("")
        lines.append(
            f"**Best routed accuracy = {best_acc:.4f} at τ = {best_t:.3f}** vs audio-only "
            f"{audio_acc_j:.4f} (Δ = {best_acc - audio_acc_j:+.4f}). "
            + (
                "Routing NEVER beats audio-only — the margin does not identify a "
                "subpopulation where video is the better source."
                if best_acc <= audio_acc_j + 1e-9
                else "Routing finds a lift; validate against Tab F1 before any SPEC claim."
            )
        )
    lines.append("")

    # Per-clip table
    lines.append("## Per-clip breakdown")
    lines.append("")
    lines.append(
        "| clip | ambig | haveCV | audio acc | video acc | both | audio-only | "
        "video-only | neither | best orient |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    clips = sorted({n.clip for n in all_notes})
    for clip in clips:
        notes = [n for n in all_notes if n.clip == clip]
        j = [n for n in notes if n.video_string is not None]
        a_acc = sum(n.audio_right for n in notes) / len(notes) if notes else 0.0
        v_acc = sum(n.video_right for n in j) / len(j) if j else 0.0
        lines.append(
            f"| {clip} | {len(notes)} | {len(j)} | {a_acc:.3f} | {v_acc:.3f} | "
            f"{sum(n.audio_right and n.video_right for n in j)} | "
            f"{sum(n.audio_right and not n.video_right for n in j)} | "
            f"{sum((not n.audio_right) and n.video_right for n in j)} | "
            f"{sum((not n.audio_right) and (not n.video_right) for n in j)} | "
            f"{per_clip_orient.get(clip, '—')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf (cache key)")
    ap.add_argument("--vote-frames", type=int, default=1)
    ap.add_argument("--vote-window-s", type=float, default=0.06)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="WS1 per-clip nonlinear fret-map calibration (the best measured video)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None, help="write markdown report here")
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    calibrate = make_fret_xs_calibrator(cfg) if args.calibrate else None
    clips = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )
    if args.limit is not None:
        clips = clips[: args.limit]

    all_notes: list[NoteJoin] = []
    per_clip_orient: dict[str, str] = {}
    for stem in clips:
        result = probe_clip(
            stem,
            args.data_root,
            args.video_cache,
            args.cache_dir,
            cfg,
            conf=args.conf,
            window_s=args.vote_window_s,
            max_frames=args.vote_frames,
            calibrate=calibrate,
        )
        if result is None:
            continue
        joins, orient = result
        all_notes.extend(joins)
        per_clip_orient[stem] = orient
        j = [n for n in joins if n.video_string is not None]
        print(
            f"  {stem}: ambig={len(joins)} haveCV={len(j)} "
            f"audio={sum(n.audio_right for n in joins) / len(joins):.3f} "
            f"video={sum(n.video_right for n in j) / len(j) if j else 0.0:.3f} "
            f"video-only-right={sum((not n.audio_right) and n.video_right for n in j)}"
        )

    if not all_notes:
        print("no clips probed")
        return 1

    report = format_report(all_notes, per_clip_orient, calibrated=bool(args.calibrate), args=args)
    print()
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
