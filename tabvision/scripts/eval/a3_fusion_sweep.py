"""A3/A4 — in-process fusion-constants sweep on val24 (the load-bearing infra).

Fusion constants (``playability`` emission/transition weights, ``chord``
clustering gap, the position-prior ``alpha``/``power``, A3's new
``FRET_PRIOR_WEIGHT``, A4's ``TRANSITION_GAP_TAU``) only affect ``fuse`` — not
the expensive audio transcription. So this harness:

1. transcribes each val24 clip **once** (highres) into raw ``AudioEvent``s and
   caches them (resumable);
2. for each grid point, rebinds the constant(s), re-applies the position prior,
   re-runs ``fuse`` (milliseconds), and scores per-tier Tab F1.

It runs **1-D marginal sweeps** around the shipped defaults (the same shape that
banked the +1.5pp ``POSITION_SHIFT_COST`` win), on the roadmap's val24 baseline
config: highres + ``guitarset-v1`` position prior, **no** sequence prior
(baseline single-line 0.4820 / strummed 0.7980 — post-A5 ``CHORD_SHAPE_BONUS=0.1``
default; strummed was 0.7951 pre-A5). The baseline row must reproduce those
numbers — that validates the harness. A5 (chord dictionary) and per-tier configs
ride this same infra.

Any grid point that beats the baseline here is a *candidate only*: the
measurement discipline still requires a 60-clip lower-95 confirm + GAPS
clean-12 no-regression before any default change (this harness does neither).

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.a3_fusion_sweep \
        --output ../docs/EVAL_REPORTS/a3_fusion_sweep_val24_2026-07-06.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabvision.eval.composite import _resolve_path, _session_from_clip
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers import get_parser
from tabvision.fusion import chord, chord_shapes, playability
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.viterbi import fuse
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

_DEFAULT_CACHE_DIR = Path.home() / ".tabvision/cache/a3_fusion_sweep"


@dataclass
class ClipData:
    clip_id: str
    tier: str
    raw_events: list[AudioEvent]
    gold: list[TabEvent]


# --- raw AudioEvent (de)serialization (no numpy fields on a raw backend event) -


def _events_to_json(events: list[AudioEvent]) -> list[dict[str, object]]:
    return [
        {
            "onset_s": e.onset_s,
            "offset_s": e.offset_s,
            "pitch_midi": e.pitch_midi,
            "velocity": e.velocity,
            "confidence": e.confidence,
            "tags": list(e.tags),
        }
        for e in events
    ]


def _events_from_json(payload: list[dict[str, Any]]) -> list[AudioEvent]:
    events: list[AudioEvent] = []
    for d in payload:
        raw_tags = d.get("tags", ())
        tags = tuple(str(t) for t in raw_tags) if isinstance(raw_tags, (list, tuple)) else ()
        events.append(
            AudioEvent(
                onset_s=float(d["onset_s"]),
                offset_s=float(d["offset_s"]),
                pitch_midi=int(d["pitch_midi"]),
                velocity=float(d["velocity"]),
                confidence=float(d["confidence"]),
                tags=tags,
            )
        )
    return events


def make_shared_audio_backend(backend_name: str) -> object:
    """Build the audio backend ONCE for reuse across every clip.

    Mirrors ``composite.make_run_pipeline_predictor``: the highres backend
    caches its ~0.5 GB checkpoint (and torchlibrosa's STFT matrices) on
    construction, so rebuilding it per clip is ~10x slower AND accumulates
    memory across a multi-clip run until an allocation fails
    (``numpy._core._exceptions._ArrayMemoryError`` from a fresh
    ``torchlibrosa.stft`` init) partway through a 60+-clip sweep/gate run.
    """
    from tabvision.audio.backend import make as make_audio_backend  # noqa: PLC0415

    return make_audio_backend(backend_name)


def _raw_events_cached(
    media_path: Path,
    session: SessionConfig,
    backend_name: str,
    cache_dir: Path,
    backend: object,
) -> list[AudioEvent]:
    """Transcribe once (highres), cache the raw AudioEvents, resume from cache.

    ``backend`` must be pre-built via :func:`make_shared_audio_backend` and
    reused across every clip in a run — see that function's docstring for why.
    """
    mtime = media_path.stat().st_mtime_ns
    key = json.dumps({"media": str(media_path.resolve()), "backend": backend_name, "mtime": mtime})
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{media_path.stem}.{digest}.json"
    if cache_path.exists():
        return _events_from_json(json.loads(cache_path.read_text(encoding="utf-8")))

    from tabvision.demux import demux  # noqa: PLC0415

    demuxed = demux(str(media_path))
    events = list(backend.transcribe(demuxed.wav, demuxed.sample_rate, session))  # type: ignore[attr-defined]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_events_to_json(events)), encoding="utf-8")
    print(f"  [transcribe] {media_path.name}: {len(events)} events")
    return events


# --- config apply / reset -------------------------------------------------------

# Captured at import (env unset in a normal run → shipped defaults).
_DEFAULTS: dict[tuple[object, str], object] = {
    (playability, "LOW_FRET_BIAS"): playability.LOW_FRET_BIAS,
    (playability, "OPEN_STRING_BONUS"): playability.OPEN_STRING_BONUS,
    (playability, "FRET_PRIOR_WEIGHT"): playability.FRET_PRIOR_WEIGHT,
    (playability, "SAME_STRING_BONUS"): playability.SAME_STRING_BONUS,
    (playability, "POSITION_SHIFT_COST"): playability.POSITION_SHIFT_COST,
    (playability, "SPAN_NORM"): playability.SPAN_NORM,
    (playability, "MAX_HAND_SPAN"): playability.MAX_HAND_SPAN,
    (playability, "HAND_SPAN_BARRIER"): playability.HAND_SPAN_BARRIER,
    (playability, "TRANSITION_GAP_TAU"): playability.TRANSITION_GAP_TAU,
    (chord, "CHORD_MAX_GAP_S"): chord.CHORD_MAX_GAP_S,
    (chord_shapes, "CHORD_SHAPE_BONUS"): chord_shapes.CHORD_SHAPE_BONUS,
}


def _reset_defaults() -> None:
    for (mod, name), value in _DEFAULTS.items():
        setattr(mod, name, value)


# Axes that rebind a module global directly. (module, attr, [values]).
_MODULE_AXES: list[tuple[object, str, list[float]]] = [
    (playability, "POSITION_SHIFT_COST", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    (playability, "FRET_PRIOR_WEIGHT", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
    (playability, "OPEN_STRING_BONUS", [0.0, 0.25, 0.5, 0.75, 1.0]),
    (playability, "SAME_STRING_BONUS", [0.0, 0.25, 0.5, 0.75, 1.0]),
    (playability, "LOW_FRET_BIAS", [0.0, 0.05, 0.10, 0.20, 0.30]),
    (playability, "HAND_SPAN_BARRIER", [2.5, 5.0, 7.5, 10.0]),
    (playability, "SPAN_NORM", [6.0, 9.0, 12.0, 18.0]),
    (chord, "CHORD_MAX_GAP_S", [0.04, 0.06, 0.08, 0.10, 0.12]),
    # A4: gap-decay time constant (inf = off/default).
    (playability, "TRANSITION_GAP_TAU", [float("inf"), 4.0, 2.0, 1.0, 0.5, 0.25]),
    # A5: chord-shape bonus magnitude (shipped default 0.1; 0.0 disables). Only
    # moves strummed — single-line clusters are singletons and never match.
    (chord_shapes, "CHORD_SHAPE_BONUS", [0.0, 0.1, 0.25, 0.5, 1.0]),
]

# Prior alpha/power are baked into the loaded prior, swept by reloading it.
_PRIOR_AXES: list[tuple[str, list[float]]] = [
    ("power", [1.0, 1.5, 2.0, 2.5, 3.0]),
    ("alpha", [0.5, 1.0, 1.5, 2.0]),
]


def _evaluate(clips: list[ClipData], prior_obj: object, cfg: GuitarConfig) -> dict[str, float]:
    """Per-tier mean Tab F1 for the current global config + prior."""
    by_tier: dict[str, list[float]] = {}
    for clip in clips:
        events = apply_pitch_position_prior(clip.raw_events, prior_obj)  # type: ignore[arg-type]
        predicted = fuse(events, [], cfg)
        f1 = tab_f1(predicted, clip.gold).f1
        by_tier.setdefault(clip.tier, []).append(f1)
    out = {tier: sum(v) / len(v) for tier, v in by_tier.items()}
    out["aggregate"] = sum(f1 for v in by_tier.values() for f1 in v) / sum(
        len(v) for v in by_tier.values()
    )
    return out


def _fmt_tiers(tiers: dict[str, float]) -> str:
    sl = tiers.get("clean_acoustic_single_line", float("nan"))
    st = tiers.get("clean_acoustic_strummed", float("nan"))
    return f"{sl:.4f} | {st:.4f} | {tiers['aggregate']:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/eval/local_gs_val24.toml"))
    ap.add_argument("--backend", default="highres")
    ap.add_argument("--position-prior", default="guitarset-v1")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--media-root", type=Path, default=None)
    ap.add_argument("--annotation-root", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    payload = tomllib.loads(args.manifest.read_text(encoding="utf-8"))

    print("transcribing / loading raw audio events...")
    shared_backend = make_shared_audio_backend(args.backend)
    clips: list[ClipData] = []
    for clip in payload.get("clips") or []:
        if clip["split"] not in splits:
            continue
        media = _resolve_path(clip["media_path"], args.media_root)
        annotation = _resolve_path(clip["annotation_path"], args.annotation_root)
        gold = get_parser(clip["annotation_format"])(annotation, cfg)
        raw = _raw_events_cached(
            media, _session_from_clip(clip), args.backend, args.cache_dir, shared_backend
        )
        clips.append(ClipData(clip["id"], clip["tier"], raw, gold))
    print(f"{len(clips)} clips ready\n")

    def prior_for(alpha: float | None, power: float | None) -> object:
        import os  # noqa: PLC0415

        if alpha is not None:
            os.environ["TABVISION_PRIOR_ALPHA"] = str(alpha)
        else:
            os.environ.pop("TABVISION_PRIOR_ALPHA", None)
        if power is not None:
            os.environ["TABVISION_PRIOR_POWER"] = str(power)
        else:
            os.environ.pop("TABVISION_PRIOR_POWER", None)
        return load_pitch_position_prior(args.position_prior, cfg=cfg)

    default_prior = prior_for(None, None)
    _reset_defaults()
    baseline = _evaluate(clips, default_prior, cfg)

    lines = ["# A3/A4 — fusion-constants sweep (val24)", ""]
    lines.append(
        f"Config: `{args.backend}` + `{args.position_prior}` prior, **no sequence prior**, "
        f"splits `{args.splits}`. 1-D marginal sweeps around defaults."
    )
    lines.append("")
    lines.append(
        f"**Baseline** (single-line | strummed | aggregate): **{_fmt_tiers(baseline)}** "
        f"(post-A5 val24 baseline 0.4820 / 0.7980, CHORD_SHAPE_BONUS=0.1 — harness validation)."
    )
    lines.append("")
    lines.append(
        "Δ columns are vs this baseline aggregate; **best** marks the top aggregate per axis."
    )

    best_overall = (baseline["aggregate"], "baseline", None)

    for mod, attr, values in _MODULE_AXES:
        default_val = _DEFAULTS[(mod, attr)]
        lines.append(f"\n## `{attr}` (default {default_val})")
        lines.append("")
        lines.append("| value | single-line | strummed | aggregate | Δ agg |")
        lines.append("|---:|---:|---:|---:|---:|")
        rows: list[tuple[float, dict[str, float]]] = []
        for v in values:
            _reset_defaults()
            setattr(mod, attr, v)
            rows.append((v, _evaluate(clips, default_prior, cfg)))
        _reset_defaults()
        best_v = max(rows, key=lambda r: r[1]["aggregate"])
        for v, tiers in rows:
            mark = (
                " **best**"
                if v == best_v[0] and best_v[1]["aggregate"] > baseline["aggregate"] + 1e-6
                else ""
            )
            d = tiers["aggregate"] - baseline["aggregate"]
            lines.append(f"| {v} | {_fmt_tiers(tiers)} | {d:+.4f}{mark} |")
            if tiers["aggregate"] > best_overall[0]:
                best_overall = (tiers["aggregate"], f"{attr}={v}", None)

    for kind, values in _PRIOR_AXES:
        lines.append(f"\n## prior `{kind}` (default {'2.0' if kind == 'power' else '1.0'})")
        lines.append("")
        lines.append("| value | single-line | strummed | aggregate | Δ agg |")
        lines.append("|---:|---:|---:|---:|---:|")
        _reset_defaults()
        rows = []
        for v in values:
            prior_obj = prior_for(v if kind == "alpha" else None, v if kind == "power" else None)
            rows.append((v, _evaluate(clips, prior_obj, cfg)))
        best_v = max(rows, key=lambda r: r[1]["aggregate"])
        for v, tiers in rows:
            mark = (
                " **best**"
                if v == best_v[0] and best_v[1]["aggregate"] > baseline["aggregate"] + 1e-6
                else ""
            )
            d = tiers["aggregate"] - baseline["aggregate"]
            lines.append(f"| {v} | {_fmt_tiers(tiers)} | {d:+.4f}{mark} |")
            if tiers["aggregate"] > best_overall[0]:
                best_overall = (tiers["aggregate"], f"prior_{kind}={v}", None)
    prior_for(None, None)  # restore env

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    delta = best_overall[0] - baseline["aggregate"]
    if delta > 1e-6:
        lines.append(
            f"Best single-axis point: **{best_overall[1]}** → aggregate {best_overall[0]:.4f} "
            f"(**{delta:+.4f}** vs baseline). Candidate only — needs the 60-clip lower-95 "
            f"confirm + GAPS clean-12 no-regression before any default change."
        )
    else:
        lines.append(
            "No single-axis point beats the baseline aggregate — the shipped constants are "
            "at (or indistinguishable from) a local optimum on val24. Banked as a wash."
        )
    lines.append("")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
