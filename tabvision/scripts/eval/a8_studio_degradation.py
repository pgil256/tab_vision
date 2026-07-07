"""A8 — studio-condition degradation eval (the eval-vs-product gap).

Every accuracy number in the repo is measured on **clean corpus WAVs**, but the
product ingests what a browser ``MediaRecorder`` produces: Opus-in-webm at 48 kHz
from whatever microphone the user has. This harness quantifies that gap by
re-encoding each clean val24 clip through a *capture-chain degradation curve*,
re-transcribing the degraded audio through the real pipeline, and re-scoring.
Gold labels are unchanged (the degradation is audio-only), so it is fully
automated and free.

**Faithful to the actual product.** The web client
(``web-client/src/components/RecordPanel.tsx``) records
``audio/webm;codecs=opus`` with ``echoCancellation``/``noiseSuppression``/
``autoGainControl`` all **disabled**. So the only *guaranteed* product-side
degradation is the Opus codec; the microphone frequency response + room noise
are environmental (user hardware). The curve therefore separates the two:

- ``opus_128`` / ``opus_64`` — Opus round-trip only, no mic model. The honest
  **floor** on product degradation (good mic, quiet room): isolates the codec.
- ``laptop_mic`` — built-in-laptop-mic band-limiting (HP/LP) + a low pink-noise
  floor + Opus. The realistic **typical** capture (AGC off, faithful to the app).
- ``noisy_room`` — harsher band + louder noise + light compression (covers the
  driver/OS-AGC case and the roadmap's "light compression" clause) + low-bitrate
  Opus. A **worst-case** stress bound.

This is a **diagnostic tier, NOT a gate** (roadmap A8 / D1-c) — it does not edit
SPEC §1.4 targets. It exists to *decide a fork*: if accuracy holds through the
chain, keep tuning clean-corpus fusion constants; if it craters, pivot to input
robustness (denoise/AGC, preflight, the B9 bad-input banner) and capture
guidance instead. Bank the number either way.

The clean baseline reuses the warm ``a3_fusion_sweep`` raw-events cache, so the
only new compute is transcribing the degraded audio (highres, ~36 s/clip). Both
the re-encoded media and the degraded raw events are cached, so the run is
resumable.

Reproduce::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.eval.a8_studio_degradation \
        --output ../docs/EVAL_REPORTS/a8_studio_degradation_val24_2026-07-07.md
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from scripts.eval.a3_fusion_sweep import _raw_events_cached, make_shared_audio_backend
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.composite import _resolve_path, _session_from_clip
from tabvision.eval.metrics import event_f1, tab_f1
from tabvision.eval.parsers import get_parser
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.viterbi import fuse
from tabvision.types import GuitarConfig

_DEFAULT_MEDIA_CACHE = Path.home() / ".tabvision/cache/a8_studio/media"
_DEFAULT_EVENTS_CACHE = Path.home() / ".tabvision/cache/a8_studio/events"
# The clean side reuses the sweep's warm highres cache (same media + backend).
_CLEAN_CACHE = Path.home() / ".tabvision/cache/a3_fusion_sweep"

# Fork-classification thresholds on Δ aggregate Tab F1 vs the clean baseline.
# Diagnostic heuristics (not gates); pinned so the verdict is reproducible/tested.
HOLDS_THRESHOLD = -0.03  # Δ >= this  → "holds"
CRATERS_THRESHOLD = -0.08  # Δ <= this → "craters"; between the two → "soft"


@dataclass(frozen=True)
class Profile:
    """One capture-chain degradation condition, compiled to ffmpeg args."""

    name: str
    description: str
    bitrate_k: int
    highpass_hz: int | None = None
    lowpass_hz: int | None = None
    noise_amp: float | None = None  # anoisesrc amplitude (0..1); None = no added noise
    compress: bool = False
    noise_color: str = "pink"
    noise_seed: int = 42  # fixed → reproducible re-encodes


PROFILES: dict[str, Profile] = {
    "opus_128": Profile(
        "opus_128",
        "Opus/webm 48k @128k, no mic model — codec-only floor (good mic, quiet room)",
        bitrate_k=128,
    ),
    "opus_64": Profile(
        "opus_64",
        "Opus/webm 48k @64k, no mic model — codec stress (constrained bitrate)",
        bitrate_k=64,
    ),
    "laptop_mic": Profile(
        "laptop_mic",
        "Built-in laptop mic: HP70/LP8000 + pink noise ~-44 dBFS (SNR~30 dB) + "
        "Opus 96k; AGC off (faithful to the app)",
        bitrate_k=96,
        highpass_hz=70,
        lowpass_hz=8000,
        noise_amp=0.006,
    ),
    "noisy_room": Profile(
        "noisy_room",
        "Worst-case: HP90/LP7000 + pink noise ~-34 dBFS (SNR~16 dB) + light compression + Opus 64k",
        bitrate_k=64,
        highpass_hz=90,
        lowpass_hz=7000,
        noise_amp=0.02,
        compress=True,
    ),
}

DEFAULT_PROFILE_ORDER = ("opus_128", "opus_64", "laptop_mic", "noisy_room")

# val24 roadmap baseline (highres + guitarset-v1, no sequence prior) — the clean
# rows must reproduce these, which validates the harness end to end.
_BASELINE_HINT = "roadmap val24 clean baseline: single-line 0.4820 / strummed 0.7951"


@dataclass(frozen=True)
class Scored:
    """Per-clip metrics under one condition."""

    clip_id: str
    tier: str
    tab: float
    onset: float
    pitch: float


def build_ffmpeg_args(ffmpeg: str, src: Path, out: Path, profile: Profile) -> list[str]:
    """Compile a profile to an ffmpeg argument list.

    Codec-only profiles use a simple ``-af`` chain; profiles with an added noise
    floor use ``-filter_complex`` to mix in an ``anoisesrc`` stream. ``amix`` uses
    ``normalize=0`` so the noise sits at its literal ``anoisesrc`` amplitude rather
    than being halved, and ``duration=first`` truncates the (infinite) noise to
    the signal length. Everything is resampled to 48 kHz mono (getUserMedia's
    capture rate) before the Opus encode.
    """
    signal_chain = ["aresample=48000"]
    if profile.highpass_hz is not None:
        signal_chain.append(f"highpass=f={profile.highpass_hz}")
    if profile.lowpass_hz is not None:
        signal_chain.append(f"lowpass=f={profile.lowpass_hz}")
    if profile.compress:
        signal_chain.append("acompressor=threshold=-18dB:ratio=2.5:attack=15:release=250")
    chain = ",".join(signal_chain)

    tail = [
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "libopus",
        "-b:a",
        f"{profile.bitrate_k}k",
        str(out),
    ]
    head = [ffmpeg, "-y", "-v", "error", "-i", str(src)]

    if profile.noise_amp is None:
        return [*head, "-af", chain, *tail]

    filter_complex = (
        f"[0:a]{chain}[s];"
        f"anoisesrc=color={profile.noise_color}:amplitude={profile.noise_amp}"
        f":sample_rate=48000:seed={profile.noise_seed}[n];"
        f"[s][n]amix=inputs=2:duration=first:normalize=0[m]"
    )
    return [*head, "-filter_complex", filter_complex, "-map", "[m]", *tail]


def reencode_cache_path(src: Path, profile: Profile, out_dir: Path) -> Path:
    """Deterministic path for a clip's degraded webm, invalidated by source mtime.

    Keyed on (resolved source, mtime, profile identity) so a changed source WAV or
    an edited profile forces a fresh encode without manual cache-busting.
    """
    mtime = src.stat().st_mtime_ns
    key = f"{src.resolve()}|{mtime}|{profile!r}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return out_dir / f"{src.stem}.{profile.name}.{digest}.webm"


def reencode(ffmpeg: str, src: Path, profile: Profile, out_dir: Path) -> Path:
    """Re-encode ``src`` through ``profile``, returning the cached degraded path."""
    out = reencode_cache_path(src, profile, out_dir)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        build_ffmpeg_args(ffmpeg, src, out, profile), capture_output=True, check=False
    )
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg re-encode failed for {src.name} / {profile.name}: "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    print(f"  [reencode] {src.stem} -> {profile.name} ({out.stat().st_size // 1024} KiB)")
    return out


def classify_fork(clean_agg: float, profile_aggs: dict[str, float]) -> tuple[str, str]:
    """Map the measured degradation curve to a fork recommendation.

    Reads the codec-only *floor* (``opus_128``) and the realistic *typical*
    condition (``laptop_mic``) when present, falling back to the harshest and
    least-harsh profiles otherwise. Thresholds are the module-level heuristics.
    """

    def delta(name: str) -> float | None:
        return profile_aggs[name] - clean_agg if name in profile_aggs else None

    def band(d: float) -> str:
        if d >= HOLDS_THRESHOLD:
            return "holds"
        if d <= CRATERS_THRESHOLD:
            return "craters"
        return "soft"

    floor = delta("opus_128")
    if floor is None and profile_aggs:
        floor = min(profile_aggs.values()) - clean_agg  # least-degrading available
    realistic = delta("laptop_mic")
    if realistic is None and profile_aggs:
        realistic = min(profile_aggs.values()) - clean_agg  # most-degrading available

    if floor is None or realistic is None:
        return ("inconclusive", "No profiles scored — cannot classify the fork.")

    floor_band, real_band = band(floor), band(realistic)

    if floor_band == "craters":
        label = "codec-dominated"
        prose = (
            f"Even the codec-only floor craters (Δagg {floor:+.4f}). The Opus round-trip "
            "itself is the dominant risk — investigate bitrate/encode settings and consider "
            "server-side re-encode before transcription."
        )
    elif real_band == "craters":
        label = "environment-dominated → pivot to input robustness"
        prose = (
            f"The codec floor holds (Δagg {floor:+.4f}) but the realistic mic/room condition "
            f"craters (Δagg {realistic:+.4f}). The gap is environmental, not the codec — the "
            "highest-value work is input robustness (denoise/AGC, preflight rejection, the B9 "
            "bad-input banner) and capture guidance, NOT more clean-corpus fusion tuning."
        )
    elif real_band == "soft":
        label = "mild degradation → keep tuning, monitor input"
        prose = (
            f"The codec floor holds (Δagg {floor:+.4f}); the realistic condition degrades mildly "
            f"(Δagg {realistic:+.4f}). Clean-corpus tuning is still worthwhile, but track input "
            "quality and keep input-robustness work on the roadmap."
        )
    else:
        label = "robust → keep tuning"
        prose = (
            f"Accuracy holds through the whole capture chain (codec floor Δagg {floor:+.4f}, "
            f"realistic Δagg {realistic:+.4f}). The eval-vs-product gap is small — clean-corpus "
            "accuracy work transfers to the product; keep tuning."
        )
    return (label, prose)


def _tier_values(rows: list[Scored], attr: str) -> dict[str, list[float]]:
    """Group one metric's per-clip values by tier, plus an ``aggregate`` bucket."""
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r.tier, []).append(getattr(r, attr))
    out["aggregate"] = [getattr(r, attr) for r in rows]
    return out


def _fmt_ci(values: list[float]) -> tuple[float, float]:
    ci = bootstrap_ci(values, n_bootstrap=10_000, seed=42)
    return ci.statistic, ci.lower


def _metric_table(
    scores: dict[str, list[Scored]],
    order: list[str],
    attr: str,
    tiers: list[str],
) -> list[str]:
    """Rows = conditions (clean first), columns = tiers + aggregate + Δagg vs clean."""
    header = ["condition"] + [f"{t} mean (lo95)" for t in [*tiers, "aggregate"]] + ["Δ agg"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    clean_agg = _tier_values(scores["clean"], attr)["aggregate"]
    clean_agg_mean = sum(clean_agg) / len(clean_agg)
    for cond in ["clean", *order]:
        by_tier = _tier_values(scores[cond], attr)
        cells: list[str] = [f"`{cond}`"]
        for t in [*tiers, "aggregate"]:
            if t in by_tier and by_tier[t]:
                mean, lo = _fmt_ci(by_tier[t])
                cells.append(f"{mean:.4f} ({lo:.4f})")
            else:
                cells.append("—")
        agg_mean = sum(by_tier["aggregate"]) / len(by_tier["aggregate"])
        d = agg_mean - clean_agg_mean
        cells.append("—" if cond == "clean" else f"{d:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/eval/local_gs_val24.toml"))
    ap.add_argument("--backend", default="highres")
    ap.add_argument("--position-prior", default="guitarset-v1")
    ap.add_argument("--splits", default="validation")
    ap.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILE_ORDER),
        help="comma-separated profile names; see PROFILES",
    )
    ap.add_argument("--media-cache", type=Path, default=_DEFAULT_MEDIA_CACHE)
    ap.add_argument("--events-cache", type=Path, default=_DEFAULT_EVENTS_CACHE)
    ap.add_argument("--clean-cache", type=Path, default=_CLEAN_CACHE)
    ap.add_argument("--media-root", type=Path, default=None)
    ap.add_argument("--annotation-root", type=Path, default=None)
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: PATH lookup)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH; see the tabvision-eval-env setup note")

    order = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in order if p not in PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s): {unknown}; known: {sorted(PROFILES)}")

    prior_name: str | None = args.position_prior
    if prior_name and prior_name.lower() == "none":
        prior_name = None

    cfg = GuitarConfig()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    payload = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    prior = load_pitch_position_prior(prior_name, cfg=cfg) if prior_name else None

    print(f"building shared {args.backend} backend...")
    backend = make_shared_audio_backend(args.backend)

    def score(events: object, gold: object, clip_id: str, tier: str) -> Scored:
        predicted = events if prior is None else apply_pitch_position_prior(events, prior)  # type: ignore[arg-type]
        predicted = fuse(predicted, [], cfg)  # type: ignore[arg-type]
        return Scored(
            clip_id,
            tier,
            tab=tab_f1(predicted, gold).f1,  # type: ignore[arg-type]
            onset=event_f1(predicted, gold, match_pitch=False).f1,  # type: ignore[arg-type]
            pitch=event_f1(predicted, gold, match_pitch=True).f1,  # type: ignore[arg-type]
        )

    scores: dict[str, list[Scored]] = {cond: [] for cond in ["clean", *order]}
    clips = [c for c in (payload.get("clips") or []) if c["split"] in splits]
    if args.limit is not None:
        clips = clips[: args.limit]
    print(f"{len(clips)} clips × ({len(order)} profiles + clean)\n")

    for i, clip in enumerate(clips, 1):
        media = _resolve_path(clip["media_path"], args.media_root)
        annotation = _resolve_path(clip["annotation_path"], args.annotation_root)
        gold = get_parser(clip["annotation_format"])(annotation, cfg)
        session = _session_from_clip(clip)
        cid, tier = clip["id"], clip["tier"]
        print(f"[{i}/{len(clips)}] {cid}")

        clean_events = _raw_events_cached(media, session, args.backend, args.clean_cache, backend)
        scores["clean"].append(score(clean_events, gold, cid, tier))

        for name in order:
            degraded = reencode(ffmpeg, media, PROFILES[name], args.media_cache)
            deg_events = _raw_events_cached(
                degraded, session, args.backend, args.events_cache, backend
            )
            scores[name].append(score(deg_events, gold, cid, tier))

    tiers = sorted({r.tier for r in scores["clean"]})
    clean_agg_vals = _tier_values(scores["clean"], "tab")["aggregate"]
    clean_agg = sum(clean_agg_vals) / len(clean_agg_vals)
    profile_aggs = {
        name: sum(v := _tier_values(scores[name], "tab")["aggregate"]) / len(v) for name in order
    }
    label, prose = classify_fork(clean_agg, profile_aggs)

    lines = ["# A8 — studio-condition degradation eval (val24)", ""]
    lines.append(
        f"Config: `{args.backend}` + `{prior_name or 'none'}` prior, splits `{args.splits}`, "
        f"{len(clips)} clips. Gold labels unchanged (audio-only degradation). "
        f"{_BASELINE_HINT} — clean rows should reproduce it."
    )
    lines.append("")
    lines.append(
        "**Diagnostic tier, NOT a gate** (roadmap A8 / D1-c): this does not touch SPEC §1.4 "
        "targets. It measures the clean-WAV-eval → real-product-capture gap to decide whether "
        "to keep tuning clean-corpus accuracy or pivot to input robustness."
    )
    lines.append("")
    lines.append("## Profiles")
    lines.append("")
    for name in order:
        lines.append(f"- **`{name}`** — {PROFILES[name].description}")
    lines.append("")
    lines.append(
        "The product records `audio/webm;codecs=opus` at 48 kHz with echoCancellation / "
        "noiseSuppression / autoGainControl **disabled** (`web-client/.../RecordPanel.tsx`), so "
        "the codec is the only guaranteed degradation; mic band + noise are environmental. "
        "Noise levels are approximate (SNR vs a ~-18 dBFS-RMS signal)."
    )
    lines.append("")

    lines.append("## Tab F1 (mean / lower-95, bootstrap N=10k)")
    lines.append("")
    lines.extend(_metric_table(scores, order, "tab", tiers))
    lines.append("")
    lines.append("## Onset F1 (detection — where codec/noise hits note recall)")
    lines.append("")
    lines.extend(_metric_table(scores, order, "onset", tiers))
    lines.append("")
    lines.append("## Pitch F1 (detection + pitch)")
    lines.append("")
    lines.extend(_metric_table(scores, order, "pitch", tiers))
    lines.append("")

    # Per-clip worst Tab F1 drops for the two realistic profiles.
    lines.append("## Worst per-clip Tab F1 drops")
    lines.append("")
    clean_by_clip = {r.clip_id: r.tab for r in scores["clean"]}
    for name in [p for p in ("laptop_mic", "noisy_room") if p in order]:
        drops = sorted(
            ((r.clip_id, r.tab - clean_by_clip[r.clip_id]) for r in scores[name]),
            key=lambda x: x[1],
        )
        worst = ", ".join(f"{c} ({d:+.3f})" for c, d in drops[:6] if d < -1e-4)
        lines.append(f"- **`{name}`**: {worst or 'no clip regressed'}")
    lines.append("")

    lines.append("## Verdict — fork")
    lines.append("")
    lines.append(f"**{label}.** {prose}")
    lines.append("")
    lines.append(
        f"(Heuristic bands on Δ aggregate Tab F1 vs clean: holds ≥ {HOLDS_THRESHOLD}, "
        f"craters ≤ {CRATERS_THRESHOLD}. Diagnostic only.)"
    )
    lines.append("")

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
