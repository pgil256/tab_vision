"""GAPS source-video acquisition + audio<->video offset alignment (v1.1 video chain).

GAPS ships no video. Each clip's performance lives on YouTube (``yt_id`` in the
metadata CSV); the local GAPS WAV is a *crop* of that upload, so the video's
frame time and the gold/onset time differ by an unknown constant offset. This
module supplies the net-new pieces the v1.1 video real-chain needs:

  1. **Download** each clip's source video at <=360p (single-file format 18,
     H.264 + AAC) via yt-dlp into a local cache. GAPS is non-commercial,
     offline-eval-only: media stays local and is never committed or
     redistributed.
  2. **Align** by recovering the per-clip offset: cross-correlate the
     onset-strength envelope of the GAPS WAV against the video's decoded audio.

The offset is defined so that ``video_time = gold_onset_s + offset_s``
(a positive offset means the video is delayed relative to the GAPS crop).

The cross-correlation core (:func:`onset_envelope_lag`) is a pure function over
two envelopes — unit-tested in ``tests/unit/test_gaps_video_align.py`` — so the
alignment logic is verifiable without network or media files.

CLI::

    python -m scripts.acquire.gaps_video --download --offsets \\
        --data-root ~/.tabvision/data --clips clean12
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# The 12 clean (>=80% gold coverage) standard-tuning GAPS test clips — the
# acceptance subset from docs/EVAL_REPORTS/v1_1_gaps_chunk5_2026-06-19.md §5.
CLEAN_12: tuple[str, ...] = (
    "027_Zpswc",
    "031_vpswc",
    "043_bc1wc",
    "063_bV1wc",
    "104_xf1wc",
    "118_VD1wc",
    "142_GD1wc",
    "179_pM1wc",
    "212_y41wc",
    "235_Ny1wc",
    "294_BSswc",
    "341_1M1wc",
)

DEFAULT_VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video"

# Envelope cross-correlation resolution. 16 kHz / hop 160 => 100 fps envelope
# (10 ms lag resolution), ample against the 50 ms onset gate and ~42 ms video
# frame period.
_XCORR_SR = 16_000
_XCORR_HOP = 160


@dataclass(frozen=True)
class AlignmentResult:
    """Recovered video<->audio crop offset for one clip.

    ``offset_s`` is defined so ``video_time = gold_onset_s + offset_s``.
    ``peak_ratio`` is the cross-correlation peak divided by the best value
    outside a +-5-frame guard band; values >~2 indicate a confident, sharp
    alignment. ``*_duration_s`` are corroborating sanity checks (the GAPS crop
    and the upload are typically near-equal length).
    """

    offset_s: float
    peak_ratio: float
    audio_duration_s: float
    video_duration_s: float


def onset_envelope_lag(
    video_env: np.ndarray,
    audio_env: np.ndarray,
    *,
    guard: int = 5,
) -> tuple[int, float]:
    """Lag (in envelope frames) that best aligns ``audio_env`` onto ``video_env``.

    Pure function. Returns ``(lag, peak_ratio)`` where a feature at audio frame
    ``n`` appears at video frame ``n + lag`` (so a positive lag means the video
    is delayed relative to the audio). ``peak_ratio`` is the correlation peak
    over the best competing value outside a ``+-guard`` band around the peak.
    """
    video_env = np.asarray(video_env, dtype=np.float64)
    audio_env = np.asarray(audio_env, dtype=np.float64)
    if video_env.size == 0 or audio_env.size == 0:
        raise ValueError("onset_envelope_lag requires non-empty envelopes")
    corr = np.correlate(video_env, audio_env, mode="full")
    k = int(np.argmax(corr))
    lag = k - (audio_env.size - 1)
    lo, hi = max(0, k - guard), min(corr.size, k + guard + 1)
    mask = np.ones(corr.size, dtype=bool)
    mask[lo:hi] = False
    next_best = float(np.abs(corr[mask]).max()) if mask.any() else 0.0
    peak_ratio = float(corr[k]) / (next_best + 1e-9)
    return lag, peak_ratio


def _decode_mono(path: Path, sr: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 at ``sr`` via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)


def _onset_envelope(wav: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Normalized spectral-flux onset-strength envelope (zero-mean, unit-std)."""
    import librosa

    env = librosa.onset.onset_strength(y=wav.astype(np.float32), sr=sr, hop_length=hop)
    return (env - env.mean()) / (env.std() + 1e-9)


def estimate_offset(
    audio_wav_path: str | Path,
    video_path: str | Path,
    *,
    sr: int = _XCORR_SR,
    hop: int = _XCORR_HOP,
) -> AlignmentResult:
    """Recover the video<->audio crop offset by onset-envelope cross-correlation.

    Decodes both media to mono at ``sr``, computes onset-strength envelopes, and
    cross-correlates. Returns an :class:`AlignmentResult` with the offset such
    that ``video_time = gold_onset_s + offset_s``.
    """
    audio = _decode_mono(Path(audio_wav_path), sr)
    video = _decode_mono(Path(video_path), sr)
    a_env = _onset_envelope(audio, sr, hop)
    v_env = _onset_envelope(video, sr, hop)
    lag, peak_ratio = onset_envelope_lag(v_env, a_env)
    return AlignmentResult(
        offset_s=lag * hop / sr,
        peak_ratio=peak_ratio,
        audio_duration_s=len(audio) / sr,
        video_duration_s=len(video) / sr,
    )


def read_yt_ids(csv_path: str | Path, stems: tuple[str, ...] | None = None) -> dict[str, str]:
    """Map ``stem -> yt_id`` from the GAPS metadata CSV (optionally filtered)."""
    wanted = set(stems) if stems is not None else None
    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row["id"]
            if wanted is not None and stem not in wanted:
                continue
            yt = (row.get("yt_id") or "").strip()
            if yt:
                out[stem] = yt
    return out


def download_video(
    yt_id: str,
    dst: str | Path,
    *,
    max_height: int = 360,
    ffmpeg_location: str | Path | None = None,
) -> Path:
    """Download a YouTube video at <=``max_height`` to ``dst`` (idempotent)."""
    import yt_dlp

    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"18/bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
    opts: dict = {
        "format": fmt,
        "outtmpl": str(dst),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if ffmpeg_location is not None:
        opts["ffmpeg_location"] = str(ffmpeg_location)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
    if not (dst.exists() and dst.stat().st_size > 0):
        raise RuntimeError(f"download produced no file for yt_id={yt_id}")
    return dst


def _resolve_stems(spec: str) -> tuple[str, ...]:
    if spec == "clean12":
        return CLEAN_12
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    ap.add_argument("--ffmpeg-location", type=Path, default=None)
    ap.add_argument("--download", action="store_true", help="download missing videos")
    ap.add_argument("--offsets", action="store_true", help="estimate + print per-clip offsets")
    ap.add_argument("--offsets-json", type=Path, default=None, help="write offsets to JSON")
    args = ap.parse_args(argv)

    gaps = args.data_root / "gaps"
    stems = _resolve_stems(args.clips)
    yt = read_yt_ids(gaps / "gaps_metadata_with_splits.csv", stems)

    if args.download:
        for stem in stems:
            dst = args.cache_dir / f"{stem}.mp4"
            status = "skip" if dst.exists() else "get"
            print(f"[{status}] {stem}  yt={yt.get(stem, '?')}")
            if status == "get":
                download_video(yt[stem], dst, ffmpeg_location=args.ffmpeg_location)

    offsets: dict[str, dict] = {}
    if args.offsets:
        print(f"\n{'stem':<12}{'offset_s':>10}{'peak_ratio':>12}{'wav_dur':>10}{'vid_dur':>10}")
        for stem in stems:
            wav = gaps / "audio" / f"{stem}.wav"
            vid = args.cache_dir / f"{stem}.mp4"
            if not (wav.exists() and vid.exists()):
                print(f"{stem:<12}  (missing media)")
                continue
            res = estimate_offset(wav, vid)
            offsets[stem] = asdict(res)
            print(
                f"{stem:<12}{res.offset_s:>+10.3f}{res.peak_ratio:>12.2f}"
                f"{res.audio_duration_s:>10.2f}{res.video_duration_s:>10.2f}"
            )
        if args.offsets_json is not None:
            args.offsets_json.write_text(json.dumps(offsets, indent=2), encoding="utf-8")
            print(f"\nwrote {args.offsets_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
