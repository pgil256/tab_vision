"""A8 studio-degradation harness — pure-function contracts.

These pin the ffmpeg-arg construction, the re-encode cache-key behaviour, and
the fork-classification thresholds without touching ffmpeg or the highres
backend (the eval run itself is the integration test).
"""

from __future__ import annotations

from pathlib import Path

from scripts.eval.a8_studio_degradation import (
    CRATERS_THRESHOLD,
    DEFAULT_PROFILE_ORDER,
    HOLDS_THRESHOLD,
    PROFILES,
    build_ffmpeg_args,
    classify_fork,
    reencode_cache_path,
)

# --- profile registry -----------------------------------------------------------


def test_default_order_matches_registry() -> None:
    assert set(DEFAULT_PROFILE_ORDER) <= set(PROFILES)
    # The codec-only floor and the realistic condition the verdict reads must exist.
    assert "opus_128" in PROFILES
    assert "laptop_mic" in PROFILES


# --- build_ffmpeg_args -----------------------------------------------------------


def test_codec_only_profile_uses_simple_af_chain() -> None:
    """opus_* profiles have no mic model: a plain -af chain, no noise mix."""
    args = build_ffmpeg_args("ffmpeg", Path("in.wav"), Path("out.webm"), PROFILES["opus_128"])
    assert "-af" in args
    assert "-filter_complex" not in args
    assert "anoisesrc" not in " ".join(args)
    # Codec-only means literally only aresample — no highpass/lowpass/compressor.
    af = args[args.index("-af") + 1]
    assert af == "aresample=48000"
    # 48 kHz mono Opus at the profile bitrate.
    assert args[-9:] == ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "128k", "out.webm"]


def test_noise_profile_uses_filter_complex_with_unnormalized_amix() -> None:
    """laptop_mic mixes an anoisesrc floor via filter_complex.

    ``normalize=0`` is load-bearing: without it amix would halve every input,
    burying the intended noise level (and attenuating the signal).
    """
    args = build_ffmpeg_args("ffmpeg", Path("in.wav"), Path("out.webm"), PROFILES["laptop_mic"])
    assert "-filter_complex" in args
    assert "-af" not in args
    fc = args[args.index("-filter_complex") + 1]
    assert "anoisesrc=color=pink" in fc
    assert "amix=inputs=2:duration=first:normalize=0" in fc
    assert "highpass=f=70" in fc
    assert "lowpass=f=8000" in fc
    assert "seed=42" in fc  # deterministic noise realization
    assert args[args.index("-map") + 1] == "[m]"
    assert "96k" in args


def test_noisy_room_includes_compression_and_low_bitrate() -> None:
    args = build_ffmpeg_args("ffmpeg", Path("in.wav"), Path("out.webm"), PROFILES["noisy_room"])
    fc = args[args.index("-filter_complex") + 1]
    assert "acompressor=" in fc
    assert "highpass=f=90" in fc
    assert "lowpass=f=7000" in fc
    assert "64k" in args


def test_ffmpeg_binary_and_paths_are_threaded_through() -> None:
    args = build_ffmpeg_args(
        "/opt/ffmpeg", Path("/data/a.wav"), Path("/tmp/a.webm"), PROFILES["opus_64"]
    )
    assert args[0] == "/opt/ffmpeg"
    assert args[args.index("-i") + 1] == str(Path("/data/a.wav"))
    assert args[-1] == str(Path("/tmp/a.webm"))


# --- reencode_cache_path ---------------------------------------------------------


def test_cache_path_is_deterministic_and_profile_scoped(tmp_path: Path) -> None:
    src = tmp_path / "clip.wav"
    src.write_bytes(b"pretend-audio")
    out_dir = tmp_path / "media"

    p1 = reencode_cache_path(src, PROFILES["opus_128"], out_dir)
    p2 = reencode_cache_path(src, PROFILES["opus_128"], out_dir)
    p_other = reencode_cache_path(src, PROFILES["laptop_mic"], out_dir)

    assert p1 == p2  # deterministic for the same (src, mtime, profile)
    assert p1 != p_other  # different profile → different file
    assert p1.name.startswith("clip.opus_128.")
    assert p1.suffix == ".webm"
    assert p_other.name.startswith("clip.laptop_mic.")


def test_cache_path_invalidates_on_source_mtime_change(tmp_path: Path) -> None:
    src = tmp_path / "clip.wav"
    src.write_bytes(b"v1")
    before = reencode_cache_path(src, PROFILES["opus_128"], tmp_path)
    # Bump mtime explicitly (write alone may land in the same ns tick on Windows).
    import os

    os.utime(src, ns=(10**18, 10**18))
    after = reencode_cache_path(src, PROFILES["opus_128"], tmp_path)
    assert before != after


# --- classify_fork ---------------------------------------------------------------


def test_fork_robust_when_everything_holds() -> None:
    label, _ = classify_fork(0.50, {"opus_128": 0.49, "opus_64": 0.49, "laptop_mic": 0.48})
    assert label == "robust → keep tuning"


def test_fork_environment_dominated_when_codec_holds_but_mic_craters() -> None:
    label, prose = classify_fork(0.50, {"opus_128": 0.49, "laptop_mic": 0.40})
    assert label.startswith("environment-dominated")
    assert "input robustness" in prose


def test_fork_codec_dominated_when_even_the_floor_craters() -> None:
    label, _ = classify_fork(0.50, {"opus_128": 0.40, "laptop_mic": 0.35})
    assert label == "codec-dominated"


def test_fork_mild_when_realistic_only_softly_degrades() -> None:
    label, _ = classify_fork(0.50, {"opus_128": 0.49, "laptop_mic": 0.45})
    assert label.startswith("mild degradation")


def test_fork_inconclusive_without_profiles() -> None:
    label, _ = classify_fork(0.50, {})
    assert label == "inconclusive"


def test_fork_thresholds_are_ordered() -> None:
    # A "craters" cut must be strictly worse than a "holds" cut, else the
    # soft band vanishes and the classification is ill-defined.
    assert CRATERS_THRESHOLD < HOLDS_THRESHOLD < 0
