"""CLI parser smoke for the transcribe-subcommand flags.

Covers ``--fusion-lambda-vision``, ``--video`` / ``--no-video``,
``--video-backend``, ``--video-stride``, and the video-flag resolution
that makes audio-only the default (DECISIONS.md 2026-07-28).
"""

from __future__ import annotations

import pytest

from tabvision.cli import _build_parser

# ---------- --fusion-lambda-vision ----------


def test_default_lambda_vision_is_one():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.fusion_lambda_vision == 1.0


def test_explicit_lambda_vision_parsed():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "2.5"])
    assert args.fusion_lambda_vision == pytest.approx(2.5)


def test_lambda_vision_zero_accepted():
    """``--fusion-lambda-vision 0`` is the audio-only ablation knob."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "0"])
    assert args.fusion_lambda_vision == 0.0


def test_lambda_vision_only_on_transcribe():
    """The ``check`` subcommand has no fusion stage, so the flag should
    not be exposed there."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--fusion-lambda-vision", "1.0"])


def test_lambda_vision_negative_rejected():
    """A negative weight flips the sign of the vision term in
    ``playability.emission_cost`` instead of disabling it (that's what 0.0
    is for) — silently wrong output, so reject it at the parser."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "-1.0"])


def test_lambda_vision_non_numeric_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "high"])


def test_lambda_vision_negative_rejected_on_diagnose():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "in.mp4", "--fusion-lambda-vision", "-0.5"])


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_lambda_vision_nonfinite_rejected(value: str):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", value])


# ---------- --video / --no-video ----------


def test_video_flags_default_off():
    """Audio-only is the shipped default (DECISIONS.md 2026-07-28)."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.video is False
    assert args.no_video is False


def test_video_flag_sets_true():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--video"])
    assert args.video is True


def test_no_video_still_accepted_for_compat():
    """Redundant since the default flip, but the desktop shell passes it."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--no-video"])
    assert args.no_video is True


def test_video_and_no_video_are_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--video", "--no-video"])


def test_video_flag_available_on_diagnose():
    parser = _build_parser()
    args = parser.parse_args(["diagnose", "in.mp4", "--video"])
    assert args.video is True


# ---------- video-flag resolution ----------


def _resolve(argv: list[str]):
    from tabvision.cli import _resolve_video_args

    return _resolve_video_args(_build_parser().parse_args(argv))


def test_resolve_video_default_is_audio_only():
    assert _resolve(["transcribe", "in.mp4"]) == (False, "legacy")


def test_resolve_video_opt_in_uses_legacy_backend():
    assert _resolve(["transcribe", "in.mp4", "--video"]) == (True, "legacy")


def test_resolve_explicit_video_backend_implies_opt_in():
    """Pre-flip ``--video-backend fretcam`` invocations keep their meaning."""
    argv = ["transcribe", "in.mp4", "--video-backend", "fretcam"]
    assert _resolve(argv) == (True, "fretcam")


def test_resolve_no_video_wins_over_explicit_backend():
    argv = ["transcribe", "in.mp4", "--no-video", "--video-backend", "fretcam"]
    assert _resolve(argv) == (False, "fretcam")


def test_resolve_warns_when_lambda_vision_is_inert(caplog):
    """A non-default weight without --video silently does nothing — say so."""
    with caplog.at_level("WARNING", logger="tabvision.cli"):
        enabled, _backend = _resolve(["transcribe", "in.mp4", "--fusion-lambda-vision", "2.5"])
    assert enabled is False
    assert any("no effect" in rec.message for rec in caplog.records)


# ---------- --video-backend ----------


def test_video_backend_defaults_to_unset_sentinel():
    """``None`` at parse time distinguishes "not given" from an explicit
    "legacy" — an explicit backend implies video opt-in, so the parser must
    keep the difference; resolution maps ``None`` → 'legacy'."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.video_backend is None


def test_fretcam_video_backend_is_explicitly_reachable():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--video-backend", "fretcam"])
    assert args.video_backend == "fretcam"


def test_video_backend_rejects_unknown_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--video-backend", "guess"])


def test_fretcam_video_backend_is_available_on_diagnose():
    parser = _build_parser()
    args = parser.parse_args(["diagnose", "in.mp4", "--video-backend", "fretcam"])
    assert args.video_backend == "fretcam"


def test_video_backend_is_not_exposed_on_check():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--video-backend", "fretcam"])


# ---------- --video-stride ----------


def test_video_stride_default_three():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.video_stride == 3


def test_video_stride_explicit_value():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--video-stride", "1"])
    assert args.video_stride == 1


def test_video_stride_only_on_transcribe():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--video-stride", "5"])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_video_stride_below_one_rejected(value):
    """``stride < 1`` would crash the pipeline mid-run; reject it up front."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--video-stride", value])


def test_video_stride_non_integer_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--video-stride", "half"])


def test_video_stride_validated_on_diagnose():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "in.mp4", "--video-stride", "0"])


# ---------- --audio-backend ----------


def test_audio_backend_default_is_auto():
    """The accepted config is the default: 'auto' resolves via
    ``audio_backend_for_session`` — clean acoustic → 'highres-ensemble'
    (promoted 2026-07-20), electric → 'highres-electric', else 'highres' —
    without a flag."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.audio_backend == "auto"


@pytest.mark.parametrize(
    "choice",
    ["basicpitch", "highres", "highres-fl", "highres-ensemble", "highres-electric", "auto"],
)
def test_audio_backend_choices_parsed(choice):
    """``highres-electric`` and ``auto`` are registered in
    ``tabvision.audio.backend`` / consumed by ``run_pipeline``'s tone-toggle
    routing (``audio_backend_for_session``, the SPEC v1 "tone toggle"
    feature), but were missing from the CLI's ``choices`` — making both
    unreachable from ``tabvision transcribe``/``diagnose``."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--audio-backend", choice])
    assert args.audio_backend == choice


def test_audio_backend_rejects_unknown_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--audio-backend", "tensorflow-magic"])


def test_audio_backend_auto_available_on_diagnose():
    parser = _build_parser()
    args = parser.parse_args(["diagnose", "in.mp4", "--audio-backend", "auto"])
    assert args.audio_backend == "auto"


# ---------- --position-prior ----------


def test_position_prior_default_auto():
    """Automatic routing applies the accepted prior only in its validated domain."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.position_prior == "auto"


def test_position_prior_none_reachable():
    """The bare decode stays reachable for ablations/evals."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--position-prior", "none"])
    assert args.position_prior == "none"


def test_position_prior_only_on_transcribe():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--position-prior", "guitarset-v1"])


# ---------- --sequence-prior ----------


def test_sequence_prior_default_auto():
    """Default-on flip (A15, 2026-07-02): 'auto' couples the sequence prior
    to --position-prior — active iff the position prior is."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.sequence_prior == "auto"


@pytest.mark.parametrize("choice", ["auto", "none", "guitarset-seq-v1"])
def test_sequence_prior_choices_parsed(choice):
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--sequence-prior", choice])
    assert args.sequence_prior == choice


def test_sequence_prior_rejects_unknown_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--sequence-prior", "dadagp-v1"])


def test_sequence_prior_only_on_transcribe():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--sequence-prior", "auto"])


# ---------- --string-evidence ----------


def test_string_evidence_default_auto():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.string_evidence == "auto"


@pytest.mark.parametrize("choice", ["auto", "none", "guitarset-timbre-v1"])
def test_string_evidence_choices_parsed(choice):
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--string-evidence", choice])
    assert args.string_evidence == choice


def test_string_evidence_only_on_transcribe():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--string-evidence", "auto"])


# ---------- --audio-filters ----------


def test_audio_filters_default_auto():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.audio_filters == "auto"


@pytest.mark.parametrize("choice", ["auto", "on", "off"])
def test_audio_filters_choices_parsed(choice):
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--audio-filters", choice])
    assert args.audio_filters == choice


def test_audio_filters_rejects_unknown_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--audio-filters", "maybe"])


def test_audio_filters_available_on_diagnose():
    parser = _build_parser()
    args = parser.parse_args(["diagnose", "in.mp4", "--audio-filters", "on"])
    assert args.audio_filters == "on"


def test_audio_filters_not_on_check():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--audio-filters", "on"])


def test_resolve_audio_filters_maps_choices():
    from tabvision.cli import _resolve_audio_filters

    assert _resolve_audio_filters("auto") is None
    assert _resolve_audio_filters("on") is True
    assert _resolve_audio_filters("off") is False
