from __future__ import annotations

import json
from pathlib import Path

import pytest

from fretcam.local_eval import (
    FINGERS,
    LocalEvalError,
    add_annotation,
    add_clip,
    coverage_report,
    default_dataset_root,
    init_dataset,
    load_dataset,
    parse_finger_spec,
    run_cli,
    schema_path,
    validate_dataset,
    validate_dataset_root,
)


def _write_media(root: Path, name: str, content: bytes = b"video") -> str:
    relative = f"media/{name}.mp4"
    (root / relative).write_bytes(content)
    return relative


def _licensed_clip_args(
    *,
    clip_id: str,
    media_path: str,
    handedness: str = "right",
    framing: str = "close",
    lighting: str = "bright",
    skin_tone: str = "medium",
    guitar: str = "acoustic_steel",
    sleeve: str = "short_sleeve",
    background: str = "plain",
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "media_path": media_path,
        "duration_ms": 2_000,
        "provenance_kind": "public_licensed",
        "source_uri": f"https://example.org/{clip_id}",
        "license_name": "CC-BY-4.0",
        "rights_verified": True,
        "appearance_metadata_rights_verified": True,
        "redistribution_allowed": False,
        "handedness": handedness,
        "framing": framing,
        "lighting": lighting,
        "skin_tone": skin_tone,
        "appearance_basis": "licensed_dataset_label",
        "guitar": guitar,
        "sleeve": sleeve,
        "background": background,
        "face_visibility": "none",
        "minors_present": False,
    }


def _hovering_fingers() -> dict[str, dict[str, object]]:
    return {finger: parse_finger_spec("visible:hovering:-") for finger in FINGERS}


def test_default_storage_is_machine_local_and_git_roots_are_rejected(
    tmp_path: Path,
) -> None:
    default = default_dataset_root()
    assert default == Path.home() / ".tabvision" / "cache" / "fretcam_local_eval"

    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(LocalEvalError, match="Git repository"):
        validate_dataset_root(repository / "data" / "local-eval")

    assert validate_dataset_root(tmp_path) == tmp_path.resolve()


def test_init_is_empty_and_does_not_record_or_extract_frames(tmp_path: Path) -> None:
    manifest = init_dataset(tmp_path, dataset_id="local-eval")

    assert manifest["policy"]["automatic_recording"] is False
    assert manifest["policy"]["automatic_frame_extraction"] is False
    assert manifest["policy"]["source_scope"] == "public_or_synthetic_only"
    assert manifest["policy"]["private_or_user_media_allowed"] is False
    assert manifest["policy"]["model_training_allowed"] is False
    assert manifest["policy"]["release_gate_allowed"] is False
    assert list((tmp_path / "media").iterdir()) == []
    assert {path.name for path in tmp_path.iterdir()} == {"manifest.json", "media"}


def test_register_clip_requires_licensed_rights_and_keeps_hashed_local_media(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")
    media = _write_media(tmp_path, "session-01")
    args = _licensed_clip_args(clip_id="clip-01", media_path=media)
    args["rights_verified"] = False

    with pytest.raises(LocalEvalError, match="verified use rights"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]

    args["rights_verified"] = True
    clip = add_clip(tmp_path, **args)  # type: ignore[arg-type]
    assert clip["media"]["path"] == "media/session-01.mp4"
    assert clip["media"]["byte_size"] == 5
    assert len(clip["media"]["sha256"]) == 64
    assert clip["provenance"]["redistribution_allowed"] is False

    (tmp_path / media).write_bytes(b"tampered")
    with pytest.raises(LocalEvalError, match="byte_size does not match"):
        load_dataset(tmp_path)


def test_paths_unlicensed_appearance_metadata_and_minors_are_rejected(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")
    media = _write_media(tmp_path, "session-01")

    args = _licensed_clip_args(clip_id="clip-01", media_path=media)
    args["appearance_metadata_rights_verified"] = False
    with pytest.raises(LocalEvalError, match="labeling rights"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]

    args = _licensed_clip_args(clip_id="clip-01", media_path="../outside.mp4")
    with pytest.raises(LocalEvalError, match="stay inside"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]

    args = _licensed_clip_args(clip_id="clip-01", media_path=media)
    args["minors_present"] = True
    with pytest.raises(LocalEvalError, match="minors"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]


def test_private_and_user_provenance_are_unconditionally_rejected(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")
    media = _write_media(tmp_path, "public-01")
    args = _licensed_clip_args(clip_id="clip-01", media_path=media)
    args["provenance_kind"] = "self_recorded"
    with pytest.raises(LocalEvalError, match="public_licensed, synthetic"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]

    args["provenance_kind"] = "third_party_consented"
    with pytest.raises(LocalEvalError, match="public_licensed, synthetic"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]


def test_synthetic_provenance_requires_specification_basis(tmp_path: Path) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")
    media = _write_media(tmp_path, "synthetic-01")
    args = _licensed_clip_args(clip_id="clip-01", media_path=media)
    args.update(
        {
            "provenance_kind": "synthetic",
            "source_uri": "https://example.org/reproducible-recipe",
            "license_name": "CC0-1.0",
        }
    )
    with pytest.raises(LocalEvalError, match="synthetic appearance metadata"):
        add_clip(tmp_path, **args)  # type: ignore[arg-type]

    args["appearance_basis"] = "synthetic_specification"
    clip = add_clip(tmp_path, **args)  # type: ignore[arg-type]
    assert clip["provenance"]["kind"] == "synthetic"


def test_finger_annotation_separates_visibility_pressing_and_contacts(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")
    media = _write_media(tmp_path, "session-01")
    add_clip(
        tmp_path,
        **_licensed_clip_args(clip_id="clip-01", media_path=media),  # type: ignore[arg-type]
    )

    fingers = _hovering_fingers()
    fingers["index"] = parse_finger_spec("visible:pressing:1@1,2@1,3@1")
    annotation = add_annotation(
        tmp_path,
        clip_id="clip-01",
        annotation_id="frame-001",
        timestamp_ms=500,
        phase="stable",
        position=1,
        technique="barre",
        fingers=fingers,
    )
    assert annotation["fingers"]["index"]["pressing"] is True
    assert annotation["fingers"]["index"]["contacts"] == [
        {"string": 1, "fret": 1},
        {"string": 2, "fret": 1},
        {"string": 3, "fret": 1},
    ]
    assert annotation["fingers"]["middle"]["pressing"] is False

    with pytest.raises(LocalEvalError, match="pressing=true"):
        parse_finger_spec("visible:pressing:-")
    with pytest.raises(LocalEvalError, match="must be null"):
        parse_finger_spec("occluded:hovering:-")


def test_coverage_gate_requires_all_positions_techniques_and_diversity(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="coverage-set")
    techniques = ("barre", "note", "chord", "stretch", "slide")
    lights = ("bright", "dim", "warm", "cool", "uneven")
    skin_tones = ("light", "medium", "deep")
    pressing_fingers = ("index", "middle", "ring", "pinky")

    for position in range(1, 13):
        clip_id = f"clip-{position:02d}"
        media = _write_media(
            tmp_path,
            clip_id,
            content=f"video-{position}".encode(),
        )
        add_clip(
            tmp_path,
            **_licensed_clip_args(  # type: ignore[arg-type]
                clip_id=clip_id,
                media_path=media,
                handedness="left" if position % 2 == 0 else "right",
                framing="full_neck" if position % 2 == 0 else "close",
                lighting=lights[(position - 1) % len(lights)],
                skin_tone=skin_tones[(position - 1) % len(skin_tones)],
                guitar=("electric_solid" if position % 2 == 0 else "acoustic_steel"),
                sleeve="long_sleeve" if position % 2 == 0 else "bare_arm",
                background="cluttered" if position % 2 == 0 else "plain",
            ),
        )
        technique = techniques[(position - 1) % len(techniques)]
        fingers = _hovering_fingers()
        pressing = pressing_fingers[(position - 1) % len(pressing_fingers)]
        contact = f"1@{position}"
        if technique == "barre":
            contact = f"1@{position},2@{position}"
        fingers[pressing] = parse_finger_spec(f"visible:pressing:{contact}")
        add_annotation(
            tmp_path,
            clip_id=clip_id,
            annotation_id=f"frame-{position:02d}",
            timestamp_ms=500,
            phase="stable",
            position=position,
            technique=technique,
            fingers=fingers,
        )

    report = validate_dataset(tmp_path)
    assert report["complete"] is True
    assert report["observed"]["positions"] == list(range(1, 13))
    assert set(report["observed"]["techniques"]) == {
        "note",
        "chord",
        "barre",
        "stretch",
        "slide",
    }
    assert report["diversity"]["skin_tone"] == {"observed": 3, "required": 3}


def test_schema_only_validation_supports_incremental_annotation(
    tmp_path: Path,
) -> None:
    init_dataset(tmp_path, dataset_id="local-eval")

    report = validate_dataset(tmp_path, require_coverage=False)
    assert report["complete"] is False
    assert "position:1" in report["missing"]
    with pytest.raises(LocalEvalError, match="coverage is incomplete"):
        validate_dataset(tmp_path)


def test_cli_lifecycle_and_packaged_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert json.loads(schema_path().read_text(encoding="utf-8"))["title"].startswith(
        "FretCam"
    )
    assert run_cli(["--root", str(tmp_path), "init", "--dataset-id", "cli-eval"]) == 0
    _write_media(tmp_path, "cli-clip")
    assert (
        run_cli(
            [
                "--root",
                str(tmp_path),
                "add-clip",
                "--clip-id",
                "clip-01",
                "--media",
                "media/cli-clip.mp4",
                "--duration-ms",
                "2000",
                "--provenance",
                "public_licensed",
                "--source-uri",
                "https://example.org/cli-clip",
                "--license",
                "CC-BY-4.0",
                "--confirm-rights",
                "--handedness",
                "right",
                "--framing",
                "close",
                "--lighting",
                "bright",
                "--guitar",
                "acoustic_steel",
                "--sleeve",
                "short_sleeve",
                "--background",
                "plain",
            ]
        )
        == 0
    )
    finger = "visible:hovering:-"
    assert (
        run_cli(
            [
                "--root",
                str(tmp_path),
                "add-annotation",
                "--clip-id",
                "clip-01",
                "--annotation-id",
                "frame-01",
                "--timestamp-ms",
                "500",
                "--phase",
                "stable",
                "--position",
                "2",
                "--technique",
                "note",
                "--thumb",
                finger,
                "--index",
                "visible:pressing:2@2",
                "--middle",
                finger,
                "--ring",
                finger,
                "--pinky",
                finger,
            ]
        )
        == 0
    )
    assert run_cli(["--root", str(tmp_path), "validate", "--schema-only"]) == 0
    output = capsys.readouterr().out
    assert '"automatic_recording": false' in output
    assert '"valid": true' in output


def test_coverage_report_does_not_treat_unknown_skin_tone_as_diversity(
    tmp_path: Path,
) -> None:
    manifest = init_dataset(tmp_path, dataset_id="local-eval")
    assert coverage_report(manifest)["diversity"]["skin_tone"]["observed"] == 0
