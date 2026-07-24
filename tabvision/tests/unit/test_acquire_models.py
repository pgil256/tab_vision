from __future__ import annotations

from pathlib import Path

from scripts.acquire import models


def test_yolo_checkpoint_path_defaults_under_data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(models.YOLO_CHECKPOINT_ENV, raising=False)

    assert models.yolo_checkpoint_path(tmp_path) == (
        tmp_path / "models" / models.YOLO_CHECKPOINT_NAME
    )


def test_yolo_checkpoint_path_honors_env(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "custom.pt"
    monkeypatch.setenv(models.YOLO_CHECKPOINT_ENV, str(custom))

    assert models.yolo_checkpoint_path(tmp_path) == custom


def test_collect_status_reports_missing_yolo_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(models.YOLO_CHECKPOINT_ENV, raising=False)
    monkeypatch.setenv(models.HAND_MODEL_ENV, str(tmp_path / "missing-hand.task"))

    statuses = {item.name: item for item in models.collect_status(tmp_path)}

    yolo = statuses["yolo-obb checkpoint"]
    assert yolo.status == "missing"
    assert str(tmp_path / "models" / models.YOLO_CHECKPOINT_NAME) == yolo.detail
    assert models.YOLO_CHECKPOINT_ENV in yolo.action


def test_hand_model_path_and_status_honor_env(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "hand_landmarker.task"
    monkeypatch.setenv(models.HAND_MODEL_ENV, str(custom))

    assert models.hand_model_path() == custom
    statuses = {item.name: item for item in models.collect_status(tmp_path)}
    hand = statuses["mediapipe hand-landmarker model"]
    assert hand.status == "missing"
    assert hand.detail == str(custom)
    assert models.HAND_MODEL_ENV in hand.action

    custom.write_bytes(b"model")
    statuses = {item.name: item for item in models.collect_status(tmp_path)}
    assert statuses["mediapipe hand-landmarker model"].status == "ready"


def test_prepare_hand_dir_prints_asset_action(tmp_path: Path, monkeypatch, capsys) -> None:
    target = tmp_path / "models" / models.HAND_MODEL_NAME
    monkeypatch.setenv(models.HAND_MODEL_ENV, str(target))

    assert models.main(["prepare-hand-dir"]) == 0

    out = capsys.readouterr().out
    assert str(target) in out
    assert models.HAND_MODEL_URL in out
    assert target.parent.is_dir()


def test_list_command_prints_supported_groups(capsys) -> None:
    assert models.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "basic-pitch audio baseline" in out
    assert "YOLO-OBB checkpoint" in out
    assert "MediaPipe hand-landmarker model" in out
