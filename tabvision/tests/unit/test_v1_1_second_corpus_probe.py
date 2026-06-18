"""Tests for the v1.1 cached/resumable second-corpus composite-eval runner."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.eval.v1_1_second_corpus_probe import (
    CachingPredictor,
    tabevents_from_json,
    tabevents_to_json,
)
from tabvision.types import SessionConfig, TabEvent


def _tab(t: float, string_idx: int, fret: int, pitch: int) -> TabEvent:
    return TabEvent(
        onset_s=t,
        duration_s=0.2,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
        techniques=("slide",),
    )


def _make_media(tmp_path: Path, name: str = "clip.wav") -> Path:
    media = tmp_path / name
    media.write_bytes(b"fake-audio")
    return media


class _CountingPredictor:
    """Fake base predictor: returns a fixed event list and counts invocations."""

    def __init__(self, events: list[TabEvent]) -> None:
        self.events = events
        self.calls = 0

    def __call__(self, media_path: Path, session: SessionConfig) -> list[TabEvent]:
        self.calls += 1
        return list(self.events)


def test_tabevents_json_round_trip_preserves_all_fields() -> None:
    events = [_tab(1.0, 5, 0, 64), _tab(2.0, 4, 3, 53)]

    restored = tabevents_from_json(tabevents_to_json(events))

    assert restored == events


def test_caching_predictor_computes_once_then_reads_cache(tmp_path: Path) -> None:
    media = _make_media(tmp_path)
    base = _CountingPredictor([_tab(1.0, 5, 0, 64)])
    predictor = CachingPredictor(
        base,
        cache_dir=tmp_path / "cache",
        key_fields={"backend": "highres", "position_prior": "none"},
    )

    first = predictor(media, SessionConfig())
    second = predictor(media, SessionConfig())

    assert base.calls == 1  # second call served from disk cache
    assert first == second == base.events
    assert predictor.cache_misses == 1
    assert predictor.cache_hits == 1


def test_caching_predictor_resumes_with_a_fresh_instance(tmp_path: Path) -> None:
    media = _make_media(tmp_path)
    cache_dir = tmp_path / "cache"
    base_a = _CountingPredictor([_tab(1.0, 5, 0, 64)])
    CachingPredictor(base_a, cache_dir=cache_dir, key_fields={"backend": "highres"})(
        media, SessionConfig()
    )

    base_b = _CountingPredictor([_tab(9.0, 0, 9, 99)])  # different output if recomputed
    restored = CachingPredictor(base_b, cache_dir=cache_dir, key_fields={"backend": "highres"})(
        media, SessionConfig()
    )

    assert base_b.calls == 0  # a brand-new process reuses the on-disk cache
    assert restored == base_a.events


def test_caching_predictor_recomputes_when_media_changes(tmp_path: Path) -> None:
    media = _make_media(tmp_path)
    base = _CountingPredictor([_tab(1.0, 5, 0, 64)])
    predictor = CachingPredictor(base, cache_dir=tmp_path / "cache", key_fields={})

    predictor(media, SessionConfig())
    # Bump mtime so the cached mtime_ns no longer matches.
    stat = media.stat()
    os.utime(media, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    predictor(media, SessionConfig())

    assert base.calls == 2


def test_caching_predictor_keys_on_settings(tmp_path: Path) -> None:
    media = _make_media(tmp_path)
    base = _CountingPredictor([_tab(1.0, 5, 0, 64)])
    cache_dir = tmp_path / "cache"

    CachingPredictor(base, cache_dir=cache_dir, key_fields={"prior": "none"})(
        media, SessionConfig()
    )
    CachingPredictor(base, cache_dir=cache_dir, key_fields={"prior": "guitarset-v1"})(
        media, SessionConfig()
    )

    assert base.calls == 2  # different settings → different cache entry


def test_caching_predictor_refresh_cache_forces_recompute(tmp_path: Path) -> None:
    media = _make_media(tmp_path)
    base = _CountingPredictor([_tab(1.0, 5, 0, 64)])

    CachingPredictor(base, cache_dir=tmp_path / "cache", key_fields={})(media, SessionConfig())
    CachingPredictor(base, cache_dir=tmp_path / "cache", key_fields={}, refresh_cache=True)(
        media, SessionConfig()
    )

    assert base.calls == 2
