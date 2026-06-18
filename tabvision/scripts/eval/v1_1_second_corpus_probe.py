"""v1.1 chunk-4 cached/resumable composite-eval runner for second-corpus checks.

The plain composite-eval CLI (``scripts.eval.composite_eval``) runs the full
pipeline once per clip and only writes its report at the very end. A 12-clip
Guitar-TECHS highres run exceeds the local 30-minute interactive budget, so an
interrupted run loses every transcription.

This probe wraps the same :func:`tabvision.eval.composite.run_composite_eval`
harness with an on-disk per-clip ``TabEvent`` cache. Each clip's prediction is
written to the cache as soon as it is computed, so the run is **resumable**:
re-running scores already-cached clips instantly and only transcribes the
remaining ones. The scoring, aggregation, and report formatting are unchanged --
the markdown is byte-identical to the composite CLI for the same inputs.

It is corpus-agnostic (any manifest works); the chunk-4 second-corpus gate runs
it over ``data/eval/local_gt_chords.toml`` with ``--splits train``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabvision.eval.composite import (
    Predictor,
    format_baseline_markdown,
    format_decomposition_markdown,
    make_run_pipeline_predictor,
    run_composite_eval,
)
from tabvision.types import SessionConfig, TabEvent

_DEFAULT_CACHE_DIR = Path.home() / ".tabvision/cache/v1_1_second_corpus"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    position_prior: str | None = args.position_prior
    if position_prior and position_prior.lower() == "none":
        position_prior = None

    base_predictor = make_run_pipeline_predictor(
        audio_backend_name=args.backend,
        position_prior=position_prior,
        melodic_prior_enabled=args.melodic_prior,
        video_enabled=args.enable_video,
    )
    key_fields = {
        "backend": args.backend,
        "position_prior": position_prior or "none",
        "melodic_prior": bool(args.melodic_prior),
        "video": bool(args.enable_video),
    }
    predictor = CachingPredictor(
        base_predictor,
        cache_dir=args.cache_dir,
        key_fields=key_fields,
        refresh_cache=args.refresh_cache,
    )

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    report = run_composite_eval(
        args.manifest,
        predictor=predictor,
        media_root=args.media_root,
        annotation_root=args.annotation_root,
        splits=splits,
        onset_tolerance_s=args.onset_tolerance_s,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    baseline_md = format_baseline_markdown(
        report,
        backend_label=args.backend,
        position_prior_label=position_prior or "none",
        eval_harness_sha=args.eval_harness_sha,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(baseline_md, encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")

    if args.decomposition_output:
        decomp_md = format_decomposition_markdown(report)
        args.decomposition_output.parent.mkdir(parents=True, exist_ok=True)
        args.decomposition_output.write_text(decomp_md, encoding="utf-8", newline="\n")
        print(f"wrote {args.decomposition_output}")

    print(
        f"scored {len(report.per_clip)} clips "
        f"({predictor.cache_hits} cached, {predictor.cache_misses} computed)"
    )
    return 0


class CachingPredictor:
    """Wrap a composite-eval predictor with an on-disk per-clip ``TabEvent`` cache.

    The wrapped ``base`` predictor is only invoked on a cache miss; the resulting
    events are written to ``cache_dir`` immediately, keyed by the resolved media
    path, its mtime, and the prediction settings (``key_fields``). This makes a
    long multi-clip run resumable across process restarts.
    """

    def __init__(
        self,
        base: Predictor,
        *,
        cache_dir: Path,
        key_fields: Mapping[str, Any],
        refresh_cache: bool = False,
    ) -> None:
        self._base = base
        self._cache_dir = Path(cache_dir)
        self._key_fields = dict(key_fields)
        self._refresh_cache = refresh_cache
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, media_path: Path, session: SessionConfig) -> list[TabEvent]:
        media_path = Path(media_path)
        mtime_ns = media_path.stat().st_mtime_ns
        cache_path = self._cache_path(media_path)

        if not self._refresh_cache and cache_path.exists():
            cached = _read_cache(cache_path)
            if cached is not None and _cache_matches(
                cached,
                media_path=media_path,
                mtime_ns=mtime_ns,
                key_fields=self._key_fields,
            ):
                self.cache_hits += 1
                print(f"  [cache] {media_path.name}: {len(cached['events'])} events")
                return tabevents_from_json(cached["events"])

        started = time.monotonic()
        events = list(self._base(media_path, session))
        elapsed = time.monotonic() - started
        self.cache_misses += 1
        print(f"  [run]   {media_path.name}: {len(events)} events in {elapsed:.1f}s")
        self._write_cache(cache_path, media_path, mtime_ns, events)
        return events

    def _cache_path(self, media_path: Path) -> Path:
        key = json.dumps(
            {"media": str(media_path.resolve()), **self._key_fields},
            sort_keys=True,
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{media_path.stem}.{digest}.json"

    def _write_cache(
        self,
        cache_path: Path,
        media_path: Path,
        mtime_ns: int,
        events: Sequence[TabEvent],
    ) -> None:
        payload = {
            "media_path": str(media_path.resolve()),
            "source_mtime_ns": mtime_ns,
            "key_fields": self._key_fields,
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "events": tabevents_to_json(events),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _read_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _cache_matches(
    cached: Mapping[str, Any],
    *,
    media_path: Path,
    mtime_ns: int,
    key_fields: Mapping[str, Any],
) -> bool:
    return (
        cached.get("media_path") == str(media_path.resolve())
        and cached.get("source_mtime_ns") == mtime_ns
        and cached.get("key_fields") == dict(key_fields)
        and isinstance(cached.get("events"), list)
    )


def tabevents_to_json(events: Sequence[TabEvent]) -> list[dict[str, Any]]:
    return [
        {
            "onset_s": event.onset_s,
            "duration_s": event.duration_s,
            "string_idx": event.string_idx,
            "fret": event.fret,
            "pitch_midi": event.pitch_midi,
            "confidence": event.confidence,
            "techniques": list(event.techniques),
        }
        for event in events
    ]


def tabevents_from_json(payload: Sequence[Mapping[str, Any]]) -> list[TabEvent]:
    return [
        TabEvent(
            onset_s=float(item["onset_s"]),
            duration_s=float(item["duration_s"]),
            string_idx=int(item["string_idx"]),
            fret=int(item["fret"]),
            pitch_midi=int(item["pitch_midi"]),
            confidence=float(item["confidence"]),
            techniques=tuple(str(tag) for tag in item.get("techniques", ())),
        )
        for item in payload
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--backend", default="highres", help="audio backend name")
    ap.add_argument(
        "--position-prior",
        default="none",
        help='position prior name; pass "none" to disable',
    )
    ap.add_argument("--melodic-prior", action="store_true")
    ap.add_argument("--enable-video", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--decomposition-output", type=Path, default=None)
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--media-root", type=Path, default=None)
    ap.add_argument("--annotation-root", type=Path, default=None)
    ap.add_argument("--bootstrap-n", type=int, default=10_000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--onset-tolerance-s", type=float, default=0.05)
    ap.add_argument("--eval-harness-sha", default="<unset>")
    ap.add_argument("--title", default="Composite per-tier baseline")
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh-cache", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
