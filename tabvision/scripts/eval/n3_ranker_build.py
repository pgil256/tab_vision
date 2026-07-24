"""Accuracy-loop N3 build — physics features in a self-contained review ranker.

The entry probe showed the physics "doubt" score is a strong wrong-position
detector (AUC 0.75 isolated). The intended build was to add it to the frozen
Phase 6 ranker and beat its 38.76% wrong-reduction @60 s. **That exact
comparison is blocked**: the Phase 6 feature cache is keyed to
``PHASE1_NOTES``, a re-decode stage not on disk and not cheaply regenerable,
and three of its ten features come from the Phase 4 timbre model. Its row
order does not match the phase0 lattice (event-id hash differs), so the cache
cannot be aligned to fresh physics measurements.

So this measures the **marginal value of physics** instead, honestly scoped:
a self-contained ranker over the phase0 dev-OOF ambiguous rows, trained and
replayed with the Phase 6 protocol (player-held nested OOF, Platt calibration,
2 s/note review, gold-in-top-3 correctable, wrong-reduction at 10/30/60 s per
clip), comparing:

- **decoder** — features derivable from the lattice alone (path margin,
  candidate count, cluster size, mode).
- **+physics** — the same plus the measured physics doubt score, fit r², and a
  fired indicator.

This is NOT the 38.76% comparison (that baseline had timbre features and a
different row set). It answers whether physics is worth adding to a
decoder-feature ranker; the delta between the two arms is the physics
contribution, reported as the assisted metric, separate from automatic Tab F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from scripts.eval.n2_muscriptor_merge import DEV_PLAYERS, _event_from_json
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.eval.review_queue import binary_roc_auc, fit_platt
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import (
    _harmonic_frequencies,
    _isolated_flags,
    _overlapping,
    estimate_inharmonicity,
    inharmonicity_matrix,
)
from tabvision.fusion.string_physics import reference_stiffness_model
from tabvision.types import GuitarConfig

SEED = 6621
TRAIN_EPOCHS = 40
BATCH_SIZE = 512
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4
ACTION_SECONDS = 2
REPLAY_BUDGETS_S = (10, 30, 60)
MIN_R2 = 0.50
SKIP_ATTACK_S = 0.030
MIN_WINDOW_S = 0.120
MAX_WINDOW_S = 0.400
SEPARATION_FACTOR = 3.0


@dataclass
class Row:
    event_id: str
    track_id: str
    player: str
    mode: str
    event_index: int
    onset_ms: int
    pitch: int
    predicted_string: int
    candidates: tuple[tuple[int, int, float], ...]  # (string, fret, cost)
    reference_in_top3: bool
    wrong: bool
    cluster_size: int


def _parse_candidates(value: str) -> tuple[tuple[int, int, float], ...]:
    out = []
    for item in value.split(";"):
        if not item:
            continue
        s, f, c = item.split(":")
        out.append((int(s), int(f), float(c)))
    return tuple(out)


def load_rows(csv_path: Path) -> list[Row]:
    cluster_counts: dict[tuple[str, str], int] = defaultdict(int)
    raw: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["condition"] != "production_equivalent":
                continue
            if r["evaluation_split"] != "development_oof":
                continue
            if r["ambiguous_pitch_match"] != "1" or int(r["candidate_count"]) < 2:
                continue
            if not r["reference_string"]:
                continue
            cluster_counts[(r["track_id"], r["cluster_index"])] += 1
            raw.append(r)
    rows: list[Row] = []
    for r in raw:
        cands = _parse_candidates(r["candidate_path"])
        ref = (int(r["reference_string"]), int(r["reference_fret"]))
        rows.append(
            Row(
                event_id=r["event_id"],
                track_id=r["track_id"],
                player=r["player"],
                mode=r["mode"],
                event_index=int(r["event_index"]),
                onset_ms=int(round(float(r["onset_s"]) * 1000)),
                pitch=int(r["pitch_midi"]),
                predicted_string=int(r["predicted_string"]),
                candidates=cands,
                reference_in_top3=ref in {(s, f) for s, f, _ in cands[:3]},
                wrong=r["reference_rank"] != "1",
                cluster_size=cluster_counts[(r["track_id"], r["cluster_index"])],
            )
        )
    return rows


def measure_physics(
    rows: list[Row], data_home: Path, cache_dir: Path
) -> dict[str, dict[str, float]]:
    """Physics doubt score / r2 / fired per event_id, from GuitarSet audio."""
    cfg = GuitarConfig()
    model = reference_stiffness_model()
    by_track: dict[str, dict[tuple[int, int], Row]] = defaultdict(dict)
    for row in rows:
        by_track[row.track_id][(row.onset_ms, row.pitch)] = row

    out: dict[str, dict[str, float]] = {}
    for track_id, notes in by_track.items():
        cache = cache_dir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            continue
        events = sorted(
            (_event_from_json(x) for x in json.loads(cache.read_text("utf-8"))),
            key=lambda e: e.onset_s,
        )
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        audio = np.asarray(wav, dtype=np.float64)
        isolated = _isolated_flags(events)
        for index, (event, is_iso) in enumerate(zip(events, isolated, strict=True)):
            row = notes.get((int(round(event.onset_s * 1000)), event.pitch_midi))
            if row is None:
                continue
            duration = event.offset_s - event.onset_s
            if duration < MIN_WINDOW_S + SKIP_ATTACK_S:
                continue
            start = int((event.onset_s + SKIP_ATTACK_S) * sr)
            stop = start + int(min(MAX_WINDOW_S, duration - SKIP_ATTACK_S) * sr)
            if start < 0 or stop > audio.size:
                continue
            blocked: list[float] = []
            separation = 0.0
            if not is_iso:
                separation = SEPARATION_FACTOR / max((stop - start) / sr, 1e-6)
                for other in _overlapping(
                    events, index, event.onset_s + SKIP_ATTACK_S, event.onset_s + duration
                ):
                    blocked.extend(_harmonic_frequencies(other.pitch_midi))
            nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
            fit = estimate_inharmonicity(
                audio[start:stop],
                int(sr),
                nominal,
                blocked_hz=blocked,
                min_separation_hz=separation,
            )
            if fit is None or fit.r2 < MIN_R2:
                continue
            matrix = inharmonicity_matrix(event.pitch_midi, cfg, fit.log_b, model)
            if matrix is None:
                continue
            prob_dec = 0.0
            for c in candidate_positions(event.pitch_midi, cfg):
                if c.string_idx == row.predicted_string:
                    prob_dec = float(matrix[c.string_idx, c.fret])
                    break
            out[row.event_id] = {"prob_decoder": prob_dec, "r2": fit.r2, "fired": 1.0}
    return out


def build_features(
    rows: list[Row], physics: dict[str, dict[str, float]], with_physics: bool
) -> np.ndarray:
    cols = 4 + (3 if with_physics else 0)
    feats = np.zeros((len(rows), cols), dtype=np.float64)
    for i, row in enumerate(rows):
        margin = row.candidates[1][2] if len(row.candidates) > 1 else 50.0
        feats[i, 0] = math.log1p(min(margin, 50.0)) / math.log1p(50.0)
        feats[i, 1] = len(row.candidates) / 6.0
        feats[i, 2] = min(row.cluster_size, 6) / 6.0
        feats[i, 3] = float(row.mode == "comp")
        if with_physics:
            p = physics.get(row.event_id)
            if p is not None:
                # 1 - prob(decoder's string): high = physics doubts the decoder.
                feats[i, 4] = 1.0 - p["prob_decoder"]
                feats[i, 5] = p["r2"]
                feats[i, 6] = 1.0
    return feats


class Net(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[Net, np.ndarray, np.ndarray]:
    _seed(seed)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    std = ((features - mean) / scale).astype(np.float32)
    net = Net(features.shape[1])
    wrong = int(labels.sum())
    pw = torch.as_tensor((len(labels) - wrong) / max(wrong, 1), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tf = torch.from_numpy(std)
    tl = torch.from_numpy(labels.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(TRAIN_EPOCHS):
        order = rng.permutation(len(labels))
        net.train()
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            loss = functional.binary_cross_entropy_with_logits(net(tf[idx]), tl[idx], pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return net, mean, scale


def _logits(net: Net, features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    std = ((features - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        return net(torch.from_numpy(std)).numpy()


def run_oof(features: np.ndarray, rows: list[Row]) -> np.ndarray:
    players = np.asarray([r.player for r in rows])
    labels = np.asarray([r.wrong for r in rows], dtype=bool)
    risk = np.zeros(len(rows), dtype=np.float64)
    for oi, held in enumerate(DEV_PLAYERS):
        held_idx = np.flatnonzero(players == held)
        train_idx = np.flatnonzero(players != held)
        train_players = players[train_idx]
        inner = np.zeros(len(train_idx), dtype=np.float64)
        for ii, ip in enumerate(p for p in DEV_PLAYERS if p != held):
            val = np.flatnonzero(train_players == ip)
            tr = np.flatnonzero(train_players != ip)
            net, mean, scale = fit(
                features[train_idx[tr]], labels[train_idx[tr]], SEED + oi * 100 + ii
            )
            inner[val] = _logits(net, features[train_idx[val]], mean, scale)
        cal = fit_platt(inner, labels[train_idx])
        net, mean, scale = fit(features[train_idx], labels[train_idx], SEED + oi * 100 + 50)
        risk[held_idx] = cal.probabilities(_logits(net, features[held_idx], mean, scale))
    return risk


def replay(rows: list[Row], risk: np.ndarray) -> dict[int, float]:
    by_track: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[r.track_id].append(i)
    out: dict[int, float] = {}
    for budget in REPLAY_BUDGETS_S:
        total_wrong = corrected = 0
        for indices in by_track.values():
            ordered = sorted(indices, key=lambda i: (-risk[i], rows[i].event_index))
            selected = ordered[: budget // ACTION_SECONDS]
            for i in selected:
                if rows[i].wrong and rows[i].reference_in_top3:
                    corrected += 1
            total_wrong += sum(rows[i].wrong for i in indices)
        out[budget] = corrected / total_wrong if total_wrong else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_full_dev_cache")

    rows = load_rows(args.lattice)
    labels = np.asarray([r.wrong for r in rows], dtype=bool)
    print(f"rows={len(rows)} wrong_rate={labels.mean():.4f}", flush=True)
    physics = measure_physics(rows, data_home, cache_dir)
    fired = sum(1 for r in rows if r.event_id in physics)
    print(f"physics fired on {fired}/{len(rows)} ({fired / len(rows):.1%})", flush=True)

    summary: dict[str, Any] = {
        "rows": len(rows),
        "wrong_rate": float(labels.mean()),
        "physics_fired": fired,
        "arms": {},
    }
    for name, with_phys in (("decoder", False), ("physics", True)):
        feats = build_features(rows, physics, with_phys)
        risk = run_oof(feats, rows)
        auc = binary_roc_auc(risk, labels)
        red = replay(rows, risk)
        summary["arms"][name] = {"auc": auc, "reduction": {str(k): v for k, v in red.items()}}
        print(
            f"  {name:>8}: AUC={auc:.4f} reduction@60s={red[60]:.4f} "
            f"(@10s {red[10]:.4f}, @30s {red[30]:.4f})",
            flush=True,
        )

    d = summary["arms"]["decoder"]["reduction"]["60"]
    p = summary["arms"]["physics"]["reduction"]["60"]
    summary["reduction_delta_60s"] = p - d
    print(f"\nwrong-reduction @60s: decoder {d:.4f} -> +physics {p:.4f} (delta {p - d:+.4f})")
    print("note: NOT the shipped 38.76% (that used 10 features incl. timbre + PHASE1 rows)")
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
