// Score-view layout (sheet mode): partition the document's time axis into
// measures and group them into systems (staff lines). Pure time math — pixel
// mapping lives in ScoreView, which stretches each system to the page width.
//
// Like the beat grid itself, this is advisory: note timestamps are never
// moved, and notes are placed proportionally inside whichever measure their
// timestamp falls in.

import { BeatGrid } from './beatGrid';

export interface ScoreMeasure {
  /** 1-based measure number, printed at system starts. */
  number: number;
  start: number;
  end: number;
}

export interface ScoreSystem {
  measures: ScoreMeasure[];
  start: number;
  end: number;
}

/** Bars per staff line. Fixed for now — proportional spacing inside the
 * measure absorbs density differences. */
export const MEASURES_PER_SYSTEM = 4;

/** Bar length used when the document has no detected beat grid. */
export const FALLBACK_BAR_S = 2;

/** Measure boundaries covering [0, duration]. With a grid, boundaries are the
 * tracked downbeats (a leading partial "measure 0→first downbeat" becomes a
 * pickup bar, and the mean bar length extends the grid to the clip end).
 * Without one, fixed FALLBACK_BAR_S bars. */
export function buildMeasures(duration: number, grid: BeatGrid | null): ScoreMeasure[] {
  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : FALLBACK_BAR_S;
  const bounds: number[] = [];

  const downbeats = grid
    ? grid.beatTimes.filter((_, i) => i % grid.beatsPerBar === 0)
    : [];

  if (downbeats.length >= 2) {
    const meanBar =
      (downbeats[downbeats.length - 1] - downbeats[0]) / (downbeats.length - 1);
    // Always start at 0. A meaningful lead-in becomes a pickup bar; a sliver
    // (< 15% of a bar) is folded into the first measure instead of becoming
    // a stretched near-empty bar of its own.
    bounds.push(0);
    bounds.push(...(downbeats[0] > meanBar * 0.15 ? downbeats : downbeats.slice(1)));
    let t = downbeats[downbeats.length - 1];
    while (t < safeDuration - 1e-6) {
      t += meanBar;
      bounds.push(t);
    }
  } else {
    for (let t = 0; t <= safeDuration + FALLBACK_BAR_S - 1e-6; t += FALLBACK_BAR_S) {
      bounds.push(t);
    }
  }

  if (bounds[0] > 1e-6) bounds.unshift(0);
  // The final boundary must reach the clip end so every note has a measure.
  if (bounds[bounds.length - 1] < safeDuration) bounds[bounds.length - 1] = safeDuration;

  const measures: ScoreMeasure[] = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    if (bounds[i + 1] - bounds[i] < 1e-6) continue;
    measures.push({ number: measures.length + 1, start: bounds[i], end: bounds[i + 1] });
  }
  return measures;
}

export function buildSystems(
  measures: ScoreMeasure[],
  perSystem = MEASURES_PER_SYSTEM
): ScoreSystem[] {
  const systems: ScoreSystem[] = [];
  for (let i = 0; i < measures.length; i += perSystem) {
    const chunk = measures.slice(i, i + perSystem);
    systems.push({
      measures: chunk,
      start: chunk[0].start,
      end: chunk[chunk.length - 1].end,
    });
  }
  return systems;
}

/** Index of the system whose time range contains `t` (clamped to the ends). */
export function systemIndexAt(systems: ScoreSystem[], t: number): number {
  if (systems.length === 0) return -1;
  for (let i = 0; i < systems.length; i++) {
    if (t < systems[i].end) return i;
  }
  return systems.length - 1;
}
