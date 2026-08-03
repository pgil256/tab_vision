// Advisory beat-grid selector + quantization grid helpers.
//
// The grid drives measure/beat lines on the canvas and export-time snapping
// only. Note timestamps in the document are never moved by it.

import { TabDocument } from '../types/tab';

export interface BeatGrid {
  tempoBpm: number;
  beatTimes: number[];
  beatsPerBar: number;
}

/** Returns the document's beat grid, or null for documents without one
 * (detection failed server-side, or the doc predates the feature). */
export function getBeatGrid(doc: TabDocument | null): BeatGrid | null {
  const meta = doc?.metadata;
  if (
    !meta ||
    typeof meta.tempoBpm !== 'number' ||
    !Array.isArray(meta.beatTimes) ||
    meta.beatTimes.length < 2
  ) {
    return null;
  }
  return {
    tempoBpm: meta.tempoBpm,
    beatTimes: meta.beatTimes,
    beatsPerBar: meta.beatsPerBar && meta.beatsPerBar > 0 ? meta.beatsPerBar : 4,
  };
}

/** All grid points (seconds) at the given subdivision: `subdivision / 4`
 * points per tracked beat pair, extended past both ends at the mean beat
 * interval so early/late notes still find a nearby point. */
export function buildQuantizeGrid(beatTimes: number[], subdivision = 16): number[] {
  if (beatTimes.length < 2) return [...beatTimes];
  const subsPerBeat = Math.max(1, Math.floor(subdivision / 4));
  const points: number[] = [];
  for (let i = 0; i < beatTimes.length - 1; i++) {
    const start = beatTimes[i];
    const end = beatTimes[i + 1];
    for (let j = 0; j < subsPerBeat; j++) {
      points.push(start + ((end - start) * j) / subsPerBeat);
    }
  }
  points.push(beatTimes[beatTimes.length - 1]);

  const meanInterval =
    (beatTimes[beatTimes.length - 1] - beatTimes[0]) / (beatTimes.length - 1);
  const subInterval = meanInterval / subsPerBeat;
  // Extend to t=0 before the first beat and one bar past the last.
  for (let t = beatTimes[0] - subInterval; t >= 0; t -= subInterval) {
    points.unshift(t);
  }
  const tail = beatTimes[beatTimes.length - 1] + meanInterval * 4;
  for (let t = beatTimes[beatTimes.length - 1] + subInterval; t <= tail; t += subInterval) {
    points.push(t);
  }
  return points;
}

/** Nearest grid point to `t` (grid must be sorted ascending). */
export function snapToGrid(t: number, grid: number[]): number {
  if (grid.length === 0) return t;
  let lo = 0;
  let hi = grid.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (grid[mid] <= t) lo = mid;
    else hi = mid;
  }
  return Math.abs(grid[lo] - t) <= Math.abs(grid[hi] - t) ? grid[lo] : grid[hi];
}
