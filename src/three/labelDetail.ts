/**
 * Level of detail of the layer titles in a crowded view: when neighbouring
 * layers come too close on screen (a deep network in a narrow view), the
 * titles first drop their details ("12 · ReLU"), then shorten to the layer
 * prefix ("H1"), so they never pile up on top of each other.
 */

/** 0 = name and details, 1 = name only, 2 = short name. */
export type LabelDetail = 0 | 1 | 2;

export const LABEL_DETAILS: readonly LabelDetail[] = [0, 1, 2];

/** Size in CSS pixels of the largest title at one level of detail. */
export interface LabelSize {
  width: number;
  height: number;
}

/** Free space kept between two titles (px). */
const GAP = 6;
/** Extra room needed to go back to a more detailed level, so the titles do not flicker at the threshold. */
const HYSTERESIS = 10;

/**
 * Most detailed level at which the titles of consecutive layers do not overlap.
 * `points` holds the screen position of each title as (x, y, w) triples, as
 * written by projectPoints; `sizes` the largest title at each level.
 */
export function chooseLabelDetail(points: ArrayLike<number>, sizes: Readonly<Record<LabelDetail, LabelSize>>, current: LabelDetail): LabelDetail {
  for (const level of [0, 1] as const) {
    if (fits(points, sizes[level], GAP + (level < current ? HYSTERESIS : 0))) return level;
  }
  // The short names are the last resort, even if they still touch.
  return 2;
}

function fits(points: ArrayLike<number>, size: LabelSize, gap: number): boolean {
  for (let k = 3; k + 2 < points.length; k += 3) {
    // Points behind the camera are NaN: their titles are hidden anyway.
    const dx = Math.abs(points[k] - points[k - 3]);
    const dy = Math.abs(points[k + 1] - points[k - 2]);
    if (dx < size.width + gap && dy < size.height + gap) return false;
  }
  return true;
}
