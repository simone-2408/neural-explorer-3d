import { describe, expect, it } from 'vitest';
import { chooseLabelDetail, type LabelSize } from './labelDetail';

const SIZES: [LabelSize, LabelSize, LabelSize] = [
  { width: 74, height: 39 },
  { width: 62, height: 22 },
  { width: 30, height: 22 },
];

/** Titles in a row, `spacing` px apart (screen x, y, clip w). */
function row(count: number, spacing: number, rise = 0): number[] {
  return Array.from({ length: count }, (_, k) => [100 + k * spacing, 200 - k * rise, 1]).flat();
}

describe('chooseLabelDetail', () => {
  it('keeps the full titles when neighbouring layers are far apart', () => {
    expect(chooseLabelDetail(row(4, 120), SIZES, 0)).toBe(0);
  });

  it('drops the details, then shortens the names as the layers get closer', () => {
    expect(chooseLabelDetail(row(8, 75), SIZES, 0)).toBe(1);
    expect(chooseLabelDetail(row(8, 45), SIZES, 0)).toBe(2);
    // Even when nothing fits, the short names are the last resort.
    expect(chooseLabelDetail(row(8, 10), SIZES, 0)).toBe(2);
  });

  it('ignores neighbours that are far apart vertically', () => {
    expect(chooseLabelDetail(row(4, 20, 60), SIZES, 0)).toBe(0);
  });

  it('needs extra room to go back to more detail (no flicker at the threshold)', () => {
    const points = row(6, 82);
    expect(chooseLabelDetail(points, SIZES, 0)).toBe(0);
    expect(chooseLabelDetail(points, SIZES, 1)).toBe(1);
    expect(chooseLabelDetail(row(6, 95), SIZES, 1)).toBe(0);
  });

  it('skips titles behind the camera', () => {
    const points = [100, 200, 1, NaN, NaN, -1, 110, 200, 1];
    expect(chooseLabelDetail(points, SIZES, 0)).toBe(0);
  });
});
