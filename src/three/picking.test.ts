import { describe, expect, it } from 'vitest';
import { pointSegmentDistance, projectPoints } from './picking';

describe('pointSegmentDistance', () => {
  it('measures the distance to the closest point of the segment', () => {
    expect(pointSegmentDistance(5, 3, 0, 0, 10, 0)).toEqual({ distance: 3, t: 0.5 });
    // beyond the end: distance to the endpoint
    const beyond = pointSegmentDistance(13, 4, 0, 0, 10, 0);
    expect(beyond.distance).toBeCloseTo(5, 12);
    expect(beyond.t).toBe(1);
    // degenerate segment
    expect(pointSegmentDistance(3, 4, 0, 0, 0, 0).distance).toBe(5);
  });
});

describe('projectPoints', () => {
  it('maps clip space to canvas pixels and flags points behind the camera', () => {
    // Identity "view-projection": clip = (x, y, z, 1)
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const out = projectPoints([0, 0, 0, 1, 1, 0, -1, -1, 0], identity, 200, 100, new Float32Array(9));
    expect(Array.from(out.slice(0, 3))).toEqual([100, 50, 1]); // centre
    expect(Array.from(out.slice(3, 6))).toEqual([200, 0, 1]); // top-right
    expect(Array.from(out.slice(6, 9))).toEqual([0, 100, 1]); // bottom-left

    // w = -1 → behind the camera
    const flip = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1];
    const behind = projectPoints([0, 0, 0], flip, 200, 100, new Float32Array(3));
    expect(Number.isNaN(behind[0])).toBe(true);
    expect(behind[2]).toBeLessThan(0);
  });
});
