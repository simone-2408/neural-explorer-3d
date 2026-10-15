/**
 * Screen-space picking math (pure functions, unit-tested).
 *
 * Connections are thin cylinders: ray-casting them would require pixel-perfect
 * aim. Instead, endpoints are projected to the screen and the cursor is
 * matched against the 2D segments with a pixel tolerance.
 */

/**
 * Projects xyz points with a column-major 4×4 view-projection matrix.
 * Writes (screenX, screenY, clipW) per point; clipW ≤ 0 means "behind the camera".
 */
export function projectPoints(
  positions: ArrayLike<number>,
  m: ArrayLike<number>,
  width: number,
  height: number,
  out: Float32Array,
): Float32Array {
  const count = positions.length / 3;
  for (let k = 0; k < count; k++) {
    const x = positions[3 * k];
    const y = positions[3 * k + 1];
    const z = positions[3 * k + 2];
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    out[3 * k + 2] = cw;
    if (cw <= 1e-6) {
      out[3 * k] = NaN;
      out[3 * k + 1] = NaN;
      continue;
    }
    out[3 * k] = (cx / cw * 0.5 + 0.5) * width;
    out[3 * k + 1] = (1 - (cy / cw * 0.5 + 0.5)) * height;
  }
  return out;
}

/** Distance from point P to segment AB. `t` (0..1) is the position of the closest point along AB. */
export function pointSegmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): { distance: number; t: number } {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSq = dx * dx + dy * dy;
  let t = lengthSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lengthSq : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const cx = ax + t * dx;
  const cy = ay + t * dy;
  return { distance: Math.hypot(px - cx, py - cy), t };
}
