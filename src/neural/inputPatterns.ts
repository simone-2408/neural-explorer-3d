/**
 * Synthetic input patterns for image-shaped input layers (e.g. 28×28).
 * Values are in [0, 1] like normalized grayscale pixels.
 */
import { clamp } from '../math/stats';
import { createRng } from '../math/random';

export type InputPattern = 'ring' | 'bar' | 'cross' | 'noise' | 'blank';

export const INPUT_PATTERNS: { id: InputPattern; label: string }[] = [
  { id: 'ring', label: '“0”' },
  { id: 'bar', label: '“1”' },
  { id: 'cross', label: '×' },
  { id: 'noise', label: 'Noise' },
  { id: 'blank', label: 'Clear' },
];

/** Soft stroke: 1 on the stroke centre, fading to 0 at `width`. */
const stroke = (distance: number, width: number) => clamp(1 - Math.max(0, distance - width * 0.35) / (width * 0.65), 0, 1);

export function makePattern(pattern: InputPattern, rows: number, cols: number, seed = 7): Float64Array {
  const out = new Float64Array(rows * cols);
  const cy = (rows - 1) / 2;
  const cx = (cols - 1) / 2;
  const unit = Math.min(rows, cols) / 28; // patterns are designed on a 28×28 canvas
  const rng = createRng(seed);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const y = (r - cy) / unit;
      const x = (c - cx) / unit;
      let v = 0;
      switch (pattern) {
        case 'ring': {
          // Ellipse with radii 6 (x) and 8.5 (y): distance from the ellipse outline.
          const d = Math.hypot(x / 6, y / 8.5);
          v = stroke(Math.abs(d - 1) * 7, 2.6);
          break;
        }
        case 'bar':
          v = Math.abs(y) < 10 ? stroke(Math.abs(x - y * 0.18), 2.2) : 0;
          break;
        case 'cross':
          v = Math.max(stroke(Math.abs(x - y) / Math.SQRT2, 2), stroke(Math.abs(x + y) / Math.SQRT2, 2)) * (Math.hypot(x, y) < 11 ? 1 : 0);
          break;
        case 'noise':
          v = rng.next();
          break;
        case 'blank':
          v = 0;
          break;
      }
      out[r * cols + c] = v;
    }
  }
  return out;
}
