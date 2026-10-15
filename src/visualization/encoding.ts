/**
 * Visual encodings: how numbers become visual properties. Kept separate from
 * three.js so the mapping is explicit, documented, testable and reusable by
 * 2D views (heatmaps, charts).
 *
 *   weight w      → connection thickness ∝ |w|, hue = sign, dashed when w < 0
 *   activation a  → neuron colour intensity and size ∝ |a|, hue = sign
 */
import type { ForwardTrace } from '../neural/forward';
import { maxAbs } from '../math/stats';

/** |v| / scale clamped to [0, 1] (0 when scale is 0). */
export function magnitude(value: number, scale: number): number {
  if (!(scale > 0)) return 0;
  const t = Math.abs(value) / scale;
  return t > 1 ? 1 : t;
}

export interface ConnectionRadiusRange {
  min: number;
  max: number;
}

/**
 * Connection radius grows linearly with |w| (relative to the largest |w| in
 * the network) from `min` (w = 0) to `max` (|w| = max|w|).
 */
export function connectionRadius(weight: number, maxAbsWeight: number, range: ConnectionRadiusRange): number {
  return range.min + (range.max - range.min) * magnitude(weight, maxAbsWeight);
}

export type ActivationScaleMode = 'global' | 'layer';

/**
 * Scale used to normalize activation intensities, per layer.
 * - global: brightness comparable across the whole network (max |a| overall)
 * - layer:  each layer uses its own max |a| (highlights the most active neurons of every layer)
 */
export function activationScales(trace: ForwardTrace, mode: ActivationScaleMode): number[] {
  const perLayer = trace.a.map((a) => maxAbs(a));
  if (mode === 'layer') return perLayer.map((m) => m || 1);
  const global = Math.max(...perLayer) || 1;
  return perLayer.map(() => global);
}

/** Perceptual boost for small magnitudes so they remain visible (monotonic, 0 → 0, 1 → 1). */
export function visibleIntensity(t: number): number {
  return Math.pow(t, 0.7);
}
