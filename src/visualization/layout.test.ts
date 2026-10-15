import { describe, expect, it } from 'vitest';
import { specFromConfig } from '../neural/config';
import { findPreset } from '../neural/presets';
import { Topology } from '../neural/topology';
import { computeLayout, DEFAULT_LAYOUT_OPTIONS, type LayoutMode } from './layout';

const spec = specFromConfig(findPreset('default')!.config);
const topology = Topology.fromSpec(spec);

describe('computeLayout', () => {
  it.each<LayoutMode>(['grid', 'column', 'ring'])('places layers along X, centred on the origin (%s)', (mode) => {
    const layout = computeLayout(spec, topology, { mode });
    expect(layout.positions.length).toBe(topology.neuronCount * 3);
    const xs = layout.frames.map((f) => f.center[0]);
    for (let l = 1; l < xs.length; l++) expect(xs[l]).toBeGreaterThan(xs[l - 1]);
    expect(xs[0] + xs[xs.length - 1]).toBeCloseTo(0, 9);
    // every neuron of a layer shares the layer's x
    for (let g = 0; g < topology.neuronCount; g++) {
      expect(layout.positions[3 * g]).toBeCloseTo(xs[topology.neuronAt(g).layer], 9);
    }
  });

  it('keeps neurons of a layer apart', () => {
    const layout = computeLayout(spec, topology, { mode: 'grid' });
    for (let l = 0; l < spec.layers.length; l++) {
      for (let i = 0; i < spec.layers[l].size; i++) {
        for (let j = i + 1; j < spec.layers[l].size; j++) {
          const a = topology.neuronIndex(l, i);
          const b = topology.neuronIndex(l, j);
          const d = Math.hypot(layout.positions[3 * a + 1] - layout.positions[3 * b + 1], layout.positions[3 * a + 2] - layout.positions[3 * b + 2]);
          expect(d).toBeGreaterThan(2 * layout.radii[a]);
        }
      }
    }
  });

  it('uses the configured image shape and bounds very large layers', () => {
    const mnist = specFromConfig(findPreset('mnist')!.config);
    const layout = computeLayout(mnist, Topology.fromSpec(mnist), { mode: 'grid' });
    const input = layout.frames[0];
    const extent = (input.halfWidth - input.neuronRadius) * 2;
    expect(extent).toBeLessThanOrEqual(DEFAULT_LAYOUT_OPTIONS.maxExtent + 2);
    // 28 columns along Z: first and last pixel of the first row are maxExtent apart.
    const first = 2; // z of neuron 0
    const last = 3 * 27 + 2; // z of neuron 27
    expect(layout.positions[last] - layout.positions[first]).toBeCloseTo(DEFAULT_LAYOUT_OPTIONS.maxExtent, 6);
  });
});
