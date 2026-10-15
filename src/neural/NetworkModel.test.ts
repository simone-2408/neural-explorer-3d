import { describe, expect, it, vi } from 'vitest';
import { specFromConfig } from './config';
import { NetworkModel } from './NetworkModel';
import { PRESETS } from './presets';

const config = {
  layers: [
    { type: 'input', size: 2 },
    { type: 'dense', size: 4, activation: 'relu' },
    { type: 'dense', size: 4, activation: 'relu' },
    { type: 'output', size: 1, activation: 'sigmoid' },
  ],
};

describe('NetworkModel', () => {
  it('runs headless: builds, initializes and computes the forward pass', () => {
    const model = NetworkModel.fromConfig(config, { input: [0.7, 0.2] });
    expect(model.totalParameters()).toBe(37);
    expect(model.output[0]).toBeGreaterThan(0);
    expect(model.output[0]).toBeLessThan(1);
  });

  it('is deterministic for a given seed', () => {
    const a = NetworkModel.fromConfig(config, { init: { seed: 7 } });
    const b = NetworkModel.fromConfig(config, { init: { seed: 7 } });
    const c = NetworkModel.fromConfig(config, { init: { seed: 8 } });
    expect(Array.from(a.weights(2))).toEqual(Array.from(b.weights(2)));
    expect(Array.from(a.weights(2))).not.toEqual(Array.from(c.weights(2)));
  });

  it('recomputes and notifies when a weight changes', () => {
    const model = NetworkModel.fromConfig(config, { input: [0.7, 0.2] });
    const params = vi.fn();
    const activations = vi.fn();
    model.events.on('params', params);
    model.events.on('activations', activations);
    const before = model.trace.z[1][0];
    model.setWeight(1, 0, 0, model.weight(1, 0, 0) + 1);
    expect(model.trace.z[1][0]).toBeCloseTo(before + 0.7, 12); // ∂z/∂w = x = 0.7
    expect(params).toHaveBeenCalledWith({ reason: 'edit', layer: 1 });
    expect(activations).toHaveBeenCalledTimes(1);
  });

  it('batches several changes into a single notification', () => {
    const model = NetworkModel.fromConfig(config);
    const activations = vi.fn();
    model.events.on('activations', activations);
    model.batch(() => {
      model.setInput(0, 0.1);
      model.setInput(1, 0.2);
      model.setBias(2, 1, 0.3);
    });
    expect(activations).toHaveBeenCalledTimes(1);
  });

  it('preserves overlapping weights when a layer grows', () => {
    const model = NetworkModel.fromConfig(config, { input: [0.7, 0.2] });
    const before = Array.from(model.weights(2));
    const bigger = specFromConfig({ ...config, layers: config.layers.map((l, i) => (i === 1 ? { ...l, size: 5 } : l)) });
    const structure = vi.fn();
    model.events.on('structure', structure);
    model.setArchitecture(bigger);
    expect(structure).toHaveBeenCalledTimes(1);
    // Layer 2 is now [4, 5]: its first 4 columns are the old [4, 4] matrix.
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) expect(model.weight(2, j, i)).toBe(before[j * 4 + i]);
    expect(Array.from(model.input)).toEqual([0.7, 0.2]);
  });

  it('builds every preset', () => {
    for (const preset of PRESETS) {
      const model = NetworkModel.fromConfig(preset.config, { input: preset.input?.(), init: preset.init });
      expect(model.output.every(Number.isFinite), preset.id).toBe(true);
    }
  });
});
