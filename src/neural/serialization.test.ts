import { describe, expect, it } from 'vitest';
import { NetworkModel } from './NetworkModel';
import { findPreset } from './presets';
import { applyLoadedModel, deserializeModel, serializeModel, stringifyModel } from './serialization';

describe('model file format', () => {
  const preset = findPreset('classifier')!;

  it('round-trips architecture, weights, init and input', () => {
    const source = NetworkModel.fromConfig(preset.config, { input: preset.input?.(), init: { seed: 123 } });
    source.setWeight(1, 2, 1, 1.2345);
    const text = stringifyModel(serializeModel(source));
    const loaded = deserializeModel(JSON.parse(text));
    expect(loaded.ok).toBe(true);
    if (!loaded.ok) return;

    const target = NetworkModel.fromConfig({ layers: [{ type: 'input', size: 1 }, { type: 'output', size: 1 }] });
    applyLoadedModel(target, loaded.model);
    expect(target.spec).toEqual(source.spec);
    expect(target.init).toEqual(source.init);
    for (let l = 1; l < source.layerCount; l++) {
      expect(Array.from(target.weights(l))).toEqual(Array.from(source.weights(l)));
      expect(Array.from(target.biases(l))).toEqual(Array.from(source.biases(l)));
    }
    expect(Array.from(target.output)).toEqual(Array.from(source.output));
  });

  it('stores weights with the PyTorch nn.Linear shape [out, in]', () => {
    const model = NetworkModel.fromConfig(preset.config);
    const data = serializeModel(model);
    expect(data.parameters[0].weight.length).toBe(5); // out
    expect(data.parameters[0].weight[0].length).toBe(3); // in
  });

  it('accepts a bare configuration', () => {
    const loaded = deserializeModel(preset.config);
    expect(loaded.ok && loaded.model.weights).toBeNull();
  });

  it('rejects inconsistent files', () => {
    const data = serializeModel(NetworkModel.fromConfig(preset.config)) as unknown as Record<string, unknown>;
    const broken = { ...data, parameters: [] };
    const result = deserializeModel(broken);
    expect(result.ok).toBe(false);
    expect(deserializeModel({ format: 'something-else' }).ok).toBe(false);
  });
});
