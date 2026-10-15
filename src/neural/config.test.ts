import { describe, expect, it } from 'vitest';
import { formatConfig, parseConfigText, specToConfig, validateConfig } from './config';

const SPEC_EXAMPLE = {
  layers: [
    { type: 'input', size: 3 },
    { type: 'dense', size: 5, activation: 'relu' },
    { type: 'dense', size: 4, activation: 'relu' },
    { type: 'output', size: 2, activation: 'softmax' },
  ],
};

describe('validateConfig', () => {
  it('accepts the example configuration from the specification', () => {
    const result = validateConfig(SPEC_EXAMPLE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.spec.layers.map((l) => [l.role, l.size, l.activation, l.name, l.prefix])).toEqual([
      ['input', 3, 'linear', 'Input', 'I'],
      ['hidden', 5, 'relu', 'Hidden 1', 'H1'],
      ['hidden', 4, 'relu', 'Hidden 2', 'H2'],
      ['output', 2, 'softmax', 'Output', 'O'],
    ]);
  });

  it('applies defaults: relu for hidden layers, linear for the output', () => {
    const result = validateConfig({ layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 3 }, { type: 'output', size: 1 }] });
    expect(result.ok && result.spec.layers.map((l) => l.activation)).toEqual(['linear', 'relu', 'linear']);
  });

  it.each([
    [{}, 'layers'],
    [{ layers: [{ type: 'input', size: 2 }] }, 'layers'],
    [{ layers: [{ type: 'dense', size: 2 }, { type: 'output', size: 1 }] }, 'layers[0].type'],
    [{ layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 1 }] }, 'layers[1].type'],
    [{ layers: [{ type: 'input', size: 0 }, { type: 'output', size: 1 }] }, 'layers[0].size'],
    [{ layers: [{ type: 'input', size: 2.5 }, { type: 'output', size: 1 }] }, 'layers[0].size'],
    [{ layers: [{ type: 'input', size: 2 }, { type: 'output', size: 1, activation: 'swish' }] }, 'layers[1].activation'],
    [{ layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 3, activation: 'softmax' }, { type: 'output', size: 1 }] }, 'layers[1].activation'],
    [{ layers: [{ type: 'input', size: 2 }, { type: 'conv2d', size: 3 }, { type: 'output', size: 1 }] }, 'layers[1].type'],
    [{ layers: [{ type: 'input', size: 6, shape: [2, 2] }, { type: 'output', size: 1 }] }, 'layers[0].shape'],
    [{ layers: [{ type: 'input', size: 2 }, { type: 'output', size: 2, labels: ['a'] }] }, 'layers[1].labels'],
    [{ layers: [{ type: 'input', size: 1000 }, { type: 'dense', size: 1000 }, { type: 'output', size: 10 }] }, 'layers'],
  ])('rejects invalid configuration %#', (config, path) => {
    const result = validateConfig(config);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.map((e) => e.path)).toContain(path);
  });

  it('warns about unknown properties instead of failing', () => {
    const result = validateConfig({ ...SPEC_EXAMPLE, colour: 'red' });
    expect(result.ok).toBe(true);
    expect(result.warnings.map((w) => w.path)).toContain('colour');
  });

  it('reports JSON syntax errors', () => {
    const result = parseConfigText('{ "layers": [ }');
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors[0].path).toBe('JSON');
  });
});

describe('specToConfig / formatConfig', () => {
  it('round-trips through the JSON editor format', () => {
    const first = validateConfig({ name: 'Round trip', ...SPEC_EXAMPLE });
    if (!first.ok) throw new Error('invalid');
    const text = formatConfig(specToConfig(first.spec));
    const second = parseConfigText(text);
    expect(second.ok).toBe(true);
    if (second.ok) expect(second.spec).toEqual(first.spec);
    expect(text).toContain('{ "type": "dense", "size": 5, "activation": "relu" }');
  });
});
