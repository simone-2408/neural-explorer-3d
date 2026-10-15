import { describe, expect, it } from 'vitest';
import { ACTIVATION_IDS, ACTIVATIONS, applyActivation, erf, sigmoid, softmax } from './activations';

describe('erf', () => {
  it('matches reference values to ~1e-12', () => {
    const reference: [number, number][] = [
      [0, 0],
      [0.5, 0.5204998778130465],
      [1, 0.8427007929497149],
      [2, 0.9953222650189527],
      [3, 0.9999779095030014],
      [4.5, 0.9999999998033839],
      [-1, -0.8427007929497149],
    ];
    for (const [x, expected] of reference) expect(erf(x)).toBeCloseTo(expected, 12);
  });
});

describe('elementwise activations', () => {
  it('evaluate known values', () => {
    expect(ACTIVATIONS.relu.kind === 'elementwise' && ACTIVATIONS.relu.f(-2)).toBe(0);
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(-800)).toBeGreaterThanOrEqual(0); // no overflow
    const gelu = ACTIVATIONS.gelu;
    if (gelu.kind !== 'elementwise') throw new Error('gelu must be elementwise');
    // PyTorch: torch.nn.functional.gelu(torch.tensor(1.0)) = 0.8413447460685429
    expect(gelu.f(1)).toBeCloseTo(0.8413447460685429, 12);
    const leaky = ACTIVATIONS.leaky_relu;
    if (leaky.kind !== 'elementwise') throw new Error('leaky_relu must be elementwise');
    expect(leaky.f(-2)).toBeCloseTo(-0.02, 12);
  });

  it('have derivatives consistent with finite differences', () => {
    const h = 1e-6;
    for (const id of ACTIVATION_IDS) {
      const info = ACTIVATIONS[id];
      if (info.kind !== 'elementwise') continue;
      for (const z of [-3, -1.3, -0.4, 0.37, 1.1, 2.9]) {
        const numeric = (info.f(z + h) - info.f(z - h)) / (2 * h);
        expect(info.df(z), `${id}'(${z})`).toBeCloseTo(numeric, 5);
      }
    }
  });
});

describe('softmax', () => {
  it('returns probabilities that sum to 1 and is numerically stable', () => {
    const p = softmax([1000, 1001, 1002], new Float64Array(3));
    expect(p.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(Number.isFinite(p[0])).toBe(true);
  });

  it('is applied to a whole layer by applyActivation', () => {
    const out = applyActivation('softmax', Float64Array.from([0, 0]), new Float64Array(2));
    expect(Array.from(out)).toEqual([0.5, 0.5]);
  });
});
