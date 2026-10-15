import { describe, expect, it } from 'vitest';
import { specFromConfig } from './config';
import { forward } from './forward';

describe('forward pass', () => {
  it('reproduces the worked example (neuron H1-3)', () => {
    // x1 = 0.7, x2 = 0.2, w1 = 0.42, w2 = −0.17, b = 0.12
    // z = 0.7·0.42 + 0.2·(−0.17) + 0.12 = 0.38, a = ReLU(0.38) = 0.38
    const spec = specFromConfig({ layers: [{ type: 'input', size: 2 }, { type: 'output', size: 1, activation: 'relu' }] });
    const trace = forward(spec, [new Float64Array(0), Float64Array.from([0.42, -0.17])], [new Float64Array(0), Float64Array.from([0.12])], [0.7, 0.2]);
    expect(trace.z[1][0]).toBeCloseTo(0.38, 12);
    expect(trace.a[1][0]).toBeCloseTo(0.38, 12);
  });

  it('computes z = W·a + b layer by layer with row-major [out, in] weights', () => {
    const spec = specFromConfig({
      layers: [
        { type: 'input', size: 2 },
        { type: 'dense', size: 2, activation: 'relu' },
        { type: 'output', size: 1, activation: 'linear' },
      ],
    });
    // W1 = [[1, -1], [0.5, 2]], b1 = [0, -1];  W2 = [[2, 3]], b2 = [0.5]
    const weights = [new Float64Array(0), Float64Array.from([1, -1, 0.5, 2]), Float64Array.from([2, 3])];
    const biases = [new Float64Array(0), Float64Array.from([0, -1]), Float64Array.from([0.5])];
    const trace = forward(spec, weights, biases, [1, 2]);
    // z1 = [1 - 2, 0.5 + 4 - 1] = [-1, 3.5] → a1 = [0, 3.5]
    expect(Array.from(trace.z[1])).toEqual([-1, 3.5]);
    expect(Array.from(trace.a[1])).toEqual([0, 3.5]);
    // y = 2·0 + 3·3.5 + 0.5 = 11
    expect(trace.a[2][0]).toBe(11);
  });

  it('produces a probability distribution with a softmax output', () => {
    const spec = specFromConfig({ layers: [{ type: 'input', size: 1 }, { type: 'output', size: 3, activation: 'softmax' }] });
    const trace = forward(spec, [new Float64Array(0), Float64Array.from([1, 2, 3])], [new Float64Array(0), new Float64Array(3)], [1]);
    const sum = trace.a[1].reduce((s, v) => s + v, 0);
    expect(sum).toBeCloseTo(1, 12);
    expect(trace.a[1][2]).toBeGreaterThan(trace.a[1][1]);
  });
});
