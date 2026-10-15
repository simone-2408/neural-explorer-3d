import { describe, expect, it } from 'vitest';
import { specFromConfig } from './config';
import { NetworkModel } from './NetworkModel';
import { normalizeFlowPerLayer, signalFlow } from './signalFlow';

function model(): NetworkModel {
  const spec = specFromConfig({
    layers: [
      { type: 'input', size: 2 },
      { type: 'dense', size: 2, activation: 'relu' },
      { type: 'output', size: 1, activation: 'linear' },
    ],
  });
  const m = new NetworkModel(spec, { input: [1, 1] });
  // W1 = [[1, 3], [2, 0]], b1 = [0, 0] → a1 = [4, 2];  W2 = [[1, 1]], b2 = [0] → y = 6
  m.setParameters(
    [new Float64Array(0), Float64Array.from([1, 3, 2, 0]), Float64Array.from([1, 1])],
    [new Float64Array(0), new Float64Array(2), new Float64Array(1)],
  );
  return m;
}

describe('signalFlow', () => {
  it('computes downstream shares of |w·a|', () => {
    const m = model();
    const flow = signalFlow(m, { kind: 'neuron', layer: 0, index: 0 });
    const { topology } = m;
    // H1 = 1·1 + 3·1 → share of x1 = 1/4; H2 = 2·1 + 0 → share 1
    expect(flow.neurons[topology.neuronIndex(1, 0)]).toBeCloseTo(0.25, 6);
    expect(flow.neurons[topology.neuronIndex(1, 1)]).toBeCloseTo(1, 6);
    // output = 4 + 2: share from x1 = (0.25·4 + 1·2) / 6 = 0.5
    expect(flow.neurons[topology.neuronIndex(2, 0)]).toBeCloseTo(0.5, 6);
    expect(flow.connections[topology.connectionIndex(1, 0, 1)]).toBe(0); // x2 → H1 is not on x1's path
  });

  it('computes upstream shares into the output', () => {
    const m = model();
    const flow = signalFlow(m, { kind: 'neuron', layer: 2, index: 0 });
    const { topology } = m;
    expect(flow.neurons[topology.neuronIndex(1, 0)]).toBeCloseTo(4 / 6, 6);
    expect(flow.neurons[topology.neuronIndex(1, 1)]).toBeCloseTo(2 / 6, 6);
    // x1 feeds 1/4 of H1 and all of H2: (4/6)(1/4) + (2/6)(1) = 0.5
    expect(flow.neurons[topology.neuronIndex(0, 0)]).toBeCloseTo(0.5, 6);
    expect(flow.neurons[topology.neuronIndex(0, 1)]).toBeCloseTo(0.5, 6);
  });

  it('stops at inactive neurons and normalizes per layer for display', () => {
    const m = model();
    m.setInput(0, -10); // H1 = -10 + 3 < 0 → inactive; H2 = -20 → inactive
    const flow = signalFlow(m, { kind: 'neuron', layer: 0, index: 1 });
    expect(flow.neurons[m.topology.neuronIndex(2, 0)]).toBe(0);
    const display = normalizeFlowPerLayer(m, signalFlow(m, { kind: 'neuron', layer: 0, index: 0 }));
    for (const v of display.neurons) expect(v).toBeLessThanOrEqual(1);
  });

  it('follows a connection in both directions', () => {
    const m = model();
    const flow = signalFlow(m, { kind: 'connection', layer: 1, target: 0, source: 1 });
    const { topology } = m;
    expect(flow.connections[topology.connectionIndex(1, 0, 1)]).toBe(1);
    expect(flow.neurons[topology.neuronIndex(1, 0)]).toBeCloseTo(0.75, 6); // 3 of H1's 4
  });
});
