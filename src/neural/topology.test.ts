import { describe, expect, it } from 'vitest';
import { specFromConfig } from './config';
import { connectionName, neuronName, neuronTitle, Topology, weightSymbol } from './topology';

describe('Topology', () => {
  const topology = new Topology([2, 4, 4, 1]);

  it('counts neurons and connections', () => {
    expect(topology.neuronCount).toBe(11);
    expect(topology.connectionCount).toBe(2 * 4 + 4 * 4 + 4 * 1);
    expect(topology.neuronOffsets).toEqual([0, 2, 6, 10, 11]);
    expect(topology.connectionOffsets).toEqual([0, 0, 8, 24, 28]);
  });

  it('maps neuron indices both ways', () => {
    for (let g = 0; g < topology.neuronCount; g++) {
      const { layer, index } = topology.neuronAt(g);
      expect(topology.neuronIndex(layer, index)).toBe(g);
    }
  });

  it('maps connection indices both ways, in weight-matrix (row-major) order', () => {
    for (let c = 0; c < topology.connectionCount; c++) {
      const ref = topology.connectionAt(c);
      expect(topology.connectionIndex(ref.layer, ref.target, ref.source)).toBe(c);
    }
    // W^{(2)}[1, 3] is the (1·4 + 3)-th weight of layer 2.
    expect(topology.connectionIndex(2, 1, 3)).toBe(8 + 1 * 4 + 3);
  });
});

describe('naming', () => {
  const spec = specFromConfig({
    layers: [
      { type: 'input', size: 2 },
      { type: 'dense', size: 4 },
      { type: 'output', size: 2, labels: ['cat', 'dog'] },
    ],
  });

  it('uses textbook-style 1-based names', () => {
    expect(neuronName(spec, 0, 0)).toBe('I1');
    expect(neuronName(spec, 1, 2)).toBe('H1-3');
    expect(neuronName(spec, 2, 1)).toBe('O2');
    expect(neuronTitle(spec, 2, 1)).toBe('O2 “dog”');
    expect(connectionName(spec, { layer: 1, target: 2, source: 0 })).toBe('I1 → H1-3');
    expect(weightSymbol({ layer: 1, target: 2, source: 0 })).toBe('W^{(1)}_{3,1}');
  });
});
