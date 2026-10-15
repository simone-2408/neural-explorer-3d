import { describe, expect, it } from 'vitest';
import { Topology } from '../neural/topology';
import { computeEmphasis, EMPHASIS } from './emphasis';

const topology = new Topology([2, 3, 1]);

describe('computeEmphasis', () => {
  it('leaves everything normal without a selection', () => {
    const e = computeEmphasis(topology, null, true);
    expect([...e.neurons].every((v) => v === EMPHASIS.normal)).toBe(true);
    expect([...e.connections].every((v) => v === EMPHASIS.normal)).toBe(true);
  });

  it('highlights a neuron with its incoming and outgoing weights', () => {
    const e = computeEmphasis(topology, { kind: 'neuron', layer: 1, index: 1 }, true);
    expect(e.neurons[topology.neuronIndex(1, 1)]).toBe(EMPHASIS.focus);
    expect(e.neurons[topology.neuronIndex(0, 0)]).toBe(EMPHASIS.related);
    expect(e.neurons[topology.neuronIndex(2, 0)]).toBe(EMPHASIS.related);
    expect(e.neurons[topology.neuronIndex(1, 0)]).toBe(EMPHASIS.dim);
    expect(e.connections[topology.connectionIndex(1, 1, 0)]).toBe(EMPHASIS.related); // incoming
    expect(e.connections[topology.connectionIndex(2, 0, 1)]).toBe(EMPHASIS.related); // outgoing
    expect(e.connections[topology.connectionIndex(1, 0, 0)]).toBe(EMPHASIS.dim);
  });

  it('highlights the weight matrix of a selected layer', () => {
    const e = computeEmphasis(topology, { kind: 'layer', layer: 1 }, false);
    for (let c = topology.connectionOffsets[1]; c < topology.connectionOffsets[2]; c++) expect(e.connections[c]).toBe(EMPHASIS.related);
    // dimming disabled: unrelated elements stay normal
    expect(e.connections[topology.connectionIndex(2, 0, 0)]).toBe(EMPHASIS.normal);
  });
});
