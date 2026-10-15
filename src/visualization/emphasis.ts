/**
 * Emphasis = how prominent each neuron / connection should be, derived from
 * the current selection. Pure function of (topology, selection), consumed by
 * the renderer as per-instance attributes.
 *
 * "Follow the Signal" (workspaces/network/FollowSignal.ts) produces the same
 * structure from signal contributions instead of from the selection.
 */
import type { Topology } from '../neural/topology';
import type { NetworkSelection } from './selection';

export const EMPHASIS = {
  /** Unrelated to the current focus (drawn faded). */
  dim: 0,
  /** Nothing is selected. */
  normal: 1,
  /** Directly connected to the focus. */
  related: 2,
  /** The selected element itself. */
  focus: 3,
} as const;

export interface Emphasis {
  neurons: Uint8Array;
  connections: Uint8Array;
}

export function createEmphasis(topology: Topology): Emphasis {
  return { neurons: new Uint8Array(topology.neuronCount), connections: new Uint8Array(topology.connectionCount) };
}

export function computeEmphasis(
  topology: Topology,
  selection: NetworkSelection | null,
  dimUnrelated: boolean,
  out: Emphasis = createEmphasis(topology),
): Emphasis {
  const { neurons, connections } = out;
  const sizes = topology.sizes;
  const background = selection && dimUnrelated ? EMPHASIS.dim : EMPHASIS.normal;
  neurons.fill(background);
  connections.fill(background);
  if (!selection) return out;

  const raise = (array: Uint8Array, index: number, level: number) => {
    if (array[index] < level) array[index] = level;
  };

  switch (selection.kind) {
    case 'neuron': {
      const { layer, index } = selection;
      raise(neurons, topology.neuronIndex(layer, index), EMPHASIS.focus);
      if (layer > 0) {
        // Incoming weights: row `index` of W^{(layer)}.
        for (let i = 0; i < sizes[layer - 1]; i++) {
          raise(connections, topology.connectionIndex(layer, index, i), EMPHASIS.related);
          raise(neurons, topology.neuronIndex(layer - 1, i), EMPHASIS.related);
        }
      }
      if (layer < sizes.length - 1) {
        // Outgoing weights: column `index` of W^{(layer+1)}.
        for (let j = 0; j < sizes[layer + 1]; j++) {
          raise(connections, topology.connectionIndex(layer + 1, j, index), EMPHASIS.related);
          raise(neurons, topology.neuronIndex(layer + 1, j), EMPHASIS.related);
        }
      }
      break;
    }
    case 'connection': {
      const { layer, target, source } = selection;
      raise(connections, topology.connectionIndex(layer, target, source), EMPHASIS.focus);
      raise(neurons, topology.neuronIndex(layer - 1, source), EMPHASIS.related);
      raise(neurons, topology.neuronIndex(layer, target), EMPHASIS.related);
      break;
    }
    case 'layer': {
      const { layer } = selection;
      for (let j = 0; j < sizes[layer]; j++) raise(neurons, topology.neuronIndex(layer, j), EMPHASIS.focus);
      // The layer's own weight matrix (its incoming connections).
      if (layer > 0) {
        const start = topology.connectionOffsets[layer];
        const end = topology.connectionOffsets[layer + 1];
        for (let c = start; c < end; c++) raise(connections, c, EMPHASIS.related);
      }
      break;
    }
  }
  return out;
}
