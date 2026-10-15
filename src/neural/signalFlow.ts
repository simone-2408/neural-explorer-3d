/**
 * "Follow the Signal": how much of the signal flowing through the network
 * starts from (downstream) or ends in (upstream) a selected element.
 *
 * The measure is deliberately simple and transparent: shares of the
 * *magnitudes* of the terms of each weighted sum,
 *
 *     m_ji = |W_ji · a_i|        total_j = Σ_i m_ji + |b_j|
 *
 *  - downstream from neuron s:  S(s) = 1,  S(j) = Σ_i S(i) · m_ji / total_j
 *    (fraction of j's input magnitude that originates, through any path, from s)
 *  - upstream into neuron t:    R(t) = 1,  R(i) = Σ_j R(j) · m_ji / total_j
 *    (fraction of t's input magnitude that comes from i, through any path)
 *
 * It is an attribution heuristic (similar in spirit to LRP with absolute
 * values), not a causal explanation: positive and negative terms are counted
 * by magnitude, and non-linear interactions are ignored. The UI says so.
 */
import type { NetworkModel } from './NetworkModel';

export interface FlowScores {
  /** Score per neuron (global index), in [0, 1]. */
  neurons: Float32Array;
  /** Score per connection (global index), in [0, 1]. */
  connections: Float32Array;
}

export type FlowSource =
  | { kind: 'neuron'; layer: number; index: number }
  | { kind: 'connection'; layer: number; target: number; source: number };

/** Magnitude of each term and the total magnitude of each weighted sum, per layer. */
function termMagnitudes(model: NetworkModel): { terms: Float64Array[]; totals: Float64Array[] } {
  const { spec, trace } = model;
  const terms: Float64Array[] = [new Float64Array(0)];
  const totals: Float64Array[] = [new Float64Array(0)];
  for (let l = 1; l < spec.layers.length; l++) {
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    const W = model.weights(l);
    const b = model.biases(l);
    const x = trace.a[l - 1];
    const m = new Float64Array(rows * cols);
    const total = new Float64Array(rows);
    for (let j = 0; j < rows; j++) {
      let sum = Math.abs(b[j]);
      for (let i = 0; i < cols; i++) {
        const v = Math.abs(W[j * cols + i] * x[i]);
        m[j * cols + i] = v;
        sum += v;
      }
      total[j] = sum;
    }
    terms.push(m);
    totals.push(total);
  }
  return { terms, totals };
}

/** Downstream shares from the given per-layer seed vector (seed[layer0] set, others ignored). */
function propagateDown(
  model: NetworkModel,
  startLayer: number,
  seed: Float64Array,
  out: FlowScores,
  magnitudes: ReturnType<typeof termMagnitudes>,
): void {
  const { spec, topology } = model;
  let share = seed;
  for (let i = 0; i < share.length; i++) bump(out.neurons, topology.neuronIndex(startLayer, i), share[i]);
  for (let l = startLayer + 1; l < spec.layers.length; l++) {
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    const m = magnitudes.terms[l];
    const total = magnitudes.totals[l];
    const next = new Float64Array(rows);
    for (let j = 0; j < rows; j++) {
      if (total[j] <= 0) continue;
      let s = 0;
      for (let i = 0; i < cols; i++) {
        if (share[i] === 0) continue;
        const part = (share[i] * m[j * cols + i]) / total[j];
        s += part;
        bump(out.connections, topology.connectionIndex(l, j, i), part);
      }
      next[j] = s;
      bump(out.neurons, topology.neuronIndex(l, j), s);
    }
    share = next;
  }
}

/** Upstream shares into the given seed vector of layer `endLayer`. */
function propagateUp(
  model: NetworkModel,
  endLayer: number,
  seed: Float64Array,
  out: FlowScores,
  magnitudes: ReturnType<typeof termMagnitudes>,
): void {
  const { spec, topology } = model;
  let share = seed;
  for (let j = 0; j < share.length; j++) bump(out.neurons, topology.neuronIndex(endLayer, j), share[j]);
  for (let l = endLayer; l >= 1; l--) {
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    const m = magnitudes.terms[l];
    const total = magnitudes.totals[l];
    const previous = new Float64Array(cols);
    for (let j = 0; j < rows; j++) {
      if (share[j] === 0 || total[j] <= 0) continue;
      for (let i = 0; i < cols; i++) {
        const part = (share[j] * m[j * cols + i]) / total[j];
        previous[i] += part;
        bump(out.connections, topology.connectionIndex(l, j, i), part);
      }
    }
    for (let i = 0; i < cols; i++) bump(out.neurons, topology.neuronIndex(l - 1, i), previous[i]);
    share = previous;
  }
}

function bump(array: Float32Array, index: number, value: number): void {
  if (value > array[index]) array[index] = value;
}

/**
 * Raw flow scores (absolute shares in [0, 1]) for a selected neuron or
 * connection: what it feeds downstream and what feeds it upstream.
 */
export function signalFlow(model: NetworkModel, source: FlowSource): FlowScores {
  const { spec, topology } = model;
  const out: FlowScores = { neurons: new Float32Array(topology.neuronCount), connections: new Float32Array(topology.connectionCount) };
  const magnitudes = termMagnitudes(model);
  if (source.kind === 'neuron') {
    const seed = new Float64Array(spec.layers[source.layer].size);
    seed[source.index] = 1;
    propagateDown(model, source.layer, seed, out, magnitudes);
    propagateUp(model, source.layer, Float64Array.from(seed), out, magnitudes);
  } else {
    const { layer, target, source: from } = source;
    const cols = spec.layers[layer - 1].size;
    const share = magnitudes.totals[layer][target] > 0 ? magnitudes.terms[layer][target * cols + from] / magnitudes.totals[layer][target] : 0;
    out.connections[topology.connectionIndex(layer, target, from)] = 1;
    const up = new Float64Array(cols);
    up[from] = 1;
    propagateUp(model, layer - 1, up, out, magnitudes);
    const down = new Float64Array(spec.layers[layer].size);
    down[target] = share;
    propagateDown(model, layer, down, out, magnitudes);
    out.neurons[topology.neuronIndex(layer, target)] = Math.max(out.neurons[topology.neuronIndex(layer, target)], share);
  }
  return out;
}

/**
 * Display version of the scores: each layer is rescaled by its maximum so the
 * strongest path is always fully visible (raw shares get small in wide layers).
 */
export function normalizeFlowPerLayer(model: NetworkModel, raw: FlowScores): FlowScores {
  const { topology } = model;
  const out: FlowScores = { neurons: raw.neurons.slice(), connections: raw.connections.slice() };
  for (let l = 0; l < topology.layerCount; l++) {
    rescale(out.neurons, topology.neuronOffsets[l], topology.neuronOffsets[l + 1]);
    if (l > 0) rescale(out.connections, topology.connectionOffsets[l], topology.connectionOffsets[l + 1]);
  }
  return out;
}

function rescale(array: Float32Array, start: number, end: number): void {
  let max = 0;
  for (let k = start; k < end; k++) max = Math.max(max, array[k]);
  if (max <= 0) return;
  for (let k = start; k < end; k++) array[k] /= max;
}
