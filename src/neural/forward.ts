/**
 * Forward pass of a fully-connected network — pure functions, no rendering.
 *
 *   z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
 *   a^{(l)} = f^{(l)}(z^{(l)})          with a^{(0)} = x
 *
 * The result is a `ForwardTrace`: every intermediate value, so that the UI can
 * explain the computation and the forward-pass player can animate it step by step. A trace is
 * the same kind of object a future PyTorch backend will return (see
 * docs/ARCHITECTURE.md → "Traces").
 */
import { applyActivation } from '../math/activations';
import { matVecAdd } from '../math/linalg';
import type { NetworkSpec } from './config';

export interface ForwardTrace {
  /** Pre-activations per layer (for the input layer: the input itself). */
  readonly z: Float64Array[];
  /** Outputs (activations) per layer; a[0] = x, a[L-1] = network output. */
  readonly a: Float64Array[];
}

export function createTrace(sizes: readonly number[]): ForwardTrace {
  return {
    z: sizes.map((n) => new Float64Array(n)),
    a: sizes.map((n) => new Float64Array(n)),
  };
}

/**
 * Runs the forward pass, writing into a pre-allocated trace (no allocations:
 * this runs on every slider movement and, later, on every training step).
 *
 * @param weights weights[l] has shape [n_l, n_{l-1}] (row-major); weights[0] is unused.
 * @param biases  biases[l] has length n_l; biases[0] is unused.
 */
export function forwardInto(
  spec: NetworkSpec,
  weights: readonly Float64Array[],
  biases: readonly Float64Array[],
  input: ArrayLike<number>,
  trace: ForwardTrace,
): ForwardTrace {
  const layers = spec.layers;
  trace.z[0].set(input);
  trace.a[0].set(input);
  for (let l = 1; l < layers.length; l++) {
    const rows = layers[l].size;
    const cols = layers[l - 1].size;
    matVecAdd(weights[l], rows, cols, trace.a[l - 1], biases[l], trace.z[l]);
    applyActivation(layers[l].activation, trace.z[l], trace.a[l]);
  }
  return trace;
}

/** Allocating convenience wrapper (tests, one-off computations). */
export function forward(
  spec: NetworkSpec,
  weights: readonly Float64Array[],
  biases: readonly Float64Array[],
  input: ArrayLike<number>,
): ForwardTrace {
  return forwardInto(spec, weights, biases, input, createTrace(spec.layers.map((l) => l.size)));
}
