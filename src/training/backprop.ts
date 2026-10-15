/**
 * Backpropagation for fully-connected networks, written out by hand (no
 * autograd) so every intermediate value can be shown:
 *
 *   δ^(L)   = ∂L/∂z^(L)                              (see loss.ts)
 *   δ^(l)   = (W^(l+1)ᵀ δ^(l+1)) ⊙ f′(z^(l))        for hidden layers
 *   ∂L/∂W^(l) = δ^(l) a^(l−1)ᵀ        ∂L/∂b^(l) = δ^(l)
 */
import { ACTIVATIONS } from '../math/activations';
import type { NetworkSpec } from '../neural/config';
import type { ForwardTrace } from '../neural/forward';
import { outputDelta, type LossId } from './loss';

export interface Gradients {
  /** ∂L/∂W per layer (same layout as the weights; [0] unused). */
  dW: Float64Array[];
  db: Float64Array[];
  /** δ = ∂L/∂z per layer ([0] unused). */
  delta: Float64Array[];
}

export function createGradients(spec: NetworkSpec): Gradients {
  const dW = [new Float64Array(0)];
  const db = [new Float64Array(0)];
  const delta = [new Float64Array(spec.layers[0].size)];
  for (let l = 1; l < spec.layers.length; l++) {
    dW.push(new Float64Array(spec.layers[l].size * spec.layers[l - 1].size));
    db.push(new Float64Array(spec.layers[l].size));
    delta.push(new Float64Array(spec.layers[l].size));
  }
  return { dW, db, delta };
}

export function zeroGradients(g: Gradients): void {
  for (const a of g.dW) a.fill(0);
  for (const a of g.db) a.fill(0);
  for (const a of g.delta) a.fill(0);
}

export function scaleGradients(g: Gradients, factor: number): void {
  for (const arrays of [g.dW, g.db, g.delta]) for (const a of arrays) for (let k = 0; k < a.length; k++) a[k] *= factor;
}

/** Scratch buffers for one sample (δ per layer). */
export function createScratch(spec: NetworkSpec): Float64Array[] {
  return spec.layers.map((l) => new Float64Array(l.size));
}

/**
 * Backpropagates one sample whose forward pass is in `trace` and ADDS its
 * gradients to `acc` (so a batch can be accumulated and then averaged).
 * `scratch` receives the per-sample δ.
 */
export function backpropSample(
  spec: NetworkSpec,
  weights: readonly Float64Array[],
  trace: ForwardTrace,
  loss: LossId,
  label: number,
  acc: Gradients,
  scratch: Float64Array[],
): void {
  const L = spec.layers.length - 1;
  outputDelta(loss, spec, trace.z[L], trace.a[L], label, scratch[L]);
  for (let l = L; l >= 1; l--) {
    const delta = scratch[l];
    const prev = trace.a[l - 1];
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    const dW = acc.dW[l];
    const db = acc.db[l];
    const accDelta = acc.delta[l];
    for (let j = 0; j < rows; j++) {
      const d = delta[j];
      db[j] += d;
      accDelta[j] += d;
      if (d === 0) continue;
      const offset = j * cols;
      for (let i = 0; i < cols; i++) dW[offset + i] += d * prev[i];
    }
    if (l > 1) {
      const W = weights[l];
      const below = scratch[l - 1];
      const activation = ACTIVATIONS[spec.layers[l - 1].activation];
      const z = trace.z[l - 1];
      for (let i = 0; i < cols; i++) {
        let sum = 0;
        for (let j = 0; j < rows; j++) sum += W[j * cols + i] * delta[j];
        below[i] = activation.kind === 'elementwise' ? sum * activation.df(z[i]) : sum;
      }
    }
  }
}
