/**
 * Forward pass of the CNN with a complete trace (every map before and after
 * the activation, the pooled maps and which input won each max-pooling
 * window), and the backward pass for training (cross-entropy on the softmax).
 *
 *   z[f, i, j] = b[f] + Σ_c Σ_u Σ_v W[f, c, u, v] · x[c, i + u − p, j + v − p]     (p = (k − 1) / 2, zero outside)
 *   a = f(z),   pooled = max / average over 2×2 windows,   logits = W_d · flatten + b_d
 */
import type { CnnModel, ConvShape } from './model';

export interface ConvTrace {
  /** Input of the layer [C, S, S]. */
  input: Float64Array;
  /** Convolution output before the activation [F, S, S]. */
  pre: Float64Array;
  /** After the activation [F, S, S]. */
  act: Float64Array;
  /** After pooling [F, S', S'] (the same array as `act` without pooling). */
  pooled: Float64Array;
  /** Max pooling: index in `act` of the winner of each window (−1 otherwise). */
  argmax: Int32Array;
}

export interface CnnTrace {
  input: Float64Array;
  layers: ConvTrace[];
  /** Flattened output of the last conv block (channel, row, column). */
  flat: Float64Array;
  logits: Float64Array;
  probs: Float64Array;
}

function activate(model: CnnModel, z: number): number {
  return model.config.activation === 'relu' ? (z > 0 ? z : 0) : Math.tanh(z);
}

/** Convolution with "same" padding, stride 1. */
function convolve(model: CnnModel, layer: number, input: Float64Array): Float64Array {
  const s = model.shapes[layer];
  const W = model.convWeight(layer);
  const b = model.convBias(layer);
  const { size: S, kernel: k, inChannels: C, outChannels: F } = s;
  const p = (k - 1) / 2;
  const out = new Float64Array(F * S * S);
  for (let f = 0; f < F; f++) {
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        let sum = b[f];
        for (let c = 0; c < C; c++) {
          for (let u = 0; u < k; u++) {
            const y = i + u - p;
            if (y < 0 || y >= S) continue;
            for (let v = 0; v < k; v++) {
              const x = j + v - p;
              if (x < 0 || x >= S) continue;
              sum += W[((f * C + c) * k + u) * k + v] * input[(c * S + y) * S + x];
            }
          }
        }
        out[(f * S + i) * S + j] = sum;
      }
    }
  }
  return out;
}

function pool(s: ConvShape, act: Float64Array): { pooled: Float64Array; argmax: Int32Array } {
  if (s.pool === 'none') return { pooled: act, argmax: new Int32Array(0) };
  const S = s.size;
  const P = s.pooledSize;
  const pooled = new Float64Array(s.outChannels * P * P);
  const argmax = new Int32Array(pooled.length).fill(-1);
  for (let f = 0; f < s.outChannels; f++) {
    for (let i = 0; i < P; i++) {
      for (let j = 0; j < P; j++) {
        const o = (f * P + i) * P + j;
        let best = -Infinity;
        let sum = 0;
        for (let di = 0; di < 2; di++) {
          for (let dj = 0; dj < 2; dj++) {
            const idx = (f * S + 2 * i + di) * S + 2 * j + dj;
            sum += act[idx];
            if (act[idx] > best) {
              best = act[idx];
              argmax[o] = idx;
            }
          }
        }
        pooled[o] = s.pool === 'max' ? best : sum / 4;
        if (s.pool === 'avg') argmax[o] = -1;
      }
    }
  }
  return { pooled, argmax };
}

export function softmax(logits: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const out = Float64Array.from(logits, (v) => Math.exp(v - max));
  const sum = out.reduce((a, b) => a + b, 0);
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function runCnn(model: CnnModel, image: ArrayLike<number>): CnnTrace {
  const input = Float64Array.from(image);
  let x: Float64Array = input;
  const layers: ConvTrace[] = model.shapes.map((s, l) => {
    const pre = convolve(model, l, x);
    const act = Float64Array.from(pre, (z) => activate(model, z));
    const { pooled, argmax } = pool(s, act);
    const trace = { input: x, pre, act, pooled, argmax };
    x = pooled;
    return trace;
  });
  const flat = x;
  const C = model.config.classes;
  const D = model.flatSize;
  const W = model.denseWeight;
  const logits = Float64Array.from(model.denseBias);
  for (let k = 0; k < C; k++) for (let d = 0; d < D; d++) logits[k] += W[k * D + d] * flat[d];
  return { input, layers, flat, logits, probs: softmax(logits) };
}

/**
 * Gradients of the cross-entropy loss −log p(label) for one image, added
 * into `grads` (same layout as model.params). Returns the loss.
 */
export function backwardCnn(model: CnnModel, trace: CnnTrace, label: number, grads: Float64Array[], weight = 1): number {
  const C = model.config.classes;
  const D = model.flatSize;
  const L = model.shapes.length;
  // dense layer: δ = p − onehot(label)
  const delta = Float64Array.from(trace.probs, (p, k) => (p - (k === label ? 1 : 0)) * weight);
  const gW = grads[2 * L];
  const gb = grads[2 * L + 1];
  const W = model.denseWeight;
  let dx: Float64Array = new Float64Array(D);
  for (let k = 0; k < C; k++) {
    gb[k] += delta[k];
    for (let d = 0; d < D; d++) {
      gW[k * D + d] += delta[k] * trace.flat[d];
      dx[d] += W[k * D + d] * delta[k];
    }
  }
  for (let l = L - 1; l >= 0; l--) {
    const s = model.shapes[l];
    const t = trace.layers[l];
    const S = s.size;
    // un-pool: the gradient goes to the winner (max) or is shared by the window (average)
    let dAct: Float64Array;
    if (s.pool === 'none') dAct = dx;
    else {
      dAct = new Float64Array(t.act.length);
      const P = s.pooledSize;
      for (let f = 0; f < s.outChannels; f++) {
        for (let i = 0; i < P; i++) {
          for (let j = 0; j < P; j++) {
            const o = (f * P + i) * P + j;
            if (s.pool === 'max') dAct[t.argmax[o]] += dx[o];
            else for (let di = 0; di < 2; di++) for (let dj = 0; dj < 2; dj++) dAct[(f * S + 2 * i + di) * S + 2 * j + dj] += dx[o] / 4;
          }
        }
      }
    }
    // through the activation
    const dz = Float64Array.from(dAct, (g, n) => (model.config.activation === 'relu' ? (t.pre[n] > 0 ? g : 0) : g * (1 - t.act[n] * t.act[n])));
    // convolution: weight and bias gradients, and the gradient of the input (unless it is the image)
    const { kernel: k, inChannels: Cin, outChannels: F } = s;
    const p = (k - 1) / 2;
    const Wc = model.convWeight(l);
    const gWc = grads[2 * l];
    const gbc = grads[2 * l + 1];
    const dInput = l > 0 ? new Float64Array(t.input.length) : null;
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < S; i++) {
        for (let j = 0; j < S; j++) {
          const g = dz[(f * S + i) * S + j];
          if (g === 0) continue;
          gbc[f] += g;
          for (let c = 0; c < Cin; c++) {
            for (let u = 0; u < k; u++) {
              const y = i + u - p;
              if (y < 0 || y >= S) continue;
              for (let v = 0; v < k; v++) {
                const x = j + v - p;
                if (x < 0 || x >= S) continue;
                const w = ((f * Cin + c) * k + u) * k + v;
                const n = (c * S + y) * S + x;
                gWc[w] += g * t.input[n];
                if (dInput) dInput[n] += g * Wc[w];
              }
            }
          }
        }
      }
    }
    if (dInput) dx = dInput;
  }
  return -Math.log(Math.max(trace.probs[label], 1e-300));
}

export function zeroGrads(model: CnnModel): Float64Array[] {
  return model.params.map((p) => new Float64Array(p.length));
}
