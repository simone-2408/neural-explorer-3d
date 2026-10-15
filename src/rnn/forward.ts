/**
 * The recurrent network unrolled through time: a forward pass that keeps
 * every intermediate value (input and recurrent parts, gates, states,
 * predictions) and backpropagation through time (BPTT), which also reports
 * how large the gradient of the loss is at each step's hidden state — the
 * quantity that vanishes (or explodes) over long sequences.
 */
import { softmax } from '../cnn/forward';
import type { RnnModel } from './model';

export interface RnnStep {
  /** Input symbol id. */
  input: number;
  hPrev: Float64Array;
  /** W_ih x_t + b_ih (all gate rows for the GRU). */
  inputPart: Float64Array;
  /** W_hh h_{t−1} + b_hh (all gate rows for the GRU). */
  recurrentPart: Float64Array;
  /** GRU gates (null for the vanilla RNN). */
  reset: Float64Array | null;
  update: Float64Array | null;
  candidate: Float64Array | null;
  h: Float64Array;
  logits: Float64Array;
  probs: Float64Array;
}

export interface RnnTrace {
  steps: RnnStep[];
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** y = W x + b for a row-major W [rows, cols]; x one-hot when `hot` is given. */
function affine(W: Float64Array, b: Float64Array, rows: number, cols: number, x: Float64Array | null, hot = -1): Float64Array {
  const out = Float64Array.from(b);
  for (let r = 0; r < rows; r++) {
    if (hot >= 0) out[r] += W[r * cols + hot];
    else for (let c = 0; c < cols; c++) out[r] += W[r * cols + c] * x![c];
  }
  return out;
}

export function runRnn(model: RnnModel, inputs: readonly number[], h0?: Float64Array): RnnTrace {
  const { inputSize: I, hiddenSize: H, outputSize: O, cell } = model.config;
  const G = model.gates * H;
  let h: Float64Array = h0 ? Float64Array.from(h0) : new Float64Array(H);
  const steps: RnnStep[] = [];
  for (const input of inputs) {
    const inputPart = affine(model.Wih, model.bih, G, I, null, input);
    const recurrentPart = affine(model.Whh, model.bhh, G, H, h);
    let next: Float64Array;
    let reset: Float64Array | null = null;
    let update: Float64Array | null = null;
    let candidate: Float64Array | null = null;
    if (cell === 'rnn') {
      next = Float64Array.from(inputPart, (v, i) => Math.tanh(v + recurrentPart[i]));
    } else {
      reset = new Float64Array(H);
      update = new Float64Array(H);
      candidate = new Float64Array(H);
      next = new Float64Array(H);
      for (let i = 0; i < H; i++) {
        reset[i] = sigmoid(inputPart[i] + recurrentPart[i]);
        update[i] = sigmoid(inputPart[H + i] + recurrentPart[H + i]);
        candidate[i] = Math.tanh(inputPart[2 * H + i] + reset[i] * recurrentPart[2 * H + i]);
        next[i] = (1 - update[i]) * candidate[i] + update[i] * h[i];
      }
    }
    const logits = affine(model.Wout, model.bout, O, H, next);
    steps.push({ input, hPrev: h, inputPart, recurrentPart, reset, update, candidate, h: next, logits, probs: softmax(logits) });
    h = next;
  }
  return { steps };
}

/**
 * BPTT for cross-entropy at the steps that have a target (target < 0 = no
 * loss at that step), each loss term multiplied by `weight`. Gradients are
 * added into `grads` (layout of model.params). Returns the summed loss and
 * ‖∂L/∂h_t‖ for every step.
 */
export function backwardRnn(model: RnnModel, trace: RnnTrace, targets: readonly number[], grads: Float64Array[], weight = 1): { loss: number; hiddenGradNorms: number[] } {
  const { inputSize: I, hiddenSize: H, outputSize: O, cell } = model.config;
  const G = model.gates * H;
  const [gWih, gWhh, gbih, gbhh, gWout, gbout] = grads;
  const T = trace.steps.length;
  let loss = 0;
  const norms = new Array<number>(T).fill(0);
  let dhNext = new Float64Array(H);
  for (let t = T - 1; t >= 0; t--) {
    const s = trace.steps[t];
    const dh = Float64Array.from(dhNext);
    const target = targets[t];
    if (target !== undefined && target >= 0) {
      loss -= Math.log(Math.max(s.probs[target], 1e-300)) * weight;
      for (let k = 0; k < O; k++) {
        const d = (s.probs[k] - (k === target ? 1 : 0)) * weight;
        gbout[k] += d;
        for (let i = 0; i < H; i++) {
          gWout[k * H + i] += d * s.h[i];
          dh[i] += model.Wout[k * H + i] * d;
        }
      }
    }
    norms[t] = Math.hypot(...dh);
    // gradients of the pre-activations of the input part (dGi) and of the recurrent part (dGh)
    const dGi = new Float64Array(G);
    const dGh = new Float64Array(G);
    const dhPrev = new Float64Array(H);
    if (cell === 'rnn') {
      for (let i = 0; i < H; i++) {
        const dz = dh[i] * (1 - s.h[i] * s.h[i]);
        dGi[i] = dz;
        dGh[i] = dz;
      }
    } else {
      const r = s.reset!;
      const z = s.update!;
      const n = s.candidate!;
      const hn = s.recurrentPart; // rows [2H, 3H) hold W_hn h + b_hn
      for (let i = 0; i < H; i++) {
        const dn = dh[i] * (1 - z[i]);
        const dzGate = dh[i] * (s.hPrev[i] - n[i]);
        dhPrev[i] += dh[i] * z[i];
        const dnPre = dn * (1 - n[i] * n[i]);
        const dr = dnPre * hn[2 * H + i];
        const drPre = dr * r[i] * (1 - r[i]);
        const dzPre = dzGate * z[i] * (1 - z[i]);
        dGi[i] = drPre;
        dGh[i] = drPre;
        dGi[H + i] = dzPre;
        dGh[H + i] = dzPre;
        dGi[2 * H + i] = dnPre;
        dGh[2 * H + i] = dnPre * r[i];
      }
    }
    for (let g = 0; g < G; g++) {
      gbih[g] += dGi[g];
      gWih[g * I + s.input] += dGi[g];
      gbhh[g] += dGh[g];
      for (let j = 0; j < H; j++) {
        gWhh[g * H + j] += dGh[g] * s.hPrev[j];
        dhPrev[j] += model.Whh[g * H + j] * dGh[g];
      }
    }
    dhNext = dhPrev;
  }
  return { loss, hiddenGradNorms: norms };
}

export function zeroRnnGrads(model: RnnModel): Float64Array[] {
  return model.params.map((p) => new Float64Array(p.length));
}

/** Scales all gradients so that their global norm is at most `max` (like torch.nn.utils.clip_grad_norm_). Returns the norm before clipping. */
export function clipGradients(grads: Float64Array[], max: number): number {
  let sum = 0;
  for (const g of grads) for (let i = 0; i < g.length; i++) sum += g[i] * g[i];
  const norm = Math.sqrt(sum);
  if (max > 0 && norm > max) {
    const k = max / (norm + 1e-6);
    for (const g of grads) for (let i = 0; i < g.length; i++) g[i] *= k;
  }
  return norm;
}
