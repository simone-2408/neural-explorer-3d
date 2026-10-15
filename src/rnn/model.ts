/**
 * A recurrent network that reads a sequence one symbol at a time, with the
 * same weights at every step, keeping a hidden state h:
 *
 *   vanilla RNN (Elman):  h_t = tanh(W_ih x_t + b_ih + W_hh h_{t−1} + b_hh)
 *   GRU (gated):          r = σ(W_ir x + b_ir + W_hr h + b_hr)          reset gate
 *                         z = σ(W_iz x + b_iz + W_hz h + b_hz)          update gate
 *                         n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))  candidate
 *                         h_t = (1 − z) ⊙ n + z ⊙ h_{t−1}
 *   output:               logits_t = W_out h_t + b_out
 *
 * These are exactly torch.nn.RNN / torch.nn.GRU (+ nn.Linear), with the same
 * parameter layouts (GRU rows ordered r, z, n), so a model can be checked
 * against PyTorch (tests/pytorch-parity.test.ts). The input x_t is one-hot.
 */
import { createRng, mixSeed } from '../math/random';

export type CellType = 'rnn' | 'gru';

export interface RnnConfig {
  cell: CellType;
  /** Number of input symbols (one-hot size). */
  inputSize: number;
  hiddenSize: number;
  /** Number of output classes. */
  outputSize: number;
  /**
   * GRU only: added to the update gate's input bias at initialization. A
   * positive value makes z ≈ 1 at the start — the cell keeps its state by
   * default — so gradients survive long sequences (0 = PyTorch's default).
   */
  updateGateBias?: number;
}

export const CELLS: Record<CellType, { label: string; description: string }> = {
  rnn: { label: 'Vanilla RNN', description: 'One tanh layer applied at every step: the new state mixes the input with the previous state.' },
  gru: { label: 'GRU', description: 'Gated recurrent unit: an update gate decides, number by number, how much of the old state to keep; a reset gate how much of it to use.' },
};

export class RnnModel {
  readonly config: RnnConfig;
  /** [W_ih, W_hh, b_ih, b_hh, W_out, b_out] — PyTorch's order for nn.RNN / nn.GRU + nn.Linear. */
  readonly params: Float64Array[];
  readonly seed: number;

  constructor(config: RnnConfig, seed = 1, params?: Float64Array[]) {
    this.config = { ...config };
    this.seed = seed;
    this.params = params ?? this.initialize(seed);
  }

  /** Rows of the gate block: 1 for the vanilla RNN, 3 (r, z, n) for the GRU. */
  get gates(): number {
    return this.config.cell === 'gru' ? 3 : 1;
  }

  /** PyTorch's default: U(−1/√hidden, 1/√hidden) for the recurrent layer, U(−1/√fan_in, …) for the output layer. */
  private initialize(seed: number): Float64Array[] {
    const { inputSize: I, hiddenSize: H, outputSize: O } = this.config;
    const G = this.gates * H;
    const uniform = (count: number, bound: number, part: number) => {
      const rng = createRng(mixSeed(seed, part));
      return Float64Array.from({ length: count }, () => rng.uniform(-bound, bound));
    };
    const k = 1 / Math.sqrt(H);
    const params = [uniform(G * I, k, 0), uniform(G * H, k, 1), uniform(G, k, 2), uniform(G, k, 3), uniform(O * H, k, 4), uniform(O, k, 5)];
    const bias = this.config.updateGateBias ?? 0;
    if (this.config.cell === 'gru' && bias) for (let i = H; i < 2 * H; i++) params[2][i] += bias;
    return params;
  }

  get Wih(): Float64Array {
    return this.params[0];
  }

  get Whh(): Float64Array {
    return this.params[1];
  }

  get bih(): Float64Array {
    return this.params[2];
  }

  get bhh(): Float64Array {
    return this.params[3];
  }

  get Wout(): Float64Array {
    return this.params[4];
  }

  get bout(): Float64Array {
    return this.params[5];
  }

  parameterCount(): number {
    return this.params.reduce((n, p) => n + p.length, 0);
  }

  clone(): RnnModel {
    return new RnnModel(this.config, this.seed, this.params.map((p) => Float64Array.from(p)));
  }
}
