/**
 * Optimizers with PyTorch semantics (torch.optim.SGD / SGD(momentum) / Adam),
 * so the same run can be reproduced in PyTorch (see the parity test).
 */
export type OptimizerId = 'sgd' | 'momentum' | 'adam';

export interface OptimizerSettings {
  id: OptimizerId;
  learningRate: number;
  /** Momentum coefficient μ (momentum optimizer). */
  momentum: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  /** L2 penalty λ added to the gradient (PyTorch `weight_decay`). */
  weightDecay: number;
}

export const DEFAULT_OPTIMIZER: OptimizerSettings = {
  id: 'adam',
  learningRate: 0.03,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  weightDecay: 0,
};

export interface OptimizerInfo {
  id: OptimizerId;
  label: string;
  /** TeX of the update rule. */
  tex: string;
  description: string;
}

export const OPTIMIZERS: Readonly<Record<OptimizerId, OptimizerInfo>> = {
  sgd: {
    id: 'sgd',
    label: 'SGD',
    tex: 'w \\leftarrow w - \\eta\\, \\frac{\\partial L}{\\partial w}',
    description: 'Plain gradient descent: every step moves each parameter against its gradient, scaled by the learning rate η.',
  },
  momentum: {
    id: 'momentum',
    label: 'Momentum',
    tex: 'v \\leftarrow \\mu\\, v + \\frac{\\partial L}{\\partial w},\\qquad w \\leftarrow w - \\eta\\, v',
    description: 'Accumulates a velocity v: consistent gradients speed up, oscillating ones cancel out.',
  },
  adam: {
    id: 'adam',
    label: 'Adam',
    tex: 'm \\leftarrow \\beta_1 m + (1-\\beta_1) g,\\; v \\leftarrow \\beta_2 v + (1-\\beta_2) g^2,\\; w \\leftarrow w - \\eta\\, \\frac{\\hat m}{\\sqrt{\\hat v} + \\epsilon}',
    description: 'Adapts the step size of every parameter using running averages of the gradient (m) and of its square (v).',
  },
};

export class Optimizer {
  readonly settings: OptimizerSettings;
  private buffers: Float64Array[] = [];
  private second: Float64Array[] = [];
  private t = 0;

  constructor(settings: OptimizerSettings) {
    this.settings = { ...settings };
  }

  get steps(): number {
    return this.t;
  }

  reset(): void {
    this.buffers = [];
    this.second = [];
    this.t = 0;
  }

  /** Internal state (step count, momentum / Adam moments) for saving a session. */
  exportState(): { t: number; first: number[][]; second: number[][] } {
    return { t: this.t, first: this.buffers.map((b) => Array.from(b)), second: this.second.map((b) => Array.from(b)) };
  }

  importState(state: { t: number; first: number[][]; second: number[][] }): void {
    this.t = state.t;
    this.buffers = state.first.map((b) => Float64Array.from(b));
    this.second = state.second.map((b) => Float64Array.from(b));
  }

  /** Updates `params[k]` in place using `grads[k]` (same shapes). */
  step(params: Float64Array[], grads: readonly Float64Array[]): void {
    const s = this.settings;
    this.t++;
    if (this.buffers.length !== params.length) {
      this.buffers = params.map((p) => new Float64Array(p.length));
      this.second = params.map((p) => new Float64Array(p.length));
    }
    const lr = s.learningRate;
    const bias1 = 1 - Math.pow(s.beta1, this.t);
    const bias2 = 1 - Math.pow(s.beta2, this.t);
    for (let k = 0; k < params.length; k++) {
      const p = params[k];
      const g = grads[k];
      const m = this.buffers[k];
      const v = this.second[k];
      for (let i = 0; i < p.length; i++) {
        const grad = g[i] + s.weightDecay * p[i];
        switch (s.id) {
          case 'sgd':
            p[i] -= lr * grad;
            break;
          case 'momentum':
            m[i] = this.t === 1 ? grad : s.momentum * m[i] + grad;
            p[i] -= lr * m[i];
            break;
          case 'adam': {
            m[i] = s.beta1 * m[i] + (1 - s.beta1) * grad;
            v[i] = s.beta2 * v[i] + (1 - s.beta2) * grad * grad;
            p[i] -= (lr * (m[i] / bias1)) / (Math.sqrt(v[i] / bias2) + s.epsilon);
            break;
          }
        }
      }
    }
  }
}
