/**
 * What an experiment measures: the change of the next-token prediction
 * between a baseline run and a run with interventions. Same definitions as
 * python/server/experiments.py (`effect`).
 */
import type { Effect, MetricId } from './types';

/** log softmax (natural log), numerically stable. */
export function logSoftmax(logits: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const lse = max + Math.log(sum);
  return Float64Array.from(logits, (v) => v - lse);
}

/** Change from `base` to `run` (both log-probability vectors of the next token). */
export function effectBetween(base: ArrayLike<number>, run: ArrayLike<number>, target: number, contrast: number | null): Effect {
  let kl = 0;
  for (let i = 0; i < base.length; i++) {
    const p = Math.exp(base[i]);
    if (p > 0) kl += p * (base[i] - run[i]);
  }
  return {
    dlogp: run[target] - base[target],
    dp: Math.exp(run[target]) - Math.exp(base[target]),
    kl: Math.max(0, kl / Math.LN2),
    dlogit: contrast === null ? null : run[target] - run[contrast] - (base[target] - base[contrast]),
  };
}

export interface MetricInfo {
  id: MetricId;
  /** Button label. */
  label: string;
  /** Formula-like name, e.g. "Δ log p". */
  symbol: string;
  /** True when the metric has a sign (KL does not). */
  signed: boolean;
  description: string;
}

export const METRICS: Record<MetricId, MetricInfo> = {
  logp: {
    id: 'logp',
    label: 'Δ log p',
    symbol: 'Δ log p',
    signed: true,
    description: 'Change of the log-probability of the target token. −0.69 halves its probability, +0.69 doubles it.',
  },
  logit: {
    id: 'logit',
    label: 'Δ logit diff',
    symbol: 'Δ(logit t − logit c)',
    signed: true,
    description: 'Change of the logit difference between the target and the contrast token: how much the model prefers one over the other (independent of all other tokens).',
  },
  p: {
    id: 'p',
    label: 'Δ p',
    symbol: 'Δ p',
    signed: true,
    description: 'Change of the probability of the target token, in percentage points.',
  },
  kl: {
    id: 'kl',
    label: 'KL',
    symbol: 'KL',
    signed: false,
    description: 'Kullback–Leibler divergence between the next-token distributions before and after, in bits: how much the whole prediction changed, whatever the target.',
  },
};

/** The value of one metric (null when it needs a contrast token that is not set). */
export function metricValue(effect: Effect, metric: MetricId): number | null {
  switch (metric) {
    case 'logp':
      return effect.dlogp;
    case 'logit':
      return effect.dlogit;
    case 'p':
      return effect.dp;
    case 'kl':
      return effect.kl;
  }
}

/**
 * How much the target relies on a component, from the effect of zeroing it:
 * positive when removing the component lowers the target (it supported it),
 * negative when removing it raises the target. KL has no sign.
 */
export function support(effect: Effect, metric: MetricId): number | null {
  const v = metricValue(effect, metric);
  if (v === null) return null;
  return METRICS[metric].signed ? -v : v;
}
