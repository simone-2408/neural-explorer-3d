/**
 * Experiments as data: a model, a prompt, the token whose
 * probability is measured, and a list of interventions. The same description
 * runs in the browser on the toy Transformer (toyEngine.ts) and in the Python
 * backend on GPT-2 small (python/server/experiments.py), and is saved and
 * loaded as JSON (serialization.ts).
 *
 * Nothing here changes a model permanently: interventions are applied to one
 * computation and compared with the untouched model.
 */

export type ExperimentModel = 'toy' | 'gpt2';

export interface TokenRef {
  id: number;
  /** Token text, e.g. "paris" (toy) or " Mary" (GPT-2, leading space included). */
  text: string;
}

/**
 * Weight matrices of the toy Transformer that can be edited:
 * token embeddings (tied to the output layer!), position embeddings, the
 * query / key / value / output matrices of one head, the two MLP matrices.
 */
export type TensorId = 'embed' | 'pos' | 'q' | 'k' | 'v' | 'o' | 'mlpIn' | 'mlpOut';

export type InterventionSpec =
  /** Output of one attention head × scale (0 = switched off, 2 = doubled). */
  | { kind: 'head'; layer: number; head: number; scale: number }
  /** Whole output of one MLP layer × scale. */
  | { kind: 'mlp'; layer: number; scale: number }
  /** One MLP neuron after its activation: multiplied by `value` ('scale') or replaced by it ('set'). */
  | { kind: 'neuron'; layer: number; unit: number; mode: 'scale' | 'set'; value: number }
  /** Token embedding at `position` → (1 − mix)·E[original] + mix·E[token]. */
  | { kind: 'embedding'; position: number; token: TokenRef; mix: number }
  /** The token at `position` is removed from the input. */
  | { kind: 'remove'; position: number }
  /** A weight matrix (toy model only) × scale, and optionally one entry set to a value. */
  | { kind: 'weight'; tensor: TensorId; layer: number; head: number; scale: number; cell: { row: number; col: number; value: number } | null };

export type InterventionKind = InterventionSpec['kind'];

export type Intervention = InterventionSpec & {
  id: string;
  /** Disabled interventions stay in the list but are not applied. */
  enabled: boolean;
};

export interface Experiment {
  model: ExperimentModel;
  text: string;
  /** Token measured at the last position; null = the original model's most likely next token. */
  target: TokenRef | null;
  /** Optional second token: the logit difference target − contrast is then measured too. */
  contrast: TokenRef | null;
  interventions: Intervention[];
}

/** A component that can be switched off or scaled. */
export type ComponentRef = { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number } | { kind: 'neuron'; layer: number; unit: number };

// ------------------------------------------------------------------ results

/** How the next-token prediction changed between two runs. */
export interface Effect {
  /** Change of log p(target) (natural log). */
  dlogp: number;
  /** Change of p(target). */
  dp: number;
  /** KL(before ‖ after) of the next-token distributions, in bits (0 = identical). */
  kl: number;
  /** Change of logit(target) − logit(contrast); null without a contrast token. */
  dlogit: number | null;
}

export type MetricId = 'logp' | 'logit' | 'p' | 'kl';

export interface Prediction {
  id: number;
  text: string;
  p: number;
}

/** What one run predicts. */
export interface Outcome {
  /** Next-token prediction (last position): the 10 most likely tokens. */
  top: Prediction[];
  pTarget: number;
  logitTarget: number;
  logitContrast: number | null;
  /** The most likely next token at every position of this run's input. */
  predictions: Prediction[];
}

export type ResultSource = 'browser' | 'live' | 'recorded';

export interface ExperimentResult {
  model: ExperimentModel;
  /** Tokens of the original text, and of the input actually run (after removals). */
  tokens: TokenRef[];
  tokensAfter: TokenRef[];
  /** For each position of the edited input, its position in the original input. */
  alignment: number[];
  truncated: boolean;
  target: TokenRef;
  contrast: TokenRef | null;
  /** The untouched model, and the model with the interventions. */
  before: Outcome;
  after: Outcome;
  effect: Effect;
  /**
   * Relative change ‖x_after − x_before‖ / ‖x_before‖ of the residual stream,
   * [stage][original position]: stage 0 = embeddings, 2l + 1 = after the
   * attention of layer l, 2l + 2 = after its MLP. null = removed token.
   */
  delta: (number | null)[][];
  layers: number;
  heads: number;
  source: ResultSource;
  seconds?: number;
}

/** Effect of zeroing each component alone, on top of a baseline. */
export interface SweepResult {
  /** [layer][head] */
  heads: Effect[][];
  /** [layer] */
  mlps: Effect[];
  /** The original model, or the model with the experiment's interventions. */
  baseline: 'original' | 'experiment';
  target: TokenRef;
  contrast: TokenRef | null;
  /** p(target) in the baseline run. */
  p: number;
  source: ResultSource;
  seconds?: number;
}

export interface VocabularyEffect {
  promoted: { id: number; text: string; value: number }[];
  suppressed: { id: number; text: string; value: number }[];
}

/** The MLP neurons of one layer: linear estimate vs measured effect of zeroing them. */
export interface NeuronScan {
  layer: number;
  /** Metric of the estimates: change of log p(target), or of the logit difference. */
  metric: 'logp' | 'logitdiff';
  /** First-order estimate of the effect of zeroing each unit, for every unit of the layer. */
  estimate: number[];
  /** Measured effect of zeroing units (all of them for the toy model, the strongest estimates for GPT-2). */
  measured: { unit: number; effect: Effect }[];
  /** Activation of the measured units at each position of the input that was run. */
  activations: Record<number, number[]>;
  /** Direct effect of the measured units on the output vocabulary (GPT-2; computed on demand for the toy model). */
  vocab: Record<number, VocabularyEffect>;
  source: ResultSource;
  seconds?: number;
}

/** p(target) (and the logit difference) while one component is scaled, the other interventions in place. */
export interface ScaleCurve {
  component: ComponentRef;
  scales: number[];
  p: number[];
  logitDiff: number[] | null;
}

/** Attention of one head, without and with the interventions. */
export interface AttentionPair {
  /** T×T rows (query) of key weights. */
  before: number[][];
  after: number[][];
  alignment: number[];
}
