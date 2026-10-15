/**
 * Experiments on the toy Transformer, computed in the browser: one forward
 * pass takes about a millisecond, so every slider move is recomputed exactly
 * (no approximation, no backend).
 *
 *   original text ──tokenize──► tokens ──► runTransformer(model)              = before
 *                                  │ remove tokens
 *                                  ▼
 *                    edited tokens ──► runTransformer(model with weight edits,
 *                                                     interventions)          = after
 */
import { isSpecial, unitVocabularyEffect } from '../transformer/analysis';
import { residualAt, runTransformer, stageCount, type Interventions, type TransformerTrace } from '../transformer/forward';
import type { Linear, TransformerModel } from '../transformer/model';
import { tokenize, type Token } from '../transformer/tokenizer';
import { effectBetween, logSoftmax } from './metrics';
import type { ComponentRef, Effect, Experiment, ExperimentResult, InterventionSpec, NeuronScan, Outcome, Prediction, ScaleCurve, SweepResult, TensorId, TokenRef, VocabularyEffect } from './types';

type WeightEdit = Extract<InterventionSpec, { kind: 'weight' }>;

// ------------------------------------------------------------------ weights

export interface TensorInfo {
  tensor: TensorId;
  layer: number;
  head: number;
  /** "Layer 1 · head 2 · W_Q" */
  label: string;
  rows: number;
  cols: number;
  /** What rows and columns index, e.g. "query dimension" / "input dimension". */
  rowName: string;
  colName: string;
  rowLabel(row: number): string;
  colLabel(col: number): string;
  /** Index of entry (row, col) in the flat array. */
  index(row: number, col: number): number;
  array(model: TransformerModel): Float64Array;
  /** What this matrix does, in one sentence. */
  role: string;
}

export const TENSORS: { id: TensorId; label: string; perHead: boolean; perLayer: boolean }[] = [
  { id: 'q', label: 'W_Q — queries of a head', perHead: true, perLayer: true },
  { id: 'k', label: 'W_K — keys of a head', perHead: true, perLayer: true },
  { id: 'v', label: 'W_V — values of a head', perHead: true, perLayer: true },
  { id: 'o', label: 'W_O — output of a head', perHead: true, perLayer: true },
  { id: 'mlpIn', label: 'MLP W_in', perHead: false, perLayer: true },
  { id: 'mlpOut', label: 'MLP W_out', perHead: false, perLayer: true },
  { id: 'pos', label: 'Position embeddings', perHead: false, perLayer: false },
  { id: 'embed', label: 'Token embeddings (tied to the output)', perHead: false, perLayer: false },
];

export function tensorInfo(model: TransformerModel, tensor: TensorId, layer: number, head: number): TensorInfo {
  const { dModel: d, nHeads, dMlp, vocab, context } = model.config;
  const dh = d / nHeads;
  const dim = (i: number) => `dim ${i + 1}`;
  const L = `Layer ${layer + 1}`;
  switch (tensor) {
    case 'embed':
      return {
        tensor, layer: 0, head: 0, label: 'Token embeddings E', rows: vocab.length, cols: d, rowName: 'word', colName: 'dimension',
        rowLabel: (r) => vocab[r], colLabel: dim, index: (r, c) => r * d + c, array: (m) => m.tokenEmbedding,
        role: 'One row per word: the vector a word starts as. The same matrix also turns the last vector into scores for every word (tied embeddings), so editing it changes both the input and the output side.',
      };
    case 'pos':
      return {
        tensor, layer: 0, head: 0, label: 'Position embeddings P', rows: context, cols: d, rowName: 'position', colName: 'dimension',
        rowLabel: (r) => `pos ${r}`, colLabel: dim, index: (r, c) => r * d + c, array: (m) => m.positionEmbedding,
        role: 'One row per position, added to the token embedding. It is the model’s only source of word order: attention by itself does not know which token comes first.',
      };
    case 'q':
    case 'k':
    case 'v': {
      const offset = (tensor === 'q' ? 0 : tensor === 'k' ? d : 2 * d) + head * dh;
      const name = { q: 'query', k: 'key', v: 'value' }[tensor];
      return {
        tensor, layer, head, label: `${L} · head ${head + 1} · W_${tensor.toUpperCase()}`, rows: dh, cols: d, rowName: `${name} dimension`, colName: 'input dimension',
        rowLabel: (r) => `${tensor}${r + 1}`, colLabel: dim, index: (r, c) => (offset + r) * d + c, array: (m) => m.blocks[layer].qkv.weight,
        role: {
          q: 'Turns each token’s vector into the question it asks: what it is looking for.',
          k: 'Turns each token’s vector into the label it offers: what it can be found by. Attention scores are query · key.',
          v: 'Turns each token’s vector into the content it passes on when it is attended to.',
        }[tensor],
      };
    }
    case 'o':
      return {
        tensor, layer, head, label: `${L} · head ${head + 1} · W_O`, rows: d, cols: dh, rowName: 'output dimension', colName: 'head dimension',
        rowLabel: dim, colLabel: (c) => `h${c + 1}`, index: (r, c) => r * d + head * dh + c, array: (m) => m.blocks[layer].proj.weight,
        role: 'Writes the head’s result back into the 32-number residual stream. Scaling it to 0 is the same as switching the head off.',
      };
    case 'mlpIn':
      return {
        tensor, layer, head: 0, label: `${L} · MLP W_in`, rows: dMlp, cols: d, rowName: 'neuron', colName: 'input dimension',
        rowLabel: (r) => `n${r}`, colLabel: dim, index: (r, c) => r * d + c, array: (m) => m.blocks[layer].fc1.weight,
        role: 'Each row is one neuron’s detector: the neuron fires (before GELU) when the vector points in the direction of its row.',
      };
    case 'mlpOut':
      return {
        tensor, layer, head: 0, label: `${L} · MLP W_out`, rows: d, cols: dMlp, rowName: 'output dimension', colName: 'neuron',
        rowLabel: dim, colLabel: (c) => `n${c}`, index: (r, c) => r * dMlp + c, array: (m) => m.blocks[layer].fc2.weight,
        role: 'Each column is what one neuron writes into the residual stream when it fires.',
      };
  }
}

/** A copy of the model with the weight edits applied (only the edited arrays are copied). */
export function applyWeightEdits(model: TransformerModel, edits: WeightEdit[]): TransformerModel {
  if (!edits.length) return model;
  const copy: TransformerModel = { ...model, blocks: model.blocks.map((b) => ({ ...b })) };
  const cloned = new Set<Float64Array>();
  const writable = (info: TensorInfo): Float64Array => {
    const current = info.array(copy);
    if (cloned.has(current)) return current;
    const array = Float64Array.from(current);
    cloned.add(array);
    const block = copy.blocks[info.layer];
    const swap = (l: Linear): Linear => ({ ...l, weight: array });
    switch (info.tensor) {
      case 'embed':
        copy.tokenEmbedding = array;
        break;
      case 'pos':
        copy.positionEmbedding = array;
        break;
      case 'q':
      case 'k':
      case 'v':
        block.qkv = swap(block.qkv);
        break;
      case 'o':
        block.proj = swap(block.proj);
        break;
      case 'mlpIn':
        block.fc1 = swap(block.fc1);
        break;
      case 'mlpOut':
        block.fc2 = swap(block.fc2);
        break;
    }
    return array;
  };
  for (const edit of edits) {
    const info = tensorInfo(model, edit.tensor, edit.layer, edit.head);
    const array = writable(info);
    if (edit.scale !== 1) for (let r = 0; r < info.rows; r++) for (let c = 0; c < info.cols; c++) array[info.index(r, c)] *= edit.scale;
    const cell = edit.cell;
    if (cell && cell.row >= 0 && cell.row < info.rows && cell.col >= 0 && cell.col < info.cols) array[info.index(cell.row, cell.col)] = cell.value;
  }
  return copy;
}

// --------------------------------------------------------------- preparing

/** An experiment translated into one forward pass of the toy model. */
export interface PreparedRun {
  model: TransformerModel;
  /** Tokens of the original text. */
  original: Token[];
  /** Tokens actually run (removed tokens left out, positions renumbered). */
  tokens: Token[];
  alignment: number[];
  interventions: Interventions;
}

export function prepare(model: TransformerModel, text: string, items: readonly InterventionSpec[]): PreparedRun {
  const original = tokenize(model, text);
  const T = original.length;
  const edited = applyWeightEdits(
    model,
    items.filter((i): i is WeightEdit => i.kind === 'weight'),
  );
  // The <s> token (position 0) starts every training sentence: it is never removed.
  const removed = new Set(items.flatMap((i) => (i.kind === 'remove' && i.position > 0 && i.position < T ? [i.position] : [])));
  const alignment = original.map((_, p) => p).filter((p) => !removed.has(p));
  const tokens = alignment.map((p, j) => ({ ...original[p], position: j }));

  const heads = new Map<string, { layer: number; head: number; scale: number }>();
  const mlps = new Map<number, number>();
  const neurons: NonNullable<Interventions['neurons']> = [];
  const embeddings = new Map<number, Float64Array>();
  const d = model.config.dModel;
  for (const item of items) {
    switch (item.kind) {
      case 'head': {
        const key = `${item.layer}.${item.head}`;
        const entry = heads.get(key) ?? { layer: item.layer, head: item.head, scale: 1 };
        entry.scale *= item.scale;
        heads.set(key, entry);
        break;
      }
      case 'mlp':
        mlps.set(item.layer, (mlps.get(item.layer) ?? 1) * item.scale);
        break;
      case 'neuron':
        neurons.push(item.mode === 'set' ? { layer: item.layer, unit: item.unit, value: item.value } : { layer: item.layer, unit: item.unit, scale: item.value });
        break;
      case 'embedding': {
        const j = alignment.indexOf(item.position);
        if (j < 0 || item.token.id < 0 || item.token.id >= model.config.vocab.length) break;
        const from = original[item.position].id;
        const add = new Float64Array(d);
        for (let i = 0; i < d; i++) add[i] = item.mix * (edited.tokenEmbedding[item.token.id * d + i] - edited.tokenEmbedding[from * d + i]);
        embeddings.set(j, add);
        break;
      }
    }
  }
  return {
    model: edited,
    original,
    tokens,
    alignment,
    interventions: {
      heads: [...heads.values()],
      mlps: [...mlps].map(([layer, scale]) => ({ layer, scale })),
      neurons,
      embeddings: [...embeddings].map(([position, add]) => ({ position, add })),
    },
  };
}

/** The interventions with one more component scaled (multiplying any scale it already has). */
function withScaled(interventions: Interventions, c: ComponentRef, scale: number): Interventions {
  switch (c.kind) {
    case 'head': {
      const heads = (interventions.heads ?? []).map((h) => (h.layer === c.layer && h.head === c.head ? { ...h, scale: h.scale * scale } : h));
      if (!heads.some((h) => h.layer === c.layer && h.head === c.head)) heads.push({ layer: c.layer, head: c.head, scale });
      return { ...interventions, heads };
    }
    case 'mlp': {
      const mlps = (interventions.mlps ?? []).map((m) => (m.layer === c.layer ? { ...m, scale: m.scale * scale } : m));
      if (!mlps.some((m) => m.layer === c.layer)) mlps.push({ layer: c.layer, scale });
      return { ...interventions, mlps };
    }
    case 'neuron':
      // Neuron edits are applied in order: this one acts on top of the others.
      return { ...interventions, neurons: [...(interventions.neurons ?? []), { layer: c.layer, unit: c.unit, scale }] };
  }
}

export function sameComponent(item: InterventionSpec, c: ComponentRef): boolean {
  if (item.kind !== c.kind) return false;
  if (c.kind === 'head') return item.kind === 'head' && item.layer === c.layer && item.head === c.head;
  if (c.kind === 'mlp') return item.kind === 'mlp' && item.layer === c.layer;
  return item.kind === 'neuron' && item.layer === c.layer && item.unit === c.unit;
}

// ----------------------------------------------------------------- running

export interface ToyExperimentRun {
  result: ExperimentResult;
  before: TransformerTrace;
  after: TransformerTrace;
  prepared: PreparedRun;
}

/** Most likely next token at the last position, special tokens (<s>, <unk>, <pad>) excluded. */
export function defaultTarget(model: TransformerModel, logits: ArrayLike<number>): number {
  let best = -1;
  for (let id = 0; id < logits.length; id++) if (!isSpecial(model, id) && (best < 0 || logits[id] > logits[best])) best = id;
  return best;
}

function ref(model: TransformerModel, id: number): TokenRef {
  return { id, text: model.config.vocab[id] };
}

function outcome(model: TransformerModel, trace: TransformerTrace, target: number, contrast: number | null): Outcome {
  const T = trace.tokens.length;
  const last = trace.logits[T - 1];
  const lp = logSoftmax(last);
  const top: Prediction[] = Array.from(lp.keys())
    .sort((a, b) => lp[b] - lp[a])
    .slice(0, 10)
    .map((id) => ({ id, text: model.config.vocab[id], p: Math.exp(lp[id]) }));
  const predictions = trace.logits.map((logits) => {
    const l = logSoftmax(logits);
    let best = 0;
    for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
    return { id: best, text: model.config.vocab[best], p: Math.exp(l[best]) };
  });
  return { top, pTarget: Math.exp(lp[target]), logitTarget: last[target], logitContrast: contrast === null ? null : last[contrast], predictions };
}

function relativeChange(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let diff = 0;
  let base = 0;
  for (let i = 0; i < a.length; i++) {
    diff += (b[i] - a[i]) ** 2;
    base += a[i] ** 2;
  }
  return Math.sqrt(diff) / Math.max(Math.sqrt(base), 1e-9);
}

export function runToyExperiment(model: TransformerModel, experiment: Experiment): ToyExperimentRun {
  const active = experiment.interventions.filter((i) => i.enabled);
  const prepared = prepare(model, experiment.text, active);
  const before = runTransformer(model, prepared.original);
  const after = runTransformer(prepared.model, prepared.tokens, prepared.interventions);
  const T = prepared.original.length;
  const target = experiment.target && experiment.target.id < model.config.vocab.length ? experiment.target.id : defaultTarget(model, before.logits[T - 1]);
  const contrast = experiment.contrast && experiment.contrast.id !== target ? experiment.contrast.id : null;
  const stages = stageCount(model);
  const delta: (number | null)[][] = [];
  for (let s = 0; s < stages; s++) {
    const x0 = residualAt(before, s);
    const x1 = residualAt(after, s);
    delta.push(prepared.original.map((_, p) => {
      const j = prepared.alignment.indexOf(p);
      return j < 0 ? null : relativeChange(x0[p], x1[j]);
    }));
  }
  const lp0 = logSoftmax(before.logits[T - 1]);
  const lp1 = logSoftmax(after.logits[after.logits.length - 1]);
  const result: ExperimentResult = {
    model: 'toy',
    tokens: prepared.original.map((t) => ({ id: t.id, text: t.text })),
    tokensAfter: prepared.tokens.map((t) => ({ id: t.id, text: t.text })),
    alignment: prepared.alignment,
    truncated: false,
    target: ref(model, target),
    contrast: contrast === null ? null : ref(model, contrast),
    before: outcome(model, before, target, contrast),
    after: outcome(prepared.model, after, target, contrast),
    effect: effectBetween(lp0, lp1, target, contrast),
    delta,
    layers: model.config.nLayers,
    heads: model.config.nHeads,
    source: 'browser',
  };
  return { result, before, after, prepared };
}

// ---------------------------------------------------------------- analyses

interface Baseline {
  prepared: PreparedRun;
  trace: TransformerTrace;
  lp: Float64Array;
  target: number;
  contrast: number | null;
}

function baselineRun(model: TransformerModel, experiment: Experiment, result: ExperimentResult, withInterventions: boolean, skip?: ComponentRef): Baseline {
  const items = withInterventions ? experiment.interventions.filter((i) => i.enabled && !(skip && sameComponent(i, skip))) : [];
  const prepared = prepare(model, experiment.text, items);
  const trace = runTransformer(prepared.model, prepared.tokens, prepared.interventions);
  return { prepared, trace, lp: logSoftmax(trace.logits[trace.logits.length - 1]), target: result.target.id, contrast: result.contrast?.id ?? null };
}

function lastLogProbs(base: Baseline, interventions: Interventions): Float64Array {
  const trace = runTransformer(base.prepared.model, base.prepared.tokens, interventions);
  return logSoftmax(trace.logits[trace.logits.length - 1]);
}

/** Zero each head alone and each MLP layer alone, on top of the original model or of the experiment. */
export function toySweep(model: TransformerModel, experiment: Experiment, result: ExperimentResult, baseline: 'original' | 'experiment'): SweepResult {
  const base = baselineRun(model, experiment, result, baseline === 'experiment');
  const { nLayers, nHeads } = model.config;
  const effectOf = (c: ComponentRef): Effect => effectBetween(base.lp, lastLogProbs(base, withScaled(base.prepared.interventions, c, 0)), base.target, base.contrast);
  return {
    heads: Array.from({ length: nLayers }, (_, layer) => Array.from({ length: nHeads }, (_, head) => effectOf({ kind: 'head', layer, head }))),
    mlps: Array.from({ length: nLayers }, (_, layer) => effectOf({ kind: 'mlp', layer })),
    baseline,
    target: result.target,
    contrast: result.contrast,
    p: Math.exp(base.lp[base.target]),
    source: 'browser',
  };
}

/**
 * Every MLP neuron of one layer: measured effect of zeroing it, and the
 * first-order estimate of that effect (the slope of the metric at the
 * neuron's current value, extrapolated to zero — what a gradient-based method
 * such as attribution patching would report).
 */
export function toyNeuronScan(model: TransformerModel, experiment: Experiment, result: ExperimentResult, baseline: 'original' | 'experiment', layer: number): NeuronScan {
  const base = baselineRun(model, experiment, result, baseline === 'experiment');
  const metric = base.contrast === null ? 'logp' : 'logitdiff';
  const value = (lp: Float64Array) => (base.contrast === null ? lp[base.target] : lp[base.target] - lp[base.contrast]);
  const m0 = value(base.lp);
  const eps = 1e-4;
  const units = model.config.dMlp;
  const estimate: number[] = [];
  const measured: NeuronScan['measured'] = [];
  const activations: Record<number, number[]> = {};
  const act = base.trace.blocks[layer].mlpAct;
  for (let unit = 0; unit < units; unit++) {
    const c: ComponentRef = { kind: 'neuron', layer, unit };
    const nudged = value(lastLogProbs(base, withScaled(base.prepared.interventions, c, 1 - eps)));
    estimate.push(-(m0 - nudged) / eps);
    measured.push({ unit, effect: effectBetween(base.lp, lastLogProbs(base, withScaled(base.prepared.interventions, c, 0)), base.target, base.contrast) });
    activations[unit] = act.map((row) => row[unit]);
  }
  return { layer, metric, estimate, measured, activations, vocab: {}, source: 'browser' };
}

/** Direct effect of a toy MLP unit on the output words (through W_out and the unembedding; later layers ignored). */
export function toyVocabularyEffect(model: TransformerModel, layer: number, unit: number): VocabularyEffect {
  const { promoted, suppressed } = unitVocabularyEffect(model, layer, unit, 5);
  return {
    promoted: promoted.map((t) => ({ id: t.id, text: t.text, value: t.value })),
    suppressed: suppressed.map((t) => ({ id: t.id, text: t.text, value: t.value })),
  };
}

/** p(target) while one component is scaled, the other interventions of the experiment in place. */
export function toyCurve(model: TransformerModel, experiment: Experiment, result: ExperimentResult, component: ComponentRef, scales: number[]): ScaleCurve {
  const base = baselineRun(model, experiment, result, true, component);
  const p: number[] = [];
  const logitDiff: number[] = [];
  for (const s of scales) {
    const lp = lastLogProbs(base, withScaled(base.prepared.interventions, component, s));
    p.push(Math.exp(lp[base.target]));
    if (base.contrast !== null) logitDiff.push(lp[base.target] - lp[base.contrast]);
  }
  return { component, scales, p, logitDiff: base.contrast === null ? null : logitDiff };
}
