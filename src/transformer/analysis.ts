/**
 * Read-outs of a Transformer trace used by the inspectors: top tokens, logit
 * lens, nearest words, attention-pattern summaries, per-head contributions,
 * the direct vocabulary effect of an MLP unit and PCA projections of
 * embeddings / residual vectors.
 *
 * Several of these are *probes*, not ground truth (logit lens, direct effect,
 * attention summaries, PCA): the UI states their limits next to the numbers.
 */
import { pca, type PCAResult } from '../math/pca';
import { layerNorm, residualAt, stageCount, unembed, type TransformerTrace } from './forward';
import type { TransformerModel } from './model';
import { softmaxWithTemperature } from './sampling';

export interface TokenScore {
  id: number;
  text: string;
  /** Probability (softmax of the logits) or similarity, depending on the function. */
  value: number;
  logit: number;
}

/** The k most likely tokens of a logit vector. */
export function topTokens(model: TransformerModel, logits: ArrayLike<number>, k = 5, temperature = 1): TokenScore[] {
  const probs = softmaxWithTemperature(logits, temperature);
  return Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, k)
    .map((id) => ({ id, text: model.config.vocab[id], value: probs[id], logit: logits[id] }));
}

/**
 * Logit lens: decode an intermediate residual vector as if it were the last
 * one (final LayerNorm + unembedding). Intermediate layers were never trained
 * to be read this way, so this is only a rough probe of what the vector "points to".
 */
export function logitLens(model: TransformerModel, residual: ArrayLike<number>, k = 5): TokenScore[] {
  return topTokens(model, unembed(model, residual), k);
}

export function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  const n = norm(a) * norm(b);
  return n > 0 ? dot / n : 0;
}

/** True for <pad>, <s>, <unk>. */
export function isSpecial(model: TransformerModel, id: number): boolean {
  const word = model.config.vocab[id];
  return word === model.config.pad || word === model.config.bos || word === model.config.unk;
}

/** Ordinary words of the vocabulary (no special tokens). */
export function wordIds(model: TransformerModel): number[] {
  return model.config.vocab.map((_, id) => id).filter((id) => !isSpecial(model, id));
}

/** Words whose embedding points in the most similar direction (cosine similarity in the full d_model space). */
export function nearestWords(model: TransformerModel, vector: ArrayLike<number>, k = 8, exclude: number[] = []): TokenScore[] {
  const d = model.config.dModel;
  return wordIds(model)
    .filter((id) => !exclude.includes(id))
    .map((id) => ({ id, text: model.config.vocab[id], value: cosine(vector, model.tokenEmbedding.subarray(id * d, (id + 1) * d)), logit: 0 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}

// ------------------------------------------------------------- attention

export interface HeadPattern {
  layer: number;
  head: number;
  /** Mean attention to the previous token (queries at positions ≥ 2). */
  previous: number;
  /** Mean attention of a token to itself. */
  self: number;
  /** Mean attention to <s>, the first position. */
  bos: number;
  /** Mean entropy of the attention rows, in bits (0 = looks at one token only). */
  entropy: number;
  /** One-line description of the dominant pattern in this sentence. */
  summary: string;
}

/**
 * Description of one attention matrix (T×T, rows = queries) on one input: mean
 * attention to the previous token, to itself and to the first position, and
 * the mean entropy of the rows (queries at positions ≥ 2). It describes the
 * numbers, not the head's "purpose".
 */
export function describeAttention(a: ArrayLike<number>, T: number, firstName = '<s>'): Omit<HeadPattern, 'layer' | 'head'> {
  let previous = 0;
  let self = 0;
  let bos = 0;
  let entropy = 0;
  let count = 0;
  for (let i = 2; i < T; i++) {
    previous += a[i * T + i - 1];
    self += a[i * T + i];
    bos += a[i * T];
    let h = 0;
    for (let j = 0; j <= i; j++) {
      const p = a[i * T + j];
      if (p > 0) h -= p * Math.log2(p);
    }
    entropy += h;
    count++;
  }
  if (count > 0) {
    previous /= count;
    self /= count;
    bos /= count;
    entropy /= count;
  }
  let summary: string;
  if (count === 0) summary = 'Too few tokens to describe a pattern.';
  else if (previous >= 0.5) summary = `Mostly looks at the previous token (${pct(previous)} on average).`;
  else if (self >= 0.5) summary = `Mostly looks at the current token itself (${pct(self)} on average).`;
  else if (bos >= 0.5) summary = `Mostly rests on ${firstName}, the first position (${pct(bos)} on average): this happens when nothing else is useful to the head.`;
  else if (entropy > 0.75 * Math.log2(Math.max(2, (T + 1) / 2))) summary = 'Spreads its attention over many tokens.';
  else summary = 'Looks at specific earlier words, depending on their content.';
  return { previous, self, bos, entropy, summary };
}

/** Summary of what each head does on this particular input (a description of the numbers, not of the head's "purpose"). */
export function headPatterns(trace: TransformerTrace): HeadPattern[] {
  const T = trace.tokens.length;
  return trace.blocks.flatMap((block, layer) => block.attention.map((a, head) => ({ layer, head, ...describeAttention(a, T) })));
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** ‖contribution‖ of each head of `layer` to the residual stream at `position`. */
export function headContributionNorms(trace: TransformerTrace, layer: number, position: number): number[] {
  return trace.blocks[layer].headContribution.map((perPosition) => norm(perPosition[position]));
}

// ------------------------------------------------------------------- MLP

/**
 * Direct effect of an MLP unit on the vocabulary: its output weights
 * (column `unit` of W_out) projected on the unembedding, through the final
 * LayerNorm's centring and gain but ignoring its normalization and all later
 * layers. A heuristic: it says which words the unit *pushes towards* when it
 * fires positively, not what the unit "means".
 */
export function unitVocabularyEffect(model: TransformerModel, layer: number, unit: number, k = 3): { promoted: TokenScore[]; suppressed: TokenScore[] } {
  const { fc2 } = model.blocks[layer];
  const d = model.config.dModel;
  const w = new Float64Array(d);
  for (let r = 0; r < d; r++) w[r] = fc2.weight[r * fc2.in + unit];
  let mean = 0;
  for (let r = 0; r < d; r++) mean += w[r] / d;
  for (let r = 0; r < d; r++) w[r] = (w[r] - mean) * model.lnFinal.weight[r];
  const scores = wordIds(model).map((id) => {
    let s = 0;
    for (let r = 0; r < d; r++) s += model.tokenEmbedding[id * d + r] * w[r];
    return { id, text: model.config.vocab[id], value: s, logit: s };
  });
  scores.sort((a, b) => b.value - a.value);
  return { promoted: scores.slice(0, k), suppressed: scores.slice(-k).reverse() };
}

/** Units of an MLP layer at a position, sorted by |activation|. */
export function topUnits(trace: TransformerTrace, layer: number, position: number, k = 8): { unit: number; pre: number; act: number }[] {
  const block = trace.blocks[layer];
  const pre = block.mlpPre[position];
  const act = block.mlpAct[position];
  return Array.from(act.keys())
    .sort((a, b) => Math.abs(act[b]) - Math.abs(act[a]))
    .slice(0, k)
    .map((unit) => ({ unit, pre: pre[unit], act: act[unit] }));
}

// ------------------------------------------------------------------- PCA

export interface Projection3D {
  pca: PCAResult;
  /** Share of the total variance kept by each of the 3 components. */
  explained: number[];
}

function projection(points: ArrayLike<number>[]): Projection3D {
  const result = pca(points, 3);
  return { pca: result, explained: result.variances.map((v) => (result.totalVariance > 0 ? v / result.totalVariance : 0)) };
}

export function project3(p: Projection3D, v: ArrayLike<number>): [number, number, number] {
  const { mean, components } = p.pca;
  const out: [number, number, number] = [0, 0, 0];
  components.forEach((c, k) => {
    let s = 0;
    for (let j = 0; j < c.length; j++) s += (v[j] - mean[j]) * c[j];
    out[k] = s;
  });
  return out;
}

/** PCA of the token embeddings of all ordinary words. */
export function vocabularyProjection(model: TransformerModel): Projection3D & { ids: number[] } {
  const d = model.config.dModel;
  const ids = wordIds(model);
  return { ...projection(ids.map((id) => model.tokenEmbedding.subarray(id * d, (id + 1) * d))), ids };
}

export type ResidualSpace = 'raw' | 'normalized';

/**
 * Residual vectors of the sentence at every stage, optionally normalized like
 * LayerNorm does (zero mean, unit variance per vector, no gain/bias): the raw
 * stream grows layer after layer, while each sub-layer reads a normalized copy.
 */
export function residualVectors(trace: TransformerTrace, model: TransformerModel, space: ResidualSpace): Float64Array[][] {
  const unit = { weight: new Float64Array(model.config.dModel).fill(1), bias: new Float64Array(model.config.dModel) };
  const stages = stageCount(model);
  return Array.from({ length: stages }, (_, s) => residualAt(trace, s).map((v) => (space === 'raw' ? v : layerNorm(v, unit, model.config.layerNormEps))));
}

/** PCA fitted on the residual vectors of every stage (positions ≥ `from`), so all stages share one 3-D space. */
export function residualProjection(vectors: Float64Array[][], from = 1): Projection3D {
  const points: Float64Array[] = [];
  for (const stage of vectors) for (let p = from; p < stage.length; p++) points.push(stage[p]);
  return projection(points);
}
