/**
 * Forward pass of the toy Transformer with a complete trace: every vector the
 * visualization may need (Q, K, V per head, scores, attention probabilities,
 * each head's contribution, residual stream before/after each sub-layer, MLP
 * activations, logits).
 *
 *   x₀ = E[token] + P[position]
 *   for each block:   x ← x + Attn(LN₁(x));   x ← x + MLP(LN₂(x))
 *   logits = LN_f(x) · Eᵀ
 *
 * Interventions (used by the Experiments tab) can scale or zero heads and MLP
 * layers, set MLP units or edit embeddings without touching the weights.
 */
import { ACTIVATIONS } from '../math/activations';
import type { Linear, Norm, TransformerModel } from './model';
import type { Token } from './tokenizer';

export interface BlockTrace {
  input: Float64Array[];
  ln1: Float64Array[];
  /** [head][position] → d_head */
  q: Float64Array[][];
  k: Float64Array[][];
  v: Float64Array[][];
  /** [head] → T×T row-major (row = query, column = key); masked entries are −∞. */
  scores: Float64Array[];
  /** [head] → T×T attention probabilities. */
  attention: Float64Array[];
  /** [head][position] → d_head: Σ_j A_ij v_j */
  headOut: Float64Array[][];
  /** [head][position] → d_model: the head's share of the output projection (bias excluded). */
  headContribution: Float64Array[][];
  /** Output of the attention sub-layer (after the projection), per position. */
  attnOut: Float64Array[];
  /** Residual stream after attention: input + attnOut. */
  mid: Float64Array[];
  ln2: Float64Array[];
  mlpPre: Float64Array[];
  mlpAct: Float64Array[];
  mlpOut: Float64Array[];
  /** Residual stream after the block: mid + mlpOut. */
  output: Float64Array[];
}

export interface TransformerTrace {
  tokens: Token[];
  tokenEmbeddings: Float64Array[];
  positionEmbeddings: Float64Array[];
  /** Residual stream entering block 1 (embedding + position). */
  x0: Float64Array[];
  blocks: BlockTrace[];
  /** LN_f(x) per position. */
  final: Float64Array[];
  /** Logits over the vocabulary, per position. */
  logits: Float64Array[];
}

export interface Interventions {
  /** Heads whose output (Σ_j A_ij v_j, before W_O) is multiplied by `scale` (0 = head switched off). */
  heads?: { layer: number; head: number; scale: number }[];
  /** MLP layers whose whole output (bias included) is multiplied by `scale`. */
  mlps?: { layer: number; scale: number }[];
  /** MLP units: value set after the activation (or multiplied by `scale`). */
  neurons?: { layer: number; unit: number; value?: number; scale?: number }[];
  /** Replace the token embedding of a position with another token's embedding, or add a vector. */
  embeddings?: { position: number; replaceWith?: number; add?: ArrayLike<number> }[];
}

export function layerNorm(x: ArrayLike<number>, n: Norm, eps: number): Float64Array {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let variance = 0;
  for (let i = 0; i < d; i++) variance += (x[i] - mean) ** 2;
  variance /= d;
  const inv = 1 / Math.sqrt(variance + eps);
  const out = new Float64Array(d);
  for (let i = 0; i < d; i++) out[i] = (x[i] - mean) * inv * n.weight[i] + n.bias[i];
  return out;
}

export function applyLinear(l: Linear, x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(l.out);
  for (let r = 0; r < l.out; r++) {
    let s = l.bias[r];
    const o = r * l.in;
    for (let c = 0; c < l.in; c++) s += l.weight[o + c] * x[c];
    out[r] = s;
  }
  return out;
}

const add = (a: ArrayLike<number>, b: ArrayLike<number>) => Float64Array.from(a, (v, i) => v + b[i]);

/** Logits of a residual vector: LN_f(x) · Eᵀ. Applied to intermediate layers this is the "logit lens". */
export function unembed(model: TransformerModel, residual: ArrayLike<number>): Float64Array {
  const { config } = model;
  const normalized = layerNorm(residual, model.lnFinal, config.layerNormEps);
  const V = config.vocab.length;
  const d = config.dModel;
  const logits = new Float64Array(V);
  for (let t = 0; t < V; t++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += model.tokenEmbedding[t * d + i] * normalized[i];
    logits[t] = s;
  }
  return logits;
}

export function runTransformer(model: TransformerModel, tokens: Token[], interventions: Interventions = {}): TransformerTrace {
  const { config } = model;
  const d = config.dModel;
  const H = config.nHeads;
  const dh = d / H;
  const T = tokens.length;
  const gelu = ACTIVATIONS.gelu;
  if (gelu.kind !== 'elementwise') throw new Error('GELU must be elementwise');

  const tokenEmbeddings = tokens.map((t, p) => {
    const edit = interventions.embeddings?.find((e) => e.position === p);
    const id = edit?.replaceWith ?? t.id;
    const v = model.tokenEmbedding.slice(id * d, (id + 1) * d);
    if (edit?.add) for (let i = 0; i < d; i++) v[i] += edit.add[i];
    return v;
  });
  const positionEmbeddings = tokens.map((_, p) => model.positionEmbedding.slice(p * d, (p + 1) * d));
  const x0 = tokenEmbeddings.map((e, p) => add(e, positionEmbeddings[p]));

  let x = x0;
  const blocks: BlockTrace[] = [];
  model.blocks.forEach((block, layer) => {
    const ln1 = x.map((v) => layerNorm(v, block.ln1, config.layerNormEps));
    const qkv = ln1.map((v) => applyLinear(block.qkv, v));
    const q: Float64Array[][] = [];
    const k: Float64Array[][] = [];
    const v: Float64Array[][] = [];
    for (let h = 0; h < H; h++) {
      q.push(qkv.map((row) => row.slice(h * dh, (h + 1) * dh)));
      k.push(qkv.map((row) => row.slice(d + h * dh, d + (h + 1) * dh)));
      v.push(qkv.map((row) => row.slice(2 * d + h * dh, 2 * d + (h + 1) * dh)));
    }
    const scale = 1 / Math.sqrt(dh);
    const scores: Float64Array[] = [];
    const attention: Float64Array[] = [];
    const headOut: Float64Array[][] = [];
    for (let h = 0; h < H; h++) {
      const s = new Float64Array(T * T).fill(-Infinity);
      const a = new Float64Array(T * T);
      const out: Float64Array[] = [];
      const headScale = interventions.heads?.find((x) => x.layer === layer && x.head === h)?.scale ?? 1;
      for (let i = 0; i < T; i++) {
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          let dot = 0;
          for (let c = 0; c < dh; c++) dot += q[h][i][c] * k[h][j][c];
          s[i * T + j] = dot * scale;
          if (s[i * T + j] > max) max = s[i * T + j];
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          a[i * T + j] = Math.exp(s[i * T + j] - max);
          sum += a[i * T + j];
        }
        const o = new Float64Array(dh);
        for (let j = 0; j <= i; j++) {
          a[i * T + j] /= sum;
          for (let c = 0; c < dh; c++) o[c] += a[i * T + j] * v[h][j][c];
        }
        if (headScale !== 1) for (let c = 0; c < dh; c++) o[c] *= headScale;
        out.push(o);
      }
      scores.push(s);
      attention.push(a);
      headOut.push(out);
    }
    // Output projection, split by head: proj(concat(heads)) = Σ_h W_O[:, h] · head_h + b.
    const headContribution: Float64Array[][] = [];
    for (let h = 0; h < H; h++) {
      headContribution.push(
        headOut[h].map((o) => {
          const c = new Float64Array(d);
          for (let r = 0; r < d; r++) {
            let s = 0;
            for (let i = 0; i < dh; i++) s += block.proj.weight[r * d + h * dh + i] * o[i];
            c[r] = s;
          }
          return c;
        }),
      );
    }
    const attnOut = Array.from({ length: T }, (_, p) => {
      const out = Float64Array.from(block.proj.bias);
      for (let h = 0; h < H; h++) for (let r = 0; r < d; r++) out[r] += headContribution[h][p][r];
      return out;
    });
    const mid = x.map((row, p) => add(row, attnOut[p]));
    const ln2 = mid.map((row) => layerNorm(row, block.ln2, config.layerNormEps));
    const mlpPre = ln2.map((row) => applyLinear(block.fc1, row));
    const mlpAct = mlpPre.map((row) => Float64Array.from(row, (z) => gelu.f(z)));
    for (const n of interventions.neurons ?? []) {
      if (n.layer !== layer) continue;
      for (const row of mlpAct) row[n.unit] = n.value ?? row[n.unit] * (n.scale ?? 0);
    }
    const mlpScale = interventions.mlps?.find((x) => x.layer === layer)?.scale ?? 1;
    const mlpOut = mlpAct.map((row) => {
      const out = applyLinear(block.fc2, row);
      if (mlpScale !== 1) for (let i = 0; i < d; i++) out[i] *= mlpScale;
      return out;
    });
    const output = mid.map((row, p) => add(row, mlpOut[p]));
    blocks.push({ input: x, ln1, q, k, v, scores, attention, headOut, headContribution, attnOut, mid, ln2, mlpPre, mlpAct, mlpOut, output });
    x = output;
  });

  const final = x.map((row) => layerNorm(row, model.lnFinal, config.layerNormEps));
  const logits = x.map((row) => unembed(model, row));
  return { tokens, tokenEmbeddings, positionEmbeddings, x0, blocks, final, logits };
}

/** Residual stream of every position at a given stage: 0 = embedding, then (attn, mlp) per layer. */
export function residualAt(trace: TransformerTrace, stage: number): Float64Array[] {
  if (stage === 0) return trace.x0;
  const layer = Math.floor((stage - 1) / 2);
  return (stage - 1) % 2 === 0 ? trace.blocks[layer].mid : trace.blocks[layer].output;
}

export function stageCount(model: TransformerModel): number {
  return 1 + 2 * model.config.nLayers;
}

export function stageName(stage: number): string {
  if (stage === 0) return 'Embedding + position';
  const layer = Math.floor((stage - 1) / 2) + 1;
  return (stage - 1) % 2 === 0 ? `Layer ${layer} · after attention` : `Layer ${layer} · after MLP`;
}
