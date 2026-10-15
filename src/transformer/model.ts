/**
 * The tiny educational Transformer: loading of the weights trained
 * in PyTorch (python/toy_transformer/train.py) into flat Float64Arrays.
 *
 * Shapes follow PyTorch nn.Linear: weight [out, in] row-major.
 */

export interface TransformerConfig {
  vocab: string[];
  dModel: number;
  nHeads: number;
  nLayers: number;
  dMlp: number;
  context: number;
  layerNormEps: number;
  bos: string;
  pad: string;
  unk: string;
}

export interface Linear {
  weight: Float64Array;
  bias: Float64Array;
  out: number;
  in: number;
}

export interface Norm {
  weight: Float64Array;
  bias: Float64Array;
}

export interface BlockWeights {
  ln1: Norm;
  /** Q, K and V projections stacked: rows [0, d) = Q, [d, 2d) = K, [2d, 3d) = V; head h uses rows h·d_h … */
  qkv: Linear;
  proj: Linear;
  ln2: Norm;
  fc1: Linear;
  fc2: Linear;
}

export interface TransformerModel {
  config: TransformerConfig;
  /** [vocab, dModel] — also the output (unembedding) matrix (tied embeddings). */
  tokenEmbedding: Float64Array;
  /** [context, dModel] */
  positionEmbedding: Float64Array;
  blocks: BlockWeights[];
  lnFinal: Norm;
  /** Word → category (country, capital, animal…), for colouring the embedding space. */
  groups: Record<string, string>;
  training: {
    sentences: number;
    steps: number;
    finalLoss: number;
    parameters: number;
    templates: string[];
    probes: { text: string; top: [string, number][] }[];
  };
  index: Map<string, number>;
}

export interface TransformerFile {
  format: string;
  formatVersion: number;
  config: TransformerConfig;
  weights: {
    tokenEmbedding: number[][];
    positionEmbedding: number[][];
    blocks: Record<'ln1' | 'qkv' | 'proj' | 'ln2' | 'fc1' | 'fc2', { weight: number[] | number[][]; bias: number[] }>[];
    lnFinal: { weight: number[]; bias: number[] };
  };
  groups: Record<string, string>;
  training: TransformerModel['training'];
  reference: { text: string; ids: number[]; logits: number[]; attention: number[][] }[];
}

const flat = (rows: number[][]) => Float64Array.from(rows.flat());

function linear(w: { weight: number[] | number[][]; bias: number[] }): Linear {
  const rows = w.weight as number[][];
  return { weight: flat(rows), bias: Float64Array.from(w.bias), out: rows.length, in: rows[0].length };
}

function norm(w: { weight: number[] | number[][]; bias: number[] }): Norm {
  return { weight: Float64Array.from(w.weight as number[]), bias: Float64Array.from(w.bias) };
}

export function loadTransformer(file: TransformerFile): TransformerModel {
  if (file.format !== 'neural-explorer-3d/toy-transformer') throw new Error(`Unknown model format ${file.format}`);
  const { config, weights } = file;
  return {
    config,
    tokenEmbedding: flat(weights.tokenEmbedding),
    positionEmbedding: flat(weights.positionEmbedding),
    blocks: weights.blocks.map((b) => ({
      ln1: norm(b.ln1),
      qkv: linear(b.qkv),
      proj: linear(b.proj),
      ln2: norm(b.ln2),
      fc1: linear(b.fc1),
      fc2: linear(b.fc2),
    })),
    lnFinal: norm(weights.lnFinal),
    groups: file.groups,
    training: file.training,
    index: new Map(config.vocab.map((w, i) => [w, i])),
  };
}

export function parameterCount(model: TransformerModel): number {
  const { config } = model;
  let total = model.tokenEmbedding.length + model.positionEmbedding.length + 2 * config.dModel;
  for (const b of model.blocks) {
    for (const l of [b.qkv, b.proj, b.fc1, b.fc2]) total += l.weight.length + l.bias.length;
    total += 4 * config.dModel;
  }
  return total;
}

/** Embedding vector of a vocabulary entry (a view, not a copy). */
export function embeddingOf(model: TransformerModel, id: number): Float64Array {
  const d = model.config.dModel;
  return model.tokenEmbedding.subarray(id * d, (id + 1) * d);
}
