/** Data exchanged with the Python backend (python/server) — see python/server/llm.py. */

export interface LlmConfig {
  layers: number;
  heads: number;
  dModel: number;
  dMlp: number;
  vocab: number;
  context: number;
  parameters: number;
  maxTokens: number;
}

export interface LlmStatus {
  model: string;
  ready: boolean;
  loading: boolean;
  error: string | null;
  config?: LlmConfig;
  loadSeconds?: number;
}

export interface LlmToken {
  id: number;
  /** Decoded text, e.g. " capital" (leading space included). */
  text: string;
  /** Byte-level BPE piece, e.g. "Ġcapital" (Ġ = a space before the word). */
  piece: string;
  /** The token's raw UTF-8 bytes, in hex ("20 63 61 …"). */
  bytes: string;
}

export interface LensLayer {
  /** [position][rank] → [token id, probability] */
  top: [number, number][][];
  /** Probability, under this layer's lens, of the token the model finally predicts at each position. */
  finalProb: number[];
  /** Rank (0 = first) of that token under this layer's lens. */
  finalRank: number[];
}

/** JSON as sent by the server. */
export interface LlmAnalysisJson {
  model: string;
  tokens: LlmToken[];
  truncated: boolean;
  /** Next-token logits of the last position, float32 little-endian, base64. */
  logits: string;
  nextTop: [number, number][];
  /** Model's top prediction (id, probability) at each position. */
  predictions: [number, number][];
  lens: LensLayer[];
  words: Record<string, string>;
  /** layers × heads × T × T attention probabilities, uint16 (p · 65535), base64. */
  attention: string;
  /** ‖residual‖ per stage (0 = embeddings … 12 = after the last block) and position. */
  norms: number[][];
  journey: { coords: number[][][]; explained: number[] };
  rollout: number[][];
  saliency: { target: number; values: number[] };
  seconds: number;
  /** Recorded continuations (demo files only). */
  demo?: { greedy: GenerateResult; sampled: GenerateResult };
}

export interface LlmAnalysis extends Omit<LlmAnalysisJson, 'logits' | 'attention'> {
  T: number;
  layers: number;
  heads: number;
  logits: Float32Array;
  /** Flattened [layer][head][query][key]. */
  attention: Float32Array;
  /** Where the analysis came from. */
  source: 'live' | 'demo';
}

export interface GenerateStep {
  id: number;
  text: string;
  p: number;
  alternatives: [number, string, number][];
}

export interface GenerateResult {
  steps: GenerateStep[];
  text: string;
}

export interface DemoEntry {
  id: string;
  label: string;
  text: string;
  description: string;
  file: string;
  /** Id of a recorded GPT-2 experiment on the same text (Experiments tab). */
  experiment?: string;
}
