/** Helpers shared by the LLM inspectors. */
import { showToken } from '../../../llm/client';
import type { LlmAnalysis } from '../../../llm/types';

/** GPT-2 small's published configuration, for when the backend's status is not available. */
export const GPT2_SMALL = { layers: 12, heads: 12, dModel: 768, dMlp: 3072, vocab: 50257, context: 1024, parameters: 124_439_808 };

export function word(a: LlmAnalysis, id: number): string {
  return showToken(a.words[String(id)] ?? `#${id}`);
}

export function tokenName(a: LlmAnalysis, position: number): string {
  return `“${showToken(a.tokens[position].text)}”`;
}

export function layerName(layer: number): string {
  return layer === 0 ? 'the embeddings' : `layer ${layer}`;
}
