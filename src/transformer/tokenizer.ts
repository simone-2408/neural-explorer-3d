/**
 * Word-level tokenizer of the toy Transformer: lower-case, split on spaces,
 * punctuation as separate tokens, apostrophes kept inside words ("didn't").
 * Unknown words become <unk>. A <s> (beginning of sequence) token is
 * prepended, as during training.
 *
 * Real LLMs use sub-word tokenizers (byte-pair encoding): a word can become
 * several tokens. The LLM tab shows GPT-2's real tokens.
 */
import type { TransformerModel } from './model';

export interface Token {
  text: string;
  id: number;
  /** Position in the sequence (0 = <s>). */
  position: number;
  /** Character span in the original text ([-1, -1] for <s>). */
  start: number;
  end: number;
  unknown: boolean;
}

const WORD = /[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]/g;

export function tokenize(model: TransformerModel, text: string): Token[] {
  const { config, index } = model;
  const tokens: Token[] = [{ text: config.bos, id: index.get(config.bos)!, position: 0, start: -1, end: -1, unknown: false }];
  const lower = text.toLowerCase();
  for (const match of lower.matchAll(WORD)) {
    if (tokens.length >= config.context) break;
    const word = match[0];
    const id = index.get(word);
    tokens.push({
      text: id === undefined ? config.unk : word,
      id: id ?? index.get(config.unk)!,
      position: tokens.length,
      start: match.index ?? 0,
      end: (match.index ?? 0) + word.length,
      unknown: id === undefined,
    });
  }
  return tokens;
}

/** The words of the original text that were not in the vocabulary. */
export function unknownWords(text: string, tokens: Token[]): string[] {
  return tokens.filter((t) => t.unknown).map((t) => text.slice(t.start, t.end));
}
