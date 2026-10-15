import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { createRng } from '../math/random';
import { loadTransformer, parameterCount, type TransformerFile } from './model';
import { residualAt, runTransformer, stageCount, unembed } from './forward';
import { distribution, entropyBits, sample } from './sampling';
import { tokenize, unknownWords } from './tokenizer';

const file = JSON.parse(readFileSync(join(process.cwd(), 'public/models/toy-transformer.json'), 'utf8')) as TransformerFile;
const model = loadTransformer(file);

describe('toy Transformer', () => {
  it('tokenizes like the training script (<s> + words, punctuation split, apostrophes kept)', () => {
    const text = "The animal didn't cross the street, because it was tired.";
    const tokens = tokenize(model, text);
    // ',' never appears in the corpus → the model sees <unk>; the original word is kept via the character span.
    expect(tokens.map((t) => t.text)).toEqual(['<s>', 'the', 'animal', "didn't", 'cross', 'the', 'street', '<unk>', 'because', 'it', 'was', 'tired', '.']);
    expect(unknownWords(text, tokens)).toEqual([',']);
    expect(tokenize(model, 'the zebra')[2].unknown).toBe(true);
    expect(tokenize(model, 'one '.repeat(40))).toHaveLength(model.config.context);
  });

  it('reproduces the PyTorch logits and attention exactly', () => {
    for (const ref of file.reference) {
      const tokens = tokenize(model, ref.text);
      expect(tokens.map((t) => t.id)).toEqual(ref.ids);
      const trace = runTransformer(model, tokens);
      const V = model.config.vocab.length;
      const T = tokens.length;
      for (let p = 0; p < T; p++) for (let t = 0; t < V; t++) expect(trace.logits[p][t]).toBeCloseTo(ref.logits[p * V + t], 8);
      ref.attention.forEach((flat, layer) => {
        // PyTorch layout: [head][query][key]
        for (let h = 0; h < model.config.nHeads; h++)
          for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) expect(trace.blocks[layer].attention[h][i * T + j]).toBeCloseTo(flat[h * T * T + i * T + j], 9);
      });
    }
  });

  it('knows its facts', () => {
    const trace = runTransformer(model, tokenize(model, 'the capital of france is'));
    const logits = trace.logits[trace.logits.length - 1];
    const best = logits.indexOf(Math.max(...logits));
    expect(model.config.vocab[best]).toBe('paris');
  });

  it('keeps residual stream bookkeeping consistent', () => {
    const trace = runTransformer(model, tokenize(model, 'one two three'));
    expect(stageCount(model)).toBe(1 + 2 * model.config.nLayers);
    const last = residualAt(trace, stageCount(model) - 1);
    const p = last.length - 1;
    // logits = unembed(final residual)
    const lens = unembed(model, last[p]);
    lens.forEach((v, t) => expect(v).toBeCloseTo(trace.logits[p][t], 10));
    // attention rows sum to 1 and respect the causal mask
    const a = trace.blocks[0].attention[0];
    const T = trace.tokens.length;
    for (let i = 0; i < T; i++) {
      let sum = 0;
      for (let j = 0; j < T; j++) {
        sum += a[i * T + j];
        if (j > i) expect(a[i * T + j]).toBe(0);
      }
      expect(sum).toBeCloseTo(1, 12);
    }
    expect(parameterCount(model)).toBe(file.training.parameters);
  });

  it('interventions change the output', () => {
    const tokens = tokenize(model, 'the capital of france is');
    const base = runTransformer(model, tokens).logits.at(-1)!;
    const allHeads = Array.from({ length: model.config.nLayers * model.config.nHeads }, (_, k) => ({ layer: Math.floor(k / model.config.nHeads), head: k % model.config.nHeads, scale: 0 }));
    const ablated = runTransformer(model, tokens, { heads: allHeads }).logits.at(-1)!;
    expect(ablated.some((v, i) => Math.abs(v - base[i]) > 1e-3)).toBe(true);
  });
});

describe('sampling', () => {
  it('applies temperature, top-k and top-p', () => {
    const logits = [2, 1, 0, -1];
    const sharp = distribution(logits, { temperature: 0.1, topK: 0, topP: 1 });
    expect(sharp.probs[0]).toBeGreaterThan(0.99);
    const flat = distribution(logits, { temperature: 100, topK: 0, topP: 1 });
    expect(entropyBits(flat.probs)).toBeGreaterThan(1.99);
    const top2 = distribution(logits, { temperature: 1, topK: 2, topP: 1 });
    expect(top2.kept).toBe(2);
    expect(top2.probs[2]).toBe(0);
    expect(top2.probs[0] + top2.probs[1]).toBeCloseTo(1, 12);
    const nucleus = distribution(logits, { temperature: 1, topK: 0, topP: 0.5 });
    expect(nucleus.kept).toBe(1); // p(top) = 0.64 ≥ 0.5
    expect(sample(nucleus, createRng(1))).toBe(0);
  });
});
