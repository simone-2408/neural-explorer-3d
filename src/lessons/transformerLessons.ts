/** Lessons 8–11: embeddings, attention, the Transformer and next-token prediction (Transformer, Experiments and LLM tabs). */
import { toyPresets } from '../experiments/presets';
import { formatProbability } from '../utils/format';
import type { LessonContext } from './context';
import type { Lesson } from './types';

const IT = "the cat didn't cross the river because it was";

async function transformerWith(ctx: LessonContext, text: string, view: 'model' | 'embeddings' = 'model') {
  const ws = await ctx.transformer();
  if (ws.text !== text) ws.loadExample(text);
  if (ws.settings.view !== view) ws.update({ view });
  ws.player?.stop();
  return ws;
}

function wordId(ctx: LessonContext, word: string): number {
  return ctx.transformerWs.model?.index.get(word) ?? -1;
}

export const TRANSFORMER_LESSONS: Lesson[] = [
  {
    id: 'embeddings',
    number: 8,
    title: 'Embeddings',
    summary: 'Words as vectors: where similar words end up, and what a 3-D projection hides.',
    tab: 'transformer',
    steps: [
      {
        title: 'Words become vectors',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT, 'embeddings');
          ws.update({ embedding: 'vocabulary' });
          ws.select(null);
          ws.resetCamera();
        },
        text: [
          'A language model cannot compute with words, only with numbers. Each word of the tiny model’s vocabulary has its own vector of 32 numbers — its *embedding*, learned during training.',
          'Each point is one word. The 32 numbers are projected to 3 axes with PCA so that they can be drawn.',
        ],
      },
      {
        title: 'Neighbours',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT, 'embeddings');
          ws.update({ embedding: 'vocabulary' });
          ws.select({ kind: 'word', id: wordId(ctx, 'paris') });
          ctx.memo.set('word', wordId(ctx, 'paris'));
        },
        text: [
          'Words that appear in similar sentences get similar vectors. The inspector lists the nearest words of “paris” by *cosine similarity*, computed with all 32 numbers.',
        ],
        tex: '\\cos(a, b) = \\frac{a \\cdot b}{\\lVert a \\rVert\\, \\lVert b \\rVert}',
        task: {
          prompt: 'Click another word in the cloud — a country, an animal — and look at its neighbours.',
          check: (ctx) => {
            const s = ctx.transformerWs.selection;
            return s?.kind === 'word' && s.id !== ctx.memo.get('word');
          },
          done: 'Groups (colours) were never given to the model: they come from the training sentences.',
        },
      },
      {
        title: 'Only a projection',
        text: [
          'The axis labels say how much of the variation each axis keeps. Whatever is not kept is invisible here: two words that look close in 3-D can be far apart in 32-D. When in doubt, trust the numbers in the inspector over the picture.',
        ],
      },
      {
        title: 'Vectors in context',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT, 'embeddings');
          ws.update({ embedding: 'sentence' });
          ws.select(null);
        },
        text: [
          'Inside a sentence, a token’s vector does not stay fixed: every layer adds information from the other tokens. Each trail follows one token of the sentence from its embedding (start) to the last layer (dot).',
        ],
      },
    ],
  },
  {
    id: 'attention',
    number: 9,
    title: 'Self-attention',
    summary: 'How a token reads from earlier tokens: queries, keys, values, the causal mask — and what arcs do not prove.',
    tab: 'transformer',
    steps: [
      {
        title: 'Reading other tokens',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT);
          ws.update({ arcLayer: 'all', arcHead: 'all', arcQueries: 'focus' });
          ws.select(null);
          ws.setFocus(8);
          ws.resetCamera();
        },
        text: [
          '*Attention* lets each token gather information from the tokens before it. The arcs start from “it” and land on the tokens it reads; thickness is the attention weight, one colour per head.',
        ],
      },
      {
        title: 'Query, key, value',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT);
          ws.setFocus(9);
          ws.select({ kind: 'head', layer: 1, head: 2, key: null });
          ws.setQkvStep(0);
        },
        text: [
          'Each token asks a question (its *query* q) and offers a label (its *key* k). The score of a pair is q·k/√d; softmax turns the scores into weights; the head returns the weighted average of the *values* v.',
          'Here: layer 2 · head 3, read from “was”.',
        ],
        tex: '\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\Big(\\frac{Q K^{\\top}}{\\sqrt{d_k}}\\Big) V',
        task: {
          prompt: 'In the inspector, step through the computation (Q · K · V → Output) with the real numbers.',
          check: (ctx) => ctx.transformerWs.qkvStep >= 3,
          done: 'From “was”, this head puts almost all its weight on “cat”.',
        },
      },
      {
        title: 'Only the past',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, IT);
          ws.select({ kind: 'attention', layer: 0 });
        },
        text: [
          'A token can only attend to itself and to earlier tokens: the cells above the diagonal are masked. That is what lets the model be trained to predict the next word without seeing it.',
        ],
      },
      {
        title: 'What arcs do not tell',
        text: [
          'A thick arc means information *can* flow; it does not prove that it matters for the answer. To find out, change the model and measure — next lesson and the Experiments tab.',
        ],
      },
    ],
  },
  {
    id: 'transformer',
    number: 10,
    title: 'The Transformer',
    summary: 'Embedding, attention and MLP blocks adding up in the residual stream — then an experiment to test it.',
    tab: 'transformer',
    steps: [
      {
        title: 'The whole computation',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, 'the capital of france is');
          ws.select(null);
          ws.resetCamera();
          ws.player?.restart();
          ws.player?.play();
        },
        text: [
          'Embedding + position, then for each of the 2 layers: attention, then an MLP. Each block *adds* its result to the vector of every token; at the end, the last vector becomes a score for every word.',
          'Press Space to pause, ← → to move step by step.',
        ],
      },
      {
        title: 'The residual stream',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, 'the capital of france is');
          ws.player?.stop();
          ws.select({ kind: 'token', position: 4, stage: 0 });
        },
        text: [
          'Each tile of 32 bars is one token’s vector at one stage (tokens from left to right, stages from bottom to top). It is never replaced, only added to — this running sum is the *residual stream*.',
        ],
        tex: 'x \\leftarrow x + \\mathrm{Attn}(\\mathrm{LN}(x)), \\qquad x \\leftarrow x + \\mathrm{MLP}(\\mathrm{LN}(x))',
        task: {
          prompt: 'Press ↑ to follow “france” up through the stages; the inspector decomposes each vector into its parts.',
          check: (ctx) => {
            const s = ctx.transformerWs.selection;
            return s?.kind === 'token' && s.stage >= 3;
          },
        },
      },
      {
        title: 'The MLP',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, 'the capital of france is');
          ws.select({ kind: 'mlp', layer: 0, position: 5 });
        },
        text: [
          'The MLP works on each token on its own: 32 numbers → 128 neurons (GELU) → 32 numbers. Attention moves information *between* tokens; the MLP transforms it *within* a token.',
        ],
      },
      {
        title: 'Test it: switch a block off',
        enter: async (ctx) => {
          const ws = await ctx.experiments();
          if (ws.modelKind !== 'toy') ws.setModel('toy');
          const preset = ws.toy ? toyPresets(ws.toy).find((p) => p.id === 'toy-fact') : null;
          if (preset) ws.loadPreset(preset);
        },
        text: (ctx) => {
          const r = ctx.experimentsWs.result;
          return [
            `The Experiments tab: the MLP of layer 1 is switched off${r ? ` and p(paris) falls from ${formatProbability(r.before.pTarget)} to ${formatProbability(r.after.pTarget)}` : ''}.`,
            'Left wall: what switching off each block alone would do. Right wall: where the change spreads in the residual stream — only upwards, from the layer of the intervention.',
          ];
        },
      },
    ],
  },
  {
    id: 'next-token',
    number: 11,
    title: 'Next-token prediction',
    summary: 'From the last vector to probabilities, choosing a word, temperature — and a real model, GPT-2.',
    tab: 'transformer',
    steps: [
      {
        title: 'Scores for every word',
        enter: async (ctx) => {
          const ws = await transformerWith(ctx, 'the lion chased the');
          ctx.memo.set('text', ws.text);
          ws.select({ kind: 'prediction', position: (ws.trace?.tokens.length ?? 1) - 1 });
        },
        text: [
          'The last vector is compared with the embedding of every word: one score (*logit*) per word. Softmax turns the scores into probabilities.',
        ],
        tex: 'p = \\mathrm{softmax}\\big(\\mathrm{LN}(x_{\\text{last}})\\, E^{\\top}\\big)',
        task: {
          prompt: 'Click *Most likely* in the Text panel (left) to append the most probable word.',
          check: (ctx) => ctx.transformerWs.text !== ctx.memo.get('text'),
          done: 'Generating text is just this, repeated: predict, choose a word, append it, predict again.',
        },
      },
      {
        title: 'Temperature',
        enter: async (ctx) => {
          const ws = await ctx.transformer();
          ws.select({ kind: 'prediction', position: (ws.trace?.tokens.length ?? 1) - 1 });
        },
        text: [
          'Instead of always taking the top word, a model can *sample*. Temperature reshapes the probabilities first: below 1 the favourite gets even more likely, above 1 the distribution flattens. Top-k and top-p cut off the unlikely tail.',
        ],
        task: {
          prompt: 'Move the temperature slider in the inspector and watch the bars change.',
          check: (ctx) => ctx.transformerWs.settings.sampling.temperature !== 1,
        },
      },
      {
        title: 'A real model',
        enter: async (ctx) => {
          const ws = await ctx.llm();
          const demo = ws.demos.find((d) => d.id === 'capital');
          if (demo && ws.text !== demo.text) await ws.openDemo(demo);
          ws.update({ view: 'tower' });
        },
        text: [
          'GPT-2 small: 124 million parameters, 12 layers, 50,257 tokens — the same architecture, scaled up. After “The capital of France is” it puts “ the” first: it predicts likely *text*, not facts.',
          'Each box decodes one layer with the logit lens: the answer often appears only in the last layers. (Recorded result if the Python backend is not running.)',
        ],
      },
      {
        title: 'Read with care',
        text: [
          'Projections, attention maps, the logit lens: useful probes, each with limits stated in the inspectors. Interventions measure what matters on one input. And a language model has no notion of truth — only of what usually comes next.',
        ],
      },
    ],
  },
];
