/** LLM inspector with nothing selected: the model, what the views show, and the limits of each read-out. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatProbability } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { LlmWorkspace } from '../LlmWorkspace';
import { GPT2_SMALL, tokenName, word } from './common';

export function overviewInspector(ws: LlmWorkspace): InspectorView {
  const a = ws.analysis;
  const c = ws.status?.config ?? GPT2_SMALL;
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Language model',
      title: 'GPT-2 small',
      subtitle: `${formatCount(c.parameters)} parameters · ${c.layers} layers · ${c.heads} heads · d = ${c.dModel} · ${formatCount(c.vocab)} tokens`,
    }),
    block(
      1,
      'What you see',
      paragraph('A real language model (OpenAI, 2019), with the same architecture as the toy Transformer but 4,000 times larger: 12 layers of attention + MLP, vectors of 768 numbers, a vocabulary of 50,257 sub-word tokens.'),
      paragraph('Layers view: one column per token, one row per layer (bottom: embeddings, top: last layer). Each box decodes that vector as if it were the last layer: its label is the word that layer would predict, its height how probable the model’s final answer already is there. Watch where the answer appears. The arcs show one attention head (choose it on the left).'),
      paragraph('Token journey: each token’s vector, layer after layer, projected to 3-D. Move the slider at the bottom.'),
    ),
    a
      ? block(
          2,
          null,
          kv('Text', `${a.T} tokens`),
          kv('Next token', a.nextTop.slice(0, 3).map(([id, p]) => `${word(a, id)} ${formatProbability(p)}`).join(' · ')),
          kv('Source', a.source === 'live' ? `computed live (${a.seconds.toFixed(2)} s)` : 'recorded example (start the backend for your own text)'),
          h('div', null, entityLink('Open the prediction →', () => ws.select({ kind: 'prediction' })), ' · ', entityLink(`Follow the information into ${tokenName(a, a.T - 1)} →`, () => ws.select({ kind: 'token', position: a.T - 1 }))),
        )
      : block(2, null, paragraph(ws.busy ? 'Loading…' : 'No analysis yet: pick an example on the left.')),
    block(
      3,
      null,
      tex('x_0 = W_E[t] + W_P[p], \\qquad x_{\\ell} = x_{\\ell-1} + \\operatorname{Attn}_\\ell(\\operatorname{LN}(x_{\\ell-1})) + \\operatorname{MLP}_\\ell(\\cdots)', { display: true }),
      tex('\\text{logit lens at layer } \\ell: \\quad p_\\ell = \\operatorname{softmax}\\big(\\operatorname{LN}_f(x_\\ell)\\, W_E^{\\top}\\big)', { display: true }),
      paragraph('At the last layer this is exactly the model’s prediction; earlier, it is a probe.'),
    ),
    block(
      null,
      'Keep in mind',
      note('Logit lens: earlier layers were never trained to be decoded directly. GPT-2’s first layers decode to odd tokens; a better-calibrated probe (e.g. a “tuned lens”) would need extra training.'),
      note('Attention shows where information can come from, not why a prediction is made. GPT-2 also parks a lot of attention on the first token (an “attention sink”).'),
      note('PCA keeps only part of the variance of 768-dimensional vectors: the 3-D journey is a projection. Rollout and gradient × input are heuristics, and they often disagree; the Experiments tab measures effects by intervening on the model.'),
      note('GPT-2 small continues text like the web pages it was trained on: it can be wrong, repetitive or biased. It has no notion of truth.'),
      h('div', null, entityLink('Compare with the toy Transformer →', () => ws.switchTo('transformer'))),
    ),
  );
  return { el, update: () => {} };
}
