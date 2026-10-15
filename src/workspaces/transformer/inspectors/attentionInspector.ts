/** An attention layer as a whole: its heads side by side (multi-head attention). */
import { headContributionNorms, headPatterns } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatPercent, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { attentionThumbnail, swatch } from '../widgets';
import { quote } from './common';

export function attentionInspector(ws: TransformerWorkspace, layer: number): InspectorView {
  const model = ws.model!;
  const trace = ws.trace!;
  const { dModel, nHeads } = model.config;
  const dk = dModel / nHeads;
  const T = trace.tokens.length;
  const i = Math.min(ws.focus, T - 1);
  const patterns = headPatterns(trace).filter((p) => p.layer === layer);
  const norms = headContributionNorms(trace, layer, i);
  const attention = trace.blocks[layer].attention;

  const cards = patterns.map((p) => {
    const a = attention[p.head];
    let best = 0;
    for (let j = 1; j <= i; j++) if (a[i * T + j] > a[i * T + best]) best = j;
    return h(
      'button',
      { type: 'button', class: 'head-card', onClick: () => ws.select({ kind: 'head', layer, head: p.head, key: null }), title: `Open head ${p.head + 1}` },
      h('span', { class: 'head-card__title' }, swatch(ws.headColor(p.head)), `Head ${p.head + 1}`),
      attentionThumbnail(a, T, ws.headColor(p.head)),
      h('span', { class: 'head-card__summary' }, p.summary),
      h('span', { class: 'head-card__meta' }, `${quote(trace.tokens[i].text)} reads ${quote(trace.tokens[best].text)} ${formatPercent(a[i * T + best], 0)} · adds ‖·‖ ${formatValue(norms[p.head], 2)}`),
    );
  });
  const params = 3 * dModel * dModel + 3 * dModel + dModel * dModel + dModel;

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Attention layer', title: `Attention ${layer + 1}`, subtitle: `${nHeads} heads × ${dk} numbers · multi-head attention` }),
    block(
      1,
      null,
      paragraph(`The ${nHeads} heads work in parallel, each with its own queries, keys and values of ${dk} numbers: they can look for different things at the same time. Each thumbnail is a head’s attention matrix for your text (rows read columns). Click one to open it.`),
      h('div', { class: 'head-grid' }, cards),
    ),
    block(
      2,
      null,
      kv('Heads × size d_k', `${nHeads} × ${dk} = ${dModel}`),
      kv('Parameters', `${formatCount(params)} (W_Q, W_K, W_V, W_O and biases)`),
      kv(`Focused token`, `${quote(trace.tokens[i].text)} (position ${i})`),
    ),
    block(
      3,
      null,
      tex(`\\operatorname{MultiHead}(X) = \\operatorname{Concat}(\\operatorname{head}_1, \\dots, \\operatorname{head}_{${nHeads}})\\, W_O + b_O`, { display: true }),
      tex('\\operatorname{head}_h = \\operatorname{softmax}\\!\\left(\\frac{(X W_Q^{h})(X W_K^{h})^{\\top}}{\\sqrt{d_k}} + M\\right) X W_V^{h}', { display: true }),
      paragraph('X are the normalized vectors LN₁(x) of all tokens. Concatenating the heads and multiplying by W_O is the same as adding one projected contribution per head — which is how the token inspector splits them.'),
    ),
    block(null, 'Keep in mind', note('The one-line descriptions summarize the numbers of this sentence only; the same head may behave differently on another text. Describing what a head does in general needs many inputs and interventions (Experiments tab).')),
  );
  return { el, update: () => {} };
}
