/** One box of the logit-lens tower: what the vector of a token decodes to after a given layer. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatProbability, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { LlmWorkspace } from '../LlmWorkspace';
import { layerName, tokenName, word } from './common';

export function cellInspector(ws: LlmWorkspace, layer: number, position: number): InspectorView {
  const a = ws.analysis!;
  const lens = a.lens[layer];
  const final = a.predictions[position][0];
  const next = a.tokens[position + 1];
  const S = a.lens.length;
  const last = layer === S - 1;
  const firstRank0 = a.lens.findIndex((l) => l.finalRank[position] === 0);

  const ranks = h(
    'div',
    { class: 'rank-strip' },
    a.lens.map((l, k) => {
      const rank = l.finalRank[position];
      const cell = h(
        'button',
        { type: 'button', class: `rank-strip__cell${k === layer ? ' is-selected' : ''}${rank === 0 ? ' is-top' : ''}`, title: `${layerName(k)}: rank ${rank + 1}, p = ${formatProbability(l.finalProb[position])}`, onClick: () => ws.select({ kind: 'cell', layer: k, position }) },
        h('span', { class: 'rank-strip__layer' }, k === 0 ? 'E' : String(k)),
        h('span', { class: 'rank-strip__rank' }, rank < 999 ? String(rank + 1) : '999+'),
      );
      cell.style.setProperty('--strength', String(Math.max(0, 1 - Math.log10(rank + 1) / 4)));
      return cell;
    }),
  );

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Logit lens', title: `${tokenName(a, position)} · ${layer === 0 ? 'embeddings' : `after layer ${layer}`}`, subtitle: `position ${position} · residual stream ${layer === 0 ? 'before the first block' : `after block ${layer}`}` }),
    block(
      1,
      null,
      paragraph(
        last
          ? `The last layer: decoding this vector gives the model’s real prediction for the token after ${tokenName(a, position)}.`
          : `The vector of ${tokenName(a, position)} after ${layerName(layer)}, decoded as if it were the last layer. It shows what this vector already “points to”, not what the model would answer.`,
      ),
    ),
    block(
      2,
      null,
      barRows(lens.top[position].map(([id, p]) => ({ label: word(a, id), value: p, highlight: id === final }))),
      kv(`Model’s final answer ${word(a, final)}`, `rank #${lens.finalRank[position] + 1} here · p = ${formatProbability(lens.finalProb[position])}`),
      next ? kv('Actual next token in the text', `${word(a, next.id)}${next.id === final ? ' (= the final answer)' : ''}`) : null,
      h('div', { class: 'field-label' }, 'Rank of the final answer at each layer (E = embeddings) — click to move'),
      ranks,
      paragraph(firstRank0 >= 0 ? `The final answer becomes the top word at ${layerName(firstRank0)} and stays ${a.lens.slice(firstRank0).every((l) => l.finalRank[position] === 0) ? 'there' : 'near the top'}.` : 'The final answer is never the top word before the last layer.'),
      kv('Length of the vector ‖x‖', formatValue(a.norms[layer][position], 1)),
    ),
    block(3, null, tex(`p_{${layer}} = \\operatorname{softmax}\\big(\\operatorname{LN}_f(x_{${layer}})\\, W_E^{\\top}\\big)`, { display: true }), paragraph('LN_f is the model’s final LayerNorm and W_E the token embedding matrix (GPT-2 uses the same matrix to read and to write tokens).')),
    block(null, 'Keep in mind', note('Intermediate layers were not trained to be read this way: the logit lens is a probe. In GPT-2 it works reasonably in the second half of the network and poorly at the very beginning.')),
  );
  return { el, update: () => {} };
}
