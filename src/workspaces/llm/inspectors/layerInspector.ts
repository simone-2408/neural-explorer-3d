/** One layer of GPT-2: the logit lens across the text and the 12 attention heads of the block that produced it. */
import { headMatrix, showToken } from '../../../llm/client';
import { describeAttention } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatProbability } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { attentionThumbnail } from '../../transformer/widgets';
import type { LlmWorkspace } from '../LlmWorkspace';
import { GPT2_SMALL, word } from './common';

export function layerInspector(ws: LlmWorkspace, layer: number): InspectorView {
  const a = ws.analysis!;
  const lens = a.lens[layer];
  const color = ws.palette.theme === 'dark' ? '#c9b8ff' : '#6a4fd6';
  const c = ws.status?.config ?? GPT2_SMALL;

  const table = h(
    'table',
    { class: 'data-table' },
    h('thead', null, h('tr', null, h('th', null, 'token'), h('th', null, 'decodes to'), h('th', null, 'final answer rank'))),
    h(
      'tbody',
      null,
      a.tokens.map((t, p) =>
        h(
          'tr',
          { class: lens.finalRank[p] === 0 ? 'is-highlight' : '' },
          h('td', null, h('button', { type: 'button', class: 'entity-link', onClick: () => ws.select({ kind: 'cell', layer, position: p }) }, `${p} · ${showToken(t.text)}`)),
          h('td', null, `${word(a, lens.top[p][0][0])} ${formatProbability(lens.top[p][0][1])}`),
          h('td', null, `#${lens.finalRank[p] + 1}`),
        ),
      ),
    ),
  );

  const heads =
    layer === 0
      ? null
      : h(
          'div',
          { class: 'head-grid head-grid--many' },
          Array.from({ length: a.heads }, (_, k) => {
            const m = headMatrix(a, layer - 1, k);
            return h(
              'button',
              { type: 'button', class: 'head-card', title: describeAttention(m, a.T, 'the first token').summary, onClick: () => ws.select({ kind: 'head', layer: layer - 1, head: k }) },
              h('span', { class: 'head-card__title' }, `${layer}.${k + 1}`),
              attentionThumbnail(m, a.T, color),
            );
          }),
        );

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Layer', title: layer === 0 ? 'Embeddings' : `Layer ${layer}`, subtitle: layer === 0 ? 'token embedding + position embedding' : `residual stream after block ${layer} of ${a.layers}` }),
    layer === 0
      ? block(
          1,
          null,
          paragraph('Each token id selects a row of the token-embedding matrix (768 numbers) and its position adds a row of the position-embedding matrix. Decoding the embeddings directly mostly gives odd tokens: the vectors still say “which word is this”, not “what comes next”.'),
          kv('Token embeddings', `${formatCount(c.vocab)} × ${c.dModel}`),
          kv('Position embeddings', `${formatCount(c.context)} × ${c.dModel}`),
        )
      : block(1, `The ${a.heads} heads of block ${layer} on this text`, paragraph('Each thumbnail is one head’s attention matrix (rows read columns). Click one to open it.'), heads),
    block(2, 'Logit lens of every token at this layer', table),
    block(null, 'Keep in mind', note('“Decodes to” applies the final LayerNorm and unembedding to this layer: a probe, exact only at the last layer.')),
  );
  return { el, update: () => {} };
}
