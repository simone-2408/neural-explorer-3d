/** One attention head of GPT-2: its attention matrix on this text and the row of the focused token. */
import { headMatrix, showToken } from '../../../llm/client';
import { describeAttention } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, note } from '../../../ui/widgets';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { attentionCanvas, attentionGrid, barRows } from '../../transformer/widgets';
import type { LlmWorkspace } from '../LlmWorkspace';
import { tokenName } from './common';

export function headInspector(ws: LlmWorkspace, layer: number, head: number): InspectorView {
  const a = ws.analysis!;
  const T = a.T;
  const matrix = headMatrix(a, layer, head);
  const labels = a.tokens.map((t) => showToken(t.text));
  const q = Math.min(ws.focus, T - 1);
  const pattern = describeAttention(matrix, T, 'the first token');
  const color = ws.palette.theme === 'dark' ? '#c9b8ff' : '#6a4fd6';
  const onCell = (i: number) => {
    ws.setFocus(i);
    ws.select({ kind: 'head', layer, head });
  };
  const view = T <= 16 ? attentionGrid({ attention: matrix, labels, color, query: q, key: null, onCell }) : attentionCanvas({ attention: matrix, labels, color, query: q, onCell });

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Attention head',
      title: `Layer ${layer + 1} · head ${head + 1}`,
      subtitle: pattern.summary,
      actions: [
        button({ icon: 'stepBack', title: 'Previous head (←)', small: true, variant: 'ghost', onClick: () => ws.select({ kind: 'head', layer, head: (head + a.heads - 1) % a.heads }) }),
        button({ icon: 'stepForward', title: 'Next head (→)', small: true, variant: 'ghost', onClick: () => ws.select({ kind: 'head', layer, head: (head + 1) % a.heads }) }),
      ],
    }),
    block(1, null, paragraph(`Rows: the token that reads; columns: the earlier tokens it can read. In 3D, this head’s arcs connect layer ${layer} (keys) to layer ${layer + 1} (queries). Click a row to change the reading token.`), view),
    block(
      2,
      `What ${tokenName(a, q)} reads`,
      barRows(Array.from({ length: q + 1 }, (_, j) => ({ label: `${j} · ${labels[j]}`, value: matrix[q * T + j], color })), { max: 1 }),
      h('div', null, entityLink(`All heads of layer ${layer + 1} →`, () => ws.select({ kind: 'layer', layer: layer + 1 }))),
    ),
    block(3, null, tex(`A^{(${layer + 1},${head + 1})} = \\operatorname{softmax}\\!\\left(\\frac{Q K^{\\top}}{\\sqrt{64}} + M\\right)`, { display: true }), paragraph('Each of the 12 heads works with queries and keys of 64 numbers (768 / 12).')),
    block(
      null,
      'Keep in mind',
      note('Many GPT-2 heads put most of their attention on the first token when they have nothing useful to do (an “attention sink”). Attention shows where a head can read from, not what it reads or whether it matters: the Experiments tab can switch a head off.'),
    ),
  );
  return { el, update: () => {} };
}
