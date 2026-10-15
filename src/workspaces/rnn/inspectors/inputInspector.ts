/** One input step: the one-hot vector and the column of W_ih it selects. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { vectorStrip } from '../../transformer/widgets';
import type { RnnWorkspace } from '../RnnWorkspace';

export function inputInspector(ws: RnnWorkspace, t: number): InspectorView {
  const step = ws.trace.steps[t];
  const m = ws.model;
  const { inputSize: I, hiddenSize: H } = m.config;
  const id = step.input;
  const symbol = ws.task.inputSymbols[id];
  const oneHot = Array.from({ length: I }, (_, k) => (k === id ? 1 : 0));
  const column = Array.from({ length: m.gates * H }, (_, r) => m.Wih[r * I + id]);
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: `Input · step ${t}`, title: `“${symbol === ' ' ? '␣' : symbol}”`, subtitle: `symbol ${id} of ${I}` }),
    block(
      1,
      null,
      paragraph(`The symbol enters as a one-hot vector: ${I} numbers, all 0 except a 1 at position ${id}. Multiplying W_ih by it simply picks column ${id} of W_ih — exactly what an embedding table does in a Transformer.`),
    ),
    block(
      2,
      null,
      vectorStrip(oneHot, { label: 'one-hot x', scale: 1, compact: true }),
      vectorStrip(column, { label: `column ${id} of W_ih`, compact: true }),
      m.config.cell === 'gru' ? paragraph('For the GRU the column has three parts: what this symbol adds to the reset gate, to the update gate and to the candidate state.') : null,
      h('div', null, entityLink('The state it feeds →', () => ws.select({ kind: 'step', t })), ' · ', entityLink('All of W_ih →', () => ws.select({ kind: 'weights', which: 'ih' }))),
    ),
    block(3, null, tex(`W_{ih}\\, x_${t} = W_{ih}\\, e_{${id}} = W_{ih}[:,\\, ${id}]`, { display: true })),
  );
  return { el, update: () => {} };
}
