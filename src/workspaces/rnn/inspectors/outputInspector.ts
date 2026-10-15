/** The prediction after one step: probabilities, the expected symbol, the loss of that step. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatProbability, formatValue, texNumber } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { RnnWorkspace } from '../RnnWorkspace';

const sym = (s: string) => (s === ' ' ? '␣' : s);

export function outputInspector(ws: RnnWorkspace, t: number): InspectorView {
  const step = ws.trace.steps[t];
  const target = ws.targets[t];
  const order = Array.from(step.probs.keys()).sort((a, b) => step.probs[b] - step.probs[a]);
  const memory = ws.settings.task === 'memory';
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: `Prediction · step ${t}`,
      title: `After “${sym(ws.task.inputSymbols[step.input])}”`,
      subtitle: target >= 0 ? `expected “${sym(ws.task.outputSymbols[target])}”` : memory ? 'not scored: only the last step counts' : 'no known answer here',
    }),
    block(1, null, paragraph('The state h is turned into one score per possible symbol (W_out h + b_out); softmax turns the scores into probabilities. The prediction uses only h: whatever the network has not kept in its state, it cannot use.')),
    block(
      2,
      null,
      barRows(
        order.slice(0, 8).map((k) => ({ label: sym(ws.task.outputSymbols[k]), value: step.probs[k], highlight: k === target, text: formatProbability(step.probs[k]) })),
        { max: 1 },
      ),
      target >= 0 ? kv('Loss at this step −log p(expected)', formatValue(-Math.log(Math.max(step.probs[target], 1e-300)), 4)) : null,
      !memory && target < 0 ? note('Several letters can follow here (for example after a space, any day can start): the network can only spread its bets.') : null,
      h('div', null, entityLink(`The state h${t} →`, () => ws.select({ kind: 'step', t }))),
    ),
    block(3, null, tex(`p_${t} = \\mathrm{softmax}(W_{out} h_${t} + b_{out})${target >= 0 ? `, \\qquad L_${t} = -\\log p_${t}[\\text{${ws.task.outputSymbols[target] === ' ' ? 'space' : ws.task.outputSymbols[target]}}] = ${texNumber(-Math.log(Math.max(step.probs[target], 1e-300)))}` : ''}`, { display: true })),
  );
  return { el, update: () => {} };
}
