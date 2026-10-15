/** RNN inspector with nothing selected: the network, how to read the scene, results and limits. */
import { CELLS } from '../../../rnn/model';
import { RNN_TASKS } from '../../../rnn/tasks';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { RnnWorkspace } from '../RnnWorkspace';

export function overviewInspector(ws: RnnWorkspace): InspectorView {
  const m = ws.model;
  const gru = m.config.cell === 'gru';
  const last = ws.trainer.history[ws.trainer.history.length - 1];
  const g = ws.gradients;
  let lastScored = -1;
  ws.targets.forEach((target, t) => {
    if (target >= 0) lastScored = t;
  });
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Recurrent network', title: `${CELLS[m.config.cell].label} · ${RNN_TASKS[ws.settings.task].label}`, subtitle: `${m.config.hiddenSize} hidden units · ${formatCount(m.parameterCount())} parameters, the same at every step` }),
    block(
      1,
      'What you see',
      paragraph('One column per step of the sequence, left to right. Bottom: the symbol read at that step. Middle: the hidden state h_t — the network’s only memory — as a tile of bars (towards you = positive). Top: what it predicts next.'),
      paragraph('The grey arrows carry the input in and the prediction out; the violet arrows carry the state from one step to the next. Every column uses exactly the same weights: “Fold” collapses the steps into the single cell with a loop you see in textbooks.'),
      gru ? paragraph('GRU: the thin bar beside each state is the update gate, averaged over the units — how much of the previous state is kept at that step.') : null,
      paragraph('The bars under the inputs (Gradients) show how strongly the loss depends on each state, from backpropagation through time: they shrink as they travel back.'),
    ),
    block(
      2,
      null,
      last ? kv('Test accuracy', `${formatValue(last.test.accuracy * 100, 1)}%${ws.settings.task === 'memory' ? ' (chance: 25%)' : ''}`) : null,
      last ? kv('Test loss', formatValue(last.test.loss, 4)) : null,
      kv('Training steps', formatCount(ws.trainer.step)),
      g && lastScored > 0 ? kv(`‖∂L/∂h‖ at step 0 / at step ${lastScored}`, `${g[0].toExponential(1)} / ${g[lastScored].toExponential(1)}`) : null,
    ),
    block(
      3,
      null,
      gru
        ? tex('\\begin{aligned} r_t &= \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\\\ z_t &= \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\\\ n_t &= \\tanh\\big(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{t-1} + b_{hn})\\big) \\\\ h_t &= (1 - z_t) \\odot n_t + z_t \\odot h_{t-1} \\end{aligned}', { display: true })
        : tex('h_t = \\tanh\\big(W_{ih}\\, x_t + b_{ih} + W_{hh}\\, h_{t-1} + b_{hh}\\big)', { display: true }),
      tex('p_t = \\mathrm{softmax}(W_{out}\\, h_t + b_{out})', { display: true }),
      tex('\\frac{\\partial L}{\\partial h_{t}} = \\frac{\\partial L_t}{\\partial h_t} + \\Big(\\frac{\\partial h_{t+1}}{\\partial h_t}\\Big)^{\\!\\top} \\frac{\\partial L}{\\partial h_{t+1}}', { display: true }),
      paragraph('Backpropagation through time multiplies by one Jacobian ∂h_{t+1}/∂h_t per step: when those factors are small (tanh slopes below 1, small recurrent weights), the gradient shrinks exponentially with the distance.'),
    ),
    block(
      null,
      'Keep in mind',
      note('These networks are tiny and the tasks are made up, so that every number can be shown. Real recurrent models (and the LSTMs/GRUs used before Transformers) are much larger, but obey the same equations.'),
      note('Transformers replaced recurrence with attention: instead of passing one state along, every token can read every earlier token directly — see the Transformer tab.'),
      note('This network checks its own maths against PyTorch (nn.RNN / nn.GRU): outputs and gradients agree to about 10⁻¹⁶.'),
    ),
  );
  return { el, update: () => {} };
}
