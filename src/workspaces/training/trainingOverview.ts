/** Inspector shown in the Training tab when nothing is selected: the training session as a whole. */
import { LOSSES, lossFor } from '../../training/loss';
import { OPTIMIZERS } from '../../training/optimizers';
import { h } from '../../ui/dom';
import { live, type InspectorView } from '../../ui/InspectorPanel';
import { tex } from '../../ui/tex';
import { kv, note } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { NetworkContext } from '../network/context';
import { block, inspectorHeader, paragraph } from '../network/inspectors/common';
import { networkFormula } from '../network/inspectors/overviewInspector';
import type { TrainingWorkspace } from './TrainingWorkspace';

export function trainingOverview(_ctx: NetworkContext, ws: TrainingWorkspace): InspectorView {
  const { spec } = ws.model;
  const loss = LOSSES[lossFor(spec)];
  const header = inspectorHeader({
    kind: 'Training session',
    title: `${ws.dataset.info.label} · ${spec.layers.map((l) => l.size).join(' → ')}`,
    subtitle: `${formatCount(ws.model.totalParameters())} parameters · ${loss.label}`,
  });

  const numbers = live(() => {
    const t = ws.trainer;
    const last = t.history[t.history.length - 1];
    const o = ws.settings.optimizer;
    return [
      kv('Training / test points', `${t.dataset.train.length} / ${t.dataset.test.length}`),
      kv('Epoch · step', `${formatCount(t.epochCount)} · ${formatCount(t.stepCount)}`),
      kv('Optimizer', `${OPTIMIZERS[o.id].label}, η = ${formatValue(o.learningRate, o.learningRate < 0.01 ? 4 : 3)}, batch ${ws.settings.batchSize >= 100000 ? 'full' : ws.settings.batchSize}`),
      last ? kv('Training loss · accuracy', `${formatValue(last.train.loss, 4)} · ${formatValue(last.train.accuracy * 100, 1)}%`, 'kv--strong') : null,
      last?.test ? kv('Test loss · accuracy', `${formatValue(last.test.loss, 4)} · ${formatValue(last.test.accuracy * 100, 1)}%`) : null,
    ];
  });

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    block(
      1,
      'What you see',
      paragraph('Left: the dataset and, in the background, what the network predicts at every point of the plane; the line is the decision boundary. Right: the same network in 3D. While it trains its weights (lines) change at every step.'),
      paragraph('Click a data point to send it through the network: the 3D view shows its forward pass and “Backprop step” animates how that point changes the weights.'),
    ),
    block(2, null, numbers.el),
    block(
      3,
      null,
      paragraph('The network is a function of the input x and of the parameters θ (all weights and biases):'),
      tex(networkFormula(spec), { display: true }),
      paragraph(`The loss of one point (${loss.label}):`),
      tex(loss.tex, { display: true }),
      paragraph('A training step averages it over a mini-batch B and moves every parameter against the gradient:'),
      tex('\\mathcal{L}(\\theta) = \\frac{1}{|B|}\\sum_{i \\in B} L\\big(f(x_i;\\theta),\\, y_i\\big),\\qquad \\theta \\leftarrow \\theta - \\eta\\, \\nabla_\\theta \\mathcal{L}', { display: true }),
      paragraph('The gradient ∇θ𝓛 is computed by backpropagation: select a neuron or a connection to see its δ and ∂L/∂w.'),
    ),
    block(null, 'Vocabulary', note('A step uses one mini-batch; an epoch is one pass over all training points. Test points are never used for training: they show how well the network generalizes.')),
  );
  return { el, update: () => numbers.update() };
}
