/** CNN inspector with nothing selected: the model, how to read the scene, results and limits. */
import { confusion } from '../../../cnn/trainer';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatValue } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { CnnWorkspace } from '../CnnWorkspace';

export function overviewInspector(ws: CnnWorkspace): InspectorView {
  const model = ws.model;
  const last = ws.trainer.history[ws.trainer.history.length - 1];
  const arch = model.shapes.map((s) => `conv ${s.kernel}×${s.kernel}×${s.outChannels}${s.pool !== 'none' ? ` → ${s.pool} pool` : ''}`).join(' → ');
  const probs = ws.trace.probs;
  let best = 0;
  for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
  // the confusion table costs a pass over the test set: skipped while training runs
  const matrix = ws.trainer.step > 0 && !ws.running ? confusion(model, ws.dataset.test) : null;
  const firstConvStage = ws.stages.findIndex((s) => s.kind === 'conv');
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Convolutional network', title: `${ws.dataset.info.label} classifier`, subtitle: `16×16 image → ${arch} → dense → ${model.config.classes} classes · ${formatCount(model.parameterCount())} parameters` }),
    block(
      1,
      'What you see',
      paragraph('Bottom: the input image, one box per pixel. Each row above is one step of the computation: the feature maps of a convolution (one map per filter), the same maps after pooling (half the size), and at the top the class probabilities.'),
      paragraph('Every map is the result of sliding one small kernel over the whole row below: the same few weights are used at every position. A box comes towards you when its value is positive and goes behind its map when it is negative; brighter means larger.'),
      paragraph('Click a cell to see the window it reads and the computation with its numbers; click “filter n” to see a kernel and what it responds to. Press Space to train and watch the filters change.'),
    ),
    block(
      2,
      null,
      kv('Prediction for this image', `${ws.classNames[best]} · ${formatValue(probs[best] * 100, 1)}%`),
      ws.inputLabel() !== null ? kv('True class', ws.classNames[ws.inputLabel()!]) : null,
      last ? kv('Test accuracy', `${formatValue(last.test.accuracy * 100, 1)}% (${formatCount(ws.dataset.test.length)} images never used for training)`) : null,
      kv('Training', `${formatCount(ws.trainer.step)} steps · epoch ${formatCount(ws.trainer.epoch)}`),
      matrix
        ? h(
            'div',
            null,
            h('div', { class: 'insp-subheading' }, 'Confusion on the test images (rows: true class, columns: prediction)'),
            h(
              'table',
              { class: 'cnn-confusion' },
              h('thead', null, h('tr', null, h('th', null, ''), ws.classNames.map((c) => h('th', null, c.length > 3 ? c.slice(0, 3) : c)))),
              h(
                'tbody',
                null,
                matrix.map((row, i) =>
                  h(
                    'tr',
                    null,
                    h('th', null, ws.classNames[i]),
                    row.map((count, j) => {
                      const cell = h('td', { class: i === j ? 'is-diagonal' : count ? 'is-error' : '' }, count ? String(count) : '');
                      return cell;
                    }),
                  ),
                ),
              ),
            ),
          )
        : paragraph(ws.running ? 'The confusion table appears when training pauses.' : 'Not trained yet: the weights are random, so the prediction is a guess.'),
      firstConvStage >= 0 ? h('div', null, entityLink('Open the first filter →', () => ws.select({ kind: 'map', stage: firstConvStage, map: 0 }))) : null,
    ),
    block(
      3,
      null,
      tex('z^{(l)}_{f,i,j} = b_f + \\sum_{c} \\sum_{u,v} W_{f,c,u,v}\\; x_{c,\\,i+u-p,\\,j+v-p}', { display: true }),
      tex('a = \\mathrm{ReLU}(z), \\qquad \\text{pool}_{i,j} = \\max\\big(a_{2i,2j},\\, a_{2i,2j+1},\\, a_{2i+1,2j},\\, a_{2i+1,2j+1}\\big)', { display: true }),
      tex('p = \\mathrm{softmax}(W_d\\, \\mathrm{flatten} + b_d), \\qquad L = -\\log p_{\\text{true class}}', { display: true }),
    ),
    block(
      null,
      'Keep in mind',
      note('The images are synthetic and simple, so a tiny network reaches high accuracy quickly. Real image datasets need many more filters and layers.'),
      note('A filter that resembles an edge detector is a numerical observation (see its similarity scores), not proof of what the network “looks for”: later layers combine filters in ways that are harder to read.'),
      note('This network checks its own maths against PyTorch (nn.Conv2d, MaxPool2d, Linear): outputs and gradients agree to about 10⁻¹⁶.'),
    ),
  );
  return { el, update: () => {} };
}
