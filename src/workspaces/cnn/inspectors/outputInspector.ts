/** One output class: logits, softmax and which feature maps vote for it. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, signedBar } from '../../../ui/widgets';
import { formatProbability, formatSigned, texNumber } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { CnnWorkspace } from '../CnnWorkspace';

export function outputInspector(ws: CnnWorkspace, index: number): InspectorView {
  const model = ws.model;
  const trace = ws.trace;
  const D = model.flatSize;
  const W = model.denseWeight;
  const last = model.shapes[model.shapes.length - 1];
  const per = last.pooledSize * last.pooledSize;
  // contribution of each flattened map to this class's logit
  const byMap = Array.from({ length: last.outChannels }, (_, m) => {
    let sum = 0;
    for (let q = 0; q < per; q++) sum += W[index * D + m * per + q] * trace.flat[m * per + q];
    return sum;
  });
  const scale = Math.max(1e-9, ...byMap.map(Math.abs));
  const label = ws.inputLabel();
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Output class', title: `“${ws.classNames[index]}”`, subtitle: `p = ${formatProbability(trace.probs[index])}` }),
    block(
      1,
      null,
      paragraph(`The last maps are flattened into ${D} numbers. Each class has one weight per number (plus a bias): its score (logit) is their weighted sum. Softmax turns the scores of all classes into probabilities.`),
      paragraph('The lines in the scene show the strongest contributions to this class (colour = sign).'),
    ),
    block(
      2,
      null,
      barRows(
        ws.classNames.map((name, k) => ({ label: name, value: trace.probs[k], highlight: k === index, text: `${formatProbability(trace.probs[k])} · logit ${formatSigned(trace.logits[k], 2)}`, onClick: () => ws.select({ kind: 'output', index: k }) })),
        { max: 1 },
      ),
      label !== null ? kv('True class of this image', ws.classNames[label]) : null,
      h('div', { class: 'insp-subheading' }, `Which maps of the last layer vote for “${ws.classNames[index]}”`),
      h('div', { class: 'cnn-sims' }, byMap.map((v, m) => h('div', { class: 'cnn-sims__row' }, h('span', null, `map ${m + 1}`), signedBar(v, scale), h('span', { class: 'cnn-sims__value' }, formatSigned(v, 2))))),
      kv('Bias', formatSigned(model.denseBias[index], 3)),
    ),
    block(
      3,
      null,
      tex(`\\ell_{${index}} = \\sum_{d=1}^{${D}} W_{${index},d}\\, x_d + b_{${index}} = ${texNumber(trace.logits[index])}`, { display: true }),
      tex(`p_{${index}} = \\frac{e^{\\ell_{${index}}}}{\\sum_k e^{\\ell_k}} = ${texNumber(trace.probs[index])}`, { display: true }),
    ),
  );
  return { el, update: () => {} };
}
