/** Inspector shown when nothing is selected: the network as a whole. */
import { ACTIVATIONS } from '../../../math/activations';
import type { NetworkSpec } from '../../../neural/config';
import { neuronTitle } from '../../../neural/topology';
import { softmaxBars } from '../../../ui/charts/activationPlot';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note, num } from '../../../ui/widgets';
import { formatCount } from '../../../utils/format';
import type { NetworkContext } from '../context';
import { block, entityLink, inspectorHeader, paragraph } from './common';

/** The whole network as one composed function. */
export function networkFormula(spec: NetworkSpec): string {
  const L = spec.layers.length - 1;
  if (L > 4) {
    return `\\begin{aligned} a^{(0)} &= x \\\\ a^{(l)} &= f^{(l)}\\!\\left(W^{(l)} a^{(l-1)} + b^{(l)}\\right),\\quad l = 1,\\dots,${L} \\\\ \\hat{y} &= a^{(${L})} \\end{aligned}`;
  }
  let inner = 'x';
  for (let l = 1; l <= L; l++) {
    const activation = ACTIVATIONS[spec.layers[l].activation];
    const affine = `W^{(${l})}${l === 1 ? 'x' : `\\,${inner}`} + b^{(${l})}`;
    const last = l === L;
    if (activation.id === 'linear') inner = last ? affine : `\\left(${affine}\\right)`;
    else inner = `${activation.tex}\\!\\left(${affine}\\right)`;
  }
  return `\\hat{y} = ${inner}`;
}

export function overviewInspector(ctx: NetworkContext): InspectorView {
  const { model, actions } = ctx;
  const { spec } = model;
  const counts = model.parameterCounts();
  const total = counts.reduce((s, c) => s + c.total, 0);
  const output = spec.layers[spec.layers.length - 1];
  const outputLayer = spec.layers.length - 1;

  const header = inspectorHeader({
    kind: 'Network',
    title: spec.name,
    subtitle: spec.layers.map((l) => (l.shape ? `${l.shape[0]}×${l.shape[1]}` : String(l.size))).join(' → '),
  });

  const palette = ctx.palette();
  const visual = h(
    'div',
    null,
    h(
      'ul',
      { class: 'legend-list' },
      h(
        'li',
        null,
        h('span', { class: 'legend-swatch legend-swatch--neuron' }),
        h('span', null, 'Sphere = neuron. Colour and size show its output (', h('strong', { class: 'num--pos' }, `${palette.positiveName} +`), ', ', h('strong', { class: 'num--neg' }, `${palette.negativeName} −`), ', grey ≈ 0).'),
      ),
      h('li', null, h('span', { class: 'legend-swatch legend-swatch--line' }), h('span', null, 'Line = weight. Thicker = larger |w|; dashed = negative.')),
      h('li', null, h('span', { class: 'legend-swatch legend-swatch--plate' }), h('span', null, 'Plate = layer. Layers are ordered along the signal direction (left → right).')),
    ),
    paragraph('Click any neuron, line or layer plate to inspect it. Drag to rotate, right-drag (or Shift-drag) to pan, scroll to zoom. Press ? for all shortcuts.'),
  );

  const outputs = live(() => {
    const y = model.output;
    if (output.activation === 'softmax') {
      const labels = Array.from({ length: output.size }, (_, j) => neuronTitle(spec, outputLayer, j));
      return softmaxBars(model.trace.z[outputLayer], y, labels, null);
    }
    return Array.from(y, (v, j) =>
      kv(entityLink(neuronTitle(spec, outputLayer, j), () => actions.select({ kind: 'neuron', layer: outputLayer, index: j })), num(v, { signed: true })),
    );
  });

  const layerTable = h(
    'table',
    { class: 'data-table' },
    h('thead', null, h('tr', null, h('th', null, 'layer'), h('th', null, 'neurons'), h('th', null, 'activation'), h('th', null, 'params'))),
    h(
      'tbody',
      null,
      spec.layers.map((l, i) =>
        h(
          'tr',
          null,
          h('td', null, entityLink(l.name, () => actions.select({ kind: 'layer', layer: i }))),
          h('td', null, formatCount(l.size)),
          h('td', null, l.role === 'input' ? '—' : ACTIVATIONS[l.activation].label),
          h('td', null, i === 0 ? '0' : formatCount(counts[i - 1].total)),
        ),
      ),
    ),
    h('tfoot', null, h('tr', null, h('td', null, 'total'), h('td', null, formatCount(model.topology.neuronCount)), h('td', null, ''), h('td', null, formatCount(total)))),
  );

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    note(`The weights are randomly initialized (${model.init.scheme}, seed ${model.init.seed}), not trained: the output is not a meaningful prediction yet. The Training tab trains a network on data.`, 'info'),
    block(1, 'How to read the scene', visual),
    block(
      2,
      null,
      kv('Layers', `${spec.layers.length} (1 input, ${spec.layers.length - 2} hidden, 1 output)`),
      kv('Neurons', formatCount(model.topology.neuronCount)),
      kv('Weights (connections)', formatCount(model.topology.connectionCount)),
      kv('Parameters (weights + biases)', formatCount(total), 'kv--strong'),
      h('h4', { class: 'insp-subheading' }, 'Current output ŷ'),
      outputs.el,
      h('h4', { class: 'insp-subheading' }, 'Layers'),
      layerTable,
    ),
    block(
      3,
      null,
      paragraph('The whole network is one function: a composition of affine maps (W·x + b) and activation functions.'),
      tex(networkFormula(spec), { display: true }),
      paragraph('Each W is a matrix (rows = neurons of the layer, columns = neurons of the previous layer); each b is a vector.'),
    ),
  );

  return { el, update: () => outputs.update() };
}
