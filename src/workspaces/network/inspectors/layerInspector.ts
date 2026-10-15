/** Layer inspector: shapes, parameter count, activations and the weight matrix of a layer. */
import { ACTIVATIONS } from '../../../math/activations';
import { summarize } from '../../../math/stats';
import { layerVectorSymbol, neuronName, neuronTitle } from '../../../neural/topology';
import { matrixView } from '../../../ui/charts/matrixView';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, num } from '../../../ui/widgets';
import { formatCount, formatValue } from '../../../utils/format';
import type { NetworkContext } from '../context';
import { activationFigure, block, entityLink, inspectorHeader, neuronState, paragraph } from './common';

const MAX_ROWS = 16;

export function layerInspector(ctx: NetworkContext, layer: number): InspectorView {
  const { model, actions } = ctx;
  const { spec } = model;
  const L = spec.layers[layer];
  const prev = layer > 0 ? spec.layers[layer - 1] : null;
  const activation = ACTIVATIONS[L.activation];
  const isInput = L.role === 'input';

  const header = inspectorHeader({
    kind: isInput ? 'Input layer' : L.role === 'output' ? 'Output layer' : 'Hidden layer · dense',
    title: L.name,
    subtitle: isInput ? `${L.size} inputs${L.shape ? ` arranged as ${L.shape[0]}×${L.shape[1]}` : ''}` : `${L.size} neurons · ${activation.label} · fully connected to ${prev!.name}`,
    actions: [
      button({ icon: 'face', title: 'Look at this layer face-on (L)', small: true, onClick: () => actions.faceLayer(layer) }),
      button({ icon: 'focus', title: 'Focus camera (F)', small: true, onClick: () => actions.focusSelection() }),
    ],
  });

  // ---- Level 1
  const visual = paragraph(
    isInput
      ? `The plate groups the ${L.size} input neurons. They hold the input vector x and have no parameters; their colour and size show the input values.`
      : `The plate groups the ${L.size} neurons of this layer. The highlighted lines are its weight matrix: every neuron of ${prev!.name} is connected to every neuron here (a dense, or fully-connected, layer). Colour and size of each sphere show its activation.`,
  );

  // ---- Level 2
  const shapes = isInput
    ? [kv('Inputs', formatCount(L.size)), kv('Parameters', '0')]
    : [
        kv('Neurons', formatCount(L.size)),
        kv('Weights W', `${L.size} × ${prev!.size} = ${formatCount(L.size * prev!.size)}`),
        kv('Biases b', formatCount(L.size)),
        kv('Parameters', formatCount(L.size * prev!.size + L.size), 'kv--strong'),
      ];

  const stats = live(() => {
    if (isInput) {
      const s = summarize(model.trace.a[layer]);
      return [kv('Input range', `${formatValue(s.min)} … ${formatValue(s.max)}`), kv('Mean input', formatValue(s.mean))];
    }
    const w = summarize(model.weights(layer));
    const a = summarize(model.trace.a[layer]);
    let inactive = 0;
    if (L.activation === 'relu') for (const z of model.trace.z[layer]) if (z <= 0) inactive++;
    return [
      kv('Weights min / max', `${formatValue(w.min)} / ${formatValue(w.max)}`),
      kv('Weights mean ± std', `${formatValue(w.mean)} ± ${formatValue(w.std)}`),
      kv('Activations mean ± std', `${formatValue(a.mean)} ± ${formatValue(a.std)}`),
      L.activation === 'relu' ? kv('Inactive neurons (z ≤ 0)', `${inactive} of ${L.size} for this input`) : null,
    ];
  });

  const activations = live(() => {
    let rows = Array.from({ length: L.size }, (_, j) => j);
    const total = rows.length;
    if (rows.length > MAX_ROWS) rows = rows.sort((p, q) => Math.abs(model.trace.a[layer][q]) - Math.abs(model.trace.a[layer][p])).slice(0, MAX_ROWS - 4);
    return [
      h(
        'table',
        { class: 'data-table' },
        h('thead', null, h('tr', null, h('th', null, 'neuron'), isInput ? null : h('th', null, 'z'), h('th', null, isInput ? 'x' : 'a'), isInput ? null : h('th', null, 'state'))),
        h(
          'tbody',
          null,
          rows.map((j) => {
            const state = isInput ? null : neuronState(model, layer, j);
            return h(
              'tr',
              null,
              h('td', null, entityLink(neuronTitle(spec, layer, j), () => actions.select({ kind: 'neuron', layer, index: j }))),
              isInput ? null : h('td', null, num(model.trace.z[layer][j], { signed: true })),
              h('td', null, num(model.trace.a[layer][j], { signed: !isInput })),
              isInput ? null : h('td', { class: 'muted' }, state?.label ?? '—'),
            );
          }),
        ),
      ),
      total > rows.length ? h('p', { class: 'insp-caption' }, `Showing the ${rows.length} most active of ${total} neurons (largest |a|).`) : null,
    ];
  });

  // ---- Level 3
  const math = live(() => {
    if (isInput) {
      return [
        tex(`x \\in \\mathbb{R}^{${L.size}}`, { display: true }),
        paragraph('The input layer is just the input vector; the first computation happens in the next layer.'),
      ];
    }
    const prevSymbol = layerVectorSymbol(spec, layer - 1);
    const outSymbol = layerVectorSymbol(spec, layer);
    const f = activation.id === 'linear' ? '' : activation.tex;
    const body = `W^{(${layer})}\\, ${prevSymbol} + b^{(${layer})}`;
    return [
      tex(`${outSymbol} = ${f ? `${f}\\!\\left(${body}\\right)` : body}`, { display: true }),
      tex(
        `W^{(${layer})} \\in \\mathbb{R}^{${L.size} \\times ${prev!.size}},\\quad ${prevSymbol} \\in \\mathbb{R}^{${prev!.size}},\\quad b^{(${layer})} \\in \\mathbb{R}^{${L.size}}`,
        { display: true },
      ),
      tex(`\\#\\text{params} = ${L.size} \\cdot ${prev!.size} + ${L.size} = ${L.size * prev!.size + L.size}`, { display: true }),
      h('h4', { class: 'insp-subheading' }, `${activation.label}: `, tex(`${activation.kind === 'vector' ? 'a_j' : 'a'} = ${activation.texDefinition}`)),
      activationFigure(model, layer, null),
      paragraph(activation.description),
      ctx.actions.openActivationExplorer && activation.kind === 'elementwise'
        ? button({ icon: 'fx', label: `Explore ${activation.label}`, small: true, onClick: () => ctx.actions.openActivationExplorer?.(activation.id) })
        : null,
      h('h4', { class: 'insp-subheading' }, h('span', null, 'Weight matrix ', tex(`W^{(${layer})}`), ' and bias ', tex(`b^{(${layer})}`))),
      matrixView({
        rows: L.size,
        cols: prev!.size,
        data: model.weights(layer),
        bias: model.biases(layer),
        rowLabels: Array.from({ length: L.size }, (_, j) => neuronName(spec, layer, j)),
        colLabels: Array.from({ length: prev!.size }, (_, i) => neuronName(spec, layer - 1, i)),
        colors: { positive: ctx.palette().positive, negative: ctx.palette().negative },
        onCellClick: ({ row, col }) => actions.select({ kind: 'connection', layer, target: row, source: col }),
      }),
      paragraph('Click an entry to select the corresponding connection in the 3D view.'),
    ];
  });

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    block(1, null, visual),
    block(2, null, shapes, stats.el, h('h4', { class: 'insp-subheading' }, isInput ? 'Values' : 'Activations for the current input'), activations.el),
    block(3, null, math.el),
  );

  return {
    el,
    update() {
      stats.update();
      activations.update();
      math.update();
    },
  };
}
