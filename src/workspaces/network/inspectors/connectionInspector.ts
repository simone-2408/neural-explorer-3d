/**
 * Connection (weight) inspector: value and sign of the weight, its
 * contribution to the target neuron and its position in the weight matrix.
 */
import { neuronName, outputSymbol, preActivationSymbol, weightSymbol } from '../../../neural/topology';
import { matrixView } from '../../../ui/charts/matrixView';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, num, slider } from '../../../ui/widgets';
import { formatPercent, formatValue, texNumber, texParen } from '../../../utils/format';
import type { NetworkContext } from '../context';
import { block, entityLink, inspectorHeader, paragraph } from './common';
import { flowBlock, flowContent } from './flowBlock';
import { connectionUpdateSection } from './trainingBlocks';

export function connectionInspector(ctx: NetworkContext, layer: number, target: number, source: number): InspectorView {
  const { model, actions } = ctx;
  const { spec } = model;
  const L = spec.layers[layer];
  const prev = spec.layers[layer - 1];
  const sourceName = neuronName(spec, layer - 1, source);
  const targetName = neuronName(spec, layer, target);
  const ref = { layer, target, source };

  const header = inspectorHeader({
    kind: 'Connection · weight',
    title: `${sourceName} → ${targetName}`,
    subtitle: h('span', null, `${prev.name} → ${L.name} · entry `, tex(weightSymbol(ref)), ` of the ${L.size}×${prev.size} matrix`),
    actions: [button({ icon: 'focus', title: 'Focus camera (F)', small: true, onClick: () => actions.focusSelection() })],
  });

  const nav = h(
    'div',
    { class: 'insp-nav' },
    'Go to: ',
    entityLink(`source ${sourceName}`, () => actions.select({ kind: 'neuron', layer: layer - 1, index: source })),
    entityLink(`target ${targetName}`, () => actions.select({ kind: 'neuron', layer, index: target })),
    entityLink(`layer ${L.name}`, () => actions.select({ kind: 'layer', layer })),
  );

  // ---- Level 1
  const visual = live(() => {
    const w = model.weight(layer, target, source);
    const palette = ctx.palette();
    let maxW = 0;
    for (let l = 1; l < spec.layers.length; l++) for (const v of model.weights(l)) maxW = Math.max(maxW, Math.abs(v));
    return [
      paragraph(
        `Line thickness grows with |w| (here |w| = ${formatValue(Math.abs(w))}; the thickest line in the network has |w| = ${formatValue(maxW)}). ` +
          `Colour and pattern show the sign: ${palette.positiveName} solid = positive, ${palette.negativeName} dashed = negative. ` +
          `The cone on the line points in the direction of the signal, from ${sourceName} to ${targetName}.`,
      ),
      h(
        'div',
        { class: `state-chip state-chip--${w >= 0 ? 'on' : 'neg'}` },
        h('strong', null, w > 0 ? 'Positive weight' : w < 0 ? 'Negative weight' : 'Zero weight'),
        h(
          'span',
          null,
          w > 0
            ? `When ${sourceName}'s output increases, ${targetName}'s weighted sum z increases.`
            : w < 0
              ? `When ${sourceName}'s output increases, ${targetName}'s weighted sum z decreases.`
              : `${sourceName} currently has no influence on ${targetName}.`,
        ),
      ),
    ];
  });

  // ---- Level 2
  const weightSlider = slider({
    label: h('span', null, 'Weight ', tex('w'), ' — drag to see the whole network react'),
    min: -3,
    max: 3,
    step: 0.01,
    value: model.weight(layer, target, source),
    diverging: true,
    ariaLabel: `Weight ${sourceName} to ${targetName}`,
    onInput: (v) => model.setWeight(layer, target, source, v),
  });

  const values = live(() => {
    const w = model.weight(layer, target, source);
    const x = model.trace.a[layer - 1][source];
    const contribution = w * x;
    const W = model.weights(layer);
    const rank = 1 + Array.from(W).filter((v) => Math.abs(v) > Math.abs(w)).length;
    let totalMagnitude = Math.abs(model.bias(layer, target));
    for (let i = 0; i < prev.size; i++) totalMagnitude += Math.abs(model.weight(layer, target, i) * model.trace.a[layer - 1][i]);
    return [
      kv('|w|', h('span', null, num(Math.abs(w), { plain: true }), h('span', { class: 'muted' }, `  · rank ${rank} of ${W.length} in this layer`))),
      kv(`Source output x (${sourceName})`, num(x, { signed: true })),
      kv('Contribution w·x', num(contribution, { signed: true }), 'kv--strong'),
      kv(
        `Share of ${targetName}'s terms`,
        h('span', { title: '|w·x| divided by the sum of |wᵢ·xᵢ| over all inputs plus |b|' }, totalMagnitude > 0 ? formatPercent(Math.abs(contribution) / totalMagnitude) : '—'),
      ),
      kv(`Target ${targetName}: z → a`, h('span', null, num(model.trace.z[layer][target], { signed: true }), ' → ', num(model.trace.a[layer][target], { signed: true }))),
      kv(
        'Gradient ∂L/∂w',
        ctx.training?.gradients()
          ? num(ctx.training.gradients()!.dW[layer][target * prev.size + source], { signed: true, digits: 5 })
          : h('span', { class: 'muted' }, ctx.training ? 'after the first training step' : 'computed in the Training tab'),
      ),
    ];
  });

  // ---- Level 3
  const math = live(() => {
    const w = model.weight(layer, target, source);
    const x = model.trace.a[layer - 1][source];
    const xs = outputSymbol(spec, layer - 1, source);
    return [
      paragraph('This weight contributes one term to the weighted sum of the target neuron:'),
      tex(`${preActivationSymbol(layer, target)} = \\cdots + \\underbrace{${weightSymbol(ref)}\\, ${xs}}_{${texParen(w)} \\cdot ${texParen(x)} \\,=\\, ${texNumber(w * x)}} + \\cdots`, {
        display: true,
      }),
      paragraph('Sensitivity of z to this weight (a local derivative — not yet the loss gradient used in training):'),
      tex(`\\frac{\\partial ${preActivationSymbol(layer, target)}}{\\partial ${weightSymbol(ref)}} = ${xs} = ${texNumber(x)}`, { display: true }),
      h('h4', { class: 'insp-subheading' }, h('span', null, 'Its place in the weight matrix ', tex(`W^{(${layer})}`))),
      matrixView({
        rows: L.size,
        cols: prev.size,
        data: model.weights(layer),
        bias: model.biases(layer),
        rowLabels: Array.from({ length: L.size }, (_, j) => neuronName(spec, layer, j)),
        colLabels: Array.from({ length: prev.size }, (_, i) => neuronName(spec, layer - 1, i)),
        selected: { row: target, col: source },
        colors: { positive: ctx.palette().positive, negative: ctx.palette().negative },
        onCellClick: ({ row, col }) => actions.select({ kind: 'connection', layer, target: row, source: col }),
      }),
      paragraph(`Row = target neuron in ${L.name}, column = source neuron in ${prev.name}. Every line between these two layers is one entry of this matrix.`),
    ];
  });

  const flow = live(() => flowContent(ctx, { kind: 'connection', layer, target, source }));
  const training = connectionUpdateSection(ctx, layer, target, source);

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    nav,
    block(1, null, visual.el),
    block(2, null, weightSlider.el, values.el),
    training.el,
    block(3, null, math.el),
    flowBlock(ctx, flow.el),
  );

  return {
    el,
    update() {
      visual.update();
      weightSlider.setValue(model.weight(layer, target, source));
      values.update();
      math.update();
      flow.update();
      training.update();
    },
  };
}
