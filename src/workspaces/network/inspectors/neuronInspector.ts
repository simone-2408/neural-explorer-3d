/**
 * Neuron inspector — the core "visual → numbers → math" view:
 * inputs, weights, bias, weighted sum, activation and output of one neuron.
 */
import { ACTIVATIONS } from '../../../math/activations';
import { maxAbs } from '../../../math/stats';
import { neuronName, neuronTitle, outputSymbol } from '../../../neural/topology';
import { paintMatrix } from '../../../ui/charts/matrixView';
import { h, type Child } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, num, signedBar, slider } from '../../../ui/widgets';
import { formatValue, texNumber } from '../../../utils/format';
import type { NetworkContext } from '../context';
import {
  activationFigure,
  activationTex,
  block,
  entityLink,
  incomingTerms,
  inspectorHeader,
  neuronState,
  paragraph,
  weightedSumTex,
} from './common';
import { flowBlock, flowContent } from './flowBlock';
import { biasUpdateSection, neuronGradientSection } from './trainingBlocks';

const MAX_ROWS = 12;

function signWord(ctx: NetworkContext, value: number): string {
  const palette = ctx.palette();
  return value > 0 ? `${palette.positiveName}: positive` : value < 0 ? `${palette.negativeName}: negative` : 'grey: zero';
}

/** Table of the outgoing connections of a neuron: where its output goes. */
function outgoingTable(ctx: NetworkContext, layer: number, index: number): Child {
  const { model, actions } = ctx;
  const { spec } = model;
  const next = spec.layers[layer + 1];
  if (!next) return null;
  const a = model.trace.a[layer][index];
  let rows = Array.from({ length: next.size }, (_, j) => ({ target: j, w: model.weight(layer + 1, j, index) }));
  const total = rows.length;
  if (rows.length > MAX_ROWS) rows = [...rows].sort((p, q) => Math.abs(q.w) - Math.abs(p.w)).slice(0, MAX_ROWS - 2);
  const scale = Math.max(...rows.map((r) => Math.abs(r.w * a)), 1e-12);
  return h(
    'div',
    null,
    h(
      'table',
      { class: 'data-table' },
      h('thead', null, h('tr', null, h('th', null, 'to'), h('th', null, 'w'), h('th', { title: 'Contribution w·a to the target weighted sum' }, 'w·a'), h('th', null, ''))),
      h(
        'tbody',
        null,
        rows.map((r) =>
          h(
            'tr',
            null,
            h('td', null, entityLink(neuronName(spec, layer + 1, r.target), () => actions.select({ kind: 'connection', layer: layer + 1, target: r.target, source: index }), 'Select this weight')),
            h('td', null, num(r.w, { signed: true })),
            h('td', null, num(r.w * a, { signed: true })),
            h('td', null, signedBar(r.w * a, scale)),
          ),
        ),
      ),
    ),
    total > rows.length ? h('p', { class: 'insp-caption' }, `Showing the ${rows.length} largest of ${total} outgoing weights by |w|.`) : null,
  );
}

function inputNeuronInspector(ctx: NetworkContext, layer: number, index: number): InspectorView {
  const { model, actions } = ctx;
  const { spec } = model;
  const L = spec.layers[layer];
  const next = spec.layers[1];

  const header = inspectorHeader({
    kind: 'Input neuron',
    title: neuronTitle(spec, layer, index),
    subtitle: `${L.name} · input ${index + 1} of ${L.size}`,
    actions: [button({ icon: 'focus', title: 'Focus camera (F)', small: true, onClick: () => actions.focusSelection() })],
  });

  const visual = live(() =>
    paragraph(
      `Input neurons perform no computation: they hold one value of the data point and send it unchanged along all ${next.size} outgoing connections to ${next.name}. Colour and size encode the value (${signWord(ctx, model.input[index])}).`,
    ),
  );

  const valueSlider = slider({
    label: h('span', null, tex(outputSymbol(spec, layer, index)), ' input value'),
    min: L.range[0],
    max: L.range[1],
    step: 0.01,
    value: model.input[index],
    diverging: L.range[0] < 0,
    ariaLabel: `Input ${neuronName(spec, layer, index)}`,
    onInput: (v) => model.setInput(index, v),
  });

  const math = live(() => [
    tex(`${outputSymbol(spec, layer, index)} = ${texNumber(model.input[index])}`, { display: true }),
    paragraph('Each neuron j of the next layer multiplies this value by its own weight and adds it to its weighted sum:'),
    tex(`z^{(1)}_j = \\cdots + W^{(1)}_{j,${index + 1}}\\, x_{${index + 1}} + \\cdots`, { display: true }),
  ]);

  const outgoing = live(() => outgoingTable(ctx, layer, index));
  const flow = live(() => flowContent(ctx, { kind: 'neuron', layer, index }));

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    block(1, null, visual.el),
    block(2, null, valueSlider.el, h('h4', { class: 'insp-subheading' }, `Outgoing → ${next.name}`), outgoing.el),
    block(3, null, math.el),
    flowBlock(ctx, flow.el),
  );

  return {
    el,
    update() {
      visual.update();
      valueSlider.setValue(model.input[index]);
      math.update();
      outgoing.update();
      flow.update();
    },
  };
}

function computeNeuronInspector(ctx: NetworkContext, layer: number, index: number): InspectorView {
  const { model, actions } = ctx;
  const { spec } = model;
  const L = spec.layers[layer];
  const prev = spec.layers[layer - 1];
  const next = spec.layers[layer + 1] ?? null;
  const activation = ACTIVATIONS[L.activation];

  const header = inspectorHeader({
    kind: L.role === 'output' ? 'Output neuron' : 'Neuron',
    title: neuronTitle(spec, layer, index),
    subtitle: `${L.name} · neuron ${index + 1} of ${L.size} · ${activation.label}`,
    actions: [button({ icon: 'focus', title: 'Focus camera (F)', small: true, onClick: () => actions.focusSelection() })],
  });

  // ---- Level 1: what the scene shows
  const visual = live(() => {
    const a = model.trace.a[layer][index];
    const state = neuronState(model, layer, index);
    return [
      paragraph(
        `The sphere's colour and size encode its output a = ${formatValue(a)} (${signWord(ctx, a)}). The highlighted lines are its ${prev.size} incoming weights from ${prev.name}${next ? ` and its ${next.size} outgoing weights to ${next.name}` : ''}.`,
      ),
      state ? h('div', { class: `state-chip state-chip--${state.tone}` }, h('strong', null, state.label), h('span', null, state.detail)) : null,
    ];
  });

  // ---- Level 2: the numbers
  const incoming = live(() => {
    const terms = incomingTerms(model, layer, index);
    const b = model.bias(layer, index);
    const z = model.trace.z[layer][index];
    let shown = terms;
    if (terms.length > MAX_ROWS) shown = [...terms].sort((p, q) => Math.abs(q.contribution) - Math.abs(p.contribution)).slice(0, MAX_ROWS - 2);
    const scale = Math.max(...shown.map((t) => Math.abs(t.contribution)), Math.abs(b), 1e-12);
    const strongest = shown.reduce((best, t) => (Math.abs(t.contribution) > Math.abs(best.contribution) ? t : best), shown[0]);
    return [
      h(
        'table',
        { class: 'data-table' },
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            h('th', null, 'from'),
            h('th', { title: 'Output of the source neuron (input xᵢ of this neuron)' }, 'xᵢ'),
            h('th', { title: 'Weight of the connection' }, 'wᵢ'),
            h('th', { title: 'Contribution wᵢ·xᵢ to the weighted sum' }, 'wᵢ·xᵢ'),
            h('th', null, ''),
          ),
        ),
        h(
          'tbody',
          null,
          shown.map((t) =>
            h(
              'tr',
              { class: t === strongest && Math.abs(t.contribution) > 0 ? 'is-strongest' : '' },
              h('td', null, entityLink(neuronName(spec, layer - 1, t.source), () => actions.select({ kind: 'connection', layer, target: index, source: t.source }), 'Select this weight')),
              h('td', null, num(t.x, { plain: true })),
              h('td', null, num(t.w, { signed: true })),
              h('td', null, num(t.contribution, { signed: true })),
              h('td', null, signedBar(t.contribution, scale)),
            ),
          ),
          h('tr', { class: 'is-bias' }, h('td', null, 'bias b'), h('td', null, ''), h('td', null, ''), h('td', null, num(b, { signed: true })), h('td', null, signedBar(b, scale))),
        ),
        h('tfoot', null, h('tr', null, h('td', null, 'z = Σ + b'), h('td', null, ''), h('td', null, ''), h('td', null, num(z, { signed: true })), h('td', null, ''))),
      ),
      terms.length > shown.length ? h('p', { class: 'insp-caption' }, `Showing the ${shown.length} largest of ${terms.length} terms by |wᵢ·xᵢ|; z uses all of them.`) : null,
    ];
  });

  const weightImage = prev.shape ? live(() => incomingWeightImage(ctx, layer, index)) : null;

  const biasSlider = slider({
    label: h('span', null, 'Bias ', tex(`b^{(${layer})}_{${index + 1}}`), ' — drag to see z and a change'),
    min: -3,
    max: 3,
    step: 0.01,
    value: model.bias(layer, index),
    diverging: true,
    ariaLabel: `Bias of ${neuronName(spec, layer, index)}`,
    onInput: (v) => model.setBias(layer, index, v),
  });

  const summary = live(() => {
    const z = model.trace.z[layer][index];
    const a = model.trace.a[layer][index];
    return [
      kv('Weighted sum z', num(z, { signed: true })),
      kv('Activation', `${activation.label}(z)`),
      kv(L.role === 'output' ? 'Output ŷ' : 'Output a', num(a, { signed: true }), 'kv--strong'),
    ];
  });

  // ---- Level 3: the math
  const explore = ctx.actions.openActivationExplorer;
  const math = live(() => [
    tex(weightedSumTex(model, layer, index), { display: true }),
    tex(activationTex(model, layer, index), { display: true }),
    activationFigure(model, layer, index),
    paragraph(activation.description),
    explore && activation.kind === 'elementwise'
      ? button({ icon: 'fx', label: `Explore ${activation.label}`, small: true, onClick: () => explore(activation.id, model.trace.z[layer][index]) })
      : null,
  ]);

  const outgoing = next ? live(() => outgoingTable(ctx, layer, index)) : null;
  const flow = live(() => flowContent(ctx, { kind: 'neuron', layer, index }));
  const gradient = neuronGradientSection(ctx, layer, index);
  const biasUpdate = biasUpdateSection(ctx, layer, index);
  if (biasUpdate) gradient.el.append(biasUpdate.el);

  const el = h(
    'div',
    { class: 'inspector-view' },
    header,
    block(1, null, visual.el),
    block(2, null, h('h4', { class: 'insp-subheading' }, `Inputs ← ${prev.name}`), incoming.el, weightImage?.el ?? null, biasSlider.el, summary.el),
    block(3, null, math.el),
    flowBlock(ctx, flow.el),
    outgoing ? block(null, `Outgoing → ${next!.name}`, outgoing.el) : block(null, 'Network output', paragraph('This is an output neuron: its value is part of the network prediction ŷ.')),
    gradient.el,
  );

  return {
    el,
    update() {
      visual.update();
      incoming.update();
      weightImage?.update();
      biasSlider.setValue(model.bias(layer, index));
      summary.update();
      math.update();
      outgoing?.update();
      flow.update();
      gradient.update();
      biasUpdate?.update();
    },
  };
}

/** For image-shaped inputs: the neuron's incoming weights arranged like the image. */
function incomingWeightImage(ctx: NetworkContext, layer: number, index: number): Child {
  const { model } = ctx;
  const prev = model.spec.layers[layer - 1];
  if (!prev.shape) return null;
  const [rows, cols] = prev.shape;
  const W = model.weights(layer);
  const row = W.subarray(index * prev.size, (index + 1) * prev.size);
  const canvas = h('canvas', { class: 'weight-image', 'aria-label': `Incoming weights as a ${rows}×${cols} image` });
  const palette = ctx.palette();
  paintMatrix(canvas, row, rows, cols, maxAbs(row), { positive: palette.positive, negative: palette.negative });
  return h(
    'figure',
    { class: 'weight-image__figure' },
    canvas,
    h(
      'figcaption',
      { class: 'insp-caption' },
      `The ${prev.size} incoming weights arranged like the ${rows}×${cols} input (${palette.positiveName} = positive, ${palette.negativeName} = negative). With random, untrained weights no meaningful pattern is expected.`,
    ),
  );
}

export function neuronInspector(ctx: NetworkContext, layer: number, index: number): InspectorView {
  return ctx.model.spec.layers[layer].role === 'input' ? inputNeuronInspector(ctx, layer, index) : computeNeuronInspector(ctx, layer, index);
}
