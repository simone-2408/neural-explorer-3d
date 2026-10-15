/** One attention head or one MLP layer: switch it off or scale it, and see what it changes. */
import { support } from '../../../experiments/metrics';
import type { NeuronScan } from '../../../experiments/types';
import { h, type Child } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, note, slider } from '../../../ui/widgets';
import { formatPercent, formatSigned, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { attentionCanvas, attentionGrid, barRows } from '../../transformer/widgets';
import { componentName, paperName } from '../describe';
import type { ExperimentsWorkspace } from '../ExperimentsWorkspace';
import { effectRows, lazyBlock, scaleCurveChart, unitStrip } from '../widgets';

type Component = { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number };

export function componentInspector(ws: ExperimentsWorkspace, c: Component): InspectorView {
  const r = ws.result!;
  const gpt2 = ws.modelKind === 'gpt2';
  const scale = ws.componentScale(c);
  const effect = !ws.sweep ? null : c.kind === 'head' ? ws.sweep.heads[c.layer]?.[c.head] : ws.sweep.mlps[c.layer];
  const name = componentName(c);
  const stage = c.kind === 'head' ? 2 * c.layer + 1 : 2 * c.layer + 2;
  const key = `scale:${JSON.stringify(c)}`;
  const editable = ws.editable;

  const scaleSlider = slider({
    label: 'Scale of its output (0 = off, 1 = untouched)',
    min: -1,
    max: 3,
    step: 0.05,
    value: scale,
    digits: 2,
    diverging: true,
    ariaLabel: `Scale of ${name}`,
    onInput: (v) => ws.setComponentScale(c, v, key),
  });
  const quick = h(
    'div',
    { class: 'button-row' },
    [0, 0.5, 2].map((s) => button({ label: s === 0 ? 'Off' : `×${s}`, small: true, variant: scale === s ? 'primary' : 'default', disabled: !editable, onClick: () => ws.setComponentScale(c, s) })),
    button({ icon: 'reset', label: 'Untouched', small: true, variant: 'ghost', disabled: !editable || scale === 1, onClick: () => ws.setComponentScale(c, 1) }),
  );
  if (!editable) scaleSlider.el.querySelectorAll('input').forEach((i) => (i.disabled = true));

  const value = effect ? support(effect, ws.metric) : null;
  const verdict =
    value === null
      ? null
      : ws.metric === 'kl'
        ? `Switched off alone, it changes the next-token distribution by ${formatValue(value, 3)} bits of KL.`
        : value > 0.05
          ? 'Switched off alone, it lowers the target: on this input the target relies on it.'
          : value < -0.05
            ? 'Switched off alone, it raises the target: on this input it works against the target.'
            : 'Switched off alone, it barely changes the target.';

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: c.kind === 'head' ? 'Attention head' : 'MLP layer',
      title: name,
      subtitle: c.kind === 'head' && gpt2 ? `“${paperName(c.layer, c.head)}” in papers, which count from 0` : `writes into the residual stream after the ${c.kind === 'head' ? 'attention' : 'MLP'} of layer ${c.layer + 1}`,
      actions: [button({ label: scale === 0 ? 'Switch on' : 'Switch off', small: true, variant: scale === 0 ? 'default' : 'danger', disabled: !editable, title: 'X', onClick: () => ws.toggleComponent(c) })],
    }),
    block(
      1,
      'Intervene',
      c.kind === 'head'
        ? paragraph('Multiplies what this head adds to the residual stream (its weighted mix of values, before W_O). Its attention pattern is not changed; later components see a different input.')
        : paragraph('Multiplies everything this MLP adds to the residual stream, bias included.'),
      scaleSlider.el,
      quick,
      editable ? null : note('Recorded preset: start the GPT-2 backend to change it.'),
      paragraph(`The highlighted row of the right wall (stage ${stage}) is where it writes.`),
    ),
    block(
      2,
      'Switched off alone',
      effect ? h('div', null, verdict ? paragraph(verdict) : null, effectRows(effect, ws.metric, { contrast: !!r.contrast }), h('div', { class: 'muted' }, ws.sweep?.baseline === 'experiment' ? 'Measured on top of your other interventions.' : 'Measured on the original model.')) : paragraph(ws.busy ? 'Measuring…' : 'Not measured.'),
      h('div', { class: 'insp-subheading' }, `Turning the knob: ${r.contrast && ws.metric === 'logit' ? `logit(${ws.format(r.target.text)}) − logit(${ws.format(r.contrast.text)})` : `p(${ws.format(r.target.text)})`}`),
      lazyBlock(ws.curve(c), (curve) => scaleCurveChart(curve, { current: scale, mode: ws.metric === 'logit' && curve.logitDiff ? 'logit' : 'p', title: `${name}: output × scale` }), 'Computing the curve…'),
      c.kind === 'head' ? headAttention(ws, c.layer, c.head) : neuronTable(ws, c.layer),
    ),
    block(
      3,
      null,
      c.kind === 'head'
        ? tex(`\\operatorname{Attn}_{${c.layer + 1}}(x) = \\sum_{h} W_O^{(h)}\\, s_h\\, \\sum_j A^{(h)}_{ij}\\, v^{(h)}_j + b_O, \\qquad s_{${c.head + 1}} = ${formatValue(scale, 2)}`, { display: true })
        : tex(`x \\leftarrow x + s\\,\\big(W_{out}\\,\\mathrm{GELU}(W_{in}\\,\\mathrm{LN}(x) + b_{in}) + b_{out}\\big), \\qquad s = ${formatValue(scale, 2)}`, { display: true }),
    ),
    block(
      null,
      'Keep in mind',
      c.kind === 'head' ? note('Attention weights show where a head reads from, not how much its output matters: a head can attend strongly and still write almost nothing useful. Switching it off measures the second.') : null,
      c.kind === 'mlp' && gpt2 && c.layer === 0 ? note('GPT-2’s first MLP matters for almost every input: interpretability work suggests it acts almost as part of the token embedding.') : null,
      note('The curve and the bar are measured on this input only, with everything else as in the experiment.'),
    ),
  );
  return { el, update: () => {} };
}

/** Where the last token's attention goes in this head, before and after the interventions. */
function headAttention(ws: ExperimentsWorkspace, layer: number, head: number): Child {
  const r = ws.result!;
  const T = r.tokens.length;
  const labels = r.tokens.map((t) => ws.format(t.text));
  let before: number[] | null = null;
  let after: (number | null)[] | null = null;
  let grid: HTMLElement | null = null;
  const color = ws.palette.theme === 'dark' ? '#c9b8ff' : '#6a4fd6';
  if (ws.modelKind === 'toy' && ws.toyRun) {
    const A = ws.toyRun.before.blocks[layer].attention[head];
    const B = ws.toyRun.after.blocks[layer].attention[head];
    const T2 = r.tokensAfter.length;
    before = Array.from({ length: T }, (_, j) => A[(T - 1) * T + j]);
    after = Array.from({ length: T }, (_, j) => {
      const k = r.alignment.indexOf(j);
      return k < 0 ? null : B[(T2 - 1) * T2 + k];
    });
    grid = T <= 16 ? attentionGrid({ attention: A, labels, color, query: T - 1, key: null }) : attentionCanvas({ attention: A, labels, color, query: T - 1 });
  }
  const render = (b: number[], a: (number | null)[]) =>
    h(
      'div',
      { class: 'stack' },
      barRows(
        labels.map((label, j) => ({ label, value: a[j] ?? 0, ghost: b[j], muted: a[j] === null, text: a[j] === null ? `${formatPercent(b[j], 0)} → removed` : `${formatPercent(b[j], 0)} → ${formatPercent(a[j]!, 0)}` })),
        { max: 1 },
      ),
      h('div', { class: 'muted' }, 'Ghost bars: original model. Scaling this head does not change its own pattern; earlier interventions can.'),
    );
  return h(
    'div',
    null,
    h('div', { class: 'insp-subheading' }, `Where “${labels[T - 1]}” looks in this head`),
    before && after ? render(before, after) : lazyBlock(ws.attention(layer, head), (pair) => {
      const T2 = pair.after.length;
      return render(pair.before[T - 1], labels.map((_, j) => {
        const k = pair.alignment.indexOf(j);
        return k < 0 ? null : pair.after[T2 - 1][k];
      }));
    }, 'Loading the attention pattern…'),
    grid ? h('details', { class: 'ex-details' }, h('summary', null, 'Full attention matrix (original model)'), grid) : null,
  );
}

/** The neurons of an MLP layer: measured vs estimated effect of zeroing each of them. */
function neuronTable(ws: ExperimentsWorkspace, layer: number): Child {
  return h(
    'div',
    null,
    h('div', { class: 'insp-subheading' }, 'Its neurons, one at a time'),
    lazyBlock(ws.neuronScan(layer), (scan) => renderScan(ws, scan), ws.modelKind === 'toy' ? `Zeroing each of the ${ws.units} neurons…` : 'Estimating 3,072 neurons, measuring the strongest…'),
  );
}

function renderScan(ws: ExperimentsWorkspace, scan: NeuronScan): HTMLElement {
  const metricName = scan.metric === 'logitdiff' ? 'Δ logit diff' : 'Δ log p';
  const value = (u: (typeof scan.measured)[number]) => (scan.metric === 'logitdiff' ? (u.effect.dlogit ?? 0) : u.effect.dlogp);
  const rows = [...scan.measured].sort((a, b) => Math.abs(value(b)) - Math.abs(value(a))).slice(0, 10);
  const max = Math.max(1e-9, ...rows.map((u) => Math.max(Math.abs(value(u)), Math.abs(scan.estimate[u.unit]))));
  const strongest = Math.abs(rows.length ? value(rows[0]) : 0);
  const toy = ws.modelKind === 'toy';
  const colors = { positive: ws.palette.positive, negative: ws.palette.negative };
  return h(
    'div',
    { class: 'stack' },
    paragraph(
      toy
        ? `All ${scan.measured.length} neurons were zeroed one at a time (measured), and their effect was also estimated from the slope at their current value (first-order estimate, like gradient-based attribution).`
        : `All ${scan.estimate.length} neurons were estimated with gradient × activation (attribution patching); the ${scan.measured.length} with the largest estimates were then zeroed one at a time and measured.`,
    ),
    h(
      'table',
      { class: 'ex-table' },
      h('thead', null, h('tr', null, h('th', null, 'neuron'), h('th', null, `measured ${metricName}`), h('th', null, 'estimate'))),
      h(
        'tbody',
        null,
        rows.map((u) =>
          h(
            'tr',
            { class: 'ex-table__row', onClick: () => ws.select({ kind: 'neuron', layer: scan.layer, unit: u.unit }), title: 'Inspect this neuron' },
            h('td', null, `#${u.unit}`),
            h('td', null, dualBar(value(u), max), h('span', { class: 'ex-table__num' }, formatSigned(value(u), 3))),
            h('td', null, dualBar(scan.estimate[u.unit], max, true), h('span', { class: 'ex-table__num' }, formatSigned(scan.estimate[u.unit], 3))),
          ),
        ),
      ),
    ),
    unitStrip(toy ? scan.measured.map(value) : scan.estimate, { colors, marked: rows.map((u) => u.unit), onClick: (unit) => ws.select({ kind: 'neuron', layer: scan.layer, unit }), title: toy ? `Measured ${metricName} of every neuron — click one` : `Estimated ${metricName} of every neuron — click one` }),
    strongest < 0.05 ? note(`No single neuron matters much here (largest |${metricName}| = ${formatValue(strongest, 3)}): what the layer does is spread over many neurons. Compare with switching off the whole layer.`) : null,
    note('The estimate is linear: it assumes the effect grows in proportion as the neuron goes from its value to zero. Where it disagrees with the measurement, the model is not linear there (GELU, LayerNorm and softmax bend the response; a confident prediction saturates).'),
  );
}

function dualBar(value: number, max: number, outline = false): HTMLElement {
  const t = Math.min(1, Math.abs(value) / max);
  const fill = h('span', { class: `ex-dual__fill ${value < 0 ? 'is-neg' : 'is-pos'}${outline ? ' is-outline' : ''}` });
  fill.style.width = `${t * 50}%`;
  fill.style.left = value < 0 ? `${50 - t * 50}%` : '50%';
  return h('span', { class: 'ex-dual', 'aria-hidden': 'true' }, fill);
}
