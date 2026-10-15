/** One MLP neuron: its activation on each token, what it writes, and the effect of changing it. */
import { toyVocabularyEffect } from '../../../experiments/toyEngine';
import type { VocabularyEffect } from '../../../experiments/types';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, note, segmented, signedBar, slider } from '../../../ui/widgets';
import { formatSigned, formatValue } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { componentName } from '../describe';
import type { ExperimentsWorkspace } from '../ExperimentsWorkspace';
import { effectRows, lazyBlock, scaleCurveChart } from '../widgets';

type Mode = 'none' | 'scale' | 'set';

export function neuronInspector(ws: ExperimentsWorkspace, layer: number, unit: number): InspectorView {
  const r = ws.result!;
  const c = { kind: 'neuron' as const, layer, unit };
  const item = ws.interventionFor(c);
  const current = item?.kind === 'neuron' ? item : null;
  const mode: Mode = current ? current.mode : 'none';
  const editable = ws.editable;
  const key = `neuron:${layer}:${unit}`;
  const format = (t: string) => ws.format(t);

  const modeControl = segmented<Mode>({
    options: [
      { value: 'none', label: 'Untouched' },
      { value: 'scale', label: 'Scale', title: 'Multiply its activation' },
      { value: 'set', label: 'Set', title: 'Replace its activation by a fixed value, at every position' },
    ],
    value: mode,
    ariaLabel: 'Intervention on this neuron',
    onChange: (m) => ws.setNeuron(layer, unit, m === 'none' ? null : m, m === 'scale' ? 0 : 3),
  });
  if (!editable) modeControl.el.querySelectorAll('button').forEach((b) => (b.disabled = true));
  const valueSlider = current
    ? slider({
        label: current.mode === 'scale' ? 'Scale (0 = zeroed)' : 'Value after GELU',
        min: current.mode === 'scale' ? -1 : -2,
        max: current.mode === 'scale' ? 3 : 8,
        step: 0.05,
        value: current.value,
        digits: 2,
        diverging: true,
        ariaLabel: 'Neuron value',
        onInput: (v) => ws.setNeuron(layer, unit, current.mode, v, key),
      })
    : null;
  if (valueSlider && !editable) valueSlider.el.querySelectorAll('input').forEach((i) => (i.disabled = true));

  // activations on each token: toy from the traces, GPT-2 from the scan
  const toy = ws.modelKind === 'toy' && ws.toyRun ? ws.toyRun : null;
  const activationBlock = (): HTMLElement => {
    if (toy) {
      const before = toy.before.blocks[layer].mlpAct.map((row) => row[unit]);
      const after = r.tokens.map((_, p) => {
        const j = r.alignment.indexOf(p);
        return j < 0 ? null : toy.after.blocks[layer].mlpAct[j][unit];
      });
      return activationTable(r.tokens.map((t) => format(t.text)), before, after);
    }
    const scan = ws.neuronScan(layer, unit);
    return h('div', null, lazyBlock(scan, (s) => {
      const acts = s.activations[unit];
      return acts ? activationTable(r.tokensAfter.map((t) => format(t.text)), acts, null) : h('div', { class: 'muted' }, 'Not available.');
    }, 'Loading its activations…'));
  };

  const vocabBlock = (v: VocabularyEffect) =>
    h(
      'div',
      { class: 'ex-vocab' },
      h('div', null, h('div', { class: 'insp-subheading' }, 'Pushes towards'), h('div', { class: 'token-chips' }, v.promoted.map((t) => h('span', { class: 'tchip' }, format(t.text), h('span', { class: 'tchip__pos' }, formatSigned(t.value, 2)))))),
      h('div', null, h('div', { class: 'insp-subheading' }, 'Pushes away from'), h('div', { class: 'token-chips' }, v.suppressed.map((t) => h('span', { class: 'tchip' }, format(t.text), h('span', { class: 'tchip__pos' }, formatSigned(t.value, 2)))))),
    );

  const scan = ws.neuronScan(layer, unit);
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'MLP neuron',
      title: componentName(c),
      subtitle: h('span', null, 'one of ', String(ws.units), ' in ', entityLink(`the MLP of layer ${layer + 1}`, () => ws.select({ kind: 'mlp', layer }))),
      actions: [button({ label: current ? 'Untouch' : 'Zero it', small: true, variant: current ? 'default' : 'danger', disabled: !editable, title: 'X', onClick: () => ws.setNeuron(layer, unit, current ? null : 'scale', 0) })],
    }),
    block(1, 'Intervene', paragraph('Changes this neuron’s output (after GELU) at every position; its output weights then carry the new value into the residual stream.'), modeControl.el, valueSlider?.el ?? null, editable ? null : note('Recorded preset: start the GPT-2 backend to change it.')),
    block(
      2,
      null,
      h('div', { class: 'insp-subheading' }, 'Its activation on each token'),
      activationBlock(),
      h('div', { class: 'insp-subheading' }, 'Zeroed alone'),
      lazyBlock(scan, (s) => {
        const m = s.measured.find((u) => u.unit === unit);
        const estimate = s.estimate[unit];
        return h(
          'div',
          null,
          m ? effectRows(m.effect, ws.metric, { contrast: !!r.contrast }) : h('div', { class: 'muted' }, 'Not measured.'),
          h('div', { class: 'muted' }, `First-order estimate (${s.metric === 'logitdiff' ? 'Δ logit diff' : 'Δ log p'}): ${formatSigned(estimate, 4)}${m ? ` · measured ${formatSigned(s.metric === 'logitdiff' ? (m.effect.dlogit ?? 0) : m.effect.dlogp, 4)}` : ''}`),
        );
      }),
      h('div', { class: 'insp-subheading' }, `Turning it up or down: p(${format(r.target.text)})`),
      lazyBlock(ws.curve(c), (curve) => scaleCurveChart(curve, { current: current?.mode === 'scale' ? current.value : 1, mode: ws.metric === 'logit' && curve.logitDiff ? 'logit' : 'p', title: `${componentName(c)}: activation × scale` })),
      h('div', { class: 'insp-subheading' }, 'What it writes, read directly as words'),
      toy ? vocabBlock(toyVocabularyEffect(ws.toy!, layer, unit)) : lazyBlock(scan, (s) => (s.vocab[unit] ? vocabBlock(s.vocab[unit]) : h('div', { class: 'muted' }, 'Not available.'))),
    ),
    block(3, null, tex(`a_{${unit}} = \\mathrm{GELU}\\big(w^{in}_{${unit}} \\cdot \\mathrm{LN}(x) + b^{in}_{${unit}}\\big), \\qquad x \\leftarrow x + a_{${unit}}\\, w^{out}_{${unit}} + \\cdots`, { display: true })),
    block(
      null,
      'Keep in mind',
      note('“Pushes towards” projects the neuron’s output weights onto the vocabulary, ignoring the layers after it: it shows what the neuron can write, not what it responds to or “means”. Many neurons respond to several unrelated things.'),
      note('Setting a value far outside the neuron’s usual range (typically between −0.17 and a few units) can produce effects the model never shows on real text.'),
    ),
  );
  return { el, update: () => {} };
}

function activationTable(labels: string[], before: number[], after: (number | null)[] | null): HTMLElement {
  const scale = Math.max(1e-9, ...before.map(Math.abs), ...(after ?? []).map((v) => Math.abs(v ?? 0)));
  return h(
    'div',
    { class: 'ex-acts' },
    labels.map((label, p) =>
      h(
        'div',
        { class: 'ex-acts__row' },
        h('span', { class: 'ex-acts__label' }, label),
        signedBar(after ? (after[p] ?? 0) : before[p], scale),
        h('span', { class: 'ex-acts__value' }, after ? (after[p] === null ? 'removed' : `${formatValue(before[p], 2)} → ${formatValue(after[p]!, 2)}`) : formatValue(before[p], 3)),
      ),
    ),
    after ? h('div', { class: 'muted' }, 'Original → with the interventions (bars: with the interventions).') : null,
  );
}
