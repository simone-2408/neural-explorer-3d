/** One step of the recurrent network: how the new state h_t is computed from the input and the previous state. */
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatProbability, formatValue, texNumber, texParen } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { vectorStrip } from '../../transformer/widgets';
import type { RnnWorkspace } from '../RnnWorkspace';

const sym = (s: string) => (s === ' ' ? '␣' : s);

export function stepInspector(ws: RnnWorkspace, t: number): InspectorView {
  const step = ws.trace.steps[t];
  const m = ws.model;
  const H = m.config.hiddenSize;
  const gru = m.config.cell === 'gru';
  const input = sym(ws.task.inputSymbols[step.input]);
  const norm = (v: ArrayLike<number>) => Math.hypot(...Array.from(v));
  // the unit with the largest |h_t|, worked out with its numbers
  let u = 0;
  for (let i = 1; i < H; i++) if (Math.abs(step.h[i]) > Math.abs(step.h[u])) u = i;
  const part = (v: Float64Array, gate: number) => v.subarray(gate * H, (gate + 1) * H);
  const strip = (label: string, values: ArrayLike<number>, scale?: number) => vectorStrip(values, { label, scale, compact: true });
  const order = Array.from(step.probs.keys()).sort((a, b) => step.probs[b] - step.probs[a]).slice(0, 3);
  const target = ws.targets[t];
  const keep = step.update ? step.update.reduce((a, b) => a + b, 0) / H : null;
  const grad = ws.gradients?.[t];

  let math: HTMLElement;
  if (!gru) {
    const a = step.inputPart[u];
    const b = step.recurrentPart[u];
    math = h(
      'div',
      null,
      tex(`h_${t} = \\tanh\\big(\\underbrace{W_{ih} x_${t} + b_{ih}}_{\\text{input}} + \\underbrace{W_{hh} h_{${t - 1}} + b_{hh}}_{\\text{recurrent}}\\big)`, { display: true }),
      paragraph(`Unit ${u + 1} (the largest |h|):`),
      tex(`h_{${t},${u + 1}} = \\tanh(${texNumber(a)} + ${texParen(b)}) = \\tanh(${texNumber(a + b)}) = ${texNumber(step.h[u])}`, { display: true }),
    );
  } else {
    const r = step.reset![u];
    const z = step.update![u];
    const n = step.candidate![u];
    const hp = step.hPrev[u];
    math = h(
      'div',
      null,
      paragraph(`Unit ${u + 1} (the largest |h|), with its numbers:`),
      tex(
        `\\begin{aligned} r &= \\sigma(${texNumber(step.inputPart[u])} + ${texParen(step.recurrentPart[u])}) = ${texNumber(r)} \\\\ z &= \\sigma(${texNumber(step.inputPart[H + u])} + ${texParen(step.recurrentPart[H + u])}) = ${texNumber(z)} \\\\ n &= \\tanh(${texNumber(step.inputPart[2 * H + u])} + ${texNumber(r)} \\cdot ${texParen(step.recurrentPart[2 * H + u])}) = ${texNumber(n)} \\\\ h_{${t}} &= (1 - ${texNumber(z)}) \\cdot ${texParen(n)} + ${texNumber(z)} \\cdot ${texParen(hp)} = ${texNumber(step.h[u])} \\end{aligned}`,
        { display: true },
      ),
    );
  }

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: `Hidden state · step ${t}`, title: `h${t} after “${input}”`, subtitle: `${H} numbers · ‖h${t}‖ = ${formatValue(norm(step.h), 3)}` }),
    block(
      1,
      null,
      gru
        ? paragraph('The GRU computes a candidate new state from the input and (part of) the previous state, then the update gate z decides, number by number, how much of the previous state to keep: z = 1 keeps it, z = 0 replaces it by the candidate.')
        : paragraph('The new state adds what the input brings (one column of W_ih) to what the previous state brings (W_hh times h), and squashes the sum with tanh. Nothing else carries memory from one step to the next.'),
      paragraph(`The same weights are used at step 0, 1, 2…: only the input and the previous state change.`),
    ),
    block(
      2,
      null,
      h(
        'div',
        { class: 'stack' },
        strip(t === 0 ? 'h₋₁ = 0' : `h${t - 1}`, step.hPrev, 1),
        gru
          ? [strip('reset r', step.reset!, 1), strip('update z', step.update!, 1), strip('candidate n', step.candidate!, 1)]
          : [strip('input part', part(step.inputPart, 0)), strip('recurrent part', part(step.recurrentPart, 0))],
        strip(`h${t}`, step.h, 1),
      ),
      kv(`‖input part‖ / ‖recurrent part‖`, `${formatValue(norm(part(step.inputPart, gru ? 2 : 0)), 3)} / ${formatValue(norm(part(step.recurrentPart, gru ? 2 : 0)), 3)}`),
      keep !== null ? kv('Kept from the previous state (mean z)', formatProbability(keep)) : null,
      h(
        'div',
        { class: 'insp-subheading' },
        'Prediction from this state: ',
        order.map((k, i) => [i ? ' · ' : '', `${sym(ws.task.outputSymbols[k])} ${formatProbability(step.probs[k])}`]),
      ),
      target >= 0 ? kv('Expected', `${sym(ws.task.outputSymbols[target])} (${order[0] === target ? 'right' : 'wrong'})`) : null,
      grad !== undefined ? kv('‖∂L/∂h‖ at this step (BPTT)', grad > 0 ? grad.toExponential(2) : '0') : null,
      h('div', null, entityLink('Its prediction →', () => ws.select({ kind: 'output', t })), ' · ', entityLink('The recurrent weights W_hh →', () => ws.select({ kind: 'weights', which: 'hh' }))),
    ),
    block(3, null, math),
    gru ? null : block(null, 'Keep in mind', note(`tanh saturates: when |input + recurrent| is large, the slope of tanh is close to 0, and so is the gradient that can flow back through this step.`)),
  );
  return { el, update: () => {} };
}
