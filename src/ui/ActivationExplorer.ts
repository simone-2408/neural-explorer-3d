/**
 * Activation function explorer (dialog): overlay several activation functions,
 * move z with the slider or by dragging on the plot, read f(z) and f′(z), and
 * see the formulas with the value substituted.
 *
 * The derivative curves matter for training: where f′ ≈ 0 (saturation, the
 * flat part of ReLU) gradients cannot flow back during training.
 */
import { ACTIVATIONS, type ActivationId, type ElementwiseActivation } from '../math/activations';
import { formatValue, texNumber } from '../utils/format';
import { h, replaceChildren, svg } from './dom';
import { openModal } from './overlays';
import { tex } from './tex';
import { slider } from './widgets';

const FUNCTIONS: { id: ActivationId; color: string; dash: string }[] = [
  // Okabe–Ito colours (distinguishable with colour-vision deficiencies) + distinct dash patterns.
  { id: 'relu', color: '#E69F00', dash: '' },
  { id: 'leaky_relu', color: '#D55E00', dash: '7 4' },
  { id: 'sigmoid', color: '#56B4E9', dash: '' },
  { id: 'tanh', color: '#009E73', dash: '2 3' },
  { id: 'gelu', color: '#CC79A7', dash: '10 3 2 3' },
  { id: 'linear', color: '#8a96a5', dash: '1 4' },
];

const W = 640;
const H = 300;
const PAD = { left: 40, right: 16, top: 14, bottom: 26 };

function info(id: ActivationId): ElementwiseActivation {
  const a = ACTIVATIONS[id];
  if (a.kind !== 'elementwise') throw new Error(`${id} is not elementwise`);
  return a;
}

/** Where the derivative is (almost) zero: the "flat" regions that block gradients. */
function flatRegions(id: ActivationId): string {
  switch (id) {
    case 'relu':
      return 'f′(z) = 0 for every z < 0: an inactive ReLU passes no gradient back.';
    case 'leaky_relu':
      return 'Never completely flat: f′(z) = 0.01 for z < 0.';
    case 'sigmoid':
      return 'f′(z) ≤ 0.25 everywhere and < 0.02 for |z| > 4 (saturation): in deep sigmoid networks these small factors multiply.';
    case 'tanh':
      return 'f′(0) = 1 but f′(z) < 0.03 for |z| > 2.5 (saturation).';
    case 'gelu':
      return 'Smooth; slightly negative for z between ≈ −2 and 0; ≈ 0 for very negative z.';
    case 'linear':
      return 'f′(z) = 1 everywhere: no saturation, but also no non-linearity.';
    default:
      return '';
  }
}

export function openActivationExplorer(options: { initial?: ActivationId; z?: number } = {}): void {
  let z = options.z ?? 0.8;
  let range = 5;
  let showDerivative = true;
  let focus: ActivationId = options.initial && ACTIVATIONS[options.initial].kind === 'elementwise' ? options.initial : 'relu';
  const enabled = new Set<ActivationId>(['relu', 'sigmoid', 'tanh', 'gelu']);
  enabled.add(focus);

  const plotHost = h('div', { class: 'explorer__plot' });
  const table = h('div', { class: 'explorer__table' });
  const card = h('div', { class: 'explorer__card' });

  const chips = FUNCTIONS.map((fn) => {
    const chip = h(
      'button',
      { type: 'button', class: 'fn-chip', title: `Show / hide ${info(fn.id).label}` },
      svg('svg', { width: 26, height: 10, viewBox: '0 0 26 10', 'aria-hidden': 'true' }, svg('line', { x1: 1, x2: 25, y1: 5, y2: 5, stroke: fn.color, 'stroke-width': 3, 'stroke-dasharray': fn.dash || undefined })),
      info(fn.id).label,
    );
    chip.addEventListener('click', () => {
      if (enabled.has(fn.id) && enabled.size > 1) enabled.delete(fn.id);
      else enabled.add(fn.id);
      if (!enabled.has(focus)) focus = [...enabled][0];
      render();
    });
    return { chip, id: fn.id };
  });

  const zSlider = slider({
    label: h('span', null, 'Input ', tex('z'), ' — or drag on the plot'),
    min: -range,
    max: range,
    step: 0.01,
    value: z,
    digits: 2,
    diverging: true,
    ariaLabel: 'Input z of the activation functions',
    onInput: (value) => {
      z = value;
      render();
    },
  });

  const derivativeToggle = h('label', { class: 'inline-check' }, h('input', { type: 'checkbox', checked: showDerivative, onChange: (e: Event) => {
    showDerivative = (e.target as HTMLInputElement).checked;
    render();
  } }), 'Show derivatives f′(z) (dashed, thin)');

  const rangeButtons = h(
    'div',
    { class: 'segmented', role: 'group', 'aria-label': 'Range of z' },
    [3, 5, 8].map((r) =>
      h('button', { type: 'button', class: 'segmented__item', dataset: { range: String(r) }, onClick: () => {
        range = r;
        z = Math.max(-range, Math.min(range, z));
        zSlider.setValue(z);
        render();
      } }, `±${r}`),
    ),
  );

  function plot(): SVGSVGElement {
    const active = FUNCTIONS.filter((fn) => enabled.has(fn.id));
    let yMin = -1.2;
    let yMax = 1.2;
    for (const fn of active) {
      const f = info(fn.id).f;
      for (let k = 0; k <= 120; k++) {
        const v = f(-range + (2 * range * k) / 120);
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
      }
    }
    yMin = Math.max(yMin, -range * 0.6) - 0.15;
    yMax = Math.min(yMax, range * 0.8) + 0.15;
    const sx = (v: number) => PAD.left + ((v + range) / (2 * range)) * (W - PAD.left - PAD.right);
    const sy = (v: number) => H - PAD.bottom - ((v - yMin) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);
    const path = (fn: (v: number) => number) => {
      let d = '';
      for (let k = 0; k <= 240; k++) {
        const v = -range + (2 * range * k) / 240;
        d += `${k ? 'L' : 'M'}${sx(v).toFixed(2)},${sy(fn(v)).toFixed(2)}`;
      }
      return d;
    };
    const ticks = [];
    for (let v = -range; v <= range; v += range <= 3 ? 1 : range <= 5 ? 1 : 2) {
      ticks.push(svg('line', { class: 'plot__gridline', x1: sx(v), x2: sx(v), y1: PAD.top, y2: H - PAD.bottom }));
      ticks.push(svg('text', { class: 'plot__tick', x: sx(v), y: H - 8, 'text-anchor': 'middle' }, formatValue(v, 0)));
    }
    for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v++) {
      ticks.push(svg('line', { class: 'plot__gridline', x1: PAD.left, x2: W - PAD.right, y1: sy(v), y2: sy(v) }));
      ticks.push(svg('text', { class: 'plot__tick', x: PAD.left - 6, y: sy(v) + 3, 'text-anchor': 'end' }, formatValue(v, 0)));
    }
    const clipId = `clip-${Math.random().toString(36).slice(2, 8)}`;
    const root = svg(
      'svg',
      { class: 'plot plot--large', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Activation functions plot' },
      svg('defs', null, svg('clipPath', { id: clipId }, svg('rect', { x: PAD.left, y: PAD.top, width: W - PAD.left - PAD.right, height: H - PAD.top - PAD.bottom }))),
      ticks,
      svg('line', { class: 'plot__axis', x1: PAD.left, x2: W - PAD.right, y1: sy(0), y2: sy(0) }),
      svg('line', { class: 'plot__axis', x1: sx(0), x2: sx(0), y1: PAD.top, y2: H - PAD.bottom }),
      svg('text', { class: 'plot__axis-label', x: W - PAD.right - 4, y: sy(0) - 6, 'text-anchor': 'end' }, 'z'),
      svg(
        'g',
        { 'clip-path': `url(#${clipId})` },
        active.map((fn) => [
          svg('path', {
            d: path(info(fn.id).f),
            fill: 'none',
            stroke: fn.color,
            'stroke-width': fn.id === focus ? 3 : 2,
            'stroke-dasharray': fn.dash || undefined,
            opacity: fn.id === focus ? 1 : 0.75,
          }),
          showDerivative
            ? svg('path', { d: path(info(fn.id).df), fill: 'none', stroke: fn.color, 'stroke-width': 1.2, 'stroke-dasharray': '3 3', opacity: 0.8 })
            : null,
        ]),
      ),
      svg('line', { class: 'plot__guide', x1: sx(z), x2: sx(z), y1: PAD.top, y2: H - PAD.bottom }),
      active.map((fn) => svg('circle', { cx: sx(z), cy: sy(info(fn.id).f(z)), r: fn.id === focus ? 5.5 : 4, fill: fn.color, stroke: 'var(--math-bg)', 'stroke-width': 1.5 })),
    );
    const setZ = (event: PointerEvent) => {
      const rect = root.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * W;
      z = Math.max(-range, Math.min(range, -range + ((x - PAD.left) / (W - PAD.left - PAD.right)) * 2 * range));
      z = Math.round(z * 100) / 100;
      zSlider.setValue(z);
      render();
    };
    let dragging = false;
    root.addEventListener('pointerdown', (event) => {
      dragging = true;
      root.setPointerCapture(event.pointerId);
      setZ(event);
    });
    root.addEventListener('pointermove', (event) => {
      if (dragging) setZ(event);
    });
    root.addEventListener('pointerup', () => (dragging = false));
    return root;
  }

  function render(): void {
    for (const { chip, id } of chips) chip.setAttribute('aria-pressed', String(enabled.has(id)));
    for (const b of rangeButtons.querySelectorAll<HTMLElement>('button')) b.setAttribute('aria-pressed', String(Number(b.dataset.range) === range));
    replaceChildren(plotHost, plot());
    replaceChildren(
      table,
      h(
        'table',
        { class: 'data-table' },
        h('thead', null, h('tr', null, h('th', null, 'function'), h('th', null, 'f(z)'), h('th', null, 'f′(z)'), h('th', null, 'range'))),
        h(
          'tbody',
          null,
          FUNCTIONS.filter((fn) => enabled.has(fn.id)).map((fn) => {
            const a = info(fn.id);
            return h(
              'tr',
              { class: fn.id === focus ? 'is-strongest' : '', style: 'cursor:pointer', onClick: () => {
                focus = fn.id;
                render();
              } },
              h('td', null, h('span', { class: 'fn-dot', style: `background:${fn.color}` }), a.label),
              h('td', null, formatValue(a.f(z), 4)),
              h('td', null, formatValue(a.df(z), 4)),
              h('td', { class: 'muted' }, a.range),
            );
          }),
        ),
      ),
    );
    const a = info(focus);
    replaceChildren(
      card,
      h('h3', { class: 'explorer__title' }, a.label),
      tex(`f(z) = ${a.texDefinition}`, { display: true }),
      tex(`f(${texNumber(z, 2)}) = ${a.texEval(z, (v) => texNumber(v, 2))} = ${texNumber(a.f(z), 4)}`, { display: true }),
      tex(`f'(z) = ${a.texDerivative}\\qquad f'(${texNumber(z, 2)}) = ${texNumber(a.df(z), 4)}`, { display: true }),
      h('p', { class: 'insp-text' }, a.description),
      h('p', { class: 'insp-text' }, h('strong', null, 'Gradient: '), flatRegions(focus)),
    );
  }

  render();
  openModal({
    title: 'Activation functions',
    wide: true,
    content: h(
      'div',
      { class: 'explorer' },
      h('p', { class: 'modal__lead' }, 'An activation function turns the weighted sum z of a neuron into its output a = f(z). Without it, a network of many layers would still compute a single linear function.'),
      h('div', { class: 'explorer__chips' }, chips.map((c) => c.chip)),
      h('div', { class: 'explorer__controls' }, zSlider.el, h('div', { class: 'explorer__options' }, rangeButtons, derivativeToggle)),
      plotHost,
      h('div', { class: 'explorer__bottom' }, table, card),
    ),
  });
}
