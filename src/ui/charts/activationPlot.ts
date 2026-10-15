/**
 * Interactive plot of an elementwise activation function a = f(z), with the
 * current neurons drawn as points on the curve (e.g. which ReLU units are in
 * the flat region, which sigmoid units are saturated). Hovering shows f(z).
 */
import { ACTIVATIONS, type ActivationId } from '../../math/activations';
import { formatValue } from '../../utils/format';
import { svg, h } from '../dom';

export interface PlotPoint {
  z: number;
  a: number;
  label: string;
  highlight?: boolean;
}

const W = 300;
const H = 150;
const PAD = { left: 30, right: 12, top: 12, bottom: 22 };

export function activationPlot(id: ActivationId, points: PlotPoint[] = []): HTMLElement {
  const info = ACTIVATIONS[id];
  if (info.kind !== 'elementwise') throw new Error('activationPlot supports elementwise activations only');
  const f = info.f;

  const zExtent = Math.max(4, ...points.map((p) => Math.abs(p.z) * 1.15));
  const zMin = -zExtent;
  const zMax = zExtent;
  const samples = 160;
  const curve: [number, number][] = [];
  let yMin = 0;
  let yMax = 0;
  for (let k = 0; k <= samples; k++) {
    const z = zMin + ((zMax - zMin) * k) / samples;
    const a = f(z);
    curve.push([z, a]);
    yMin = Math.min(yMin, a);
    yMax = Math.max(yMax, a);
  }
  const span = yMax - yMin || 1;
  yMin -= span * 0.08;
  yMax += span * 0.08;

  const sx = (z: number) => PAD.left + ((z - zMin) / (zMax - zMin)) * (W - PAD.left - PAD.right);
  const sy = (a: number) => H - PAD.bottom - ((a - yMin) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);
  const zFromX = (x: number) => zMin + ((x - PAD.left) / (W - PAD.left - PAD.right)) * (zMax - zMin);

  const path = curve.map(([z, a], k) => `${k ? 'L' : 'M'}${sx(z).toFixed(2)},${sy(a).toFixed(2)}`).join('');
  const guide = svg('line', { class: 'plot__guide', y1: PAD.top, y2: H - PAD.bottom, visibility: 'hidden' });
  const guideDot = svg('circle', { class: 'plot__guide-dot', r: 3.5, visibility: 'hidden' });
  const readout = h('div', { class: 'plot__readout' }, `${info.label}: hover the plot to evaluate f(z)`);

  const root = svg(
    'svg',
    { class: 'plot', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Plot of the ${info.label} activation function` },
    // axes through the origin
    svg('line', { class: 'plot__axis', x1: PAD.left, x2: W - PAD.right, y1: sy(0), y2: sy(0) }),
    svg('line', { class: 'plot__axis', x1: sx(0), x2: sx(0), y1: PAD.top, y2: H - PAD.bottom }),
    svg('text', { class: 'plot__tick', x: PAD.left, y: H - 6, 'text-anchor': 'start' }, formatValue(zMin, 1)),
    svg('text', { class: 'plot__tick', x: W - PAD.right, y: H - 6, 'text-anchor': 'end' }, formatValue(zMax, 1)),
    svg('text', { class: 'plot__tick', x: PAD.left - 4, y: PAD.top + 8, 'text-anchor': 'end' }, formatValue(yMax, 1)),
    svg('text', { class: 'plot__tick', x: PAD.left - 4, y: H - PAD.bottom, 'text-anchor': 'end' }, formatValue(yMin, 1)),
    svg('text', { class: 'plot__axis-label', x: W - PAD.right - 2, y: sy(0) - 5, 'text-anchor': 'end' }, 'z'),
    svg('text', { class: 'plot__axis-label', x: sx(0) + 5, y: PAD.top + 8 }, 'a'),
    svg('path', { class: 'plot__curve', d: path }),
    guide,
    guideDot,
    points.map((p) =>
      svg(
        'circle',
        {
          class: `plot__point${p.highlight ? ' is-highlight' : ''}${p.a < 0 ? ' is-neg' : ''}`,
          cx: sx(Math.max(zMin, Math.min(zMax, p.z))),
          cy: sy(p.a),
          r: p.highlight ? 5 : 3.5,
        },
        svg('title', null, `${p.label}: z = ${formatValue(p.z)}, a = ${formatValue(p.a)}`),
      ),
    ),
  );

  root.addEventListener('pointermove', (event) => {
    const rect = root.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * W;
    const z = zFromX(x);
    if (z < zMin || z > zMax) return;
    const a = f(z);
    guide.setAttribute('x1', String(sx(z)));
    guide.setAttribute('x2', String(sx(z)));
    guide.setAttribute('visibility', 'visible');
    guideDot.setAttribute('cx', String(sx(z)));
    guideDot.setAttribute('cy', String(sy(a)));
    guideDot.setAttribute('visibility', 'visible');
    readout.textContent = `${info.label}(${formatValue(z, 2)}) = ${formatValue(a, 3)}`;
  });
  root.addEventListener('pointerleave', () => {
    guide.setAttribute('visibility', 'hidden');
    guideDot.setAttribute('visibility', 'hidden');
    readout.textContent = `${info.label}: hover the plot to evaluate f(z)`;
  });

  return h('figure', { class: 'plot-figure' }, root, readout);
}

/**
 * Softmax "plot": one bar per output neuron (probability), with its logit z.
 */
export function softmaxBars(z: ArrayLike<number>, p: ArrayLike<number>, labels: string[], highlight: number | null): HTMLElement {
  const rows = [];
  for (let j = 0; j < p.length; j++) {
    const fill = h('span', { class: 'bars__fill' });
    fill.style.width = `${(p[j] * 100).toFixed(1)}%`;
    rows.push(
      h(
        'div',
        { class: `bars__row${highlight === j ? ' is-highlight' : ''}` },
        h('span', { class: 'bars__label' }, labels[j]),
        h('span', { class: 'bars__track' }, fill),
        h('span', { class: 'bars__value' }, `${formatValue(p[j] * 100, 1)}%`),
        h('span', { class: 'bars__logit' }, `z = ${formatValue(z[j], 2)}`),
      ),
    );
  }
  return h('div', { class: 'bars', role: 'img', 'aria-label': 'Softmax probabilities per output neuron' }, rows);
}
