/** Small displays used by the Experiments inspectors. */
import { METRICS } from '../../experiments/metrics';
import type { Effect, MetricId, Prediction, ScaleCurve } from '../../experiments/types';
import { maxAbs } from '../../math/stats';
import { h, svg, type Child } from '../../ui/dom';
import { kv, num } from '../../ui/widgets';
import { formatProbability, formatSigned, formatValue } from '../../utils/format';
import { barRows } from '../transformer/widgets';
import type { Lazy } from './ExperimentsWorkspace';

/** The four measures of an effect as key/value rows; the chosen metric is highlighted. */
export function effectRows(effect: Effect, metric: MetricId, o: { contrast: boolean }): HTMLElement {
  const row = (m: MetricId, value: number | null, text: Child) => kv(METRICS[m].symbol, value === null ? '—' : text, m === metric ? 'is-highlight' : '');
  return h(
    'div',
    { class: 'kv-list' },
    row('logp', effect.dlogp, num(effect.dlogp, { signed: true })),
    o.contrast ? row('logit', effect.dlogit, num(effect.dlogit ?? 0, { signed: true })) : null,
    row('p', effect.dp, h('span', { class: `num ${effect.dp > 0 ? 'num--pos' : effect.dp < 0 ? 'num--neg' : 'num--zero'}` }, `${formatSigned(effect.dp * 100, 1)} pts`)),
    row('kl', effect.kl, `${formatValue(effect.kl, effect.kl < 0.01 ? 4 : 3)} bits`),
  );
}

/** Next-token probabilities after the interventions (bars) over the original ones (ghost bars). */
export function compareBars(before: Prediction[], after: Prediction[], format: (t: string) => string, o: { target: number; limit?: number }): HTMLElement {
  const limit = o.limit ?? 8;
  const ids: number[] = [];
  for (const t of [...after.slice(0, limit), ...before.slice(0, limit)]) if (!ids.includes(t.id)) ids.push(t.id);
  const pOf = (list: Prediction[], id: number) => list.find((t) => t.id === id)?.p;
  const rows = ids
    .map((id) => {
      const text = (after.find((t) => t.id === id) ?? before.find((t) => t.id === id))!.text;
      const b = pOf(before, id);
      const a = pOf(after, id);
      return { id, text, b, a };
    })
    .sort((x, y) => (y.a ?? 0) - (x.a ?? 0))
    .slice(0, limit + 2);
  return h(
    'div',
    { class: 'stack' },
    barRows(
      rows.map((r) => ({
        label: format(r.text),
        value: r.a ?? 0,
        ghost: r.b ?? 0,
        highlight: r.id === o.target,
        text: `${r.b === undefined ? '…' : formatProbability(r.b)} → ${r.a === undefined ? '…' : formatProbability(r.a)}`,
        title: r.a === undefined || r.b === undefined ? 'Outside the top 10 of one of the two runs' : undefined,
      })),
      { max: 1 },
    ),
    h('div', { class: 'muted ex-bars-legend' }, h('span', { class: 'ex-ghost-swatch' }), ' original model  ', h('span', { class: 'ex-fill-swatch' }), ' with the interventions'),
  );
}

/**
 * p(target) (or the logit difference) as a function of a component's scale:
 * 0 = switched off, 1 = untouched. The original value is the dashed line.
 */
export function scaleCurveChart(curve: ScaleCurve, o: { current: number; mode: 'p' | 'logit'; title: string }): HTMLElement {
  const W = 320;
  const H = 150;
  const pad = { left: 40, right: 10, top: 10, bottom: 26 };
  const ys = o.mode === 'logit' && curve.logitDiff ? curve.logitDiff : curve.p;
  const xs = curve.scales;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = o.mode === 'p' ? 0 : Math.min(...ys, 0);
  let yMax = o.mode === 'p' ? 1 : Math.max(...ys, 0);
  if (yMax - yMin < 1e-6) {
    yMin -= 1;
    yMax += 1;
  }
  const sx = (x: number) => pad.left + ((x - xMin) / (xMax - xMin)) * (W - pad.left - pad.right);
  const sy = (y: number) => H - pad.bottom - ((y - yMin) / (yMax - yMin)) * (H - pad.top - pad.bottom);
  const at = (s: number) => {
    // linear interpolation between measured scales
    for (let k = 1; k < xs.length; k++) {
      if (s <= xs[k]) {
        const t = (s - xs[k - 1]) / (xs[k] - xs[k - 1] || 1);
        return ys[k - 1] + t * (ys[k] - ys[k - 1]);
      }
    }
    return ys[ys.length - 1];
  };
  const original = at(1);
  const path = xs.map((x, k) => `${k ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(ys[k]).toFixed(1)}`).join(' ');
  const ticksX = [xMin, 0, 1, 2, xMax].filter((v, i, a) => a.indexOf(v) === i && v >= xMin && v <= xMax);
  const fmtY = (v: number) => (o.mode === 'p' ? formatProbability(v) : formatValue(v, 1));
  const ticksY = o.mode === 'p' ? [0, 0.5, 1] : [yMin, (yMin + yMax) / 2, yMax];
  const current = Math.max(xMin, Math.min(xMax, o.current));
  const chart = svg(
    'svg',
    { class: 'ex-curve', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.title },
    ticksY.map((t) => [svg('line', { class: 'ex-curve__grid', x1: pad.left, x2: W - pad.right, y1: sy(t), y2: sy(t) }), svg('text', { class: 'ex-curve__tick', x: pad.left - 4, y: sy(t) + 3, 'text-anchor': 'end' }, fmtY(t))]),
    ticksX.map((t) => [svg('line', { class: `ex-curve__grid${t === 0 ? ' is-zero' : ''}`, x1: sx(t), x2: sx(t), y1: pad.top, y2: H - pad.bottom }), svg('text', { class: 'ex-curve__tick', x: sx(t), y: H - pad.bottom + 12, 'text-anchor': 'middle' }, t === 0 ? '0 off' : t === 1 ? '1 orig.' : `×${t}`)]),
    svg('line', { class: 'ex-curve__original', x1: pad.left, x2: W - pad.right, y1: sy(original), y2: sy(original) }),
    svg('path', { class: 'ex-curve__line', d: path }),
    xs.map((x, k) => svg('circle', { class: 'ex-curve__point', cx: sx(x), cy: sy(ys[k]), r: 2.2 })),
    svg('circle', { class: 'ex-curve__current', cx: sx(current), cy: sy(at(current)), r: 5 }),
    svg('text', { class: 'ex-curve__label', x: W - pad.right, y: H - 4, 'text-anchor': 'end' }, 'scale'),
  );
  return h('figure', { class: 'ex-curve-figure' }, h('figcaption', { class: 'line-chart__title' }, o.title), chart, h('div', { class: 'muted' }, `Now: ${fmtY(at(current))} at scale ${formatValue(current, 2)} · dashed line: the original model.`));
}

/** Placeholder / error / value of an on-demand analysis. */
export function lazyBlock<T>(lazy: Lazy<T>, render: (value: T) => Child, pendingText = 'Computing…'): Child {
  if (lazy.error) return h('div', { class: 'muted' }, lazy.error);
  if (lazy.pending || lazy.value === null) return h('div', { class: 'muted ex-pending' }, pendingText);
  return render(lazy.value);
}

/** Many units in one strip (canvas): bar height = |value|, colour = sign; the chosen units are marked. */
export function unitStrip(values: ArrayLike<number>, o: { colors: { positive: string; negative: string }; marked?: number[]; onClick?(unit: number): void; title: string }): HTMLElement {
  const n = values.length;
  const width = Math.min(n, 768);
  const per = n / width;
  const canvas = h('canvas', { class: 'ex-strip', width, height: 48, 'aria-label': o.title });
  const ctx = canvas.getContext('2d');
  const scale = maxAbs(values) || 1;
  if (ctx) {
    for (let x = 0; x < width; x++) {
      // several units per pixel for wide layers: draw the strongest
      let v = 0;
      for (let u = Math.floor(x * per); u < Math.floor((x + 1) * per); u++) if (Math.abs(values[u]) > Math.abs(v)) v = values[u];
      const t = Math.abs(v) / scale;
      ctx.fillStyle = v < 0 ? o.colors.negative : o.colors.positive;
      const hgt = Math.max(1, t * 22);
      ctx.fillRect(x, v < 0 ? 24 : 24 - hgt, 1, hgt);
    }
    ctx.fillStyle = 'rgba(128,128,128,0.5)';
    ctx.fillRect(0, 24, width, 0.5);
  }
  const readout = h('div', { class: 'attn-grid__readout' }, o.title);
  const unitAt = (event: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    return Math.min(n - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * n)));
  };
  canvas.addEventListener('pointermove', (event) => {
    const u = unitAt(event);
    readout.textContent = `neuron #${u}: ${formatSigned(values[u], 4)}`;
  });
  if (o.onClick) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (event) => o.onClick?.(unitAt(event)));
  }
  const marks = h('div', { class: 'ex-strip__marks' }, (o.marked ?? []).map((u) => {
    const m = h('span', { class: 'ex-strip__mark' });
    m.style.left = `${((u + 0.5) / n) * 100}%`;
    return m;
  }));
  return h('div', { class: 'ex-strip-wrap' }, h('div', { class: 'ex-strip-box' }, canvas, marks), readout);
}
