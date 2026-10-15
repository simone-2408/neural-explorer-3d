/**
 * Small canvas line chart (loss, accuracy, gradient norms…), redrawn cheaply
 * every frame during training. Colours come from CSS variables so it follows
 * the theme.
 */
import { formatValue } from '../../utils/format';
import { h } from '../dom';

export interface Series {
  label: string;
  color: string;
  values: readonly number[];
  /** Optional x values (default: index). */
  xs?: readonly number[];
  dash?: number[];
  width?: number;
}

export interface LineChartOptions {
  title: string;
  /** Logarithmic y axis (values ≤ 0 are skipped). */
  log?: boolean;
  yMin?: number;
  yMax?: number;
  xLabel?: string;
  /** Formats the latest value in the legend. */
  format?: (v: number) => string;
  height?: number;
}

function cssVar(el: Element, name: string, fallback: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim() || fallback;
}

export class LineChart {
  readonly el: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly legend: HTMLElement;
  private options: LineChartOptions;

  constructor(options: LineChartOptions) {
    this.options = options;
    this.canvas = h('canvas', { class: 'line-chart__canvas', 'aria-label': options.title });
    this.canvas.style.height = `${options.height ?? 110}px`;
    this.legend = h('div', { class: 'line-chart__legend' });
    this.el = h('figure', { class: 'line-chart' }, h('figcaption', { class: 'line-chart__title' }, options.title), this.canvas, this.legend);
  }

  setOptions(patch: Partial<LineChartOptions>): void {
    this.options = { ...this.options, ...patch };
  }

  draw(series: readonly Series[]): void {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = canvas.clientWidth || 300;
    const height = canvas.clientHeight || 110;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const grid = cssVar(this.el, '--border', '#333');
    const text = cssVar(this.el, '--text-faint', '#888');
    const pad = { left: 38, right: 8, top: 6, bottom: 16 };
    const { log } = this.options;
    const tf = (v: number) => (log ? Math.log10(v) : v);

    let xMax = 1;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const s of series) {
      for (let k = 0; k < s.values.length; k++) {
        const v = s.values[k];
        if (!Number.isFinite(v) || (log && v <= 0)) continue;
        yMin = Math.min(yMin, tf(v));
        yMax = Math.max(yMax, tf(v));
        xMax = Math.max(xMax, s.xs ? s.xs[k] : k);
      }
    }
    if (this.options.yMin !== undefined) yMin = tf(this.options.yMin);
    if (this.options.yMax !== undefined) yMax = tf(this.options.yMax);
    if (!Number.isFinite(yMin)) {
      yMin = 0;
      yMax = 1;
    }
    if (yMax - yMin < 1e-9) {
      yMax += 0.5;
      yMin -= 0.5;
    }
    const sx = (x: number) => pad.left + (x / xMax) * (width - pad.left - pad.right);
    const sy = (y: number) => height - pad.bottom - ((y - yMin) / (yMax - yMin)) * (height - pad.top - pad.bottom);

    // grid + y labels
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.fillStyle = text;
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    const ticks = log ? Array.from({ length: Math.floor(yMax) - Math.ceil(yMin) + 1 }, (_, k) => Math.ceil(yMin) + k) : [yMin, (yMin + yMax) / 2, yMax];
    for (const t of ticks.slice(0, 8)) {
      const y = sy(t);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillText(log ? `1e${t}` : formatValue(t, Math.abs(yMax - yMin) < 3 ? 2 : 1), 2, y + 3);
    }
    if (this.options.xLabel) ctx.fillText(`${this.options.xLabel}: ${formatValue(xMax, 0)}`, pad.left, height - 3);

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.6;
      ctx.setLineDash(s.dash ?? []);
      ctx.beginPath();
      let started = false;
      for (let k = 0; k < s.values.length; k++) {
        const v = s.values[k];
        if (!Number.isFinite(v) || (log && v <= 0)) {
          started = false;
          continue;
        }
        const x = sx(s.xs ? s.xs[k] : k);
        const y = sy(Math.min(yMax, Math.max(yMin, tf(v))));
        if (started) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        started = true;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const format = this.options.format ?? ((v: number) => formatValue(v, 3));
    this.legend.replaceChildren(
      ...series.map((s) => {
        const last = s.values.length ? s.values[s.values.length - 1] : NaN;
        const swatch = h('span', { class: 'line-chart__swatch' });
        swatch.style.background = s.color;
        if (s.dash) swatch.style.backgroundImage = `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`;
        if (s.dash) swatch.style.background = '';
        return h('span', { class: 'line-chart__item' }, swatch, `${s.label} `, h('strong', null, Number.isFinite(last) ? format(last) : '—'));
      }),
    );
  }
}
