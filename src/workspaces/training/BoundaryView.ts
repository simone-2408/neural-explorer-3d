/**
 * Decision-boundary view: the 2-D input plane coloured by the network's
 * prediction, the boundary line (p = 0.5) and the data points.
 *
 * Classes are told apart by shape AND colour (circle / triangle / square);
 * training points are filled, test points hollow, misclassified points get a
 * ring. Clicking feeds that point (or any point of the plane) to the network
 * shown in 3D.
 */
import { createTrace, forwardInto } from '../../neural/forward';
import type { NetworkModel } from '../../neural/NetworkModel';
import { formatValue } from '../../utils/format';
import { h } from '../../ui/dom';
import type { ScenePalette } from '../../visualization/palette';
import type { Dataset } from '../../training/datasets';
import { predictedClass } from '../../training/loss';

export type Probe = { kind: 'sample'; index: number } | { kind: 'point'; x: number; y: number };

export const DOMAIN = 1.3;
const GRID = 64;
const THIRD_CLASS = '#009E73';

export function classColors(palette: ScenePalette): string[] {
  return [palette.negative, palette.positive, THIRD_CLASS];
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
}

/** Segments of the p = 0.5 iso-line on a grid (marching squares). */
function contour(values: Float64Array, n: number, level: number): [number, number, number, number][] {
  const segments: [number, number, number, number][] = [];
  const at = (r: number, c: number) => values[r * n + c] - level;
  const lerp = (a: number, b: number) => a / (a - b);
  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const tl = at(r, c);
      const tr = at(r, c + 1);
      const br = at(r + 1, c + 1);
      const bl = at(r + 1, c);
      const points: [number, number][] = [];
      if (tl > 0 !== tr > 0) points.push([c + lerp(tl, tr), r]);
      if (tr > 0 !== br > 0) points.push([c + 1, r + lerp(tr, br)]);
      if (bl > 0 !== br > 0) points.push([c + lerp(bl, br), r + 1]);
      if (tl > 0 !== bl > 0) points.push([c, r + lerp(tl, bl)]);
      if (points.length === 2) segments.push([points[0][0], points[0][1], points[1][0], points[1][1]]);
      else if (points.length === 4) {
        segments.push([points[0][0], points[0][1], points[1][0], points[1][1]]);
        segments.push([points[2][0], points[2][1], points[3][0], points[3][1]]);
      }
    }
  }
  return segments;
}

export class BoundaryView {
  readonly el: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly readout = h('div', { class: 'boundary__readout' }, 'Hover the plane to read the prediction; click a point to send it through the network.');
  private readonly offscreen = document.createElement('canvas');
  private readonly model: NetworkModel;
  private readonly getDataset: () => Dataset;
  private readonly getPalette: () => ScenePalette;
  private readonly onProbe: (probe: Probe) => void;
  private probe: Probe | null = null;
  showTest = true;
  private readonly grid = new Float64Array(GRID * GRID);
  private readonly gridClass = new Uint8Array(GRID * GRID);

  constructor(options: { model: NetworkModel; dataset: () => Dataset; palette: () => ScenePalette; onProbe: (probe: Probe) => void }) {
    this.model = options.model;
    this.getDataset = options.dataset;
    this.getPalette = options.palette;
    this.onProbe = options.onProbe;
    this.canvas = h('canvas', { class: 'boundary__canvas', 'aria-label': 'Decision boundary and data points' });
    this.offscreen.width = GRID;
    this.offscreen.height = GRID;
    this.el = h('div', { class: 'boundary' }, this.canvas, this.readout);

    this.canvas.addEventListener('pointermove', (event) => {
      const [x, y] = this.toData(event);
      const p = this.predict(x, y);
      const ds = this.getDataset();
      this.readout.textContent =
        ds.classes > 2
          ? `x₁ = ${formatValue(x, 2)}, x₂ = ${formatValue(y, 2)} → predicted class ${p.cls} (p = ${formatValue(p.score, 2)})`
          : `x₁ = ${formatValue(x, 2)}, x₂ = ${formatValue(y, 2)} → p(class 1) = ${formatValue(p.score, 3)}`;
    });
    this.canvas.addEventListener('click', (event) => {
      const [x, y] = this.toData(event);
      const index = this.nearestSample(event);
      this.onProbe(index >= 0 ? { kind: 'sample', index } : { kind: 'point', x, y });
    });
  }

  setProbe(probe: Probe | null): void {
    this.probe = probe;
    this.render();
  }

  render(): void {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = Math.max(120, Math.floor(canvas.clientWidth || 320));
    if (canvas.width !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
    }
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const palette = this.getPalette();
    const ds = this.getDataset();
    const colors = classColors(palette);
    const rgb = colors.map(hexToRgb);
    const bg = hexToRgb(palette.background);

    // 1. Prediction map on a GRID × GRID lattice.
    const off = this.offscreen.getContext('2d')!;
    const image = off.createImageData(GRID, GRID);
    const multi = ds.classes > 2;
    const evaluation = this.evalContext();
    for (let r = 0; r < GRID; r++) {
      const y = DOMAIN - (2 * DOMAIN * r) / (GRID - 1);
      for (let c = 0; c < GRID; c++) {
        const x = -DOMAIN + (2 * DOMAIN * c) / (GRID - 1);
        const p = this.predict(x, y, evaluation);
        const k = r * GRID + c;
        this.grid[k] = multi ? p.score : p.p1;
        this.gridClass[k] = p.cls;
        let color: [number, number, number];
        let strength: number;
        if (multi) {
          color = rgb[p.cls] ?? rgb[0];
          strength = Math.max(0, (p.score - 1 / ds.classes) / (1 - 1 / ds.classes));
        } else {
          color = p.p1 >= 0.5 ? rgb[1] : rgb[0];
          strength = Math.abs(p.p1 - 0.5) * 2;
        }
        const t = 0.12 + 0.5 * strength;
        image.data[4 * k] = bg[0] + (color[0] - bg[0]) * t;
        image.data[4 * k + 1] = bg[1] + (color[1] - bg[1]) * t;
        image.data[4 * k + 2] = bg[2] + (color[2] - bg[2]) * t;
        image.data[4 * k + 3] = 255;
      }
    }
    off.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.offscreen, 0, 0, size, size);

    // 2. Axes.
    const toPx = (v: number) => ((v + DOMAIN) / (2 * DOMAIN)) * size;
    const toPy = (v: number) => ((DOMAIN - v) / (2 * DOMAIN)) * size;
    ctx.strokeStyle = palette.theme === 'dark' ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toPx(0), 0);
    ctx.lineTo(toPx(0), size);
    ctx.moveTo(0, toPy(0));
    ctx.lineTo(size, toPy(0));
    ctx.stroke();
    ctx.fillStyle = palette.theme === 'dark' ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.55)';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('x₁', size - 16, toPy(0) - 5);
    ctx.fillText('x₂', toPx(0) + 5, 12);

    // 3. Decision boundary.
    const segments = multi ? this.classEdges() : contour(this.grid, GRID, 0.5);
    ctx.strokeStyle = palette.theme === 'dark' ? 'rgba(255,255,255,0.85)' : 'rgba(10,15,20,0.85)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const scale = size / (GRID - 1);
    for (const [x1, y1, x2, y2] of segments) {
      ctx.moveTo(x1 * scale, y1 * scale);
      ctx.lineTo(x2 * scale, y2 * scale);
    }
    ctx.stroke();

    // 4. Data points.
    const isTest = new Uint8Array(ds.size);
    for (const k of ds.test) isTest[k] = 1;
    const outline = palette.theme === 'dark' ? '#0b0f15' : '#ffffff';
    const warn = palette.theme === 'dark' ? '#ffffff' : '#000000';
    for (let k = 0; k < ds.size; k++) {
      if (isTest[k] && !this.showTest) continue;
      const x = ds.x[2 * k];
      const y = ds.x[2 * k + 1];
      const wrong = this.predict(x, y, evaluation).cls !== ds.y[k];
      this.drawPoint(ctx, toPx(x), toPy(y), ds.y[k], colors[ds.y[k]], !!isTest[k], wrong, outline, warn);
    }

    // 5. Probe.
    const probe = this.probe;
    if (probe) {
      const px = probe.kind === 'sample' ? ds.x[2 * probe.index] : probe.x;
      const py = probe.kind === 'sample' ? ds.x[2 * probe.index + 1] : probe.y;
      ctx.strokeStyle = warn;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(toPx(px), toPy(py), 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(toPx(px) - 13, toPy(py));
      ctx.lineTo(toPx(px) - 6, toPy(py));
      ctx.moveTo(toPx(px) + 6, toPy(py));
      ctx.lineTo(toPx(px) + 13, toPy(py));
      ctx.stroke();
    }
  }

  dispose(): void {
    this.el.remove();
  }

  // -------------------------------------------------------------- internals

  private predict(x: number, y: number, ctx = this.evalContext()): { p1: number; cls: number; score: number } {
    const { spec } = this.model;
    ctx.input[0] = x;
    ctx.input[1] = y;
    forwardInto(spec, ctx.weights, ctx.biases, ctx.input, ctx.trace);
    const out = ctx.trace.a[spec.layers.length - 1];
    const cls = predictedClass(spec, out);
    if (out.length === 1) {
      const activation = spec.layers[spec.layers.length - 1].activation;
      const p1 = activation === 'sigmoid' ? out[0] : activation === 'tanh' ? (out[0] + 1) / 2 : 1 / (1 + Math.exp(-out[0]));
      return { p1, cls, score: cls === 1 ? p1 : 1 - p1 };
    }
    return { p1: out[1] ?? 0, cls, score: out[cls] };
  }

  /** Current parameters + scratch buffers for a series of evaluations. */
  private evalContext() {
    const { spec } = this.model;
    return {
      weights: Array.from({ length: this.model.layerCount }, (_, l) => this.model.weights(l) as Float64Array),
      biases: Array.from({ length: this.model.layerCount }, (_, l) => this.model.biases(l) as Float64Array),
      trace: createTrace(spec.layers.map((l) => l.size)),
      input: new Float64Array(2),
    };
  }

  /** Boundaries between predicted classes (multi-class): segments between grid cells of different class. */
  private classEdges(): [number, number, number, number][] {
    const segments: [number, number, number, number][] = [];
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const k = r * GRID + c;
        if (c + 1 < GRID && this.gridClass[k] !== this.gridClass[k + 1]) segments.push([c + 0.5, r - 0.5, c + 0.5, r + 0.5]);
        if (r + 1 < GRID && this.gridClass[k] !== this.gridClass[k + GRID]) segments.push([c - 0.5, r + 0.5, c + 0.5, r + 0.5]);
      }
    }
    return segments;
  }

  private drawPoint(ctx: CanvasRenderingContext2D, x: number, y: number, cls: number, color: string, hollow: boolean, wrong: boolean, outline: string, warn: string): void {
    const r = 4.6;
    ctx.beginPath();
    if (cls === 0) ctx.arc(x, y, r, 0, Math.PI * 2);
    else if (cls === 1) {
      ctx.moveTo(x, y - r * 1.2);
      ctx.lineTo(x + r * 1.1, y + r * 0.8);
      ctx.lineTo(x - r * 1.1, y + r * 0.8);
      ctx.closePath();
    } else ctx.rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8);
    if (hollow) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = outline;
      ctx.stroke();
    }
    if (wrong) {
      ctx.beginPath();
      ctx.arc(x, y, r + 3, 0, Math.PI * 2);
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = warn;
      ctx.stroke();
    }
  }

  private toData(event: MouseEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const x = -DOMAIN + ((event.clientX - rect.left) / rect.width) * 2 * DOMAIN;
    const y = DOMAIN - ((event.clientY - rect.top) / rect.height) * 2 * DOMAIN;
    return [x, y];
  }

  private nearestSample(event: MouseEvent): number {
    const rect = this.canvas.getBoundingClientRect();
    const ds = this.getDataset();
    let best = -1;
    let bestD = 10;
    for (let k = 0; k < ds.size; k++) {
      const px = ((ds.x[2 * k] + DOMAIN) / (2 * DOMAIN)) * rect.width;
      const py = ((DOMAIN - ds.x[2 * k + 1]) / (2 * DOMAIN)) * rect.height;
      const d = Math.hypot(event.clientX - rect.left - px, event.clientY - rect.top - py);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
    return best;
  }
}
