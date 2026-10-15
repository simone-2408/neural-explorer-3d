/**
 * CnnView — a convolutional network in 3D, flowing bottom → top like the
 * other tabs: the input image at the bottom, then each layer's feature maps
 * side by side (one map per filter), the pooled maps, and the output classes
 * at the top.
 *
 * Every cell is a box facing the viewer; its value pushes it towards the
 * viewer (positive) or behind its map (negative), and sets its colour. All
 * cells share one size (one pixel of the input = one cell), so downsampling
 * is visible: a map after 2×2 pooling is half as wide.
 *
 * Selecting a cell shows what it reads: the k×k window of every map below it
 * (frames), one line per weight (colour = sign of the weight, opacity =
 * size of its contribution), and the whole region of the input image that
 * can influence it.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { receptiveField } from '../cnn/analysis';
import type { CnnTrace } from '../cnn/forward';
import type { CnnModel } from '../cnn/model';
import { formatProbability, formatSigned } from '../utils/format';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import type { Viewport } from './Viewport';

export type CnnStageKind = 'input' | 'conv' | 'pool' | 'output';

export interface CnnStage {
  kind: CnnStageKind;
  /** Conv layer index (conv / pool stages), −1 otherwise. */
  layer: number;
  maps: number;
  /** Width = height of each map (1 for the output stage). */
  size: number;
}

export type CnnPick =
  | { kind: 'cell'; stage: number; map: number; i: number; j: number }
  | { kind: 'map'; stage: number; map: number }
  | { kind: 'output'; index: number };

const P = 0.24;
const CELL = 0.2;
const DEPTH = 0.05;
const MAX_Z = 0.45;
const MAP_GAP = 0.55;
const ROW_GAP = 1.7;
/** Minimum spacing of the output cells; long class names get more (their labels have a fixed pixel size). */
const OUT_PITCH = 2.3;
const OUT_PITCH_PER_CHAR = 0.32;

interface Label {
  object: CSS2DObject;
  el: HTMLElement;
}

/** The stages drawn for a model: input, (conv, pool) per layer, output. */
export function cnnStages(model: CnnModel): CnnStage[] {
  const stages: CnnStage[] = [{ kind: 'input', layer: -1, maps: 1, size: model.config.size }];
  model.shapes.forEach((s, l) => {
    stages.push({ kind: 'conv', layer: l, maps: s.outChannels, size: s.size });
    if (s.pool !== 'none') stages.push({ kind: 'pool', layer: l, maps: s.outChannels, size: s.pooledSize });
  });
  stages.push({ kind: 'output', layer: -1, maps: model.config.classes, size: 1 });
  return stages;
}

/** Values of a stage, [map, row, column]. */
export function stageValues(stage: CnnStage, trace: CnnTrace): Float64Array {
  switch (stage.kind) {
    case 'input':
      return trace.input;
    case 'conv':
      return trace.layers[stage.layer].act;
    case 'pool':
      return trace.layers[stage.layer].pooled;
    case 'output':
      return trace.probs;
  }
}

export class CnnView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: CnnPick) => void;
  private model: CnnModel | null = null;
  private trace: CnnTrace | null = null;
  private stages: CnnStage[] = [];
  private offsets: number[] = [];
  private rowY: number[] = [];
  private classNames: string[] = [];
  private cells: THREE.InstancedMesh | null = null;
  private pick: THREE.InstancedMesh | null = null;
  private readonly cellMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  private readonly pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
  private readonly lines: THREE.LineSegments;
  private readonly frames: THREE.LineSegments;
  private readonly labels = new THREE.Group();
  private stageLabels: Label[] = [];
  private mapLabels: Label[][] = [];
  private outputLabels: Label[] = [];
  private selection: CnnPick | null = null;
  private hover: CnnPick | null = null;
  /** Slide animation: in this map, only the first `count` cells are computed so far. */
  private reveal: { stage: number; map: number; count: number } | null = null;
  private showValues = true;
  private readonly raycaster = new THREE.Raycaster();

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: CnnPick): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.root.name = 'CnnView';
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
    this.frames = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }));
    for (const l of [this.lines, this.frames]) {
      l.frustumCulled = false;
      l.raycast = () => {};
    }
    const lights = new THREE.Group();
    lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(-3, 6, 9);
    lights.add(sun);
    this.root.add(lights, this.lines, this.frames, this.labels);
    (this.frames.material as THREE.LineBasicMaterial).color.set(o.palette.plateFocus);
    o.viewport.scene.add(this.root);
  }

  // ----------------------------------------------------------------- layout

  get stageList(): readonly CnnStage[] {
    return this.stages;
  }

  private mapWidth(stage: CnnStage): number {
    return stage.kind === 'output' ? CELL : stage.size * P;
  }

  private get outPitch(): number {
    const longest = Math.max(1, ...this.classNames.map((name) => name.length));
    return Math.max(OUT_PITCH, OUT_PITCH_PER_CHAR * longest);
  }

  private rowWidth(stage: CnnStage): number {
    if (stage.kind === 'output') return (stage.maps - 1) * this.outPitch;
    return stage.maps * this.mapWidth(stage) + (stage.maps - 1) * MAP_GAP;
  }

  /** Centre of map `map` of stage `s`. */
  mapCenter(s: number, map: number): Vec3 {
    const stage = this.stages[s];
    if (stage.kind === 'output') return [-this.rowWidth(stage) / 2 + map * this.outPitch, this.rowY[s], 0];
    const w = this.mapWidth(stage);
    return [-this.rowWidth(stage) / 2 + map * (w + MAP_GAP) + w / 2, this.rowY[s], 0];
  }

  /** Centre of cell (i, j) of map `map` of stage `s` (row i counted from the top, like an image). */
  cellCenter(s: number, map: number, i: number, j: number): Vec3 {
    const stage = this.stages[s];
    const [cx, cy] = this.mapCenter(s, map);
    if (stage.kind === 'output') return [cx, cy, 0];
    const half = ((stage.size - 1) * P) / 2;
    return [cx - half + j * P, cy + half - i * P, 0];
  }

  bounds(): { min: Vec3; max: Vec3 } {
    const half = Math.max(2, ...this.stages.map((s) => this.rowWidth(s) / 2));
    // The extra room on the left keeps the row titles (HTML labels left of each row) in view.
    return { min: [-half - 4.4, -2.2 - (this.stages[0] ? (this.stages[0].size * P) / 2 : 0), -MAX_Z], max: [half + 0.8, (this.rowY[this.rowY.length - 1] ?? 0) + 1.6, MAX_Z] };
  }

  private computeLayout(): void {
    this.rowY = [];
    let y = 0;
    this.stages.forEach((stage, s) => {
      const h = stage.kind === 'output' ? 0.6 : stage.size * P;
      if (s === 0) y = 0;
      else {
        const prev = this.stages[s - 1];
        const prevH = prev.kind === 'output' ? 0.6 : prev.size * P;
        y += prevH / 2 + ROW_GAP + h / 2;
      }
      this.rowY.push(y);
    });
    this.offsets = [];
    let n = 0;
    for (const stage of this.stages) {
      this.offsets.push(n);
      n += stage.maps * stage.size * stage.size;
    }
  }

  private get cellCount(): number {
    const last = this.stages[this.stages.length - 1];
    return this.offsets[this.offsets.length - 1] + last.maps * last.size * last.size;
  }

  /** Stage, map and position of instance k. */
  private decode(k: number): { s: number; map: number; i: number; j: number } {
    let s = this.offsets.length - 1;
    while (s > 0 && this.offsets[s] > k) s--;
    const stage = this.stages[s];
    const local = k - this.offsets[s];
    const per = stage.size * stage.size;
    const map = Math.floor(local / per);
    const q = local % per;
    return { s, map, i: Math.floor(q / stage.size), j: q % stage.size };
  }

  // ------------------------------------------------------------------- data

  /** New model (rebuilds the scene if the architecture changed) and/or new values. */
  setData(model: CnnModel, trace: CnnTrace, classNames: string[]): void {
    const stages = cnnStages(model);
    const changed = JSON.stringify(stages) !== JSON.stringify(this.stages) || classNames.join() !== this.classNames.join();
    this.model = model;
    this.trace = trace;
    this.classNames = classNames;
    if (changed) {
      this.stages = stages;
      this.computeLayout();
      this.build();
    }
    this.update();
  }

  private build(): void {
    for (const mesh of [this.cells, this.pick]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    const n = this.cellCount;
    const cells = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.cellMaterial, n);
    const pick = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, n);
    const m = new THREE.Matrix4();
    for (let k = 0; k < n; k++) {
      const { s, map, i, j } = this.decode(k);
      const [x, y] = this.cellCenter(s, map, i, j);
      const out = this.stages[s].kind === 'output';
      m.makeScale(out ? 0.9 : P, out ? 0.9 : P, MAX_Z * 2).setPosition(x, y, 0);
      pick.setMatrixAt(k, m);
    }
    cells.raycast = () => {};
    cells.frustumCulled = false;
    pick.computeBoundingSphere();
    this.cells = cells;
    this.pick = pick;
    this.root.add(cells, pick);
    this.buildLabels();
  }

  setShowValues(show: boolean): void {
    this.showValues = show;
    this.update();
  }

  /** Slide animation: only the first `count` cells of one map are shown as computed. */
  setReveal(reveal: { stage: number; map: number; count: number } | null): void {
    this.reveal = reveal;
    this.update();
    this.updateOverlays();
  }

  /** Recomputes cell sizes and colours from the trace. */
  update(): void {
    const { cells, trace } = this;
    if (!cells || !trace) return;
    const p = this.palette;
    const dark = p.theme === 'dark';
    const plate = new THREE.Color(p.plateFill);
    const ink = new THREE.Color(dark ? '#f1f3f5' : '#11161d');
    const pos = new THREE.Color(p.positive);
    const neg = new THREE.Color(p.negative);
    const accent = new THREE.Color(dark ? '#a99bff' : '#5b46d6');
    const white = new THREE.Color(1, 1, 1);
    const c = new THREE.Color();
    const m = new THREE.Matrix4();
    this.stages.forEach((stage, s) => {
      const values = stageValues(stage, trace);
      let scale = stage.kind === 'conv' || stage.kind === 'pool' ? 0 : 1;
      if (stage.kind === 'conv' || stage.kind === 'pool') {
        // per stage: the largest |value| gets the full height and the full colour
        for (let q = 0; q < values.length; q++) scale = Math.max(scale, Math.abs(values[q]));
        scale = Math.max(scale, 1e-6);
      }
      const per = stage.size * stage.size;
      for (let q = 0; q < values.length; q++) {
        const k = this.offsets[s] + q;
        const map = Math.floor(q / per);
        let v = values[q];
        const hidden = this.reveal && this.reveal.stage === s && this.reveal.map === map && q - map * per >= this.reveal.count;
        if (hidden) v = 0;
        const { i, j } = { i: Math.floor((q % per) / stage.size), j: (q % per) % stage.size };
        const [x, y] = this.cellCenter(s, map, i, j);
        const hovered = this.hover?.kind === 'cell' && this.hover.stage === s && this.hover.map === map && this.hover.i === i && this.hover.j === j;
        if (stage.kind === 'output') {
          const len = DEPTH + (this.showValues ? v * MAX_Z * 2.5 : 0);
          m.makeScale(0.6, 0.6, len).setPosition(x, y, len / 2);
          c.copy(plate).lerp(accent, 0.25 + 0.75 * v);
        } else if (stage.kind === 'input') {
          const len = DEPTH + (this.showValues ? v * 0.25 : 0);
          m.makeScale(CELL, CELL, len).setPosition(x, y, len / 2);
          c.copy(plate).lerp(ink, Math.min(1, Math.max(0, v)));
        } else {
          const t = Math.min(1, Math.abs(v) / scale);
          const len = DEPTH + (this.showValues ? t * MAX_Z : 0);
          m.makeScale(CELL, CELL, len).setPosition(x, y, v < 0 ? -len / 2 : len / 2);
          c.copy(plate).lerp(v < 0 ? neg : pos, hidden || t < 1e-3 ? 0 : 0.15 + 0.85 * Math.pow(t, 0.8));
        }
        if (hidden) c.lerp(plate, 0.6);
        if (hovered) c.lerp(white, 0.4);
        cells.setMatrixAt(k, m);
        cells.setColorAt(k, c);
      }
    });
    cells.instanceMatrix.needsUpdate = true;
    if (cells.instanceColor) cells.instanceColor.needsUpdate = true;
    this.updateOutputLabels();
    this.viewport.requestRender();
  }

  // ----------------------------------------------------------------- labels

  private label(className: string, text: string, pick: CnnPick | null, title?: string): HTMLElement {
    const el = document.createElement(pick ? 'button' : 'div');
    if (el instanceof HTMLButtonElement) el.type = 'button';
    el.className = className;
    el.textContent = text;
    if (title) el.title = title;
    if (pick) el.addEventListener('click', () => this.onLabelClick(pick));
    return el;
  }

  private add(el: HTMLElement, position: Vec3, center: [number, number] = [0.5, 0.5]): Label {
    const object = new CSS2DObject(el);
    object.position.set(...position);
    object.center.set(...center);
    this.labels.add(object);
    return { object, el };
  }

  private stageTitle(stage: CnnStage): [string, string] {
    const model = this.model!;
    switch (stage.kind) {
      case 'input':
        return ['Input', `${stage.size}×${stage.size} pixels`];
      case 'conv': {
        const s = model.shapes[stage.layer];
        return [`Conv ${stage.layer + 1}`, `${s.outChannels} filters ${s.kernel}×${s.kernel} · ${model.config.activation === 'relu' ? 'ReLU' : 'tanh'}`];
      }
      case 'pool':
        return [`${model.shapes[stage.layer].pool === 'max' ? 'Max' : 'Average'} pool ${stage.layer + 1}`, `2×2 → ${stage.size}×${stage.size}`];
      case 'output':
        return ['Output', `dense → softmax · ${stage.maps} classes`];
    }
  }

  private buildLabels(): void {
    for (const l of [...this.stageLabels, ...this.mapLabels.flat(), ...this.outputLabels]) this.labels.remove(l.object);
    this.stageLabels = [];
    this.mapLabels = [];
    this.outputLabels = [];
    this.stages.forEach((stage, s) => {
      const [title, sub] = this.stageTitle(stage);
      const el = document.createElement('div');
      el.className = 'cnn-stage';
      const t = document.createElement('span');
      t.className = 'cnn-stage__title';
      t.textContent = title;
      const m = document.createElement('span');
      m.className = 'cnn-stage__meta';
      m.textContent = sub;
      el.append(t, m);
      const left = -this.rowWidth(stage) / 2 - (stage.kind === 'output' ? 0.8 : 0.35);
      this.stageLabels.push(this.add(el, [left, this.rowY[s], 0], [1, 0.5]));
      const maps: Label[] = [];
      if (stage.kind === 'conv' || stage.kind === 'pool') {
        for (let f = 0; f < stage.maps; f++) {
          const [x, y] = this.mapCenter(s, f);
          const pick: CnnPick = { kind: 'map', stage: s, map: f };
          maps.push(this.add(this.label('cnn-map', stage.kind === 'conv' ? `filter ${f + 1}` : `${f + 1}`, pick, stage.kind === 'conv' ? 'This filter’s kernel and what it responds to' : 'Pooled map'), [x, y - (stage.size * P) / 2 - 0.08, 0], [0.5, 0]));
        }
      }
      this.mapLabels.push(maps);
      if (stage.kind === 'output') {
        for (let k = 0; k < stage.maps; k++) {
          const [x, y] = this.mapCenter(s, k);
          this.outputLabels.push(this.add(this.label('cnn-out', this.classNames[k] ?? String(k), { kind: 'output', index: k }), [x, y + 0.55, 0], [0.5, 1]));
        }
      }
    });
    this.updateOutputLabels();
  }

  private updateOutputLabels(): void {
    const trace = this.trace;
    if (!trace) return;
    let best = 0;
    for (let k = 1; k < trace.probs.length; k++) if (trace.probs[k] > trace.probs[best]) best = k;
    this.outputLabels.forEach((label, k) => {
      label.el.replaceChildren();
      const name = document.createElement('span');
      name.className = 'cnn-out__name';
      name.textContent = this.classNames[k] ?? String(k);
      const p = document.createElement('span');
      p.className = 'cnn-out__p';
      p.textContent = formatProbability(trace.probs[k] ?? 0);
      label.el.append(name, p);
      label.el.classList.toggle('is-top', k === best);
    });
  }

  // -------------------------------------------------------------- selection

  setSelection(pick: CnnPick | null): void {
    this.selection = pick;
    this.mapLabels.forEach((maps, s) => maps.forEach((l, f) => l.el.classList.toggle('is-selected', pick?.kind === 'map' && pick.stage === s && pick.map === f)));
    this.outputLabels.forEach((l, k) => l.el.classList.toggle('is-selected', pick?.kind === 'output' && pick.index === k));
    this.updateOverlays();
  }

  /** The cell whose window is drawn: the selection, or the cell being computed during a slide. */
  private focusCell(): { s: number; map: number; i: number; j: number } | null {
    if (this.reveal) {
      const stage = this.stages[this.reveal.stage];
      const q = Math.min(this.reveal.count, stage.size * stage.size - 1);
      return { s: this.reveal.stage, map: this.reveal.map, i: Math.floor(q / stage.size), j: q % stage.size };
    }
    const sel = this.selection;
    return sel?.kind === 'cell' ? { s: sel.stage, map: sel.map, i: sel.i, j: sel.j } : null;
  }

  private updateOverlays(): void {
    const model = this.model;
    const trace = this.trace;
    const rects: { s: number; map: number; top: number; left: number; bottom: number; right: number; strong?: boolean }[] = [];
    const segs: { a: Vec3; b: Vec3; color: THREE.Color; alpha: number }[] = [];
    const pos = new THREE.Color(this.palette.positive);
    const neg = new THREE.Color(this.palette.negative);
    if (model && trace) {
      const sel = this.selection;
      if (sel?.kind === 'map') {
        const stage = this.stages[sel.stage];
        rects.push({ s: sel.stage, map: sel.map, top: 0, left: 0, bottom: stage.size - 1, right: stage.size - 1, strong: true });
      }
      const cell = this.focusCell();
      if (cell) {
        const stage = this.stages[cell.s];
        rects.push({ s: cell.s, map: cell.map, top: cell.i, left: cell.j, bottom: cell.i, right: cell.j, strong: true });
        const below = cell.s - 1;
        const prev = this.stages[below];
        const to = this.cellCenter(cell.s, cell.map, cell.i, cell.j);
        if (stage.kind === 'conv') {
          const s = model.shapes[stage.layer];
          const p = (s.kernel - 1) / 2;
          const input = trace.layers[stage.layer].input;
          const S = s.size;
          let max = 1e-9;
          const terms: { c: number; y: number; x: number; w: number; t: number }[] = [];
          for (let c = 0; c < s.inChannels; c++) {
            rects.push({ s: below, map: c, top: Math.max(0, cell.i - p), left: Math.max(0, cell.j - p), bottom: Math.min(S - 1, cell.i + p), right: Math.min(S - 1, cell.j + p) });
            for (let u = 0; u < s.kernel; u++) {
              for (let v = 0; v < s.kernel; v++) {
                const y = cell.i + u - p;
                const x = cell.j + v - p;
                if (y < 0 || x < 0 || y >= S || x >= S) continue;
                const w = model.convWeight(stage.layer)[model.weightIndex(stage.layer, cell.map, c, u, v)];
                const t = w * input[(c * S + y) * S + x];
                max = Math.max(max, Math.abs(t));
                terms.push({ c, y, x, w, t });
              }
            }
          }
          for (const term of terms) segs.push({ a: this.cellCenter(below, term.c, term.y, term.x), b: to, color: term.w >= 0 ? pos : neg, alpha: 0.15 + 0.85 * Math.abs(term.t) / max });
          // the whole region of the input image that can influence this cell
          if (below > 0) {
            const r = receptiveField(model, stage.layer, cell.i, cell.j);
            rects.push({ s: 0, map: 0, ...r });
          }
        } else if (stage.kind === 'pool' && prev) {
          const top = 2 * cell.i;
          const left = 2 * cell.j;
          rects.push({ s: below, map: cell.map, top, left, bottom: top + 1, right: left + 1 });
          const argmax = trace.layers[stage.layer].argmax[(cell.map * stage.size + cell.i) * stage.size + cell.j];
          for (let di = 0; di < 2; di++) {
            for (let dj = 0; dj < 2; dj++) {
              const idx = (cell.map * prev.size + top + di) * prev.size + left + dj;
              segs.push({ a: this.cellCenter(below, cell.map, top + di, left + dj), b: to, color: pos, alpha: argmax < 0 ? 0.6 : idx === argmax ? 1 : 0.12 });
            }
          }
        }
      }
      if (sel?.kind === 'output') {
        // the strongest contributions w·x of the flattened features to this class
        const D = model.flatSize;
        const W = model.denseWeight;
        const last = this.stages.length - 2;
        const lastStage = this.stages[last];
        const contributions = Array.from({ length: D }, (_, d) => ({ d, t: W[sel.index * D + d] * trace.flat[d] })).sort((a, b) => Math.abs(b.t) - Math.abs(a.t)).slice(0, 24);
        const max = Math.max(1e-9, Math.abs(contributions[0]?.t ?? 0));
        const to = this.mapCenter(this.stages.length - 1, sel.index);
        const per = lastStage.size * lastStage.size;
        for (const { d, t } of contributions) {
          const map = Math.floor(d / per);
          const q = d % per;
          segs.push({ a: this.cellCenter(last, map, Math.floor(q / lastStage.size), q % lastStage.size), b: to, color: t >= 0 ? pos : neg, alpha: 0.2 + 0.8 * Math.abs(t) / max });
        }
      }
    }
    // frames
    const fp: number[] = [];
    for (const r of rects) {
      const a = this.cellCenter(r.s, r.map, r.top, r.left);
      const b = this.cellCenter(r.s, r.map, r.bottom, r.right);
      const pad = r.strong ? P * 0.62 : P * 0.55;
      const x0 = a[0] - pad;
      const x1 = b[0] + pad;
      const y0 = b[1] - pad;
      const y1 = a[1] + pad;
      const z = 0.06;
      fp.push(x0, y0, z, x1, y0, z, x1, y0, z, x1, y1, z, x1, y1, z, x0, y1, z, x0, y1, z, x0, y0, z);
    }
    this.frames.geometry.dispose();
    this.frames.geometry = new THREE.BufferGeometry();
    this.frames.geometry.setAttribute('position', new THREE.Float32BufferAttribute(fp, 3));
    // lines (colour faded towards the background by the alpha)
    const bg = new THREE.Color(this.palette.background);
    const lp: number[] = [];
    const lc: number[] = [];
    const c = new THREE.Color();
    for (const seg of segs) {
      lp.push(seg.a[0], seg.a[1], 0.08, seg.b[0], seg.b[1], 0.08);
      c.copy(bg).lerp(seg.color, seg.alpha);
      lc.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
    this.lines.geometry.dispose();
    this.lines.geometry = new THREE.BufferGeometry();
    this.lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
    this.lines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(lc, 3));
    this.viewport.requestRender();
  }

  setHover(pick: CnnPick | null): void {
    const before = JSON.stringify(this.hover);
    this.hover = pick;
    if (before !== JSON.stringify(pick)) this.update();
  }

  pickAt(x: number, y: number): CnnPick | null {
    if (!this.pick || !this.root.visible) return null;
    const { width, height } = this.viewport.size;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1), this.viewport.camera);
    const hit = this.raycaster.intersectObject(this.pick, false)[0];
    if (hit?.instanceId === undefined) return null;
    const { s, map, i, j } = this.decode(hit.instanceId);
    return this.stages[s].kind === 'output' ? { kind: 'output', index: map } : { kind: 'cell', stage: s, map, i, j };
  }

  /** Text for the tooltip of a cell. */
  describe(pick: CnnPick): string | null {
    const trace = this.trace;
    if (!trace) return null;
    if (pick.kind === 'output') return `${this.classNames[pick.index] ?? pick.index}: ${formatProbability(trace.probs[pick.index])}`;
    if (pick.kind === 'map') return null;
    const stage = this.stages[pick.stage];
    const v = stageValues(stage, trace)[(pick.map * stage.size + pick.i) * stage.size + pick.j];
    return `${this.stageTitle(stage)[0]}${stage.maps > 1 ? ` · map ${pick.map + 1}` : ''} · row ${pick.i}, column ${pick.j}: ${formatSigned(v, 3)}`;
  }

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    (this.frames.material as THREE.LineBasicMaterial).color.set(palette.plateFocus);
    this.update();
    this.updateOverlays();
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  dispose(): void {
    for (const mesh of [this.cells, this.pick]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    for (const l of [...this.stageLabels, ...this.mapLabels.flat(), ...this.outputLabels]) this.labels.remove(l.object);
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
    this.frames.geometry.dispose();
    (this.frames.material as THREE.Material).dispose();
    this.cellMaterial.dispose();
    this.pickMaterial.dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}
