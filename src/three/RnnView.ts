/**
 * RnnView — a recurrent network unrolled through time: one column per step,
 * left → right; in each column the input symbol (bottom), the hidden state
 * h_t as a tile of bars (middle; up = positive, down = negative… here:
 * towards / away from the viewer) and the prediction (top).
 *
 * Arrows: input → state, state → prediction, and the recurrent arrow from
 * each state to the next. Every column uses the same weights: the arrows of
 * one kind are drawn identically and selecting one selects the shared matrix.
 *
 * Folding collapses the columns into the single cell with a loop — the usual
 * textbook drawing — and unfolding shows that the loop is just the same cell
 * repeated. Optional bars under the inputs show ‖∂L/∂h_t‖ from
 * backpropagation through time (log scale).
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { easeInOutCubic } from '../animation/easing';
import type { RnnTrace } from '../rnn/forward';
import { formatProbability } from '../utils/format';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import { ArcMesh, type ArcInstance } from './ArcMesh';
import type { Viewport } from './Viewport';

export type RnnWeights = 'ih' | 'hh' | 'out';

export type RnnPick =
  | { kind: 'step'; t: number }
  | { kind: 'input'; t: number }
  | { kind: 'output'; t: number }
  | { kind: 'weights'; which: RnnWeights };

export interface RnnViewData {
  trace: RnnTrace;
  inputSymbols: string[];
  outputSymbols: string[];
  /** Target at each step (−1 = none). */
  targets: number[];
  /** ‖∂L/∂h_t‖ per step, or null. */
  gradients: number[] | null;
}

const DX = 2.1;
const PITCH = 0.2;
const CELL = 0.17;
const MAX_Z = 0.6;
const Y_IN = 0;
const Y_H = 2.1;
const Y_OUT = 3.9;
const Y_GRAD = -1.0;

interface Label {
  object: CSS2DObject;
  el: HTMLElement;
}

export class RnnView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: RnnPick) => void;
  private data: RnnViewData | null = null;
  private T = 0;
  private H = 0;
  private cols = 4;
  private rows = 1;
  private cells: THREE.InstancedMesh | null = null;
  private gates: THREE.InstancedMesh | null = null;
  private gradBars: THREE.InstancedMesh | null = null;
  private pickBoxes: THREE.InstancedMesh | null = null;
  private readonly material = new THREE.MeshLambertMaterial({ color: 0xffffff });
  private readonly pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
  private readonly arcs = new ArcMesh();
  private arcKinds: RnnWeights[] = [];
  private readonly frame: THREE.LineSegments;
  private readonly labels = new THREE.Group();
  private inputLabels: Label[] = [];
  private outputLabels: Label[] = [];
  private gradLabels: Label[] = [];
  private h0Label: Label | null = null;
  private card: Label | null = null;
  private selection: RnnPick | null = null;
  private hover: RnnPick | null = null;
  /** 1 = unrolled through time, 0 = folded into one cell with a loop. */
  private unfold = 1;
  private readonly raycaster = new THREE.Raycaster();

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: RnnPick): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.root.name = 'RnnView';
    this.frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: o.palette.plateFocus }));
    this.frame.visible = false;
    this.frame.raycast = () => {};
    const lights = new THREE.Group();
    lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(-3, 6, 9);
    lights.add(sun);
    this.root.add(lights, this.arcs.mesh, this.frame, this.labels);
    this.applyPalette();
    o.viewport.scene.add(this.root);
  }

  // ----------------------------------------------------------------- layout

  /** x of step t (folded: every step on the last column). */
  x(t: number): number {
    const unrolled = (t - (this.T - 1) / 2) * DX;
    const folded = 0;
    return folded + (unrolled - folded) * this.unfold;
  }

  private get tileW(): number {
    return this.cols * PITCH;
  }

  private get tileH(): number {
    return this.rows * PITCH;
  }

  stepCenter(t: number): Vec3 {
    return [this.x(t), Y_H, 0];
  }

  /** Content box for a fold state (default: the current one). */
  bounds(unfold = this.unfold): { min: Vec3; max: Vec3 } {
    const half = Math.max(2, (((this.T - 1) * DX) / 2) * unfold + 1.2);
    return { min: [-half - 1.2, Y_GRAD - 1.2, -MAX_Z], max: [half + 0.6, Y_OUT + 1.4, MAX_Z] };
  }

  /** Visibility of step t: every step when unrolled; only the last one when folded. */
  private presence(t: number): number {
    return t === this.T - 1 ? 1 : this.unfold;
  }

  // ------------------------------------------------------------------- data

  setData(data: RnnViewData): void {
    const T = data.trace.steps.length;
    const H = data.trace.steps[0]?.h.length ?? this.H;
    const rebuild = T !== this.T || H !== this.H;
    this.data = data;
    this.T = T;
    this.H = H;
    if (rebuild) this.build();
    this.buildLabels();
    this.update();
  }

  private build(): void {
    for (const mesh of [this.cells, this.gates, this.gradBars, this.pickBoxes]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    this.cols = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(this.H * 1.5))));
    this.rows = Math.ceil(this.H / this.cols);
    const n = Math.max(1, this.T);
    this.cells = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.material, Math.max(1, this.T * this.H));
    this.gates = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.material, n);
    this.gradBars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.material, n);
    this.pickBoxes = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, n);
    for (const mesh of [this.cells, this.gates, this.gradBars]) {
      mesh.raycast = () => {};
      mesh.frustumCulled = false;
    }
    this.root.add(this.cells, this.gates, this.gradBars, this.pickBoxes);
  }

  /** Positions, sizes and colours from the data and the fold state. */
  private update(): void {
    const { data, cells, gates, gradBars, pickBoxes } = this;
    if (!data || !cells || !gates || !gradBars || !pickBoxes) return;
    const p = this.palette;
    const plate = new THREE.Color(p.plateFill);
    const pos = new THREE.Color(p.positive);
    const neg = new THREE.Color(p.negative);
    const accent = new THREE.Color(p.theme === 'dark' ? '#a99bff' : '#5b46d6');
    const white = new THREE.Color(1, 1, 1);
    const c = new THREE.Color();
    const m = new THREE.Matrix4();
    const { steps } = data.trace;
    // gradient bars: log scale between the smallest and the largest norm
    const grads = data.gradients;
    const logs = grads?.filter((g) => g > 0).map(Math.log10) ?? [];
    const gMin = logs.length ? Math.min(...logs) : 0;
    const gMax = logs.length ? Math.max(...logs) : 1;
    steps.forEach((step, t) => {
      const presence = this.presence(t);
      const x = this.x(t);
      const hovered = this.hover?.kind === 'step' && this.hover.t === t;
      for (let u = 0; u < this.H; u++) {
        const v = step.h[u];
        const col = u % this.cols;
        const row = Math.floor(u / this.cols);
        const cx = x - this.tileW / 2 + (col + 0.5) * PITCH;
        const cy = Y_H + this.tileH / 2 - (row + 0.5) * PITCH;
        const len = (0.03 + Math.min(1, Math.abs(v)) * MAX_Z) * presence;
        m.makeScale(CELL * presence + 1e-4, CELL * presence + 1e-4, len + 1e-4).setPosition(cx, cy, v < 0 ? -len / 2 : len / 2);
        cells.setMatrixAt(t * this.H + u, m);
        c.copy(plate).lerp(v < 0 ? neg : pos, 0.15 + 0.85 * Math.min(1, Math.abs(v)));
        if (hovered) c.lerp(white, 0.35);
        cells.setColorAt(t * this.H + u, c);
      }
      // GRU: mean update gate (how much of the old state is kept) as a bar beside the tile
      const keep = step.update ? step.update.reduce((a, b) => a + b, 0) / step.update.length : 0;
      const gh = Math.max(1e-4, keep * this.tileH * presence);
      m.makeScale(0.07 * presence + 1e-4, gh, 0.07).setPosition(x - this.tileW / 2 - 0.16, Y_H - this.tileH / 2 + gh / 2, 0);
      gates.setMatrixAt(t, m);
      gates.setColorAt(t, accent);
      // gradient bar
      const g = grads?.[t] ?? 0;
      const level = grads && g > 0 ? (gMax > gMin ? (Math.log10(g) - gMin) / (gMax - gMin) : 1) : 0;
      const bh = grads ? Math.max(0.04, 0.04 + level * 1.0) * presence : 1e-4;
      m.makeScale(0.5 * presence + 1e-4, bh, 0.12).setPosition(x, Y_GRAD + bh / 2 - 0.5, 0);
      gradBars.setMatrixAt(t, m);
      c.copy(plate).lerp(accent, 0.2 + 0.8 * level);
      gradBars.setColorAt(t, c);
      // pick box around the hidden tile
      m.makeScale((this.tileW + 0.3) * presence + 1e-4, this.tileH + 0.3, MAX_Z * 2).setPosition(x, Y_H, 0);
      pickBoxes.setMatrixAt(t, m);
    });
    gates.visible = !!steps[0]?.update;
    gradBars.visible = !!grads;
    for (const mesh of [cells, gates, gradBars, pickBoxes]) {
      mesh.count = mesh === cells ? steps.length * this.H : steps.length;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    pickBoxes.computeBoundingSphere();
    this.updateArcs();
    this.placeLabels();
    this.updateFrame();
    this.viewport.requestRender();
  }

  private updateArcs(): void {
    const data = this.data;
    if (!data) return;
    const p = this.palette;
    const dark = p.theme === 'dark';
    const neutral = dark ? '#8d96a0' : '#7b8696';
    const recurrent = dark ? '#c9b8ff' : '#6a4fd6';
    const arcs: ArcInstance[] = [];
    const kinds: RnnWeights[] = [];
    const sel = this.selection?.kind === 'weights' ? this.selection.which : null;
    const emphasis = (kind: RnnWeights) => (sel === null ? 1 : sel === kind ? 2 : 0);
    const z = 0.02;
    for (let t = 0; t < this.T; t++) {
      const a = this.presence(t);
      if (a < 0.02) continue;
      const x = this.x(t);
      // input → state
      arcs.push({ start: [x, Y_IN + 0.35, z], control: [x, (Y_IN + Y_H) / 2, z], end: [x, Y_H - this.tileH / 2 - 0.12, z], width: 0.07, alpha: 0.75 * a, color: neutral, emphasis: emphasis('ih'), group: 0 });
      kinds.push('ih');
      // state → prediction
      arcs.push({ start: [x, Y_H + this.tileH / 2 + 0.12, z], control: [x, (Y_H + Y_OUT) / 2 - 0.2, z], end: [x, Y_OUT - 0.55, z], width: 0.07, alpha: 0.75 * a, color: neutral, emphasis: emphasis('out'), group: 0 });
      kinds.push('out');
    }
    // recurrent arrows: unrolled, from each state to the next; folded, a loop on the one cell
    if (this.unfold > 0.02) {
      for (let t = 1; t < this.T; t++) {
        const a = Math.min(this.presence(t - 1), this.presence(t));
        const x0 = this.x(t - 1) + this.tileW / 2 + 0.08;
        const x1 = this.x(t) - this.tileW / 2 - 0.08;
        if (x1 - x0 < 0.05) continue;
        arcs.push({ start: [x0, Y_H, z], control: [(x0 + x1) / 2, Y_H + 0.35, z], end: [x1, Y_H, z], width: 0.1, alpha: 0.9 * a, color: recurrent, emphasis: emphasis('hh'), group: 0 });
        kinds.push('hh');
      }
    }
    if (this.unfold < 0.98 && this.T > 0) {
      const x = this.x(this.T - 1);
      const r = this.tileW / 2 + 0.1;
      arcs.push({ start: [x + r, Y_H + 0.1, z], control: [x + r + 0.9, Y_H + this.tileH + 1.1, z], end: [x - 0.1, Y_H + this.tileH / 2 + 0.15, z], width: 0.1, alpha: 0.9 * (1 - this.unfold), color: recurrent, emphasis: emphasis('hh'), group: 0 });
      kinds.push('hh');
    }
    this.arcs.set(arcs);
    this.arcKinds = kinds;
  }

  // ----------------------------------------------------------------- labels

  private label(className: string, pick: RnnPick | null, title?: string): HTMLElement {
    const el = document.createElement(pick ? 'button' : 'div');
    if (el instanceof HTMLButtonElement) el.type = 'button';
    el.className = className;
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

  private buildLabels(): void {
    for (const l of [...this.inputLabels, ...this.outputLabels, ...this.gradLabels]) this.labels.remove(l.object);
    if (this.h0Label) this.labels.remove(this.h0Label.object);
    this.inputLabels = [];
    this.outputLabels = [];
    this.gradLabels = [];
    const data = this.data!;
    data.trace.steps.forEach((step, t) => {
      const input = this.label('tt-token rnn-in', { kind: 'input', t }, `Step ${t}: input “${data.inputSymbols[step.input]}”`);
      const text = document.createElement('span');
      text.textContent = data.inputSymbols[step.input] === ' ' ? '␣' : data.inputSymbols[step.input];
      const pos = document.createElement('span');
      pos.className = 'tt-token__pos';
      pos.textContent = `t=${t}`;
      input.append(text, pos);
      this.inputLabels.push(this.add(input, [0, Y_IN, 0]));
      // compact prediction: the most likely symbol (✓ / ✗ against the expected one)
      const best = step.probs.indexOf(Math.max(...step.probs));
      const target = data.targets[t];
      const mini = this.label(`rnn-out${target >= 0 ? (best === target ? ' is-right' : ' is-wrong') : ' is-unscored'}`, { kind: 'output', t }, `Prediction after step ${t}${target >= 0 ? ` — expected “${data.outputSymbols[target]}”` : ''}`);
      const w = document.createElement('span');
      w.className = 'rnn-out__symbol';
      w.textContent = data.outputSymbols[best] === ' ' ? '␣' : data.outputSymbols[best];
      const v = document.createElement('span');
      v.className = 'rnn-out__p';
      v.textContent = formatProbability(step.probs[best]);
      mini.append(w, v);
      this.outputLabels.push(this.add(mini, [0, Y_OUT - 0.45, 0], [0.5, 1]));
      if (data.gradients) {
        const g = this.label('rnn-grad', null, `‖∂L/∂h_${t}‖ from backpropagation through time`);
        g.textContent = data.gradients[t] > 0 ? data.gradients[t].toExponential(0) : '0';
        this.gradLabels.push(this.add(g, [0, Y_GRAD - 0.62, 0], [0.5, 0]));
      }
    });
    const h0 = this.label('rnn-h0', null, 'The state before the first step is all zeros');
    h0.textContent = 'h₀ = 0';
    this.h0Label = this.add(h0, [0, Y_H, 0], [1, 0.5]);
    this.applySelectionClasses();
  }

  private placeLabels(): void {
    this.inputLabels.forEach((l, t) => {
      l.object.position.set(this.x(t), Y_IN, 0);
      l.object.visible = this.presence(t) > 0.3;
    });
    this.outputLabels.forEach((l, t) => {
      l.object.position.set(this.x(t), Y_OUT - 0.45, 0);
      l.object.visible = this.presence(t) > 0.3;
    });
    this.gradLabels.forEach((l, t) => {
      l.object.position.set(this.x(t), Y_GRAD - 0.62, 0);
      l.object.visible = this.presence(t) > 0.3 && (this.T <= 16 || t % 2 === (this.T - 1) % 2);
    });
    if (this.h0Label) {
      this.h0Label.object.position.set(this.x(0) - this.tileW / 2 - 0.35, Y_H, 0);
      this.h0Label.object.visible = this.unfold > 0.5;
    }
    const s = this.selection;
    if (this.card && s && s.kind !== 'weights') {
      this.card.object.position.set(this.x(s.t), Y_OUT + 0.15, 0);
      this.card.object.visible = this.presence(s.t) > 0.3;
    }
  }

  // -------------------------------------------------------------- selection

  setSelection(pick: RnnPick | null): void {
    this.selection = pick;
    this.applySelectionClasses();
    this.updateArcs();
    this.updateFrame();
    this.viewport.requestRender();
  }

  private applySelectionClasses(): void {
    const s = this.selection;
    this.inputLabels.forEach((l, t) => l.el.classList.toggle('is-selected', (s?.kind === 'input' || s?.kind === 'step') && s.t === t));
    this.outputLabels.forEach((l, t) => l.el.classList.toggle('is-selected', s?.kind !== 'weights' && s?.t === t));
    this.buildCard(s && s.kind !== 'weights' ? s.t : null);
  }

  /** Top-3 card above the compact label of the selected step. */
  private buildCard(t: number | null): void {
    if (this.card) this.labels.remove(this.card.object);
    this.card = null;
    const data = this.data;
    if (t === null || !data || t >= data.trace.steps.length) return;
    const step = data.trace.steps[t];
    const target = data.targets[t];
    const el = this.label('tt-pred tt-pred--main rnn-card', { kind: 'output', t }, `Prediction after step ${t}`);
    for (const k of Array.from(step.probs.keys()).sort((a, b) => step.probs[b] - step.probs[a]).slice(0, 3)) {
      const row = document.createElement('span');
      row.className = `tt-pred__row${k === target ? ' is-target' : ''}`;
      const w = document.createElement('span');
      w.className = 'tt-pred__word';
      w.textContent = data.outputSymbols[k] === ' ' ? '␣' : data.outputSymbols[k];
      const bar = document.createElement('span');
      bar.className = 'tt-pred__bar';
      bar.style.setProperty('--p', String(step.probs[k]));
      const v = document.createElement('span');
      v.className = 'tt-pred__value';
      v.textContent = formatProbability(step.probs[k]);
      row.append(w, bar, v);
      el.append(row);
    }
    this.card = this.add(el, [this.x(t), Y_OUT + 0.15, 0], [0.5, 1]);
  }

  private updateFrame(): void {
    const s = this.selection;
    if ((s?.kind === 'step' || s?.kind === 'output' || s?.kind === 'input') && s.t < this.T && this.presence(s.t) > 0.5) {
      this.frame.visible = true;
      this.frame.scale.set(this.tileW + 0.26, this.tileH + 0.26, MAX_Z * 2 + 0.1);
      this.frame.position.set(this.x(s.t), Y_H, 0);
    } else this.frame.visible = false;
  }

  setHover(pick: RnnPick | null): void {
    const before = JSON.stringify(this.hover);
    this.hover = pick;
    this.arcs.setHover(-1);
    if (JSON.stringify(pick) !== before) this.update();
  }

  pickAt(x: number, y: number): RnnPick | null {
    if (!this.data || !this.root.visible) return null;
    const arc = this.arcs.pick(x, y, this.viewport, this.root.matrixWorld);
    if (arc) return { kind: 'weights', which: this.arcKinds[arc.index] };
    if (!this.pickBoxes) return null;
    const { width, height } = this.viewport.size;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1), this.viewport.camera);
    const hit = this.raycaster.intersectObject(this.pickBoxes, false)[0];
    if (hit?.instanceId === undefined || this.presence(hit.instanceId) < 0.5) return null;
    return { kind: 'step', t: hit.instanceId };
  }

  describe(pick: RnnPick): string | null {
    const data = this.data;
    if (!data) return null;
    switch (pick.kind) {
      case 'step': {
        const step = data.trace.steps[pick.t];
        const norm = Math.hypot(...step.h);
        return `h${pick.t} after “${data.inputSymbols[step.input]}” · ‖h‖ = ${norm.toFixed(2)}${step.update ? ` · keeps ${formatProbability(step.update.reduce((a, b) => a + b, 0) / step.update.length)} of the old state` : ''}`;
      }
      case 'weights':
        return { ih: 'W_ih: input → state (the same at every step)', hh: 'W_hh: state → next state (the same at every step)', out: 'W_out: state → prediction (the same at every step)' }[pick.which];
      default:
        return null;
    }
  }

  // ------------------------------------------------------------------ fold

  /** Animates between the folded cell (0) and the unrolled sequence (1). */
  setUnfold(target: number, animate = true): void {
    const from = this.unfold;
    const apply = (v: number) => {
      this.unfold = v;
      this.update();
    };
    if (!animate) {
      apply(target);
      return;
    }
    this.viewport.animations.tween({ channel: 'rnn-fold', duration: 1.1, easing: easeInOutCubic, onUpdate: (k) => apply(from + (target - from) * k) });
  }

  get unfolded(): number {
    return this.unfold;
  }

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPalette();
    this.update();
  }

  private applyPalette(): void {
    (this.frame.material as THREE.LineBasicMaterial).color.set(this.palette.plateFocus);
    this.arcs.setColors(this.palette.selection, this.palette.theme === 'dark' ? 0.16 : 0.22);
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  dispose(): void {
    this.viewport.animations.cancelChannel('rnn-fold');
    for (const mesh of [this.cells, this.gates, this.gradBars, this.pickBoxes]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    for (const l of [...this.inputLabels, ...this.outputLabels, ...this.gradLabels]) this.labels.remove(l.object);
    if (this.h0Label) this.labels.remove(this.h0Label.object);
    if (this.card) this.labels.remove(this.card.object);
    this.arcs.dispose();
    this.material.dispose();
    this.pickMaterial.dispose();
    this.frame.geometry.dispose();
    (this.frame.material as THREE.Material).dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}
