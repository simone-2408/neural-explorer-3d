/**
 * LlmView — a real language model (GPT-2 small) as a "logit-lens tower".
 *
 *   x  token positions (left → right)
 *   y  the residual stream after each block (0 = embeddings, 12 = last layer)
 *
 * Each box is one (layer, position) pair. Its height and colour show how
 * probable the model's final answer for that position already is if the
 * vector were decoded at that layer (logit lens), and its label shows the word
 * that layer would predict. Arcs show the attention of one chosen head.
 *
 * Only what the backend measured is drawn; the 768 numbers of each vector are
 * not (see the Transformer tab for a model small enough to show them all).
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { headMatrix, showToken } from '../llm/client';
import type { LlmAnalysis } from '../llm/types';
import { formatProbability } from '../utils/format';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import { ArcMesh, arcStyle } from './ArcMesh';
import type { Viewport } from './Viewport';

export type LlmPick =
  | { kind: 'cell'; layer: number; position: number }
  | { kind: 'token'; position: number }
  | { kind: 'layer'; layer: number }
  | { kind: 'arc'; layer: number; head: number; query: number; key: number; weight: number }
  | { kind: 'prediction' };

export interface LlmArcOptions {
  layer: number;
  head: number;
  /** Query positions whose arcs are drawn. */
  queries: number[];
  min: number;
  color: string;
}

const COL = 1.9;
const ROW = 1;
const BOX_W = 1.3;
const BOX_D = 0.7;

interface Label {
  object: CSS2DObject;
  el: HTMLElement;
}

export class LlmView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: LlmPick) => void;
  private analysis: LlmAnalysis | null = null;
  private T = 0;
  private S = 0;
  private boxes: THREE.InstancedMesh | null = null;
  private pickBoxes: THREE.InstancedMesh | null = null;
  private readonly boxMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  private readonly pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
  private readonly arcs = new ArcMesh();
  private arcSpecs: { layer: number; head: number; query: number; key: number; weight: number }[] = [];
  private readonly frameMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  private readonly rowFrame: THREE.LineSegments;
  private readonly cellFrame: THREE.LineSegments;
  private readonly labels = new THREE.Group();
  private tokenLabels: Label[] = [];
  private rowLabels: Label[] = [];
  private cellLabels: Label[] = [];
  private prediction: Label | null = null;
  private showWords = true;
  private hover: LlmPick | null = null;
  private selectedCell: { layer: number; position: number } | null = null;
  private layer: number | null = null;
  private focus: number | null = null;
  private readonly raycaster = new THREE.Raycaster();

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: LlmPick): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.root.name = 'LlmView';
    const box = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    this.rowFrame = new THREE.LineSegments(box, this.frameMaterial);
    this.rowFrame.visible = false;
    this.cellFrame = new THREE.LineSegments(box, this.frameMaterial);
    this.cellFrame.visible = false;
    const lights = new THREE.Group();
    lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.3));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(-3, 8, 6);
    lights.add(sun);
    this.root.add(lights, this.arcs.mesh, this.rowFrame, this.cellFrame, this.labels);
    this.applyPalette();
    o.viewport.scene.add(this.root);
  }

  x(position: number): number {
    return (position - (this.T - 1) / 2) * COL;
  }

  y(layer: number): number {
    return layer * ROW;
  }

  cellCenter(layer: number, position: number): Vec3 {
    return [this.x(position), this.y(layer) + 0.35, 0];
  }

  bounds(): { min: Vec3; max: Vec3 } {
    return { min: [this.x(0) - COL / 2 - 1.6, -1.2, -BOX_D], max: [this.x(this.T - 1) + COL / 2, this.y(this.S - 1) + 2.4, 1.2] };
  }

  setAnalysis(analysis: LlmAnalysis): void {
    this.analysis = analysis;
    this.T = analysis.T;
    this.S = analysis.lens.length;
    this.build();
  }

  private build(): void {
    const a = this.analysis!;
    for (const mesh of [this.boxes, this.pickBoxes]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    const { T, S } = this;
    const count = T * S;
    const boxes = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), this.boxMaterial, count);
    const pick = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, count);
    const m = new THREE.Matrix4();
    for (let l = 0; l < S; l++) {
      for (let p = 0; p < T; p++) {
        const k = l * T + p;
        const prob = a.lens[l].finalProb[p];
        const h = 0.08 + 0.72 * Math.sqrt(prob);
        m.makeScale(BOX_W, h, BOX_D).setPosition(this.x(p), this.y(l), 0);
        boxes.setMatrixAt(k, m);
        m.makeScale(BOX_W + 0.2, ROW * 0.92, BOX_D + 0.2).setPosition(this.x(p), this.y(l) + ROW * 0.42, 0);
        pick.setMatrixAt(k, m);
      }
    }
    boxes.raycast = () => {};
    boxes.frustumCulled = false;
    pick.computeBoundingSphere();
    this.boxes = boxes;
    this.pickBoxes = pick;
    this.root.add(boxes, pick);
    this.recolor();
    this.buildLabels();
    this.updateFrames();
    this.viewport.requestRender();
  }

  private recolor(): void {
    const { boxes, analysis: a } = this;
    if (!boxes || !a) return;
    const base = new THREE.Color(this.palette.neuronOff);
    const accent = new THREE.Color(this.palette.theme === 'dark' ? '#a99bff' : '#5b46d6');
    const top = new THREE.Color(this.palette.positive);
    const white = new THREE.Color(1, 1, 1);
    const c = new THREE.Color();
    for (let l = 0; l < this.S; l++) {
      for (let p = 0; p < this.T; p++) {
        const k = l * this.T + p;
        const prob = a.lens[l].finalProb[p];
        c.copy(base).lerp(accent, Math.min(1, 0.15 + Math.sqrt(prob)));
        if (a.lens[l].finalRank[p] === 0) c.lerp(top, 0.55);
        const hovered = this.hover?.kind === 'cell' && this.hover.layer === l && this.hover.position === p;
        if (hovered) c.lerp(white, 0.35);
        boxes.setColorAt(k, c);
      }
    }
    if (boxes.instanceColor) boxes.instanceColor.needsUpdate = true;
  }

  // --------------------------------------------------------------- labels

  private label(className: string, text: string, pick: LlmPick | null, title?: string): HTMLElement {
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

  private buildLabels(): void {
    for (const l of [...this.tokenLabels, ...this.rowLabels, ...this.cellLabels]) this.labels.remove(l.object);
    if (this.prediction) this.labels.remove(this.prediction.object);
    this.tokenLabels = [];
    this.rowLabels = [];
    this.cellLabels = [];
    const a = this.analysis!;
    const word = (id: number) => showToken(a.words[String(id)] ?? `#${id}`);
    a.tokens.forEach((t, p) => {
      const el = this.label('tt-token llm-token', showToken(t.text), { kind: 'token', position: p }, `token ${p} · id ${t.id} · BPE “${t.piece}”`);
      this.tokenLabels.push(this.add(el, [this.x(p), -0.25, BOX_D / 2 + 0.1], [0.5, 0]));
    });
    for (let l = 0; l < this.S; l++) {
      const el = this.label('tt-stage llm-row', l === 0 ? 'embed' : `layer ${l}`, { kind: 'layer', layer: l }, l === 0 ? 'Token + position embeddings' : `Residual stream after block ${l}`);
      this.rowLabels.push(this.add(el, [this.x(0) - COL / 2 - 0.1, this.y(l) + 0.35, BOX_D / 2], [1, 0.5]));
      for (let p = 0; p < this.T; p++) {
        const [id] = a.lens[l].top[p][0];
        const final = a.lens[l].finalRank[p] === 0;
        const el = this.label(`llm-cell${final ? ' is-final' : ''}`, word(id), null);
        this.cellLabels.push(this.add(el, [this.x(p), this.y(l) + 0.35, BOX_D / 2 + 0.05]));
      }
    }
    // next-token card above the last position
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'tt-pred tt-pred--main';
    card.title = 'Next-token probabilities — click for details and sampling';
    for (const [id, p] of a.nextTop.slice(0, 4)) {
      const row = document.createElement('span');
      row.className = 'tt-pred__row';
      const w = document.createElement('span');
      w.className = 'tt-pred__word';
      w.textContent = word(id);
      const bar = document.createElement('span');
      bar.className = 'tt-pred__bar';
      bar.style.setProperty('--p', String(p));
      const v = document.createElement('span');
      v.className = 'tt-pred__value';
      v.textContent = formatProbability(p);
      row.append(w, bar, v);
      card.append(row);
    }
    card.addEventListener('click', () => this.onLabelClick({ kind: 'prediction' }));
    this.prediction = this.add(card, [this.x(this.T - 1), this.y(this.S - 1) + 1.15, BOX_D / 2], [0.5, 1]);
    this.applyWordVisibility();
  }

  setShowWords(show: boolean): void {
    this.showWords = show;
    this.applyWordVisibility();
  }

  private applyWordVisibility(): void {
    this.cellLabels.forEach((label, k) => {
      const l = Math.floor(k / this.T);
      const p = k % this.T;
      const focused = this.focus === p || this.layer === l;
      label.object.visible = this.showWords || focused;
    });
    this.viewport.requestRender();
  }

  // ----------------------------------------------------------------- arcs

  setArcs(o: LlmArcOptions | null): void {
    const a = this.analysis;
    if (!a || !o || o.layer < 0 || o.layer >= a.layers) {
      this.arcSpecs = [];
      this.arcs.set([]);
      this.viewport.requestRender();
      return;
    }
    const T = a.T;
    const matrix = headMatrix(a, o.layer, o.head);
    const specs: typeof this.arcSpecs = [];
    for (const q of o.queries) for (let k = 0; k <= q; k++) if (matrix[q * T + k] >= o.min) specs.push({ layer: o.layer, head: o.head, query: q, key: k, weight: matrix[q * T + k] });
    this.arcSpecs = specs;
    // Attention of block L reads row L (its input) and writes row L + 1.
    const front = BOX_D / 2 + 0.15;
    this.arcs.set(
      specs.map((s) => {
        const start: Vec3 = [this.x(s.key), this.y(s.layer) + 0.55, front];
        const end: Vec3 = [this.x(s.query), this.y(s.layer + 1) + 0.15, front];
        const control: Vec3 = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.2, front + 0.4 + 0.12 * Math.abs(s.query - s.key) * COL];
        return { start, control, end, ...arcStyle(s.weight), color: o.color, emphasis: 1, group: 0 };
      }),
    );
    this.viewport.requestRender();
  }

  // ------------------------------------------------------------ selection

  /** `layer` and `focus` also choose which words are shown when not all of them are (a row and a column). */
  setHighlight(o: { layer: number | null; cell: { layer: number; position: number } | null; token: number | null; focus: number | null }): void {
    this.layer = o.layer;
    this.focus = o.focus;
    this.selectedCell = o.cell;
    this.tokenLabels.forEach((label, p) => label.el.classList.toggle('is-selected', o.token === p || o.cell?.position === p));
    this.rowLabels.forEach((label, l) => label.el.classList.toggle('is-selected', o.layer === l));
    this.updateFrames();
    this.applyWordVisibility();
  }

  private updateFrames(): void {
    if (this.layer !== null && this.layer < this.S && this.T) {
      this.rowFrame.visible = true;
      const width = (this.T - 1) * COL + BOX_W + 0.4;
      this.rowFrame.scale.set(width, ROW * 0.96, BOX_D + 0.3);
      this.rowFrame.position.set((this.x(0) + this.x(this.T - 1)) / 2, this.y(this.layer) + ROW * 0.42, 0);
    } else this.rowFrame.visible = false;
    const cell = this.selectedCell;
    if (cell && cell.position < this.T && cell.layer < this.S) {
      this.cellFrame.visible = true;
      this.cellFrame.scale.set(BOX_W + 0.16, ROW * 0.86, BOX_D + 0.16);
      this.cellFrame.position.set(this.x(cell.position), this.y(cell.layer) + ROW * 0.42, 0);
    } else this.cellFrame.visible = false;
    this.viewport.requestRender();
  }

  setHover(pick: LlmPick | null): void {
    const before = this.hover;
    this.hover = pick;
    let arc = -1;
    if (pick?.kind === 'arc') arc = this.arcSpecs.findIndex((s) => s.query === pick.query && s.key === pick.key);
    this.arcs.setHover(arc);
    if (pick?.kind === 'cell' || before?.kind === 'cell') this.recolor();
    this.viewport.requestRender();
  }

  pick(x: number, y: number): LlmPick | null {
    if (!this.analysis || !this.root.visible) return null;
    const arc = this.arcs.pick(x, y, this.viewport, this.root.matrixWorld);
    if (arc) {
      const s = this.arcSpecs[arc.index];
      if (s.weight >= 0.15 || arc.distance <= 1.5) return { kind: 'arc', ...s };
    }
    if (!this.pickBoxes) return null;
    const { width, height } = this.viewport.size;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1), this.viewport.camera);
    const hit = this.raycaster.intersectObject(this.pickBoxes, false)[0];
    if (hit?.instanceId === undefined) return null;
    return { kind: 'cell', layer: Math.floor(hit.instanceId / this.T), position: hit.instanceId % this.T };
  }

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPalette();
    this.recolor();
    this.viewport.requestRender();
  }

  private applyPalette(): void {
    this.frameMaterial.color.set(this.palette.plateFocus);
    this.arcs.setColors(this.palette.selection, this.palette.theme === 'dark' ? 0.16 : 0.22);
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  dispose(): void {
    for (const mesh of [this.boxes, this.pickBoxes]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    for (const l of [...this.tokenLabels, ...this.rowLabels, ...this.cellLabels]) this.labels.remove(l.object);
    if (this.prediction) this.labels.remove(this.prediction.object);
    this.arcs.dispose();
    this.boxMaterial.dispose();
    this.pickMaterial.dispose();
    this.frameMaterial.dispose();
    this.rowFrame.geometry.dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}
