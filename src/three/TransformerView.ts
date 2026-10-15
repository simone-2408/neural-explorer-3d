/**
 * TransformerView — the toy Transformer in 3D, read from a TransformerTrace.
 *
 *   x (columns)  token positions, left → right
 *   y (rows)     stages of the residual stream, bottom → top:
 *                embedding + position, then "+ attention" and "+ MLP" per layer
 *   tile         each vector is a small tile of bars, one per number
 *                (d_model = 32 → 4 × 8), positive up, negative down
 *
 * Between two rows sits the sub-layer that adds its output to the stream:
 * attention (arcs from the earlier tokens that are read — the only place where
 * positions exchange information) or the MLP (one block per position: the
 * same small network applied to each token separately). A vertical line per
 * token, the residual "highway", carries the vector upward.
 *
 * Draw calls: cells (1 instanced mesh), arcs (1), MLP blocks (1), highways (1),
 * invisible pick boxes, selection frames and CSS2D labels.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { TokenScore } from '../transformer/analysis';
import { stageCount, type TransformerTrace } from '../transformer/forward';
import type { TransformerModel } from '../transformer/model';
import { formatProbability } from '../utils/format';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import { ArcMesh, arcStyle } from './ArcMesh';
import { createCellMaterial } from './transformerMaterials';
import type { Viewport } from './Viewport';
import { withGlow } from './fx';

export const TRANSFORMER_LAYOUT = {
  tokenSpacing: 2,
  slabWidth: 1.08,
  /** The d numbers of a vector form a tile of `columns` × d/columns bars. */
  columns: 4,
  cellPitchZ: 0.135,
  cellFill: 0.8,
  rowSpacing: 2.8,
  barHeight: 0.6,
};

export type TransformerPick =
  | { kind: 'cell'; position: number; stage: number; dim: number }
  | { kind: 'arc'; layer: number; head: number; query: number; key: number; weight: number }
  | { kind: 'mlp'; layer: number; position: number }
  | { kind: 'sublayer'; layer: number; part: 'attention' | 'mlp' }
  | { kind: 'prediction'; position: number }
  | { kind: 'token'; position: number }
  | { kind: 'stage'; stage: number };

export interface ArcSpec {
  layer: number;
  head: number;
  query: number;
  key: number;
  weight: number;
  /** 0 dimmed, 1 normal, 2 highlighted. */
  emphasis: number;
}

export interface TransformerHighlight {
  /** Slab outlined (token at a stage), or a whole column when stage is null. */
  slab: { position: number; stage: number | null } | null;
  mlp: { layer: number; position: number } | null;
  sublayer: { layer: number; part: 'attention' | 'mlp' } | null;
  /** Columns other than this one are dimmed (null = none). */
  focus: number | null;
  /** A dimension highlighted across every vector (null = none). */
  dim: number | null;
}

export interface PredictionLabel {
  position: number;
  top: TokenScore[];
}

interface Label {
  object: CSS2DObject;
  el: HTMLElement;
}

export class TransformerView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: TransformerPick) => void;
  private readonly onLabelHover: (pick: TransformerPick | null) => void;

  private T = 0;
  private S = 0;
  private L = 0;
  private d = 0;
  private trace: TransformerTrace | null = null;
  private model: TransformerModel | null = null;
  private stage = 1e6;

  private readonly cellMaterial = createCellMaterial();
  private cells: THREE.InstancedMesh | null = null;
  private pickBoxes: THREE.InstancedMesh | null = null;
  private readonly pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
  private mlpBlocks: THREE.InstancedMesh | null = null;
  private readonly mlpMaterial = withGlow(new THREE.MeshLambertMaterial({ color: 0xffffff }), 1.0, 0.5);
  private mlpNorms: number[][] = [];
  private highways: THREE.InstancedMesh | null = null;
  private readonly highwayMaterial = new THREE.MeshBasicMaterial({ color: 0x808080, transparent: true, opacity: 0.7 });

  private readonly arcMesh = new ArcMesh();
  private arcs: ArcSpec[] = [];

  private readonly frame: THREE.LineSegments;
  private readonly mlpFrame: THREE.LineSegments;
  private readonly frameMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  private readonly lights = new THREE.Group();

  private readonly labelGroup = new THREE.Group();
  private tokenLabels: Label[] = [];
  private stageLabels: Label[] = [];
  private gapLabels: Label[] = [];
  private predictionLabels: Label[] = [];
  private predictionsVisible = true;
  private highlight: TransformerHighlight = { slab: null, mlp: null, sublayer: null, focus: null, dim: null };
  private hover: TransformerPick | null = null;
  private readonly raycaster = new THREE.Raycaster();

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: TransformerPick): void; onLabelHover?(pick: TransformerPick | null): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.onLabelHover = o.onLabelHover ?? (() => {});
    this.root.name = 'TransformerView';

    const box = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    this.frame = new THREE.LineSegments(box, this.frameMaterial);
    this.frame.visible = false;
    this.mlpFrame = new THREE.LineSegments(box, this.frameMaterial);
    this.mlpFrame.visible = false;

    this.lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(-3, 8, 6);
    this.lights.add(sun);

    this.root.add(this.arcMesh.mesh, this.frame, this.mlpFrame, this.lights, this.labelGroup);
    this.applyPalette();
    o.viewport.scene.add(this.root);
  }

  // --------------------------------------------------------------- layout

  x(position: number): number {
    return (position - (this.T - 1) / 2) * TRANSFORMER_LAYOUT.tokenSpacing;
  }

  y(stage: number): number {
    return stage * TRANSFORMER_LAYOUT.rowSpacing;
  }

  private get tileRows(): number {
    return Math.ceil(this.d / TRANSFORMER_LAYOUT.columns);
  }

  /** Offset (x, z) of dimension `dim` inside a tile. */
  cellOffset(dim: number): [number, number] {
    const { columns, slabWidth, cellPitchZ } = TRANSFORMER_LAYOUT;
    const col = dim % columns;
    const row = Math.floor(dim / columns);
    return [(col - (columns - 1) / 2) * (slabWidth / columns), (row - (this.tileRows - 1) / 2) * cellPitchZ];
  }

  private get halfDepth(): number {
    return (this.tileRows / 2) * TRANSFORMER_LAYOUT.cellPitchZ;
  }

  private get zFront(): number {
    return this.halfDepth + 0.06;
  }

  /** Centre of a residual slab (token `position` at `stage`). */
  slabCenter(position: number, stage: number): Vec3 {
    return [this.x(position), this.y(stage), 0];
  }

  mlpCenter(layer: number, position: number): Vec3 {
    return [this.x(position), this.y(2 * layer + 1.5), 0];
  }

  bounds(): { min: Vec3; max: Vec3 } {
    const { tokenSpacing, rowSpacing } = TRANSFORMER_LAYOUT;
    const left = this.x(0) - tokenSpacing * 0.5 - 2.6;
    const right = this.x(Math.max(0, this.T - 1)) + tokenSpacing * 0.5;
    return { min: [left, -1.3, -this.halfDepth], max: [right, this.y(this.S - 1) + rowSpacing * 0.9, this.zFront + 0.6] };
  }

  // ----------------------------------------------------------------- data

  /** Shows a trace. Geometry is rebuilt only when the number of tokens changes. */
  setData(model: TransformerModel, trace: TransformerTrace): void {
    const T = trace.tokens.length;
    const rebuild = T !== this.T || model !== this.model;
    this.model = model;
    this.trace = trace;
    if (rebuild) this.build(model, T);
    this.updateValues();
    this.updateTokenLabels();
    this.setStage(this.stage);
  }

  private build(model: TransformerModel, T: number): void {
    this.disposeMeshes();
    this.T = T;
    this.L = model.config.nLayers;
    this.S = stageCount(model);
    this.d = model.config.dModel;
    const { S, d } = this;
    const { slabWidth, barHeight, columns, cellPitchZ, cellFill } = TRANSFORMER_LAYOUT;
    const cellWidth = (slabWidth / columns) * cellFill;
    const cellDepth = cellPitchZ * cellFill;

    // residual cells
    const cellGeometry = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
    const count = S * T * d;
    const cells = new THREE.InstancedMesh(cellGeometry, this.cellMaterial, count);
    const cellInfo = new Float32Array(count * 3);
    const m = new THREE.Matrix4();
    let k = 0;
    for (let s = 0; s < S; s++) {
      for (let p = 0; p < T; p++) {
        for (let i = 0; i < d; i++) {
          const [dx, dz] = this.cellOffset(i);
          m.makeScale(cellWidth, 1, cellDepth).setPosition(this.x(p) + dx, this.y(s), dz);
          cells.setMatrixAt(k, m);
          cellInfo[3 * k] = s;
          cellInfo[3 * k + 1] = p;
          cellInfo[3 * k + 2] = i;
          k++;
        }
      }
    }
    cellGeometry.setAttribute('aCell', new THREE.InstancedBufferAttribute(cellInfo, 3));
    cellGeometry.setAttribute('aValue', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    cellGeometry.setAttribute('aPrev', new THREE.InstancedBufferAttribute(new Float32Array(count), 1));
    cells.frustumCulled = false;
    cells.raycast = () => {};
    this.cells = cells;

    // invisible boxes for picking whole slabs
    const pick = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, S * T);
    k = 0;
    for (let s = 0; s < S; s++) {
      for (let p = 0; p < T; p++) {
        m.makeScale(slabWidth + 0.24, 2 * barHeight + 0.3, 2 * this.halfDepth + 0.1).setPosition(this.x(p), this.y(s), 0);
        pick.setMatrixAt(k++, m);
      }
    }
    pick.computeBoundingSphere();
    this.pickBoxes = pick;

    // MLP blocks
    const mlp = new THREE.InstancedMesh(new THREE.BoxGeometry(0.66, 0.66, 0.66), this.mlpMaterial, this.L * T);
    mlp.frustumCulled = false;
    this.mlpBlocks = mlp;

    // residual highways
    const highwayGeometry = new THREE.CylinderGeometry(0.05, 0.05, 1, 10).translate(0, 0.5, 0);
    this.highways = new THREE.InstancedMesh(highwayGeometry, this.highwayMaterial, T);
    this.highways.frustumCulled = false;
    this.highways.raycast = () => {};

    this.root.add(cells, pick, mlp, this.highways);
    this.buildLabels();
  }

  private updateValues(): void {
    const { trace, cells } = this;
    if (!trace || !cells) return;
    const { T, S, d } = this;
    const values = cells.geometry.getAttribute('aValue') as THREE.InstancedBufferAttribute;
    const prev = cells.geometry.getAttribute('aPrev') as THREE.InstancedBufferAttribute;
    const rows = Array.from({ length: S }, (_, s) => this.residual(s));
    const magnitudes: number[] = [];
    let k = 0;
    for (let s = 0; s < S; s++) {
      for (let p = 0; p < T; p++) {
        const v = rows[s][p];
        const before = s === 0 ? trace.tokenEmbeddings[p] : rows[s - 1][p];
        for (let i = 0; i < d; i++) {
          values.array[k] = v[i];
          prev.array[k] = before[i];
          magnitudes.push(Math.abs(v[i]));
          k++;
        }
      }
    }
    values.needsUpdate = true;
    prev.needsUpdate = true;
    // Bar heights are relative to the 97th percentile of |value| (the largest bars are capped).
    magnitudes.sort((a, b) => a - b);
    const scale = magnitudes[Math.floor(0.97 * (magnitudes.length - 1))] || 1;
    this.cellMaterial.uniforms.uScale.value = 1 / scale;

    // MLP blocks: brightness ∝ size of the MLP output at that position
    this.mlpNorms = trace.blocks.map((block) => block.mlpOut.map((v) => Math.hypot(...v)));
    this.updateMlpBlocks();
    this.viewport.requestRender();
  }

  /** Vectors drawn on row `stage`. */
  private residual(stage: number): Float64Array[] {
    const trace = this.trace!;
    if (stage === 0) return trace.x0;
    const block = trace.blocks[Math.floor((stage - 1) / 2)];
    return (stage - 1) % 2 === 0 ? block.mid : block.output;
  }

  /** Bar scale used by the cells: values ≥ this are drawn at full height. */
  get valueScale(): number {
    return 1 / this.cellMaterial.uniforms.uScale.value;
  }

  // ---------------------------------------------------------------- stage

  /** Pipeline progress: rows 0 … ⌈stage⌉−1 are (being) computed; stage ≥ S shows everything. */
  setStage(stage: number): void {
    this.stage = stage;
    this.cellMaterial.uniforms.uStage.value = stage;
    this.arcMesh.setStage(stage);
    const { highways, T, S } = this;
    if (highways) {
      const top = Math.min(Math.max(stage - 1, 0), S - 1);
      const m = new THREE.Matrix4();
      for (let p = 0; p < T; p++) {
        m.makeScale(1, Math.max(top * TRANSFORMER_LAYOUT.rowSpacing, 1e-4), 1).setPosition(this.x(p), this.y(0), 0);
        highways.setMatrixAt(p, m);
      }
      highways.instanceMatrix.needsUpdate = true;
      highways.visible = stage > 1;
    }
    this.updateMlpBlocks();
    this.stageLabels.forEach((label, s) => label.el.classList.toggle('is-pending', stage <= s));
    this.gapLabels.forEach((label) => {
      const row = Number(label.el.dataset.row);
      label.el.classList.toggle('is-pending', stage <= row);
      label.el.classList.toggle('is-active', stage > row && stage < row + 1);
    });
    this.updatePredictionVisibility();
    this.viewport.requestRender();
  }

  private updateMlpBlocks(): void {
    const { mlpBlocks, T, L } = this;
    if (!mlpBlocks || !this.trace) return;
    const m = new THREE.Matrix4();
    const base = new THREE.Color(this.palette.plateBorder);
    const accent = new THREE.Color(this.palette.theme === 'dark' ? '#a99bff' : '#5b46d6');
    const color = new THREE.Color();
    const max = Math.max(1e-9, ...this.mlpNorms.flat());
    for (let l = 0; l < L; l++) {
      const row = 2 * l + 2; // the MLP of layer l produces row 2l + 2
      const shown = Math.min(Math.max(this.stage - (row - 1), 0), 1);
      for (let p = 0; p < T; p++) {
        const k = l * T + p;
        const [cx, cy, cz] = this.mlpCenter(l, p);
        const s = shown > 0 ? 0.55 + 0.45 * shown : 0.0001;
        m.makeScale(s, s, s).setPosition(cx, cy, cz);
        mlpBlocks.setMatrixAt(k, m);
        const t = Math.sqrt((this.mlpNorms[l]?.[p] ?? 0) / max);
        color.copy(base).lerp(accent, 0.25 + 0.75 * t);
        if (this.hover?.kind === 'mlp' && this.hover.layer === l && this.hover.position === p) color.lerp(new THREE.Color(1, 1, 1), 0.35);
        mlpBlocks.setColorAt(k, color);
      }
    }
    mlpBlocks.instanceMatrix.needsUpdate = true;
    if (mlpBlocks.instanceColor) mlpBlocks.instanceColor.needsUpdate = true;
    mlpBlocks.boundingSphere = null;
    mlpBlocks.computeBoundingSphere();
  }

  // ----------------------------------------------------------------- arcs

  setArcs(arcs: ArcSpec[], headColors: string[]): void {
    this.arcs = arcs;
    const { tokenSpacing, barHeight } = TRANSFORMER_LAYOUT;
    const zFront = this.zFront + 0.05;
    this.arcMesh.set(
      arcs.map((a) => {
        const start: Vec3 = [this.x(a.key), this.y(2 * a.layer) + barHeight + 0.12, zFront];
        const end: Vec3 = [this.x(a.query), this.y(2 * a.layer + 1) - barHeight - 0.12, zFront];
        const bulge = 0.3 + 0.13 * Math.abs(a.query - a.key) * tokenSpacing + 0.2 * a.head;
        const control: Vec3 = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, zFront + bulge];
        return { start, control, end, ...arcStyle(a.weight), color: headColors[a.head] ?? '#ffffff', emphasis: a.emphasis, group: a.layer };
      }),
    );
    if (this.hover?.kind === 'arc') this.setHover(this.hover);
    this.viewport.requestRender();
  }

  // ---------------------------------------------------------- predictions

  setPredictions(items: PredictionLabel[]): void {
    for (const label of this.predictionLabels) this.labelGroup.remove(label.object);
    this.predictionLabels = [];
    const y = this.y(this.S - 1) + TRANSFORMER_LAYOUT.rowSpacing * 0.42;
    for (const item of items) {
      const last = item.position === this.T - 1;
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `tt-pred${last ? ' tt-pred--main' : ''}`;
      el.title = last ? 'Next-token prediction — click for details, temperature and sampling' : `What the model predicts after “${this.trace?.tokens[item.position].text}”`;
      for (const t of item.top) {
        const row = document.createElement('span');
        row.className = 'tt-pred__row';
        const word = document.createElement('span');
        word.className = 'tt-pred__word';
        word.textContent = t.text;
        const bar = document.createElement('span');
        bar.className = 'tt-pred__bar';
        bar.style.setProperty('--p', String(t.value));
        const value = document.createElement('span');
        value.className = 'tt-pred__value';
        value.textContent = formatProbability(t.value);
        row.append(word, bar, value);
        el.append(row);
      }
      el.addEventListener('click', () => this.onLabelClick({ kind: 'prediction', position: item.position }));
      const object = new CSS2DObject(el);
      object.position.set(this.x(item.position), y, this.zFront);
      object.center.set(0.5, 1);
      this.labelGroup.add(object);
      this.predictionLabels.push({ object, el });
    }
    this.updatePredictionVisibility();
    this.viewport.requestRender();
  }

  setPredictionsVisible(visible: boolean): void {
    this.predictionsVisible = visible;
    this.updatePredictionVisibility();
  }

  private updatePredictionVisibility(): void {
    const show = this.predictionsVisible && this.stage >= this.S;
    for (const label of this.predictionLabels) label.object.visible = show;
    this.viewport.requestRender();
  }

  // --------------------------------------------------------------- labels

  private makeLabel(className: string, pick: TransformerPick | null, ...content: (string | HTMLElement)[]): HTMLElement {
    const el = document.createElement(pick ? 'button' : 'div');
    if (el instanceof HTMLButtonElement) el.type = 'button';
    el.className = className;
    el.append(...content);
    if (pick) {
      el.addEventListener('click', () => this.onLabelClick(pick));
      el.addEventListener('pointerenter', () => this.onLabelHover(pick));
      el.addEventListener('pointerleave', () => this.onLabelHover(null));
    }
    return el;
  }

  private span(className: string, text: string): HTMLElement {
    const el = document.createElement('span');
    el.className = className;
    el.textContent = text;
    return el;
  }

  private addLabel(el: HTMLElement, position: Vec3, center: [number, number] = [0.5, 0.5]): Label {
    const object = new CSS2DObject(el);
    object.position.set(...position);
    object.center.set(...center);
    this.labelGroup.add(object);
    return { object, el };
  }

  private buildLabels(): void {
    for (const label of [...this.tokenLabels, ...this.stageLabels, ...this.gapLabels]) this.labelGroup.remove(label.object);
    this.tokenLabels = [];
    this.stageLabels = [];
    this.gapLabels = [];
    const left = this.x(0) - TRANSFORMER_LAYOUT.tokenSpacing * 0.5 - 0.15;
    const zFront = this.zFront;

    for (let p = 0; p < this.T; p++) {
      const el = this.makeLabel('tt-token', { kind: 'token', position: p }, this.span('tt-token__text', ''), this.span('tt-token__pos', String(p)));
      this.tokenLabels.push(this.addLabel(el, [this.x(p), -TRANSFORMER_LAYOUT.barHeight - 0.35, zFront], [0.5, 0]));
    }
    for (let s = 0; s < this.S; s++) {
      const layer = Math.floor((s - 1) / 2) + 1;
      const [title, meta] = s === 0 ? ['x₀', 'token + position'] : (s - 1) % 2 === 0 ? [`x after attn ${layer}`, 'residual stream'] : [`x after MLP ${layer}`, 'residual stream'];
      const el = this.makeLabel('tt-stage', { kind: 'stage', stage: s }, this.span('tt-stage__title', title), this.span('tt-stage__meta', meta));
      this.stageLabels.push(this.addLabel(el, [left, this.y(s), zFront], [1, 0.5]));
    }
    for (let l = 0; l < this.L; l++) {
      const attention = this.makeLabel('tt-gap tt-gap--attention', { kind: 'sublayer', layer: l, part: 'attention' }, this.span('tt-gap__title', `Attention ${l + 1}`), this.span('tt-gap__meta', 'mixes positions'));
      attention.dataset.row = String(2 * l + 1);
      this.gapLabels.push(this.addLabel(attention, [left, this.y(2 * l + 0.5), zFront], [1, 0.5]));
      const mlp = this.makeLabel('tt-gap tt-gap--mlp', { kind: 'sublayer', layer: l, part: 'mlp' }, this.span('tt-gap__title', `MLP ${l + 1}`), this.span('tt-gap__meta', 'each position alone'));
      mlp.dataset.row = String(2 * l + 2);
      this.gapLabels.push(this.addLabel(mlp, [left, this.y(2 * l + 1.5), zFront], [1, 0.5]));
    }
    const unembed = this.makeLabel('tt-gap tt-gap--out', { kind: 'prediction', position: this.T - 1 }, this.span('tt-gap__title', 'Next token'), this.span('tt-gap__meta', 'LayerNorm · Eᵀ · softmax'));
    unembed.dataset.row = String(this.S);
    this.gapLabels.push(this.addLabel(unembed, [left, this.y(this.S - 1) + TRANSFORMER_LAYOUT.rowSpacing * 0.55, zFront], [1, 0.5]));
  }

  private updateTokenLabels(): void {
    const trace = this.trace;
    if (!trace) return;
    this.tokenLabels.forEach((label, p) => {
      const token = trace.tokens[p];
      (label.el.firstChild as HTMLElement).textContent = token.text;
      label.el.classList.toggle('is-unknown', token.unknown);
      label.el.classList.toggle('is-special', p === 0);
      label.el.title = token.unknown ? 'Not in the vocabulary: the model sees <unk>' : `Token “${token.text}” (id ${token.id}) at position ${p} — click to inspect`;
    });
    this.applyHighlightClasses();
  }

  // ------------------------------------------------------------ highlight

  setHighlight(highlight: TransformerHighlight): void {
    this.highlight = highlight;
    const { slab, mlp } = highlight;
    const { slabWidth, barHeight } = TRANSFORMER_LAYOUT;
    if (slab && slab.position < this.T) {
      this.frame.visible = true;
      if (slab.stage === null) {
        const top = this.y(this.S - 1);
        this.frame.scale.set(slabWidth + 0.3, top + 2 * barHeight + 0.5, 2 * this.halfDepth + 0.3);
        this.frame.position.set(this.x(slab.position), top / 2, 0);
      } else {
        this.frame.scale.set(slabWidth + 0.24, 2 * barHeight + 0.3, 2 * this.halfDepth + 0.24);
        this.frame.position.set(...this.slabCenter(slab.position, slab.stage));
      }
    } else this.frame.visible = false;
    if (mlp && mlp.position < this.T) {
      this.mlpFrame.visible = true;
      this.mlpFrame.scale.set(0.95, 0.95, 0.95);
      this.mlpFrame.position.set(...this.mlpCenter(mlp.layer, mlp.position));
    } else this.mlpFrame.visible = false;
    this.cellMaterial.uniforms.uFocus.value = highlight.focus ?? -1;
    this.cellMaterial.uniforms.uDimHighlight.value = highlight.dim ?? -1;
    this.applyHighlightClasses();
    this.viewport.requestRender();
  }

  private applyHighlightClasses(): void {
    const { slab, sublayer, focus } = this.highlight;
    this.tokenLabels.forEach((label, p) => {
      label.el.classList.toggle('is-selected', slab?.position === p);
      label.el.classList.toggle('is-focus', focus === p);
    });
    this.stageLabels.forEach((label, s) => label.el.classList.toggle('is-selected', slab?.stage === s));
    this.gapLabels.forEach((label) => {
      const row = Number(label.el.dataset.row);
      const layer = Math.floor((row - 1) / 2);
      const part = row === this.S ? null : (row - 1) % 2 === 0 ? 'attention' : 'mlp';
      label.el.classList.toggle('is-selected', !!sublayer && part === sublayer.part && layer === sublayer.layer);
    });
  }

  setHover(pick: TransformerPick | null): void {
    const previous = this.hover;
    this.hover = pick;
    const hoverCell = pick?.kind === 'cell' ? pick : null;
    (this.cellMaterial.uniforms.uHover.value as THREE.Vector2).set(hoverCell?.position ?? -1, hoverCell?.stage ?? -1);
    let arcIndex = -1;
    if (pick?.kind === 'arc') arcIndex = this.arcs.findIndex((a) => a.layer === pick.layer && a.head === pick.head && a.query === pick.query && a.key === pick.key);
    this.arcMesh.setHover(arcIndex);
    if (pick?.kind === 'mlp' || previous?.kind === 'mlp') this.updateMlpBlocks();
    this.tokenLabels.forEach((label, p) => label.el.classList.toggle('is-hover', (pick?.kind === 'token' || pick?.kind === 'cell') && pick.position === p));
    this.viewport.requestRender();
  }

  // -------------------------------------------------------------- picking

  pick(x: number, y: number): TransformerPick | null {
    if (!this.trace || !this.root.visible) return null;
    const arc = this.pickArc(x, y);
    // Arcs are drawn in front of the tiles: a strong arc under the pointer wins,
    // a faint one only when the pointer is right on it.
    if (arc && (arc.weight >= 0.15 || arc.distance <= 1.5)) return arc.pick;
    const solid = this.pickSolid(x, y);
    return solid ?? arc?.pick ?? null;
  }

  private pickSolid(x: number, y: number): TransformerPick | null {
    const { width, height } = this.viewport.size;
    const ndc = new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.viewport.camera);
    const targets = [this.pickBoxes, this.mlpBlocks].filter((o): o is THREE.InstancedMesh => !!o);
    const hits = this.raycaster.intersectObjects(targets, false);
    for (const hit of hits) {
      if (hit.instanceId === undefined) continue;
      if (hit.object === this.mlpBlocks) {
        const layer = Math.floor(hit.instanceId / this.T);
        if (this.stage <= 2 * layer + 1) continue;
        return { kind: 'mlp', layer, position: hit.instanceId % this.T };
      }
      const stage = Math.floor(hit.instanceId / this.T);
      if (this.stage <= stage) continue;
      const position = hit.instanceId % this.T;
      // The bars stand on the tile's mid-plane: find the dimension where the ray crosses it.
      const ray = this.raycaster.ray;
      const t = Math.abs(ray.direction.y) > 1e-6 ? (this.y(stage) - ray.origin.y) / ray.direction.y : -1;
      const local = this.root.worldToLocal(t > 0 ? ray.at(t, new THREE.Vector3()) : hit.point.clone());
      const { columns, slabWidth, cellPitchZ } = TRANSFORMER_LAYOUT;
      const col = Math.min(columns - 1, Math.max(0, Math.round((local.x - this.x(position)) / (slabWidth / columns) + (columns - 1) / 2)));
      const row = Math.min(this.tileRows - 1, Math.max(0, Math.round(local.z / cellPitchZ + (this.tileRows - 1) / 2)));
      const dim = Math.min(this.d - 1, row * columns + col);
      return { kind: 'cell', position, stage, dim };
    }
    return null;
  }

  /** Closest arc within its drawn half-width (+ a few pixels) of the pointer. */
  private pickArc(x: number, y: number): { pick: TransformerPick; weight: number; distance: number } | null {
    const hit = this.arcMesh.pick(x, y, this.viewport, this.root.matrixWorld);
    if (!hit) return null;
    const a = this.arcs[hit.index];
    if (a.weight < 0.01) return null;
    return { pick: { kind: 'arc', layer: a.layer, head: a.head, query: a.query, key: a.key, weight: a.weight }, weight: a.weight, distance: hit.distance };
  }

  // -------------------------------------------------------------- palette

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPalette();
    this.updateMlpBlocks();
    this.viewport.requestRender();
  }

  private applyPalette(): void {
    const p = this.palette;
    const u = this.cellMaterial.uniforms;
    (u.uPositive.value as THREE.Color).set(p.positive);
    (u.uNegative.value as THREE.Color).set(p.negative);
    (u.uNeutral.value as THREE.Color).set(p.neuronOff);
    (u.uEmpty.value as THREE.Color).set(p.theme === 'dark' ? '#1a222d' : '#dde3ea');
    (u.uBackground.value as THREE.Color).set(p.background);
    (u.uHighlight.value as THREE.Color).set(p.selection);
    this.arcMesh.setColors(p.selection, p.theme === 'dark' ? 0.16 : 0.22);
    this.highwayMaterial.color.set(p.axis);
    this.frameMaterial.color.set(p.plateFocus);
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  // -------------------------------------------------------------- cleanup

  private disposeMeshes(): void {
    for (const mesh of [this.cells, this.pickBoxes, this.mlpBlocks, this.highways]) {
      if (!mesh) continue;
      this.root.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    this.cells = this.pickBoxes = this.mlpBlocks = this.highways = null;
  }

  dispose(): void {
    this.disposeMeshes();
    for (const label of [...this.tokenLabels, ...this.stageLabels, ...this.gapLabels, ...this.predictionLabels]) {
      this.labelGroup.remove(label.object);
      label.el.remove();
    }
    this.arcMesh.dispose();
    this.cellMaterial.dispose();
    this.mlpMaterial.dispose();
    this.highwayMaterial.dispose();
    this.pickMaterial.dispose();
    this.frame.geometry.dispose();
    this.frameMaterial.dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}
