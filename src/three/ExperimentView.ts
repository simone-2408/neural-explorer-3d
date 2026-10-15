/**
 * ExperimentView — the Experiments tab in 3D: two walls facing the camera,
 * sharing one vertical axis (depth in the network, bottom → top).
 *
 *   Components wall (left): one tile per attention head (columns) and per
 *   MLP (last column) of each layer. The MLP of a layer sits half a row
 *   above its heads, because it runs after them. Bars come out of the wall
 *   towards the viewer when the target token relies on the component
 *   (switching it off lowers the target), and go behind the wall when
 *   switching it off raises the target. Tiles of switched-off or scaled
 *   components are tinted and labelled.
 *
 *   Residual wall (right): one column per token, one row per stage of the
 *   residual stream (embeddings, then after attention / after the MLP of each
 *   layer, aligned with the components that write there). Bar length shows
 *   how much the vector changed because of the interventions: the change
 *   starts where an intervention acts and spreads upwards and to the right.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { easeInOutCubic } from '../animation/easing';
import type { ExperimentResult } from '../experiments/types';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import type { Viewport } from './Viewport';

export type ExperimentPick =
  | { kind: 'head'; layer: number; head: number }
  | { kind: 'mlp'; layer: number }
  | { kind: 'cell'; stage: number; position: number }
  | { kind: 'token'; position: number };

export type ComponentState = 'normal' | 'off' | 'scaled';

export interface ComponentDisplay {
  /** How much the target relies on the component (see experiments/metrics `support`); null = not measured. */
  value: number | null;
  state: ComponentState;
  /** Label on the tile of an intervened component ("off", "×2"). */
  badge: string | null;
}

export interface ExperimentHighlight {
  component: { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number } | null;
  position: number | null;
  stage: number | null;
}

const CX = 1;
const TILE = 0.8;
const TILE_D = 0.08;
const MLP_GAP = 0.45;
const WALL_GAP = 2.6;
const PX = 1.3;
const CELL_W = 0.95;
const BAR_W = 0.46;
const BAR_MAX = 1.8;
const RBAR_W = 0.56;
const RBAR_MAX = 1.4;

/** Vertical pitch of one layer: models with few layers get taller rows, so the scene is not a thin strip. */
function rowPitch(layers: number): number {
  return layers <= 4 ? 3 : 1.3;
}

interface Label {
  object: CSS2DObject;
  el: HTMLElement;
}

/** Bar length (0…1) of a residual change: √ so that small changes stay visible; 100 % change = full length. */
export function changeLength(delta: number): number {
  return Math.min(1, Math.sqrt(Math.max(0, delta)));
}

export class ExperimentView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: ExperimentPick) => void;

  private L = 0;
  private H = 0;
  /** Vertical pitch of one layer (see rowPitch). */
  private RY = 1.3;
  private T = 0;
  private S = 0;
  private components: { heads: ComponentDisplay[][]; mlps: ComponentDisplay[] } = { heads: [], mlps: [] };
  private signed = true;
  private result: ExperimentResult | null = null;

  private readonly tileMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false });
  private readonly barMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  private readonly pickMaterial = new THREE.MeshBasicMaterial({ visible: false });
  private readonly frameMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  private readonly bandMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
  private tiles: THREE.InstancedMesh | null = null;
  private bars: THREE.InstancedMesh | null = null;
  private pickComponents: THREE.InstancedMesh | null = null;
  private cellTiles: THREE.InstancedMesh | null = null;
  private cellBars: THREE.InstancedMesh | null = null;
  private pickCells: THREE.InstancedMesh | null = null;
  private readonly componentFrame: THREE.LineSegments;
  private readonly rowBand: THREE.LineSegments;
  private readonly cellFrame: THREE.LineSegments;
  private readonly columnFrame: THREE.LineSegments;

  /** Signed bar lengths (component: −1…1, cell: 0…1), animated from → to. */
  private barFrom = new Float32Array(0);
  private barTo = new Float32Array(0);
  private barNow = new Float32Array(0);
  private cellFrom = new Float32Array(0);
  private cellTo = new Float32Array(0);
  private cellNow = new Float32Array(0);

  private readonly labels = new THREE.Group();
  private wallLabels: Label[] = [];
  private badgeLabels: Label[] = [];
  private tokenLabels: Label[] = [];
  private predictionLabels: Label[] = [];

  private hover: ExperimentPick | null = null;
  private highlight: ExperimentHighlight = { component: null, position: null, stage: null };
  private readonly raycaster = new THREE.Raycaster();

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: ExperimentPick): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.root.name = 'ExperimentView';
    const box = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    this.componentFrame = new THREE.LineSegments(box, this.frameMaterial);
    this.rowBand = new THREE.LineSegments(box, this.bandMaterial);
    this.cellFrame = new THREE.LineSegments(box, this.frameMaterial);
    this.columnFrame = new THREE.LineSegments(box, this.bandMaterial);
    for (const f of [this.componentFrame, this.rowBand, this.cellFrame, this.columnFrame]) {
      f.visible = false;
      f.raycast = () => {};
    }
    const lights = new THREE.Group();
    lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(-4, 7, 9);
    lights.add(sun);
    this.root.add(lights, this.componentFrame, this.rowBand, this.cellFrame, this.columnFrame, this.labels);
    this.applyPalette();
    o.viewport.scene.add(this.root);
  }

  // ----------------------------------------------------------------- layout

  stageY(stage: number): number {
    return ((stage - 1) * this.RY) / 2;
  }

  private get cellH(): number {
    return (this.RY / 2) * 0.8;
  }

  /** Height of a component tile (taller rows → taller tiles and bars). */
  private get tileH(): number {
    return Math.min(1.3, Math.max(TILE, this.RY * 0.42));
  }

  componentCenter(c: { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number }): Vec3 {
    return c.kind === 'head' ? [c.head * CX, this.stageY(2 * c.layer + 1), 0] : [this.H * CX + MLP_GAP, this.stageY(2 * c.layer + 2), 0];
  }

  private residualX(position: number): number {
    return this.H * CX + MLP_GAP + WALL_GAP + position * PX;
  }

  cellCenter(stage: number, position: number): Vec3 {
    return [this.residualX(position), this.stageY(stage), 0];
  }

  bounds(): { min: Vec3; max: Vec3 } {
    const top = this.stageY(Math.max(0, this.S - 1));
    const right = this.residualX(Math.max(0, this.T - 1)) + 1.2;
    return { min: [-1.6, this.stageY(0) - this.RY / 4 - 1.2, -BAR_MAX * 0.6], max: [right, top + this.RY / 4 + 2.4, BAR_MAX * 0.8] };
  }

  // ------------------------------------------------------------- components

  /** Component tiles and bars. `scale` = the |value| that gets a full-length bar. */
  setComponents(layers: number, heads: number, data: { heads: ComponentDisplay[][]; mlps: ComponentDisplay[] }, o: { signed: boolean; scale: number }): void {
    if (layers !== this.L || heads !== this.H) {
      this.L = layers;
      this.H = heads;
      this.RY = rowPitch(layers);
      this.buildComponents();
      if (this.result) this.buildResidual();
    }
    this.components = data;
    this.signed = o.signed;
    const n = this.L * (this.H + 1);
    const target = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const d = this.display(k);
      const v = d?.value ?? 0;
      target[k] = o.scale > 0 ? Math.max(-1, Math.min(1, v / o.scale)) : 0;
    }
    this.barFrom = Float32Array.from(this.barNow);
    this.barTo = target;
    this.recolorComponents();
    this.buildBadges();
    this.animate();
  }

  /** Instance k: heads of layer l at l·(H+1) + h, the MLP at l·(H+1) + H. */
  private display(k: number): ComponentDisplay | undefined {
    const l = Math.floor(k / (this.H + 1));
    const c = k % (this.H + 1);
    return c < this.H ? this.components.heads[l]?.[c] : this.components.mlps[l];
  }

  private componentAt(k: number): { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number } {
    const layer = Math.floor(k / (this.H + 1));
    const c = k % (this.H + 1);
    return c < this.H ? { kind: 'head', layer, head: c } : { kind: 'mlp', layer };
  }

  private buildComponents(): void {
    for (const mesh of [this.tiles, this.bars, this.pickComponents]) this.removeMesh(mesh);
    const n = this.L * (this.H + 1);
    const tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.tileMaterial, n);
    const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.barMaterial, n);
    const pick = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, n);
    const m = new THREE.Matrix4();
    for (let k = 0; k < n; k++) {
      const [x, y] = this.componentCenter(this.componentAt(k));
      m.makeScale(TILE, this.tileH, TILE_D).setPosition(x, y, 0);
      tiles.setMatrixAt(k, m);
      m.makeScale(CX * 0.96, this.tileH + 0.12, BAR_MAX * 2).setPosition(x, y, 0);
      pick.setMatrixAt(k, m);
    }
    tiles.raycast = () => {};
    bars.raycast = () => {};
    tiles.renderOrder = 3; // transparent tiles after the opaque bars behind them
    pick.computeBoundingSphere();
    this.tiles = tiles;
    this.bars = bars;
    this.pickComponents = pick;
    this.root.add(bars, tiles, pick);
    this.barNow = new Float32Array(n);
    this.barFrom = new Float32Array(n);
    this.barTo = new Float32Array(n);
    this.writeBars();
    this.buildWallLabels();
  }

  private writeBars(): void {
    const { bars } = this;
    if (!bars) return;
    const m = new THREE.Matrix4();
    const n = this.L * (this.H + 1);
    for (let k = 0; k < n; k++) {
      const [x, y] = this.componentCenter(this.componentAt(k));
      const v = this.barNow[k];
      const len = Math.max(0.0001, Math.abs(v) * BAR_MAX);
      const z = Math.sign(v || 1) * (TILE_D / 2 + len / 2);
      m.makeScale(BAR_W, this.tileH * 0.58, len).setPosition(x, y, z);
      bars.setMatrixAt(k, m);
    }
    bars.instanceMatrix.needsUpdate = true;
    bars.computeBoundingSphere();
  }

  private recolorComponents(): void {
    const { tiles, bars } = this;
    if (!tiles || !bars) return;
    const p = this.palette;
    const dark = p.theme === 'dark';
    const base = new THREE.Color(p.plateFill).lerp(new THREE.Color(p.plateBorder), 0.5);
    const off = new THREE.Color(dark ? '#ff7d7d' : '#cf3434');
    const scaled = new THREE.Color(dark ? '#ffc861' : '#b8770f');
    const pos = new THREE.Color(p.positive);
    const neg = new THREE.Color(p.negative);
    const neutral = new THREE.Color(dark ? '#a99bff' : '#5b46d6');
    const white = new THREE.Color(1, 1, 1);
    const c = new THREE.Color();
    const n = this.L * (this.H + 1);
    for (let k = 0; k < n; k++) {
      const d = this.display(k);
      const comp = this.componentAt(k);
      const hovered = this.hover && (this.hover.kind === 'head' || this.hover.kind === 'mlp') && samePick(this.hover, comp);
      c.copy(d?.state === 'off' ? off : d?.state === 'scaled' ? scaled : base);
      if (hovered) c.lerp(white, 0.35);
      tiles.setColorAt(k, c);
      const v = this.barTo[k];
      c.copy(!this.signed ? neutral : v >= 0 ? pos : neg);
      if (d?.value === null || d?.value === undefined) c.lerp(base, 0.8);
      if (hovered) c.lerp(white, 0.25);
      bars.setColorAt(k, c);
    }
    if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;
    if (bars.instanceColor) bars.instanceColor.needsUpdate = true;
    this.viewport.requestRender();
  }

  private buildBadges(): void {
    for (const l of this.badgeLabels) this.labels.remove(l.object);
    this.badgeLabels = [];
    const n = this.L * (this.H + 1);
    for (let k = 0; k < n; k++) {
      const d = this.display(k);
      if (!d?.badge) continue;
      const comp = this.componentAt(k);
      const [x, y] = this.componentCenter(comp);
      const el = this.label(`ex-badge is-${d.state}`, d.badge, comp, d.state === 'off' ? 'Switched off by the experiment' : 'Scaled by the experiment');
      this.badgeLabels.push(this.add(el, [x, y + this.tileH / 2 - 0.1, TILE_D / 2 + 0.02]));
    }
    this.viewport.requestRender();
  }

  // --------------------------------------------------------------- residual

  setResult(result: ExperimentResult, format: (text: string) => string): void {
    const T = result.tokens.length;
    const S = result.delta.length;
    this.result = result;
    if (T !== this.T || S !== this.S || !this.cellTiles) {
      this.T = T;
      this.S = S;
      this.buildResidual();
    }
    const target = new Float32Array(T * S);
    for (let s = 0; s < S; s++) for (let p = 0; p < T; p++) target[s * T + p] = changeLength(result.delta[s][p] ?? 0);
    this.cellFrom = Float32Array.from(this.cellNow);
    this.cellTo = target;
    this.recolorCells();
    this.buildResidualLabels(format);
    this.buildWallLabels();
    this.updateFrames();
    this.animate();
  }

  private buildResidual(): void {
    for (const mesh of [this.cellTiles, this.cellBars, this.pickCells]) this.removeMesh(mesh);
    const { T, S } = this;
    const n = Math.max(1, T * S);
    const tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.tileMaterial, n);
    const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.barMaterial, n);
    const pick = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), this.pickMaterial, n);
    tiles.count = bars.count = pick.count = T * S;
    const m = new THREE.Matrix4();
    for (let s = 0; s < S; s++) {
      for (let p = 0; p < T; p++) {
        const k = s * T + p;
        const [x, y] = this.cellCenter(s, p);
        m.makeScale(CELL_W, this.cellH, TILE_D).setPosition(x, y, 0);
        tiles.setMatrixAt(k, m);
        m.makeScale(PX * 0.96, (this.RY / 2) * 0.96, RBAR_MAX + 0.3).setPosition(x, y, (RBAR_MAX + 0.3) / 2 - 0.15);
        pick.setMatrixAt(k, m);
      }
    }
    tiles.raycast = () => {};
    bars.raycast = () => {};
    tiles.renderOrder = 3;
    pick.computeBoundingSphere();
    this.cellTiles = tiles;
    this.cellBars = bars;
    this.pickCells = pick;
    this.root.add(bars, tiles, pick);
    this.cellNow = new Float32Array(T * S);
    this.cellFrom = new Float32Array(T * S);
    this.cellTo = new Float32Array(T * S);
    this.writeCells();
  }

  private writeCells(): void {
    const { cellBars } = this;
    if (!cellBars) return;
    const m = new THREE.Matrix4();
    for (let s = 0; s < this.S; s++) {
      for (let p = 0; p < this.T; p++) {
        const k = s * this.T + p;
        const [x, y] = this.cellCenter(s, p);
        const len = Math.max(0.0001, this.cellNow[k] * RBAR_MAX);
        m.makeScale(RBAR_W, this.cellH * 0.7, len).setPosition(x, y, TILE_D / 2 + len / 2);
        cellBars.setMatrixAt(k, m);
      }
    }
    cellBars.instanceMatrix.needsUpdate = true;
    cellBars.computeBoundingSphere();
  }

  private recolorCells(): void {
    const { cellTiles, cellBars, result } = this;
    if (!cellTiles || !cellBars || !result) return;
    const p = this.palette;
    const dark = p.theme === 'dark';
    const base = new THREE.Color(p.plateFill).lerp(new THREE.Color(p.plateBorder), 0.35);
    const removed = new THREE.Color(dark ? '#ff7d7d' : '#cf3434');
    const low = new THREE.Color(p.neuronOff);
    const high = new THREE.Color(dark ? '#c3b8ff' : '#5b46d6');
    const white = new THREE.Color(1, 1, 1);
    const c = new THREE.Color();
    for (let s = 0; s < this.S; s++) {
      for (let q = 0; q < this.T; q++) {
        const k = s * this.T + q;
        const gone = result.delta[s][q] === null;
        const hovered = this.hover?.kind === 'cell' && this.hover.stage === s && this.hover.position === q;
        c.copy(gone ? removed : base);
        if (gone) c.lerp(base, 0.55);
        if (hovered) c.lerp(white, 0.35);
        cellTiles.setColorAt(k, c);
        c.copy(low).lerp(high, 0.35 + 0.65 * this.cellTo[k]);
        if (hovered) c.lerp(white, 0.25);
        cellBars.setColorAt(k, c);
      }
    }
    if (cellTiles.instanceColor) cellTiles.instanceColor.needsUpdate = true;
    if (cellBars.instanceColor) cellBars.instanceColor.needsUpdate = true;
    this.viewport.requestRender();
  }

  // ------------------------------------------------------------- animation

  private animate(): void {
    const { animations } = this.viewport;
    const barFrom = this.barFrom;
    const barTo = this.barTo;
    const cellFrom = this.cellFrom.length === this.cellTo.length ? this.cellFrom : new Float32Array(this.cellTo.length);
    const cellTo = this.cellTo;
    animations.tween({
      channel: 'experiment-bars',
      duration: 0.45,
      easing: easeInOutCubic,
      onUpdate: (t) => {
        for (let k = 0; k < barTo.length; k++) this.barNow[k] = barFrom[k] + (barTo[k] - barFrom[k]) * t;
        for (let k = 0; k < cellTo.length; k++) this.cellNow[k] = cellFrom[k] + (cellTo[k] - cellFrom[k]) * t;
        this.writeBars();
        this.writeCells();
        this.viewport.requestRender();
      },
    });
  }

  // ----------------------------------------------------------------- labels

  private label(className: string, text: string, pick: ExperimentPick | null, title?: string): HTMLElement {
    const el = document.createElement(pick ? 'button' : 'div');
    if (el instanceof HTMLButtonElement) el.type = 'button';
    el.className = className;
    el.textContent = text;
    if (title) el.title = title;
    if (pick) el.addEventListener('click', () => this.onLabelClick(pick));
    return el;
  }

  private title(text: string, sub: string): HTMLElement {
    const el = document.createElement('div');
    el.className = 'ex-title';
    const main = document.createElement('span');
    main.textContent = text;
    const small = document.createElement('span');
    small.className = 'ex-title__sub';
    small.textContent = sub;
    el.append(main, small);
    return el;
  }

  private add(el: HTMLElement, position: Vec3, center: [number, number] = [0.5, 0.5]): Label {
    const object = new CSS2DObject(el);
    object.position.set(...position);
    object.center.set(...center);
    this.labels.add(object);
    return { object, el };
  }

  private buildWallLabels(): void {
    for (const l of this.wallLabels) this.labels.remove(l.object);
    this.wallLabels = [];
    const { L, H } = this;
    if (!L) return;
    const titleY = this.stageY(2 * L) + this.RY / 4 + 2.1;
    this.wallLabels.push(this.add(this.title('Components', 'bars: effect of switching each one off alone'), [((H - 1) * CX + MLP_GAP) / 2 + 0.2, titleY, 0], [0.5, 1]));
    for (let h = 0; h < H; h++) this.wallLabels.push(this.add(this.label('ex-col', `h${h + 1}`, null, `Attention head ${h + 1} of each layer`), [h * CX, this.stageY(1) - this.tileH / 2 - 0.12, 0], [0.5, 0]));
    this.wallLabels.push(this.add(this.label('ex-col ex-col--mlp', 'MLP', null, 'The MLP of each layer (runs after its attention)'), [H * CX + MLP_GAP, this.stageY(1) - this.tileH / 2 - 0.12, 0], [0.5, 0]));
    for (let l = 0; l < L; l++) {
      this.wallLabels.push(this.add(this.label('ex-row', `layer ${l + 1}`, null), [-TILE / 2 - 0.15, this.stageY(2 * l + 1), 0], [1, 0.5]));
    }
    if (this.T) {
      const x0 = this.residualX(0);
      const x1 = this.residualX(this.T - 1);
      this.wallLabels.push(this.add(this.title('Residual stream', 'bars: how much each vector changed'), [(x0 + x1) / 2, titleY, 0], [0.5, 1]));
      // Deep models: only the embeddings and the output of each layer are named (the rows in between are “after attention”).
      const compact = this.L > 4;
      for (let s = 0; s < this.S; s++) {
        const layer = Math.floor((s - 1) / 2) + 1;
        const attn = s > 0 && (s - 1) % 2 === 0;
        if (compact && attn) continue;
        const name = s === 0 ? 'embed' : compact ? `L${layer}` : `L${layer} ${attn ? 'attn' : 'MLP'}`;
        const el = this.label(`ex-stage${s === this.S - 1 ? ' is-last' : ''}`, name, null, s === 0 ? 'Token + position embeddings' : `Residual stream after the ${attn ? 'attention' : 'MLP'} of layer ${layer}`);
        this.wallLabels.push(this.add(el, [x0 - CELL_W / 2 - 0.15, this.stageY(s), 0], [1, 0.5]));
      }
    }
    this.viewport.requestRender();
  }

  private buildResidualLabels(format: (text: string) => string): void {
    for (const l of [...this.tokenLabels, ...this.predictionLabels]) this.labels.remove(l.object);
    this.tokenLabels = [];
    this.predictionLabels = [];
    const r = this.result!;
    const removed = new Set(r.tokens.map((_, p) => p).filter((p) => !r.alignment.includes(p)));
    const yTop = this.stageY(this.S - 1);
    r.tokens.forEach((t, p) => {
      const j = r.alignment.indexOf(p);
      const after = j >= 0 ? r.tokensAfter[j] : null;
      const edited = after && after.id !== t.id;
      const el = this.label(`tt-token ex-token${removed.has(p) ? ' is-removed' : ''}${p === this.highlight.position ? ' is-selected' : ''}`, '', { kind: 'token', position: p }, `Token ${p} · click for details`);
      const text = document.createElement('span');
      text.textContent = format(t.text);
      el.append(text);
      if (removed.has(p)) {
        const tag = document.createElement('span');
        tag.className = 'tt-token__pos';
        tag.textContent = 'removed';
        el.append(tag);
      } else {
        const pos = document.createElement('span');
        pos.className = 'tt-token__pos';
        pos.textContent = edited ? `→ ${format(after.text)}` : String(p);
        el.append(pos);
      }
      const rotated = this.T > 8;
      const wrap = document.createElement('div');
      wrap.className = `ex-token-wrap${rotated ? ' is-rotated' : ''}`;
      wrap.append(el);
      const label = this.add(wrap, [this.residualX(p), this.stageY(0) - this.cellH / 2 - 0.12, 0], rotated ? [1, 0] : [0.5, 0]);
      this.tokenLabels.push({ object: label.object, el });
      // the model's next-token guess at this position, shown where the interventions changed it
      const before = r.before.predictions[p];
      const afterPrediction = j >= 0 ? r.after.predictions[j] : null;
      if (!afterPrediction || afterPrediction.id === before.id) return;
      const pred = this.label('ex-pred is-changed', `${format(before.text)} → ${format(afterPrediction.text)}`, { kind: 'cell', stage: this.S - 1, position: p }, `The model's guess for the next token after “${format(t.text)}” changed`);
      const tilted = this.T > 8;
      const holder = document.createElement('div');
      holder.className = `ex-pred-wrap${tilted ? ' is-rotated' : ''}`;
      holder.append(pred);
      const placed = this.add(holder, [this.residualX(p), yTop + this.cellH / 2 + 0.12, 0], tilted ? [0, 1] : [0.5, 1]);
      this.predictionLabels.push({ object: placed.object, el: pred });
    });
    this.viewport.requestRender();
  }

  // -------------------------------------------------------------- selection

  setHighlight(h: ExperimentHighlight): void {
    this.highlight = h;
    this.tokenLabels.forEach((label, p) => label.el.classList.toggle('is-selected', h.position === p));
    this.updateFrames();
  }

  private updateFrames(): void {
    const h = this.highlight;
    const c = h.component;
    if (c && this.L && c.layer < this.L) {
      const [x, y] = this.componentCenter(c);
      this.componentFrame.visible = true;
      this.componentFrame.scale.set(TILE + 0.14, this.tileH + 0.14, TILE_D + 0.2);
      this.componentFrame.position.set(x, y, 0);
      // the residual row this component writes into
      if (this.T) {
        const stage = c.kind === 'head' ? 2 * c.layer + 1 : 2 * c.layer + 2;
        const x0 = this.residualX(0);
        const x1 = this.residualX(this.T - 1);
        this.rowBand.visible = true;
        this.rowBand.scale.set(x1 - x0 + PX, this.RY / 2, 0.3);
        this.rowBand.position.set((x0 + x1) / 2, this.stageY(stage), 0);
      } else this.rowBand.visible = false;
    } else {
      this.componentFrame.visible = false;
      this.rowBand.visible = false;
    }
    if (h.position !== null && h.position < this.T) {
      const x = this.residualX(h.position);
      if (h.stage !== null && h.stage < this.S) {
        this.cellFrame.visible = true;
        this.cellFrame.scale.set(PX * 0.92, (this.RY / 2) * 0.92, 0.3);
        this.cellFrame.position.set(x, this.stageY(h.stage), 0);
      } else this.cellFrame.visible = false;
      this.columnFrame.visible = true;
      this.columnFrame.scale.set(PX * 0.98, this.stageY(this.S - 1) - this.stageY(0) + this.RY / 2, 0.3);
      this.columnFrame.position.set(x, (this.stageY(0) + this.stageY(this.S - 1)) / 2, 0);
    } else {
      this.cellFrame.visible = false;
      this.columnFrame.visible = false;
    }
    this.viewport.requestRender();
  }

  setHover(pick: ExperimentPick | null): void {
    const before = this.hover;
    this.hover = pick;
    if (samePick(before, pick)) return;
    const touches = (p: ExperimentPick | null) => p?.kind === 'head' || p?.kind === 'mlp';
    if (touches(before) || touches(pick)) this.recolorComponents();
    if (before?.kind === 'cell' || pick?.kind === 'cell') this.recolorCells();
    this.viewport.requestRender();
  }

  pick(x: number, y: number): ExperimentPick | null {
    if (!this.root.visible) return null;
    const { width, height } = this.viewport.size;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1), this.viewport.camera);
    const targets = [this.pickComponents, this.pickCells].filter((m): m is THREE.InstancedMesh => !!m);
    const hit = this.raycaster.intersectObjects(targets, false)[0];
    if (hit?.instanceId === undefined) return null;
    if (hit.object === this.pickComponents) return this.componentAt(hit.instanceId);
    return { kind: 'cell', stage: Math.floor(hit.instanceId / this.T), position: hit.instanceId % this.T };
  }

  // ------------------------------------------------------------------ misc

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPalette();
    this.recolorComponents();
    this.recolorCells();
  }

  private applyPalette(): void {
    this.frameMaterial.color.set(this.palette.plateFocus);
    this.bandMaterial.color.set(this.palette.theme === 'dark' ? '#c3b8ff' : '#5b46d6');
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  private removeMesh(mesh: THREE.InstancedMesh | null): void {
    if (!mesh) return;
    this.root.remove(mesh);
    mesh.geometry.dispose();
    mesh.dispose();
  }

  dispose(): void {
    this.viewport.animations.cancelChannel('experiment-bars');
    for (const mesh of [this.tiles, this.bars, this.pickComponents, this.cellTiles, this.cellBars, this.pickCells]) this.removeMesh(mesh);
    for (const l of [...this.wallLabels, ...this.badgeLabels, ...this.tokenLabels, ...this.predictionLabels]) this.labels.remove(l.object);
    this.tileMaterial.dispose();
    this.barMaterial.dispose();
    this.pickMaterial.dispose();
    this.frameMaterial.dispose();
    this.bandMaterial.dispose();
    this.componentFrame.geometry.dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}

function samePick(a: ExperimentPick | null, b: ExperimentPick | null): boolean {
  if (!a || !b) return a === b;
  return JSON.stringify(a) === JSON.stringify(b);
}
