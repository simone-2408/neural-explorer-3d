/**
 * EmbeddingView — points in a 3-D projection of a high-dimensional space
 * (PCA of token embeddings or of residual vectors). Each point can have a
 * path (one position per stage of the residual stream): `setStage()` moves
 * the points along their paths, and faint trails show the whole journey.
 *
 * The axes are labelled with the share of variance each principal component
 * keeps, because the projection is lossy: points that look close in 3-D may
 * be far apart in the original space.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import { projectPoints } from './picking';
import type { Viewport } from './Viewport';

export type EmbeddingPick = { kind: 'word'; id: number } | { kind: 'point'; position: number };

export interface EmbeddingPoint {
  /** Coordinates per stage (a single entry for static points). */
  path: Vec3[];
  label: string;
  color: string;
  pick: EmbeddingPick;
  /** Drawn smaller and fainter (e.g. <s>). */
  muted?: boolean;
}

const RADIUS = 6;

const samePick = (a: EmbeddingPick | null, b: EmbeddingPick | null) =>
  !!a && !!b && a.kind === b.kind && (a.kind === 'word' ? a.id === (b as typeof a).id : a.position === (b as Extract<EmbeddingPick, { kind: 'point' }>).position);

export class EmbeddingView {
  readonly root = new THREE.Group();
  private readonly viewport: Viewport;
  private palette: ScenePalette;
  private readonly onLabelClick: (pick: EmbeddingPick) => void;
  private points: EmbeddingPoint[] = [];
  private scale = 1;
  private stage = 0;
  private mesh: THREE.InstancedMesh | null = null;
  private readonly sphere = new THREE.SphereGeometry(0.13, 18, 12);
  private readonly material = new THREE.MeshLambertMaterial({ color: 0xffffff });
  private trails: THREE.LineSegments | null = null;
  private readonly trailMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.45 });
  private readonly axes = new THREE.Group();
  private readonly axisMaterial = new THREE.LineBasicMaterial({ color: 0x808080 });
  private labels: { object: CSS2DObject; el: HTMLElement }[] = [];
  private axisLabels: CSS2DObject[] = [];
  private current = new Float32Array(0);
  private projected = new Float32Array(0);
  private selected: EmbeddingPick | null = null;
  private hover: EmbeddingPick | null = null;

  constructor(o: { viewport: Viewport; palette: ScenePalette; onLabelClick(pick: EmbeddingPick): void }) {
    this.viewport = o.viewport;
    this.palette = o.palette;
    this.onLabelClick = o.onLabelClick;
    this.root.name = 'EmbeddingView';
    const lights = new THREE.Group();
    lights.add(new THREE.HemisphereLight(0xffffff, 0x445066, 2.3));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(-3, 8, 6);
    lights.add(sun);
    this.root.add(lights, this.axes);
    this.applyPalette();
    o.viewport.scene.add(this.root);
  }

  /** Replaces the points. `axisLabels` name the three axes (e.g. "PC1 · 11%"). */
  setPoints(points: EmbeddingPoint[], axisLabels: string[], options: { trails: boolean }): void {
    this.clear();
    this.points = points;
    // Robust scale: a few outliers must not squeeze everything else into the centre.
    const radii: number[] = [];
    for (const p of points) for (const v of p.path) radii.push(Math.hypot(...v));
    radii.sort((a, b) => a - b);
    const reference = radii[Math.floor(0.92 * (radii.length - 1))] || 1e-9;
    this.scale = RADIUS / reference;

    const mesh = new THREE.InstancedMesh(this.sphere, this.material, Math.max(1, points.length));
    mesh.count = points.length;
    mesh.frustumCulled = false;
    mesh.raycast = () => {};
    const color = new THREE.Color();
    points.forEach((p, k) => mesh.setColorAt(k, color.set(p.color)));
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.mesh = mesh;
    this.root.add(mesh);

    if (options.trails && points.some((p) => p.path.length > 1)) {
      const positions: number[] = [];
      const colors: number[] = [];
      for (const p of points) {
        color.set(p.color);
        for (let s = 0; s + 1 < p.path.length; s++) {
          const a = p.path[s].map((v) => v * this.scale);
          const b = p.path[s + 1].map((v) => v * this.scale);
          positions.push(...a, ...b);
          const fade = 0.35 + 0.65 * ((s + 1) / (p.path.length - 1));
          colors.push(color.r * fade, color.g * fade, color.b * fade, color.r * fade, color.g * fade, color.b * fade);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      this.trails = new THREE.LineSegments(geometry, this.trailMaterial);
      this.trails.raycast = () => {};
      this.root.add(this.trails);
    }

    for (const p of points) {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = `tt-word${p.muted ? ' is-muted' : ''}`;
      el.textContent = p.label;
      el.style.setProperty('--group', p.color);
      el.addEventListener('click', () => this.onLabelClick(p.pick));
      const object = new CSS2DObject(el);
      object.center.set(-0.12, 0.5);
      this.root.add(object);
      this.labels.push({ object, el });
    }
    this.buildAxes(axisLabels);
    this.current = new Float32Array(points.length * 3);
    this.setStage(this.stage);
  }

  private buildAxes(labels: string[]): void {
    const length = RADIUS * 1.08;
    const directions: Vec3[] = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    directions.forEach((dir, k) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...dir).multiplyScalar(-length), new THREE.Vector3(...dir).multiplyScalar(length)]);
      this.axes.add(new THREE.Line(geometry, this.axisMaterial));
      const el = document.createElement('div');
      el.className = 'axis-label';
      el.textContent = labels[k] ?? '';
      const object = new CSS2DObject(el);
      object.position.set(...(dir.map((v) => v * (length + 0.5)) as Vec3));
      this.axes.add(object);
      this.axisLabels.push(object);
    });
  }

  /** Moves the points along their paths (stage may be fractional). */
  setStage(stage: number): void {
    this.stage = stage;
    const { mesh } = this;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    this.points.forEach((p, k) => {
      const last = p.path.length - 1;
      const s = Math.min(Math.max(stage, 0), last);
      const i = Math.floor(s);
      const j = Math.min(i + 1, last);
      const t = s - i;
      const pos = [0, 1, 2].map((c) => (p.path[i][c] + (p.path[j][c] - p.path[i][c]) * t) * this.scale) as Vec3;
      this.current.set(pos, 3 * k);
      const size = (p.muted ? 0.65 : 1) * (samePick(p.pick, this.selected) ? 1.8 : samePick(p.pick, this.hover) ? 1.45 : 1);
      m.makeScale(size, size, size).setPosition(...pos);
      mesh.setMatrixAt(k, m);
      this.labels[k]?.object.position.set(...pos);
    });
    mesh.instanceMatrix.needsUpdate = true;
    this.viewport.requestRender();
  }

  setSelection(pick: EmbeddingPick | null): void {
    this.selected = pick;
    this.labels.forEach((label, k) => label.el.classList.toggle('is-selected', samePick(this.points[k].pick, pick)));
    this.setStage(this.stage);
  }

  setHover(pick: EmbeddingPick | null): void {
    if (samePick(pick, this.hover) || (!pick && !this.hover)) return;
    this.hover = pick;
    this.labels.forEach((label, k) => label.el.classList.toggle('is-hover', samePick(this.points[k].pick, pick)));
    this.setStage(this.stage);
  }

  pick(x: number, y: number): EmbeddingPick | null {
    if (!this.root.visible || !this.points.length) return null;
    if (this.projected.length < this.current.length) this.projected = new Float32Array(this.current.length);
    const vp = this.viewport.viewProjection().multiply(this.root.matrixWorld);
    projectPoints(this.current, vp.elements, this.viewport.size.width, this.viewport.size.height, this.projected);
    let best = -1;
    let bestDistance = 16;
    for (let k = 0; k < this.points.length; k++) {
      if (!(this.projected[3 * k + 2] > 0)) continue;
      const distance = Math.hypot(this.projected[3 * k] - x, this.projected[3 * k + 1] - y);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = k;
      }
    }
    return best >= 0 ? this.points[best].pick : null;
  }

  /** Current 3-D position of a point (for camera focus). */
  positionOf(pick: EmbeddingPick): Vec3 | null {
    const k = this.points.findIndex((p) => samePick(p.pick, pick));
    return k >= 0 ? [this.current[3 * k], this.current[3 * k + 1], this.current[3 * k + 2]] : null;
  }

  bounds(): { min: Vec3; max: Vec3 } {
    // Most points lie well inside RADIUS (a roughly Gaussian cloud): frame the bulk of it.
    const r = RADIUS * 0.7;
    return { min: [-r, -r, -r], max: [r, r, r] };
  }

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPalette();
    this.viewport.requestRender();
  }

  private applyPalette(): void {
    this.axisMaterial.color.set(this.palette.axis);
    this.trailMaterial.opacity = this.palette.theme === 'dark' ? 0.45 : 0.6;
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  private clear(): void {
    if (this.mesh) {
      this.root.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
    }
    if (this.trails) {
      this.root.remove(this.trails);
      this.trails.geometry.dispose();
      this.trails = null;
    }
    for (const label of this.labels) this.root.remove(label.object);
    this.labels = [];
    for (const child of [...this.axes.children]) {
      this.axes.remove(child);
      if (child instanceof THREE.Line) child.geometry.dispose();
    }
    this.axisLabels = [];
  }

  dispose(): void {
    this.clear();
    this.sphere.dispose();
    this.material.dispose();
    this.trailMaterial.dispose();
    this.axisMaterial.dispose();
    this.viewport.scene.remove(this.root);
    this.viewport.requestRender();
  }
}
