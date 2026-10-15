/**
 * ArcMesh — many curved, camera-facing ribbons in one draw call (quadratic
 * Bézier curves evaluated in the vertex shader, see transformerMaterials.ts).
 * Used for attention arcs by the Transformer and LLM views.
 */
import * as THREE from 'three';
import type { Vec3 } from '../visualization/layout';
import { pointSegmentDistance, projectPoints } from './picking';
import { createArcMaterial, createRibbonGeometry } from './transformerMaterials';
import type { Viewport } from './Viewport';

export interface ArcInstance {
  start: Vec3;
  control: Vec3;
  end: Vec3;
  /** World units. */
  width: number;
  /** 0…1 */
  alpha: number;
  color: string;
  /** 0 dimmed, 1 normal, 2 highlighted. */
  emphasis: number;
  /** Arcs of group g grow while the stage goes from 2g+1 to 2g+1.5 (pipeline animation). */
  group: number;
}

const SAMPLES = 12;

export class ArcMesh {
  readonly mesh: THREE.Mesh;
  private readonly material = createArcMaterial();
  private readonly geometry = createRibbonGeometry();
  private capacity = 0;
  private arcs: ArcInstance[] = [];
  private stage = 1e6;
  private projected = new Float32Array(0);

  constructor() {
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    this.mesh.raycast = () => {};
    this.mesh.visible = false;
  }

  get count(): number {
    return this.arcs.length;
  }

  set(arcs: ArcInstance[]): void {
    this.arcs = arcs;
    const n = arcs.length;
    const g = this.geometry;
    if (n > this.capacity) {
      this.capacity = Math.max(64, Math.ceil(n * 1.5));
      const c = this.capacity;
      g.setAttribute('aStart', new THREE.InstancedBufferAttribute(new Float32Array(c * 3), 3));
      g.setAttribute('aControl', new THREE.InstancedBufferAttribute(new Float32Array(c * 3), 3));
      g.setAttribute('aEnd', new THREE.InstancedBufferAttribute(new Float32Array(c * 3), 3));
      g.setAttribute('aStyle', new THREE.InstancedBufferAttribute(new Float32Array(c * 4), 4));
      g.setAttribute('aColor', new THREE.InstancedBufferAttribute(new Float32Array(c * 3), 3));
    }
    if (this.capacity === 0) {
      this.mesh.visible = false;
      return;
    }
    const attr = (name: string) => g.getAttribute(name) as THREE.InstancedBufferAttribute;
    const start = attr('aStart');
    const control = attr('aControl');
    const end = attr('aEnd');
    const style = attr('aStyle');
    const color = attr('aColor');
    const c = new THREE.Color();
    arcs.forEach((a, k) => {
      start.setXYZ(k, ...a.start);
      control.setXYZ(k, ...a.control);
      end.setXYZ(k, ...a.end);
      style.setXYZW(k, a.width, a.alpha, a.group, a.emphasis);
      c.set(a.color);
      color.setXYZ(k, c.r, c.g, c.b);
    });
    for (const a of [start, control, end, style, color]) a.needsUpdate = true;
    g.instanceCount = n;
    this.mesh.visible = n > 0;
  }

  setStage(stage: number): void {
    this.stage = stage;
    this.material.uniforms.uStage.value = stage;
  }

  setHover(index: number): void {
    this.material.uniforms.uHover.value = index;
  }

  setColors(highlight: string, dimAlpha: number): void {
    (this.material.uniforms.uHighlight.value as THREE.Color).set(highlight);
    this.material.uniforms.uDimAlpha.value = dimAlpha;
  }

  private grown(a: ArcInstance): boolean {
    return this.stage >= 2 * a.group + 1.4;
  }

  /** Arc under the canvas point (x, y): within its drawn half-width + 3 px. */
  pick(x: number, y: number, viewport: Viewport, matrixWorld: THREE.Matrix4, accept: (a: ArcInstance) => boolean = () => true): { index: number; distance: number } | null {
    const candidates: number[] = [];
    this.arcs.forEach((a, k) => {
      if (this.grown(a) && accept(a)) candidates.push(k);
    });
    if (!candidates.length || !this.mesh.visible) return null;
    const points = new Float32Array(candidates.length * (SAMPLES + 1) * 3);
    let p = 0;
    for (const k of candidates) {
      const a = this.arcs[k];
      for (let s = 0; s <= SAMPLES; s++) {
        const t = s / SAMPLES;
        const u = 1 - t;
        for (let c = 0; c < 3; c++) points[p++] = u * u * a.start[c] + 2 * u * t * a.control[c] + t * t * a.end[c];
      }
    }
    if (this.projected.length < points.length) this.projected = new Float32Array(points.length);
    const vp = viewport.viewProjection().multiply(matrixWorld);
    projectPoints(points, vp.elements, viewport.size.width, viewport.size.height, this.projected);
    const pixelsPerUnit = viewport.size.height / (2 * Math.tan(THREE.MathUtils.degToRad(viewport.camera.fov) / 2));
    let best = -1;
    let bestScore = Infinity;
    candidates.forEach((k, index) => {
      const base = index * (SAMPLES + 1);
      const halfWidth = this.arcs[k].width / 2;
      for (let s = 0; s < SAMPLES; s++) {
        const i = 3 * (base + s);
        const j = 3 * (base + s + 1);
        if (!(this.projected[i + 2] > 0 && this.projected[j + 2] > 0)) continue;
        const { distance } = pointSegmentDistance(x, y, this.projected[i], this.projected[i + 1], this.projected[j], this.projected[j + 1]);
        const score = distance - (halfWidth * pixelsPerUnit) / this.projected[i + 2];
        if (score < bestScore) {
          bestScore = score;
          best = k;
        }
      }
    });
    return best >= 0 && bestScore <= 3 ? { index: best, distance: Math.max(0, bestScore) } : null;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/** Width and opacity of an attention arc for weight w ∈ [0, 1]. */
export function arcStyle(weight: number): { width: number; alpha: number } {
  const w = Math.max(0, Math.min(1, weight));
  return { width: 0.035 + 0.3 * Math.pow(w, 0.7), alpha: 0.1 + 0.9 * Math.pow(w, 0.6) };
}
