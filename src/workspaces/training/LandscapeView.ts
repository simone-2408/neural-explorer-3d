/**
 * 3-D loss surface on a 2-D slice of parameter space (see training/landscape.ts),
 * rendered in its own small Viewport: height and colour = loss (viridis,
 * perceptually uniform), a marker for the current parameters projected on the
 * plane, and the projected trajectory of the optimizer.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { Viewport } from '../../three/Viewport';
import { sampleGrid, type LandscapeGrid } from '../../training/landscape';
import { h } from '../../ui/dom';

const SIZE = 3; // surface spans [-SIZE, SIZE] in x and z
const HEIGHT = 2.4;

/** Viridis colour map (5 control points, linear interpolation). */
const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.229, 0.322, 0.546],
  [0.128, 0.567, 0.551],
  [0.369, 0.789, 0.383],
  [0.993, 0.906, 0.144],
];

function viridis(t: number, out: THREE.Color): THREE.Color {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  return out.setRGB(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, THREE.SRGBColorSpace);
}

export class LandscapeView {
  readonly el: HTMLElement;
  private readonly container: HTMLElement;
  private viewport: Viewport | null = null;
  private readonly group = new THREE.Group();
  private surface: THREE.Mesh<THREE.BufferGeometry, THREE.MeshLambertMaterial> | null = null;
  private lines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial> | null = null;
  private readonly marker: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  private readonly stem: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private path: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial> | null = null;
  private grid: LandscapeGrid | null = null;
  logScale = true;

  constructor() {
    this.container = h('div', { class: 'landscape__viewport' });
    this.el = h('div', { class: 'landscape' }, this.container);
    this.marker = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 14), new THREE.MeshBasicMaterial({ color: '#ffffff' }));
    this.stem = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.6 }));
    this.marker.visible = false;
    this.stem.visible = false;
    this.group.add(this.marker, this.stem);
  }

  /** Creates the WebGL viewport (call once the element is in the document). */
  attach(theme: 'dark' | 'light'): void {
    if (this.viewport) return;
    this.viewport = new Viewport(this.container);
    const { scene, camera, controls } = this.viewport;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(3, 6, 4);
    scene.add(sun, this.group);
    camera.position.set(5.6, 4.6, 6.4);
    controls.target.set(0, 0.9, 0);
    controls.zoomToCursor = false;
    camera.lookAt(controls.target);
    controls.update();
    for (const [text, position] of [
      ['α (direction 1)', new THREE.Vector3(SIZE + 0.35, 0, 0)],
      ['β (direction 2)', new THREE.Vector3(0, 0, SIZE + 0.35)],
      ['loss', new THREE.Vector3(-SIZE, HEIGHT + 0.3, -SIZE)],
    ] as const) {
      const label = h('div', { class: 'axis-label' }, text);
      const object = new CSS2DObject(label);
      object.position.copy(position);
      scene.add(object);
    }
    const axes = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-SIZE, 0, 0),
        new THREE.Vector3(SIZE, 0, 0),
        new THREE.Vector3(0, 0, -SIZE),
        new THREE.Vector3(0, 0, SIZE),
        new THREE.Vector3(-SIZE, 0, -SIZE),
        new THREE.Vector3(-SIZE, HEIGHT, -SIZE),
      ]),
      new THREE.LineBasicMaterial({ color: '#8a96a5', transparent: true, opacity: 0.6 }),
    );
    scene.add(axes);
    this.setTheme(theme);
  }

  setTheme(theme: 'dark' | 'light'): void {
    this.viewport?.setBackground(theme === 'dark' ? '#0d1219' : '#f7f8fb');
    this.marker.material.color.set(theme === 'dark' ? '#ffffff' : '#111111');
    this.stem.material.color.set(theme === 'dark' ? '#ffffff' : '#111111');
  }

  get hasGrid(): boolean {
    return this.grid !== null;
  }

  /** Rebuilds the surface from a (possibly partially computed) grid. */
  setGrid(grid: LandscapeGrid): void {
    this.grid = grid;
    const { size, values } = grid;
    if (grid.rowsDone < size) return;
    const geometry = new THREE.PlaneGeometry(SIZE * 2, SIZE * 2, size - 1, size - 1);
    geometry.rotateX(-Math.PI / 2);
    const positions = geometry.getAttribute('position') as THREE.BufferAttribute;
    const colors = new Float32Array(positions.count * 3);
    const color = new THREE.Color();
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        // PlaneGeometry vertices go row by row from -z (top) to +z; our rows go from β = -range (z = -SIZE).
        const vertex = r * size + c;
        const t = this.normalized(values[r * size + c]);
        positions.setY(vertex, t * HEIGHT);
        viridis(t, color);
        colors[3 * vertex] = color.r;
        colors[3 * vertex + 1] = color.g;
        colors[3 * vertex + 2] = color.b;
      }
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    this.surface?.geometry.dispose();
    this.surface?.removeFromParent();
    this.surface = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    this.group.add(this.surface);

    // Grid lines every few rows for readability.
    const points: THREE.Vector3[] = [];
    const step = Math.max(1, Math.floor(size / 10));
    const p = (r: number, c: number) => new THREE.Vector3(positions.getX(r * size + c), positions.getY(r * size + c) + 0.004, positions.getZ(r * size + c));
    for (let r = 0; r < size; r += step) for (let c = 0; c < size - 1; c++) points.push(p(r, c), p(r, c + 1));
    for (let c = 0; c < size; c += step) for (let r = 0; r < size - 1; r++) points.push(p(r, c), p(r + 1, c));
    this.lines?.geometry.dispose();
    this.lines?.removeFromParent();
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: '#000000', transparent: true, opacity: 0.18 }));
    this.group.add(this.lines);
    this.viewport?.requestRender();
  }

  /** Current point and trajectory, given in plane coordinates (α, β). */
  setPath(points: readonly [number, number][], current: [number, number] | null): void {
    const grid = this.grid;
    if (!grid || grid.rowsDone < grid.size) return;
    const toWorld = (alpha: number, beta: number) =>
      new THREE.Vector3((alpha / grid.range) * SIZE, this.normalized(sampleGrid(grid, alpha, beta)) * HEIGHT + 0.03, (beta / grid.range) * SIZE);
    const inside = ([a, b]: [number, number]) => Math.abs(a) <= grid.range && Math.abs(b) <= grid.range;
    this.path?.geometry.dispose();
    this.path?.removeFromParent();
    const visible = points.filter(inside);
    if (visible.length > 1) {
      this.path = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(visible.map(([a, b]) => toWorld(a, b))),
        new THREE.LineBasicMaterial({ color: '#ff6b6b' }),
      );
      this.group.add(this.path);
    } else this.path = null;
    if (current && inside(current)) {
      const p = toWorld(current[0], current[1]);
      this.marker.position.copy(p);
      this.stem.geometry.setFromPoints([new THREE.Vector3(p.x, 0, p.z), p]);
      this.marker.visible = true;
      this.stem.visible = true;
    } else {
      this.marker.visible = false;
      this.stem.visible = false;
    }
    this.viewport?.requestRender();
  }

  dispose(): void {
    this.surface?.geometry.dispose();
    this.lines?.geometry.dispose();
    this.path?.geometry.dispose();
    this.viewport?.dispose();
    this.viewport = null;
    this.el.remove();
  }

  private normalized(loss: number): number {
    const grid = this.grid!;
    const f = (v: number) => (this.logScale ? Math.log10(Math.max(v, 1e-6)) : v);
    const lo = f(grid.min);
    const hi = f(Math.min(grid.max, grid.min * 1e4 + 10));
    return hi - lo > 1e-12 ? Math.max(0, Math.min(1, (f(loss) - lo) / (hi - lo))) : 0;
  }
}
