/**
 * Viewport — owns the WebGL renderer, scene, camera, orbit controls, the HTML
 * label layer and the render loop. Shared by every workspace: each workspace
 * adds its own objects to `scene`.
 *
 * Rendering is *on demand*: a frame is drawn only when something changed
 * (camera movement, an animation, `requestRender()`), so an idle app costs
 * almost nothing on laptops and older GPUs.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { AnimationController } from '../animation/AnimationController';

export interface RenderStats {
  calls: number;
  triangles: number;
}

export class Viewport {
  readonly container: HTMLElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly labelRenderer: CSS2DRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly animations = new AnimationController();
  readonly size = { width: 1, height: 1 };
  readonly stats: RenderStats = { calls: 0, triangles: 0 };

  private dirty = true;
  private rafId = 0;
  private lastTime = 0;
  private readonly beforeRender = new Set<(camera: THREE.PerspectiveCamera) => void>();
  private readonly afterRender = new Set<() => void>();
  private readonly resizeListeners = new Set<() => void>();
  private readonly resizeObserver: ResizeObserver;

  constructor(container: HTMLElement) {
    this.container = container;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.domElement.classList.add('viewport-canvas');
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.classList.add('label-layer');
    container.appendChild(this.labelRenderer.domElement);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 2000);
    this.camera.position.set(-6, 5, 12);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 400;
    this.controls.autoRotateSpeed = 0.8;
    this.controls.addEventListener('change', () => this.requestRender());
    // A user gesture always wins over an automatic camera flight.
    this.controls.addEventListener('start', () => this.animations.cancelChannel('camera'));

    this.animations.reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.rafId = requestAnimationFrame(this.loop);
  }

  /** Marks the scene as changed: a frame will be drawn on the next animation frame. */
  requestRender(): void {
    this.dirty = true;
  }

  /** Runs right before each rendered frame (e.g. billboarding). */
  onBeforeRender(fn: (camera: THREE.PerspectiveCamera) => void): () => void {
    this.beforeRender.add(fn);
    return () => this.beforeRender.delete(fn);
  }

  onAfterRender(fn: () => void): () => void {
    this.afterRender.add(fn);
    return () => this.afterRender.delete(fn);
  }

  /** Called after the drawing area changed size (window resize, panels opened/closed). */
  onResize(fn: () => void): () => void {
    this.resizeListeners.add(fn);
    return () => this.resizeListeners.delete(fn);
  }

  setBackground(color: string): void {
    this.scene.background = new THREE.Color(color);
    this.requestRender();
  }

  /** Converts client (page) coordinates to canvas CSS-pixel coordinates. */
  toCanvas(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  /** projection × view matrix for the current camera (column-major elements). */
  viewProjection(target = new THREE.Matrix4()): THREE.Matrix4 {
    this.camera.updateMatrixWorld();
    return target.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId);
    this.resizeObserver.disconnect();
    this.animations.cancelAll();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }

  private resize(): void {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    if (width === this.size.width && height === this.size.height) return;
    this.size.width = width;
    this.size.height = height;
    this.renderer.setSize(width, height);
    this.labelRenderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
    for (const fn of this.resizeListeners) fn();
  }

  private readonly loop = (time: number): void => {
    this.rafId = requestAnimationFrame(this.loop);
    const dt = this.lastTime ? Math.min((time - this.lastTime) / 1000, 0.1) : 1 / 60;
    this.lastTime = time;

    let needsRender = this.dirty;
    if (this.animations.update(dt)) needsRender = true;
    if (this.controls.update(dt)) needsRender = true;
    if (!needsRender) return;

    this.dirty = false;
    for (const fn of this.beforeRender) fn(this.camera);
    this.renderer.render(this.scene, this.camera);
    this.stats.calls = this.renderer.info.render.calls;
    this.stats.triangles = this.renderer.info.render.triangles;
    this.labelRenderer.render(this.scene, this.camera);
    for (const fn of this.afterRender) fn();
  };
}
