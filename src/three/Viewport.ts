/**
 * Viewport — owns the WebGL renderer, scene, camera, orbit controls, the HTML
 * label layer and the render loop. Shared by every workspace: each workspace
 * adds its own objects to `scene`.
 *
 * Rendering is *on demand*: a frame is drawn only when something changed
 * (camera movement, an animation, `requestRender()`), so an idle app costs
 * almost nothing on laptops and older GPUs. With the visual effects on (the
 * Atmosphere: glow, backdrop, dust), the ambient motion adds frames at 30 fps
 * unless the user prefers reduced motion; without effects nothing changes.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { AnimationController } from '../animation/AnimationController';
import { Atmosphere, type AtmosphereColors } from './Atmosphere';
import { FX } from './fx';

export interface RenderStats {
  calls: number;
  triangles: number;
}

export interface ViewportOptions {
  /** Visual-effects layer (main viewport only). */
  atmosphere?: boolean;
}

/** Interval between frames of the ambient motion, in seconds. */
const AMBIENT_FRAME = 1 / 30;
/** Resolution cap of the post-processed image (the HTML labels stay sharp regardless). */
const EFFECTS_PIXEL_RATIO = 1.5;

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
  private readonly atmosphere: Atmosphere | null = null;
  private effects = false;
  private colors: AtmosphereColors | null = null;
  private background = new THREE.Color('#0b0f15');
  private ambientElapsed = 0;
  private readonly effectsListeners = new Set<(on: boolean) => void>();
  private readonly beforeRender = new Set<(camera: THREE.PerspectiveCamera) => void>();
  private readonly afterRender = new Set<() => void>();
  private readonly resizeListeners = new Set<() => void>();
  private readonly resizeObserver: ResizeObserver;

  constructor(container: HTMLElement, options: ViewportOptions = {}) {
    this.container = container;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.domElement.classList.add('viewport-canvas');
    // Counted per frame (all passes), not per render call.
    this.renderer.info.autoReset = false;
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.classList.add('label-layer');
    container.appendChild(this.labelRenderer.domElement);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 2000);
    this.camera.position.set(-6, 5, 12);

    if (options.atmosphere) {
      this.atmosphere = new Atmosphere(this.renderer, this.scene, this.camera);
      this.atmosphere.group.visible = false;
      this.scene.add(this.atmosphere.group);
    }

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
    this.background.set(color);
    this.applyBackground();
  }

  /** Whether the visual effects are drawn (they need the atmosphere layer). */
  get effectsActive(): boolean {
    return this.effects;
  }

  /** Turns the visual effects on or off and gives them the theme's colours. */
  setEffects(on: boolean, colors: AtmosphereColors): void {
    const atmosphere = this.atmosphere;
    const active = on && atmosphere !== null;
    this.colors = colors;
    atmosphere?.setColors(colors);
    const changed = active !== this.effects;
    this.effects = active;
    const dark = colors.theme === 'dark';
    FX.uGlow.value = active && dark ? 1 : 0;
    FX.uMotion.value = active && dark && !this.animations.reducedMotion ? 1 : 0;
    this.renderer.toneMapping = active ? THREE.NeutralToneMapping : THREE.NoToneMapping;
    if (atmosphere) atmosphere.group.visible = active;
    this.applyBackground();
    this.resizeRenderTargets();
    if (changed) for (const fn of this.effectsListeners) fn(active);
  }

  onEffectsChange(fn: (on: boolean) => void): () => void {
    this.effectsListeners.add(fn);
    return () => this.effectsListeners.delete(fn);
  }

  /** Bounding box of the content: places the floor and the dust of the atmosphere. */
  setContentBounds(min: THREE.Vector3, max: THREE.Vector3): void {
    this.atmosphere?.setBounds(min, max);
    this.resizeRenderTargets();
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
    this.atmosphere?.dispose();
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
    this.resizeRenderTargets();
    this.requestRender();
    for (const fn of this.resizeListeners) fn();
  }

  private get effectsPixelRatio(): number {
    return Math.min(window.devicePixelRatio, EFFECTS_PIXEL_RATIO);
  }

  private resizeRenderTargets(): void {
    if (this.effects) this.atmosphere?.setSize(this.size.width, this.size.height, this.effectsPixelRatio);
  }

  private applyBackground(): void {
    // With effects the backdrop of the atmosphere replaces the flat background.
    this.scene.background = this.effects ? null : this.background;
    if (this.effects && this.colors && this.atmosphere) this.atmosphere.setColors({ ...this.colors, background: '#' + this.background.getHexString() });
    this.requestRender();
  }

  private readonly loop = (time: number): void => {
    this.rafId = requestAnimationFrame(this.loop);
    const dt = this.lastTime ? Math.min((time - this.lastTime) / 1000, 0.1) : 1 / 60;
    this.lastTime = time;

    let needsRender = this.dirty;
    if (this.animations.update(dt)) needsRender = true;
    if (this.controls.update(dt)) needsRender = true;
    if (this.effects && FX.uMotion.value > 0) {
      FX.uTime.value += dt;
      this.ambientElapsed += dt;
      if (this.ambientElapsed >= AMBIENT_FRAME) needsRender = true;
    }
    if (!needsRender) return;

    this.dirty = false;
    this.ambientElapsed = 0;
    for (const fn of this.beforeRender) fn(this.camera);
    this.renderer.info.reset();
    if (this.effects && this.atmosphere) this.atmosphere.render(this.camera, this.effectsPixelRatio);
    else this.renderer.render(this.scene, this.camera);
    this.stats.calls = this.renderer.info.render.calls;
    this.stats.triangles = this.renderer.info.render.triangles;
    this.labelRenderer.render(this.scene, this.camera);
    for (const fn of this.afterRender) fn();
  };
}
