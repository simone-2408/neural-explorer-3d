/**
 * CnnWorkspace — the "CNN" tab: a small convolutional network trained live
 * in the browser on 16×16 images (lines, shapes, digits).
 *
 * The 3D view shows every feature map; the inspectors show each convolution
 * with its numbers; training updates the filters while you watch them turn
 * into edge and line detectors. "Slide" animates a kernel moving over its
 * input, cell by cell.
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { IMAGE_DATASETS, IMAGE_SIZE, makeImageDataset, type ImageDataset, type ImageDatasetId } from '../../cnn/datasets';
import { runCnn, type CnnTrace } from '../../cnn/forward';
import { CnnModel, DEFAULT_CNN, type CnnConfig } from '../../cnn/model';
import { CnnTrainer } from '../../cnn/trainer';
import type { AnimationHandle } from '../../animation/AnimationController';
import type { CameraView } from '../../three/CameraController';
import { CnnView, type CnnPick } from '../../three/CnnView';
import { h, isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, Tooltip } from '../../ui/overlays';
import { frameThrottle } from '../../utils/schedule';
import type { Vec3 } from '../../visualization/layout';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import { DEFAULT_OPTIMIZER, Optimizer, type OptimizerSettings } from '../../training/optimizers';
import type { Panel } from '../network/context';
import { CnnBar } from './CnnBar';
import { CnnToolbar } from './CnnToolbar';
import { cellInspector } from './inspectors/cellInspector';
import { filterInspector } from './inspectors/filterInspector';
import { outputInspector } from './inspectors/outputInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { ArchitecturePanel, DataPanel, TrainingPanel } from './panels';

export type CnnSelection = CnnPick;

export type CnnSpeed = 'slow' | 'normal' | 'fast';

export interface CnnSettings {
  dataset: ImageDatasetId;
  architecture: Omit<CnnConfig, 'classes'>;
  optimizer: OptimizerSettings;
  batchSize: number;
  seed: number;
  speed: CnnSpeed;
  showValues: boolean;
}

export type CnnEvents = {
  /** Model, dataset or architecture rebuilt. */
  model: undefined;
  /** Weights changed (training step, edit). */
  progress: undefined;
  /** The input image changed. */
  input: undefined;
  running: boolean;
  selection: undefined;
  settings: undefined;
  slide: undefined;
};

export type CnnInput = { kind: 'sample'; index: number } | { kind: 'drawing' };

const VIEWS: Record<CameraView, Vec3> = {
  iso: [-0.35, 0.3, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.25, 0.4],
};

export class CnnWorkspace implements Workspace {
  readonly id = 'cnn' as const;
  readonly events = new Emitter<CnnEvents>();
  settings: CnnSettings = {
    dataset: 'lines',
    architecture: structuredClone(DEFAULT_CNN),
    optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.01 },
    batchSize: 16,
    seed: 1,
    speed: 'normal',
    showValues: true,
  };
  dataset: ImageDataset;
  model: CnnModel;
  trainer: CnnTrainer;
  input: CnnInput = { kind: 'sample', index: 0 };
  drawing = new Float64Array(IMAGE_SIZE * IMAGE_SIZE);
  trace: CnnTrace;
  selection: CnnSelection | null = null;
  /** Kernel slide animation: the map being computed and how many of its cells are done. */
  slide: { stage: number; map: number; count: number; playing: boolean } | null = null;

  private runningValue = false;
  private loop: AnimationHandle | null = null;
  private slideLoop: AnimationHandle | null = null;
  private lastRecord = 0;
  private lastViewUpdate = 0;
  private host: WorkspaceHost | null = null;
  private view: CnnView | null = null;
  private inspector: InspectorPanel | null = null;
  private tooltip: Tooltip | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private paletteValue: ScenePalette = scenePalette('dark', 'blue-orange');

  constructor() {
    this.dataset = makeImageDataset(this.settings.dataset, 1);
    this.model = this.newModel();
    this.trainer = this.newTrainer();
    this.trainer.record();
    this.trace = runCnn(this.model, this.inputPixels());
  }

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  get running(): boolean {
    return this.runningValue;
  }

  get classNames(): string[] {
    return this.dataset.info.classes;
  }

  private newModel(): CnnModel {
    return new CnnModel({ ...structuredClone(this.settings.architecture), classes: IMAGE_DATASETS[this.settings.dataset].classes.length }, this.settings.seed);
  }

  private newTrainer(): CnnTrainer {
    return new CnnTrainer(this.model, this.dataset, this.settings.optimizer, this.settings.batchSize, this.settings.seed + 7);
  }

  /** The image fed to the network now. */
  inputPixels(): Float64Array {
    return this.input.kind === 'drawing' ? this.drawing : this.dataset.test[this.input.index].pixels;
  }

  /** The label of the input, when it is a sample. */
  inputLabel(): number | null {
    return this.input.kind === 'sample' ? this.dataset.test[this.input.index].label : null;
  }

  // ------------------------------------------------------------- lifecycle

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.bag = new DisposeBag();
    host.setSplit(null);
    const { viewport, slots, store } = host;
    const v = store.state.view;
    this.paletteValue = scenePalette(v.theme, v.weightPalette);
    viewport.setBackground(this.paletteValue.background);
    this.view = new CnnView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.select(pick) });
    this.inspector = new InspectorPanel(slots.right);
    this.tooltip = new Tooltip(slots.stage);
    this.panels = [new DataPanel(this), new ArchitecturePanel(this), new TrainingPanel(this)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new CnnBar(this);
    slots.bottom.append(bar.el);
    const toolbar = new CnnToolbar(this);
    slots.stage.append(toolbar.el);
    this.panels.push(bar, toolbar);
    this.bag.add(
      store.watch(
        (s) => s.view.theme + s.view.weightPalette,
        () => {
          const view = store.state.view;
          this.paletteValue = scenePalette(view.theme, view.weightPalette);
          viewport.setBackground(this.paletteValue.background);
          this.view?.setPalette(this.paletteValue);
          this.showInspector(true);
        },
      ),
    );
    this.bindPointer();
    this.bindKeyboard();
    this.refreshView();
    this.view.setSelection(this.selection);
    this.fitView('iso', false);
    this.showInspector();
  }

  unmount(): void {
    this.pause();
    this.stopSlide();
    this.bag.dispose();
    for (const panel of this.panels) {
      panel.dispose();
      panel.el.remove();
    }
    this.panels = [];
    this.inspector?.dispose();
    this.inspector = null;
    this.tooltip?.el.remove();
    this.tooltip = null;
    this.view?.dispose();
    this.view = null;
    this.host = null;
  }

  // ---------------------------------------------------------------- model

  private recompute(): void {
    this.trace = runCnn(this.model, this.inputPixels());
  }

  private refreshView(): void {
    this.view?.setData(this.model, this.trace, this.classNames);
    this.view?.setShowValues(this.settings.showValues);
    this.resetCameraBounds();
  }

  /** Rebuilds the model (new architecture, dataset or seed) and restarts training. */
  private rebuild(): void {
    this.pause();
    this.stopSlide();
    this.model = this.newModel();
    this.trainer = this.newTrainer();
    this.trainer.record();
    this.lastRecord = performance.now();
    this.selection = null;
    this.recompute();
    this.refreshView();
    this.view?.setSelection(null);
    this.showInspector();
    this.events.emit('model', undefined);
    this.events.emit('progress', undefined);
    this.events.emit('selection', undefined);
  }

  update(patch: Partial<CnnSettings>): void {
    const before = this.settings;
    this.settings = { ...before, ...patch };
    if (patch.dataset && patch.dataset !== before.dataset) {
      this.dataset = makeImageDataset(patch.dataset, 1);
      this.input = { kind: 'sample', index: 0 };
      this.rebuild();
      this.fitView('iso', true);
    } else if (patch.architecture || patch.seed !== undefined) {
      this.rebuild();
      this.fitView('iso', true);
    } else {
      if (patch.optimizer) {
        if (patch.optimizer.id !== before.optimizer.id) this.trainer.optimizer = new Optimizer(this.settings.optimizer);
        else this.trainer.optimizer.settings.learningRate = this.settings.optimizer.learningRate;
      }
      if (patch.batchSize) this.trainer.batchSize = patch.batchSize;
      if (patch.showValues !== undefined) this.view?.setShowValues(patch.showValues);
    }
    this.events.emit('settings', undefined);
  }

  /** New random initial weights (same architecture). */
  reset(): void {
    this.update({ seed: this.settings.seed + 1 });
  }

  setInputSample(index: number): void {
    this.input = { kind: 'sample', index: Math.max(0, Math.min(this.dataset.test.length - 1, index)) };
    this.inputChanged();
  }

  setDrawing(pixels: ArrayLike<number>): void {
    this.drawing = Float64Array.from(pixels);
    this.input = { kind: 'drawing' };
    this.inputChanged();
  }

  private inputChanged(): void {
    this.recompute();
    this.refreshView();
    this.showInspector(true);
    this.events.emit('input', undefined);
  }

  /** Edits one weight of a convolution kernel (from the filter inspector). */
  setKernelWeight(layer: number, f: number, c: number, u: number, v: number, value: number): void {
    this.model.convWeight(layer)[this.model.weightIndex(layer, f, c, u, v)] = value;
    this.recompute();
    this.view?.setData(this.model, this.trace, this.classNames);
    this.inspector?.refresh();
    this.events.emit('progress', undefined);
  }

  // -------------------------------------------------------------- training

  play(): void {
    if (this.runningValue || !this.host) return;
    this.stopSlide();
    this.runningValue = true;
    this.events.emit('running', true);
    this.loop = this.host.viewport.animations.add({ update: () => this.tick() }, { channel: 'cnn-training' });
  }

  pause(): void {
    if (!this.runningValue) return;
    this.runningValue = false;
    this.loop?.cancel();
    this.loop = null;
    this.trainer.record();
    this.afterTraining(true);
    this.events.emit('running', false);
  }

  toggle(): void {
    if (this.runningValue) this.pause();
    else this.play();
  }

  stepOnce(): void {
    this.trainer.trainStep();
    this.trainer.record();
    this.afterTraining(true);
  }

  stepEpoch(): void {
    for (let s = 0; s < this.trainer.stepsPerEpoch; s++) this.trainer.trainStep();
    this.trainer.record();
    this.afterTraining(true);
  }

  private tick(): boolean {
    if (!this.runningValue) return true;
    const budget = { slow: 0, normal: 1, fast: 6 }[this.settings.speed];
    const now = performance.now();
    if (budget === 0) {
      if (now - this.lastViewUpdate > 150) this.trainer.trainStep();
    } else for (let k = 0; k < budget; k++) this.trainer.trainStep();
    if (now - this.lastRecord > 1500) {
      this.trainer.record();
      this.lastRecord = now;
    }
    this.afterTraining(false);
    return false;
  }

  /** View and inspector after weights changed (throttled while training). */
  private afterTraining(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.lastViewUpdate < 150) return;
    this.lastViewUpdate = now;
    this.recompute();
    this.view?.setData(this.model, this.trace, this.classNames);
    this.showInspector(true);
    this.events.emit('progress', undefined);
  }

  // ---------------------------------------------------------------- slide

  /** Animates filter `map` of conv stage `stage` sweeping over its input, one output cell at a time. */
  startSlide(stage: number, map: number): void {
    if (!this.host || !this.view) return;
    this.pause();
    this.slideLoop?.cancel();
    const st = this.view.stageList[stage];
    if (!st || st.kind !== 'conv') return;
    const total = st.size * st.size;
    this.slide = { stage, map, count: 0, playing: true };
    let time = 0;
    const perCell = st.size > 10 ? 0.06 : 0.14;
    this.slideLoop = this.host.viewport.animations.add(
      {
        update: (dt: number) => {
          const s = this.slide;
          if (!s || !s.playing) return false;
          time += dt;
          const count = Math.min(total, Math.floor(time / perCell));
          if (count !== s.count) {
            s.count = count;
            this.view?.setReveal({ stage, map, count });
            this.events.emit('slide', undefined);
          }
          if (count >= total) {
            s.playing = false;
            this.events.emit('slide', undefined);
            return true;
          }
          return false;
        },
      },
      { channel: 'cnn-slide' },
    );
    this.view.setReveal({ stage, map, count: 0 });
    this.events.emit('slide', undefined);
  }

  stopSlide(): void {
    this.slideLoop?.cancel();
    this.slideLoop = null;
    if (!this.slide) return;
    this.slide = null;
    this.view?.setReveal(null);
    this.events.emit('slide', undefined);
  }

  // ------------------------------------------------------------- selection

  select(selection: CnnSelection | null): void {
    this.selection = selection;
    this.view?.setSelection(selection);
    this.showInspector();
    this.events.emit('selection', undefined);
  }

  showInspector(keepScroll = false): void {
    this.inspector?.show(this.inspect(), keepScroll);
  }

  private inspect(): InspectorView {
    const s = this.selection;
    if (!s || !this.view) return overviewInspector(this);
    const stage = this.view.stageList[s.kind === 'output' ? this.view.stageList.length - 1 : s.stage];
    if (s.kind === 'output') return outputInspector(this, s.index);
    if (s.kind === 'map' && stage?.kind === 'conv') return filterInspector(this, s.stage, stage.layer, s.map);
    if (s.kind === 'cell') return cellInspector(this, s);
    return overviewInspector(this);
  }

  get stages() {
    return this.view?.stageList ?? [];
  }

  // ---------------------------------------------------------------- camera

  private resetCameraBounds(): void {
    const b = this.view?.bounds();
    if (b) this.host?.camera.setBounds(b.min, b.max);
  }

  private fitView(view: CameraView, animate: boolean): void {
    this.resetCameraBounds();
    this.host?.camera.fitDirection(VIEWS[view], animate);
  }

  setCameraView(view: CameraView): void {
    this.fitView(view, true);
  }

  focusSelection(): void {
    const s = this.selection;
    const view = this.view;
    if (!s || !view || !this.host) return;
    if (s.kind === 'cell') this.host.camera.focus(view.cellCenter(s.stage, s.map, s.i, s.j), 2.5);
    else if (s.kind === 'map') this.host.camera.focus(view.mapCenter(s.stage, s.map), 3);
    else this.host.camera.focus(view.mapCenter(view.stageList.length - 1, s.index), 3);
  }

  // ----------------------------------------------------------------- input

  private bindPointer(): void {
    const { viewport } = this.host!;
    const canvas = viewport.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    let last: { clientX: number; clientY: number } | null = null;
    const hoverAt = frameThrottle(() => {
      if (!last || down) return;
      const { x, y } = viewport.toCanvas(last.clientX, last.clientY);
      const hit = this.view?.pickAt(x, y) ?? null;
      this.view?.setHover(hit);
      canvas.style.cursor = hit ? 'pointer' : '';
      const text = hit ? this.view?.describe(hit) : null;
      if (text) this.tooltip?.show(x, y, h('div', null, text));
      else this.tooltip?.hide();
    });
    this.bag.add(() => hoverAt.cancel());
    this.bag.listen(canvas, 'pointermove', (event) => {
      last = { clientX: event.clientX, clientY: event.clientY };
      if (!down) hoverAt();
    });
    this.bag.listen(canvas, 'pointerdown', (event) => {
      down = { x: event.clientX, y: event.clientY };
      this.tooltip?.hide();
    });
    this.bag.listen(canvas, 'pointerup', (event) => {
      if (!down) return;
      const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
      down = null;
      if (moved < 5 && event.button === 0) {
        const { x, y } = viewport.toCanvas(event.clientX, event.clientY);
        this.select(this.view?.pickAt(x, y) ?? null);
      }
    });
    this.bag.listen(canvas, 'pointerleave', () => {
      last = null;
      this.view?.setHover(null);
      this.tooltip?.hide();
      canvas.style.cursor = '';
    });
    this.bag.listen(canvas, 'dblclick', () => this.focusSelection());
  }

  private bindKeyboard(): void {
    this.bag.listen(window, 'keydown', (event) => {
      const e = event as KeyboardEvent;
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isModalOpen() || isTypingTarget(e.target)) return;
      if (e.key.startsWith('Arrow') && e.target instanceof HTMLInputElement && e.target.type === 'range') return;
      if (this.handleKey(e)) e.preventDefault();
    });
  }

  private handleKey(e: KeyboardEvent): boolean {
    const s = this.selection;
    switch (e.key) {
      case ' ':
        this.toggle();
        return true;
      case 'Escape':
        if (this.slide) this.stopSlide();
        else if (s) this.select(null);
        else return false;
        return true;
      case 'n':
      case 'N':
        if (this.input.kind === 'sample') this.setInputSample((this.input.index + 1) % this.dataset.test.length);
        else this.setInputSample(0);
        return true;
      case 'ArrowLeft':
      case 'ArrowRight':
      case 'ArrowUp':
      case 'ArrowDown': {
        if (s?.kind !== 'cell') return false;
        const stage = this.stages[s.stage];
        const di = e.key === 'ArrowUp' ? -1 : e.key === 'ArrowDown' ? 1 : 0;
        const dj = e.key === 'ArrowLeft' ? -1 : e.key === 'ArrowRight' ? 1 : 0;
        this.select({ ...s, i: Math.max(0, Math.min(stage.size - 1, s.i + di)), j: Math.max(0, Math.min(stage.size - 1, s.j + dj)) });
        return true;
      }
      case 'r':
      case 'R':
        this.setCameraView('iso');
        return true;
      case 'f':
      case 'F':
        this.focusSelection();
        return true;
      case '1':
      case '2':
      case '3':
      case '4':
        this.setCameraView((['iso', 'side', 'top', 'front'] as const)[Number(e.key) - 1]);
        return true;
      default:
        return false;
    }
  }
}
