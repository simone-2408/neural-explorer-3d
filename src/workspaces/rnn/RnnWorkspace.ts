/**
 * RnnWorkspace — the "RNN" tab: a recurrent network (vanilla RNN or GRU)
 * trained live on sequences, shown unrolled through time.
 *
 * Two tasks: predicting the next letter of the days of the week (memory of
 * the previous word decides "saturday" vs "sunday"), and remembering a letter
 * across a gap of meaningless symbols — where backpropagation through time
 * makes the gradient vanish for a vanilla RNN and a gated cell helps.
 */
import type { AnimationHandle } from '../../animation/AnimationController';
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { createRng } from '../../math/random';
import { backwardRnn, runRnn, zeroRnnGrads, type RnnTrace } from '../../rnn/forward';
import { RnnModel, type CellType } from '../../rnn/model';
import { makeTask, SIGNALS, type RnnTask, type RnnTaskId } from '../../rnn/tasks';
import { generate, RnnTrainer } from '../../rnn/trainer';
import type { CameraView } from '../../three/CameraController';
import { RnnView, type RnnPick } from '../../three/RnnView';
import { DEFAULT_OPTIMIZER, Optimizer, type OptimizerSettings } from '../../training/optimizers';
import { h, isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, toast, Tooltip } from '../../ui/overlays';
import { frameThrottle } from '../../utils/schedule';
import type { Vec3 } from '../../visualization/layout';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import type { Panel } from '../network/context';
import { inputInspector } from './inspectors/inputInspector';
import { outputInspector } from './inspectors/outputInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { stepInspector } from './inspectors/stepInspector';
import { weightsInspector } from './inspectors/weightsInspector';
import { NetworkPanel, PresetsPanel, TaskPanel, TrainingPanel } from './panels';
import { RnnBar } from './RnnBar';
import { RnnToolbar } from './RnnToolbar';

export type RnnSelection = RnnPick;
export type RnnSpeed = 'slow' | 'normal' | 'fast';

export interface RnnSettings {
  task: RnnTaskId;
  /** Memory task: number of meaningless symbols between the letter and "?". */
  gap: number;
  cell: CellType;
  hidden: number;
  /** GRU: initial bias of the update gate (0 = PyTorch's default). */
  updateGateBias: number;
  optimizer: OptimizerSettings;
  batchSize: number;
  /** Maximum gradient norm (0 = no clipping). */
  clip: number;
  seed: number;
  speed: RnnSpeed;
  showGradients: boolean;
}

export type RnnEvents = {
  model: undefined;
  progress: undefined;
  input: undefined;
  running: boolean;
  selection: undefined;
  settings: undefined;
};

export interface RnnPreset {
  id: string;
  label: string;
  description: string;
  settings: Partial<RnnSettings>;
}

export const RNN_PRESETS: RnnPreset[] = [
  {
    id: 'days',
    label: 'Next letter (GRU)',
    description: 'The days of the week, letter by letter. Train, then type “friday s”: the next letter is “a” only if the network remembers “friday”.',
    settings: { task: 'days', cell: 'gru', hidden: 24, updateGateBias: 0 },
  },
  {
    id: 'memory-short',
    label: 'Short memory (vanilla RNN)',
    description: 'Remember the first letter across 6 symbols. A vanilla RNN learns it quickly.',
    settings: { task: 'memory', gap: 6, cell: 'rnn', hidden: 16, updateGateBias: 0 },
  },
  {
    id: 'memory-long-rnn',
    label: 'Long memory (vanilla RNN)',
    description: 'The same with 30 symbols. Train and watch the gradient bars: at the first step the gradient is many orders of magnitude smaller than at the last. Learning is slow and unreliable — with some initial weights it stalls far below 100% (chance is 25%).',
    settings: { task: 'memory', gap: 30, cell: 'rnn', hidden: 16, updateGateBias: 0 },
  },
  {
    id: 'memory-long-gru',
    label: 'Long memory (GRU)',
    description: 'A GRU whose update gate starts close to 1 (it keeps its state unless the input says otherwise). The gradient reaching the first step stays within about two orders of magnitude of the last one, and the task is learned in a few hundred steps.',
    settings: { task: 'memory', gap: 30, cell: 'gru', hidden: 16, updateGateBias: 2 },
  },
];

const VIEWS: Record<CameraView, Vec3> = {
  // nearly frontal: long sequences use the full width; the slight angle still shows the bars' depth
  iso: [-0.12, 0.28, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.3, 0.5],
};

export class RnnWorkspace implements Workspace {
  readonly id = 'rnn' as const;
  readonly events = new Emitter<RnnEvents>();
  settings: RnnSettings = {
    task: 'days',
    gap: 8,
    cell: 'gru',
    hidden: 24,
    updateGateBias: 0,
    optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.02 },
    batchSize: 16,
    clip: 5,
    seed: 1,
    speed: 'normal',
    showGradients: true,
  };
  task: RnnTask;
  model: RnnModel;
  trainer: RnnTrainer;
  text = 'friday saturday s';
  inputs: number[] = [];
  targets: number[] = [];
  trace: RnnTrace = { steps: [] };
  /** ‖∂L/∂h_t‖ for the current input (backpropagation through time), or null without scored steps. */
  gradients: number[] | null = null;
  selection: RnnSelection | null = null;
  preset: RnnPreset | null = RNN_PRESETS[0];

  private unfoldTarget = true;
  private runningValue = false;
  private loop: AnimationHandle | null = null;
  private lastRecord = 0;
  private lastViewUpdate = 0;
  private readonly rng = createRng(11);
  private host: WorkspaceHost | null = null;
  private view: RnnView | null = null;
  private inspector: InspectorPanel | null = null;
  private tooltip: Tooltip | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private paletteValue: ScenePalette = scenePalette('dark', 'blue-orange');

  constructor() {
    this.task = makeTask(this.settings.task, this.settings.gap);
    this.model = this.newModel();
    this.trainer = this.newTrainer();
    this.trainer.record();
    this.encodeText();
    this.recompute();
  }

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  get running(): boolean {
    return this.runningValue;
  }

  private newModel(): RnnModel {
    const s = this.settings;
    return new RnnModel({ cell: s.cell, inputSize: this.task.inputSymbols.length, hiddenSize: s.hidden, outputSize: this.task.outputSymbols.length, updateGateBias: s.cell === 'gru' ? s.updateGateBias : 0 }, s.seed);
  }

  private newTrainer(): RnnTrainer {
    const s = this.settings;
    return new RnnTrainer(this.model, this.task, s.optimizer, { batchSize: s.batchSize, clip: s.clip, seed: s.seed + 5 });
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
    this.view = new RnnView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.select(pick) });
    this.inspector = new InspectorPanel(slots.right);
    this.tooltip = new Tooltip(slots.stage);
    this.panels = [new PresetsPanel(this), new TaskPanel(this), new NetworkPanel(this), new TrainingPanel(this)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new RnnBar(this);
    slots.bottom.append(bar.el);
    const toolbar = new RnnToolbar(this);
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
    this.view.setUnfold(this.unfoldTarget ? 1 : 0, false);
    this.view.setSelection(this.selection);
    this.fitView('iso', false);
    this.showInspector();
  }

  unmount(): void {
    this.pause();
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

  // ------------------------------------------------------------- the input

  private encodeText(): void {
    let inputs = this.task.encode(this.text).slice(0, 40);
    if (!inputs.length) inputs = this.task.encode(this.settings.task === 'days' ? 'monday' : 'A x y ?');
    this.inputs = inputs;
    this.targets = this.task.targetsFor(inputs);
  }

  /** Forward pass (and BPTT gradient norms) on the current input. */
  private recompute(): void {
    this.trace = runRnn(this.model, this.inputs);
    if (this.targets.some((t) => t >= 0)) {
      const { hiddenGradNorms } = backwardRnn(this.model, this.trace, this.targets, zeroRnnGrads(this.model));
      this.gradients = hiddenGradNorms;
    } else this.gradients = null;
  }

  private refreshView(): void {
    this.view?.setData({
      trace: this.trace,
      inputSymbols: this.task.inputSymbols,
      outputSymbols: this.task.outputSymbols,
      targets: this.targets,
      gradients: this.settings.showGradients ? this.gradients : null,
    });
    this.resetCameraBounds();
  }

  setText(text: string): void {
    this.text = text;
    this.encodeText();
    this.recompute();
    if (this.selection && 't' in this.selection && this.selection.t >= this.inputs.length) this.selection = null;
    this.refreshView();
    this.view?.setSelection(this.selection);
    this.fitView('iso', true);
    this.showInspector(true);
    this.events.emit('input', undefined);
  }

  /** A fresh example of the task (memory: a random letter and gap). */
  newExample(): void {
    if (this.settings.task === 'days') {
      const s = this.task.sample(this.rng);
      this.setText(s.inputs.map((i) => this.task.inputSymbols[i]).join('').trim());
    } else {
      const s = this.task.sample(this.rng);
      this.setText(s.inputs.map((i) => this.task.inputSymbols[i]).join(' '));
    }
  }

  /** Days task: appends letters predicted by the network. */
  continueText(mode: 'greedy' | 'sample'): void {
    if (this.settings.task !== 'days') return;
    const next = generate(this.model, this.inputs, 12, mode === 'greedy' ? 0 : 0.8, this.rng);
    this.setText((this.inputs.map((i) => this.task.inputSymbols[i]).join('') + next.map((i) => this.task.outputSymbols[i]).join('')).slice(-40));
  }

  // -------------------------------------------------------------- settings

  update(patch: Partial<RnnSettings>, o: { preset?: RnnPreset } = {}): void {
    const before = this.settings;
    this.settings = { ...before, ...patch };
    const s = this.settings;
    const taskChanged = s.task !== before.task || (s.task === 'memory' && s.gap !== before.gap);
    const modelChanged = taskChanged || s.cell !== before.cell || s.hidden !== before.hidden || s.updateGateBias !== before.updateGateBias || s.seed !== before.seed;
    this.preset = o.preset ?? (modelChanged ? null : this.preset);
    if (modelChanged) {
      this.pause();
      if (taskChanged) {
        this.task = makeTask(s.task, s.gap);
        this.text = s.task === 'days' ? 'friday saturday s' : ['B', ...Array.from({ length: s.gap }, (_, k) => 'xyz'[k % 3]), '?'].join(' ');
        this.encodeText();
      }
      this.model = this.newModel();
      this.trainer = this.newTrainer();
      this.trainer.record();
      this.lastRecord = performance.now();
      this.selection = null;
      this.recompute();
      this.refreshView();
      this.view?.setSelection(null);
      this.fitView('iso', true);
      this.showInspector();
      this.events.emit('model', undefined);
      this.events.emit('progress', undefined);
      this.events.emit('input', undefined);
    } else {
      if (patch.optimizer) {
        if (patch.optimizer.id !== before.optimizer.id) this.trainer.optimizer = new Optimizer(s.optimizer);
        else this.trainer.optimizer.settings.learningRate = s.optimizer.learningRate;
      }
      if (patch.batchSize) this.trainer.batchSize = s.batchSize;
      if (patch.clip !== undefined) this.trainer.clip = s.clip;
      if (patch.showGradients !== undefined) this.refreshView();
    }
    this.events.emit('settings', undefined);
  }

  applyPreset(preset: RnnPreset): void {
    this.update({ ...preset.settings, seed: this.settings.seed + 1 }, { preset });
  }

  reset(): void {
    this.update({ seed: this.settings.seed + 1 });
  }

  // -------------------------------------------------------------- training

  play(): void {
    if (this.runningValue || !this.host) return;
    this.runningValue = true;
    this.events.emit('running', true);
    this.loop = this.host.viewport.animations.add({ update: () => this.tick() }, { channel: 'rnn-training' });
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

  /** Runs `count` steps at once (e.g. 100), then shows the result. */
  stepMany(count: number): void {
    for (let k = 0; k < count; k++) this.trainer.trainStep();
    this.trainer.record();
    this.afterTraining(true);
  }

  private tick(): boolean {
    if (!this.runningValue) return true;
    const now = performance.now();
    const steps = { slow: 0, normal: 1, fast: 4 }[this.settings.speed];
    if (steps === 0) {
      if (now - this.lastViewUpdate > 200) this.trainer.trainStep();
    } else for (let k = 0; k < steps; k++) this.trainer.trainStep();
    if (now - this.lastRecord > 1500) {
      this.trainer.record();
      this.lastRecord = now;
    }
    this.afterTraining(false);
    return false;
  }

  private afterTraining(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.lastViewUpdate < 160) return;
    this.lastViewUpdate = now;
    this.recompute();
    this.refreshView();
    this.showInspector(true);
    this.events.emit('progress', undefined);
  }

  // ------------------------------------------------------------- selection

  select(selection: RnnSelection | null): void {
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
    if (!s) return overviewInspector(this);
    switch (s.kind) {
      case 'step':
        return stepInspector(this, s.t);
      case 'input':
        return inputInspector(this, s.t);
      case 'output':
        return outputInspector(this, s.t);
      case 'weights':
        return weightsInspector(this, s.which);
    }
  }

  /** Unrolled through time (true) or folded into one cell with a loop. */
  setUnfold(unfold: boolean): void {
    this.unfoldTarget = unfold;
    this.view?.setUnfold(unfold ? 1 : 0);
    this.fitView('iso', true);
    this.events.emit('settings', undefined);
  }

  get unfolded(): boolean {
    return this.unfoldTarget;
  }

  /** Letter of a memory-task sequence, for the descriptions. */
  signalOf(inputs: number[]): string | null {
    return inputs[0] !== undefined && inputs[0] < SIGNALS.length ? SIGNALS[inputs[0]] : null;
  }

  // ---------------------------------------------------------------- camera

  private resetCameraBounds(): void {
    const b = this.view?.bounds(this.unfoldTarget ? 1 : 0);
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
    if (!s || !this.view || !this.host || s.kind === 'weights') return;
    this.host.camera.focus(this.view.stepCenter(s.t), 3);
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
    const T = this.inputs.length;
    switch (e.key) {
      case ' ':
        this.toggle();
        return true;
      case 'Escape':
        if (!s) return false;
        this.select(null);
        return true;
      case 'ArrowLeft':
      case 'ArrowRight': {
        const d = e.key === 'ArrowLeft' ? -1 : 1;
        if (s && s.kind !== 'weights') this.select({ ...s, t: Math.max(0, Math.min(T - 1, s.t + d)) });
        else this.select({ kind: 'step', t: d < 0 ? T - 1 : 0 });
        return true;
      }
      case 'u':
      case 'U':
        this.setUnfold(!this.unfolded);
        return true;
      case 'g':
      case 'G':
        this.update({ showGradients: !this.settings.showGradients });
        return true;
      case 'n':
      case 'N':
        this.newExample();
        return true;
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

  /** Text shown when the typed input has no symbols the task knows. */
  warnUnknown(): void {
    toast(this.settings.task === 'days' ? 'Only the letters of the days of the week and spaces are known.' : 'Use A–D, x, y, z and ? (spaces are ignored).', 'info');
  }
}
