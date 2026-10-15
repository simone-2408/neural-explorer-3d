/**
 * TrainingWorkspace — the "Training" tab: a small network learns a 2-D
 * dataset in real time. The stage is split: on the left the data and the
 * decision boundary (or the loss landscape, or the gradients), on the right
 * the same network in 3D, whose weights and activations change while it
 * trains. Every training step is a real mini-batch gradient-descent step
 * (verified against PyTorch in tests/pytorch-parity.test.ts).
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import type { AnimationHandle } from '../../animation/AnimationController';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { NetworkModel } from '../../neural/NetworkModel';
import { makeDataset, type Dataset } from '../../training/datasets';
import { flattenParameters, type LandscapeGrid, type Plane } from '../../training/landscape';
import { LOSSES, lossFor, sampleLoss } from '../../training/loss';
import { OPTIMIZERS } from '../../training/optimizers';
import { Trainer, type TrainerConfig } from '../../training/Trainer';
import { button } from '../../ui/widgets';
import type { InspectorView } from '../../ui/InspectorPanel';
import { toast } from '../../ui/overlays';
import type { NetworkSelection } from '../../visualization/selection';
import type { NetworkContext, Panel, TrainingInspectorInfo } from '../network/context';
import { connectionInspector } from '../network/inspectors/connectionInspector';
import { layerInspector } from '../network/inspectors/layerInspector';
import { neuronInspector } from '../network/inspectors/neuronInspector';
import { NetworkScene } from '../network/NetworkScene';
import { ViewPanel } from '../network/panels/ViewPanel';
import { ViewportToolbar } from '../network/ViewportToolbar';
import { BackpropPlayer } from './BackpropPlayer';
import type { Probe } from './BoundaryView';
import { diagnose, type Diagnosis } from '../../training/diagnostics';
import { DatasetPanel, OptimizerPanel, TrainingNetworkPanel, TrainingViewPanel } from './panels';
import { DEFAULT_TRAINING, sessionSpec, type TrainingSettings } from '../../training/session';
import { exportSession, importSession } from './sessionFile';
import { ScenarioPanel } from './ScenarioPanel';
import type { Scenario } from '../../training/scenarios';
import { SideView, type SideTab } from './SideView';
import { TrainingBar } from './TrainingBar';
import { trainingOverview } from './trainingOverview';

export type TrainingEvents = {
  /** Settings changed (dataset, architecture, optimizer…). */
  settings: undefined;
  /** Training made progress (steps, metrics). */
  progress: undefined;
  running: boolean;
  /** The model/dataset were rebuilt and the training restarted. */
  reset: undefined;
};

export type EncodeMode = 'weights' | 'gradients';

export class TrainingWorkspace implements Workspace {
  readonly id = 'training' as const;
  readonly events = new Emitter<TrainingEvents>();
  settings: TrainingSettings = structuredClone(DEFAULT_TRAINING);
  readonly model: NetworkModel;
  dataset: Dataset;
  trainer: Trainer;
  probe: Probe = { kind: 'sample', index: 0 };
  /** What the lines / neurons of the 3D network encode. */
  encode: EncodeMode = 'weights';
  gradientLog = true;
  showDelta = false;
  /** Parameter snapshots for the loss landscape (PCA of the trajectory). */
  readonly trajectory: Float64Array[] = [];
  plane: Plane | null = null;
  grid: LandscapeGrid | null = null;
  diagnosis: Diagnosis | null = null;
  /** 'break' shows the Break-the-Network scenarios. */
  mode: 'train' | 'break' = 'train';
  scenario: Scenario | null = null;
  scenarioFixed = false;

  private host: WorkspaceHost | null = null;
  private scene: NetworkScene | null = null;
  private ctx!: NetworkContext;
  private side: SideView | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private loop: AnimationHandle | null = null;
  private runningValue = false;
  private frame = 0;
  private lastBoundary = 0;
  private backprop: BackpropPlayer | null = null;

  constructor() {
    this.dataset = makeDataset(this.settings.dataset, this.settings.datasetOptions);
    this.model = new NetworkModel(sessionSpec(this.settings), { init: this.settings.init, input: this.probeInput() });
    this.trainer = new Trainer(this.model, this.dataset, this.trainerConfig());
    this.trainer.record();
    this.snapshotTrajectory(true);
  }

  get running(): boolean {
    return this.runningValue;
  }

  get networkScene(): NetworkScene | null {
    return this.scene;
  }

  get backpropPlayer(): BackpropPlayer | null {
    return this.backprop;
  }

  /** Shows a tab of the 2-D side view (lessons drive it). */
  showSideTab(tab: SideTab): void {
    this.side?.setTab(tab);
  }

  // ------------------------------------------------------------- lifecycle

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.bag = new DisposeBag();
    host.setSplit(0.42);
    const { store, viewport, slots } = host;

    this.scene = new NetworkScene({ host, model: this.model, inspect: (s) => this.inspect(s) });
    const scene = this.scene;
    this.ctx = {
      model: this.model,
      store,
      palette: () => scene.palette,
      actions: {
        select: (s) => scene.select(s),
        setHover: (s) => scene.setHover(s),
        focusSelection: () => scene.focusSelection(),
        resetCamera: () => scene.resetCamera(),
        setCameraView: (v) => scene.setCameraView(v),
        faceLayer: (l) => scene.faceLayer(l),
        navigate: (d) => scene.navigate(d),
        setView: (p) => scene.setView(p),
      },
      training: this.inspectorInfo(),
    };
    this.backprop = new BackpropPlayer(this, scene.view, viewport.animations);

    this.side = new SideView(this, () => scene.palette);
    slots.side.append(this.side.el);
    this.side.attach();

    const tools = [
      button({ icon: 'bolt', label: 'Backprop step', title: 'Animate one training step on the probe point: forward, loss, backward, update (B)', small: true, variant: 'ghost', onClick: () => this.animateBackprop() }),
    ];
    this.panels = [
      ...(this.mode === 'break' ? [new ScenarioPanel(this)] : []),
      new DatasetPanel(this),
      new TrainingNetworkPanel(this),
      new OptimizerPanel(this),
      new TrainingViewPanel(this),
      new ViewPanel(this.ctx),
    ];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new TrainingBar(this);
    slots.bottom.append(bar.el);
    const toolbar = new ViewportToolbar(this.ctx, tools);
    slots.stage.append(toolbar.el);
    this.panels.push(bar, toolbar);
    scene.showInspector();
    this.applyEncoding();
    this.refreshSide(true);

    this.bag.add(store.watch((s) => s.view.theme, (theme) => this.side?.setTheme(theme)));
    this.bag.add(
      store.watch(
        (s) => s.view.weightPalette,
        () => this.refreshSide(true),
      ),
    );
    this.bag.listen(window, 'keydown', (event) => {
      const e = event as KeyboardEvent;
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) return;
      if (e.key === ' ') {
        this.toggle();
        e.preventDefault();
      } else if (e.key === 'b' || e.key === 'B') {
        this.animateBackprop();
        e.preventDefault();
      }
    });
  }

  unmount(): void {
    this.pause();
    this.backprop?.stop();
    this.backprop = null;
    this.bag.dispose();
    for (const panel of this.panels) {
      panel.dispose();
      panel.el.remove();
    }
    this.panels = [];
    this.side?.dispose();
    this.side = null;
    this.scene?.dispose();
    this.scene = null;
    this.host?.setSplit(null);
    this.host = null;
  }

  // -------------------------------------------------------------- settings

  /** Applies new settings; rebuilds what depends on them. */
  update(patch: Partial<TrainingSettings>): void {
    const before = this.settings;
    const next: TrainingSettings = { ...before, ...patch };
    this.settings = next;
    const datasetChanged = patch.dataset !== undefined || patch.datasetOptions !== undefined;
    const architectureChanged = patch.hidden !== undefined || (patch.dataset !== undefined && patch.dataset !== before.dataset);
    const initChanged = patch.init !== undefined;

    if (datasetChanged) this.dataset = makeDataset(next.dataset, next.datasetOptions);
    if (architectureChanged || initChanged || datasetChanged) {
      this.restart({ rebuild: architectureChanged });
    } else {
      this.trainer.setConfig(this.trainerConfig());
    }
    this.events.emit('settings', undefined);
  }

  /** Re-initializes the weights and the training state (keeps the settings). */
  reset(): void {
    this.restart({ rebuild: false });
    this.events.emit('settings', undefined);
  }

  private restart({ rebuild }: { rebuild: boolean }): void {
    this.pause();
    this.backprop?.stop();
    const spec = sessionSpec(this.settings);
    this.host?.store.set({ selection: null, hover: null });
    this.model.batch(() => {
      if (rebuild || spec.layers.length !== this.model.layerCount || spec.layers.some((l, i) => l.size !== this.model.spec.layers[i].size || l.activation !== this.model.spec.layers[i].activation)) {
        this.model.setArchitecture(spec, { preserveParams: false, reason: 'config' });
      }
      this.model.reinitialize(this.settings.init);
    });
    if (this.probe.kind === 'sample' && this.probe.index >= this.dataset.size) this.probe = { kind: 'sample', index: 0 };
    this.model.setInputs(this.probeInput());
    this.trainer = new Trainer(this.model, this.dataset, this.trainerConfig());
    this.trainer.record();
    this.trajectory.length = 0;
    this.snapshotTrajectory(true);
    this.plane = null;
    this.grid = null;
    this.diagnosis = null;
    this.applyEncoding();
    this.refreshSide(true);
    this.events.emit('reset', undefined);
    this.events.emit('progress', undefined);
  }

  private trainerConfig(): TrainerConfig {
    return { optimizer: { ...this.settings.optimizer }, batchSize: this.settings.batchSize, seed: this.settings.datasetOptions.seed + 101 };
  }

  // ------------------------------------------------------------- scenarios

  applyScenario(scenario: Scenario): void {
    this.scenario = scenario;
    this.scenarioFixed = false;
    this.update(structuredClone(scenario.settings));
  }

  applyScenarioFix(): void {
    if (!this.scenario) return;
    this.scenarioFixed = true;
    this.update(structuredClone(this.scenario.fixed));
  }

  clearScenario(): void {
    this.scenario = null;
    this.scenarioFixed = false;
    this.events.emit('settings', undefined);
    this.refreshSide(true);
  }

  // -------------------------------------------------------------- training

  play(): void {
    if (this.runningValue) return;
    this.backprop?.stop();
    if (this.diagnosis?.severity === 'stop') this.diagnosis = null;
    this.runningValue = true;
    this.events.emit('running', true);
    if (!this.host) return;
    this.loop = this.host.viewport.animations.add({ update: () => this.tick() }, { channel: 'training-loop' });
  }

  pause(): void {
    if (!this.runningValue) return;
    this.runningValue = false;
    this.loop?.cancel();
    this.loop = null;
    this.events.emit('running', false);
    this.afterSteps(true);
  }

  toggle(): void {
    if (this.runningValue) this.pause();
    else this.play();
  }

  /** One mini-batch step. */
  stepOnce(): void {
    this.backprop?.stop();
    this.runSteps(1);
    this.afterSteps(true);
  }

  /** Steps until one more epoch is completed. */
  stepEpoch(): void {
    this.backprop?.stop();
    this.runSteps(this.trainer.stepsPerEpoch);
    this.afterSteps(true);
  }

  animateBackprop(): void {
    if (!this.backprop) return;
    this.pause();
    if (this.probe.kind !== 'sample') {
      toast('Click a data point (not an empty spot) to animate a training step on it: the step needs its label.', 'info', 4000);
      return;
    }
    this.backprop.start(this.probe.index);
  }

  /** Called by the backprop player after its real training step. */
  noteExternalStep(): void {
    this.afterStepBookkeeping();
    this.afterSteps(true);
  }

  private tick(): boolean {
    if (!this.runningValue) return true;
    this.frame++;
    const speed = this.settings.speed;
    if (speed === 'slow') {
      if (this.frame % 8 === 0) this.runSteps(1);
    } else if (speed === 'normal') this.runSteps(1);
    else if (speed === 'fast') this.runSteps(12);
    else {
      const start = performance.now();
      this.model.batch(() => {
        while (performance.now() - start < 9) {
          this.trainer.step();
          if (this.afterStepBookkeeping()) break;
        }
      });
    }
    this.afterSteps(false);
    return !this.runningValue;
  }

  private runSteps(count: number): void {
    this.model.batch(() => {
      for (let k = 0; k < count; k++) {
        this.trainer.step();
        if (this.afterStepBookkeeping()) break;
      }
    });
  }

  /** Cheap per-step check: returns true when training must stop (divergence, epoch target). */
  private afterStepBookkeeping(): boolean {
    const trainer = this.trainer;
    let stop = !Number.isFinite(trainer.lastBatchLoss);
    if (this.settings.stopAtEpoch > 0 && trainer.epochCount >= this.settings.stopAtEpoch) stop = true;
    if (stop && this.runningValue) queueMicrotask(() => this.pause());
    return stop;
  }

  /** Once per frame (or after a manual step): history, trajectory, diagnostics and views. */
  private afterSteps(force: boolean): void {
    const trainer = this.trainer;
    const last = trainer.history[trainer.history.length - 1];
    if (!last || trainer.epochCount > last.epoch || (force && trainer.stepCount > last.step)) {
      trainer.record();
      this.snapshotTrajectory(false);
    }
    this.diagnosis = diagnose(this);
    if (this.diagnosis?.severity === 'stop' && this.runningValue) queueMicrotask(() => this.pause());
    this.applyEncoding();
    const now = performance.now();
    const interval = this.model.topology.connectionCount > 300 ? 200 : 90;
    const full = force || now - this.lastBoundary > interval;
    if (full) this.lastBoundary = now;
    this.refreshSide(full);
    this.events.emit('progress', undefined);
  }

  private snapshotTrajectory(force: boolean): void {
    const last = this.trajectory[this.trajectory.length - 1];
    const current = flattenParameters(this.model);
    if (!force && last) {
      let d = 0;
      for (let k = 0; k < current.length; k++) d += (current[k] - last[k]) ** 2;
      if (d < 1e-10) return;
    }
    this.trajectory.push(current);
    if (this.trajectory.length > 400) {
      // keep the start, thin the middle
      this.trajectory.splice(1, 2);
    }
  }

  // ----------------------------------------------------------------- probe

  setProbe(probe: Probe): void {
    this.probe = probe;
    this.model.setInputs(this.probeInput());
    this.refreshSide(true);
    this.events.emit('progress', undefined);
  }

  probeInput(): [number, number] {
    const p = this.probe;
    if (p.kind === 'point') return [p.x, p.y];
    const k = Math.min(p.index, this.dataset.size - 1);
    return [this.dataset.x[2 * k], this.dataset.x[2 * k + 1]];
  }

  probeLabel(): number | null {
    return this.probe.kind === 'sample' ? this.dataset.y[Math.min(this.probe.index, this.dataset.size - 1)] : null;
  }

  // ------------------------------------------------------------- encoding

  setEncoding(patch: { encode?: EncodeMode; gradientLog?: boolean; showDelta?: boolean }): void {
    if (patch.encode !== undefined) this.encode = patch.encode;
    if (patch.gradientLog !== undefined) this.gradientLog = patch.gradientLog;
    if (patch.showDelta !== undefined) this.showDelta = patch.showDelta;
    this.applyEncoding();
    this.events.emit('settings', undefined);
  }

  /** Pushes the chosen encodings (weights/gradients, activations/δ) to the 3D view. */
  applyEncoding(): void {
    const view = this.scene?.view;
    if (!view || this.backprop?.active) return;
    const grads = this.trainer.gradients();
    view.setConnectionValues(this.encode === 'gradients' && grads ? { values: grads.dW, log: this.gradientLog } : null);
    view.setNeuronValues(this.showDelta && grads ? grads.delta : null);
  }

  refreshSide(full: boolean): void {
    this.side?.update(full);
  }

  // ------------------------------------------------------------- inspector

  private inspect(selection: NetworkSelection | null): InspectorView {
    if (!selection) return trainingOverview(this.ctx, this);
    if (selection.kind === 'neuron') return neuronInspector(this.ctx, selection.layer, selection.index);
    if (selection.kind === 'connection') return connectionInspector(this.ctx, selection.layer, selection.target, selection.source);
    return layerInspector(this.ctx, selection.layer);
  }

  private inspectorInfo(): TrainingInspectorInfo {
    return {
      gradients: () => this.trainer.gradients(),
      previousWeights: () => this.trainer.previousWeights(),
      previousBiases: () => this.trainer.previousBiases(),
      learningRate: () => this.settings.optimizer.learningRate,
      optimizerLabel: () => OPTIMIZERS[this.settings.optimizer.id].label,
      updateRuleTex: () => OPTIMIZERS[this.settings.optimizer.id].tex,
      lossInfo: () => {
        const label = this.probeLabel();
        const id = lossFor(this.model.spec);
        if (label === null) return null;
        return { value: sampleLoss(id, this.model.spec, this.model.output, label), tex: LOSSES[id].tex, label: LOSSES[id].label };
      },
    };
  }

  // ---------------------------------------------------------------- files

  exportSession(): void {
    exportSession(this);
  }

  async importSession(): Promise<void> {
    await importSession(this);
  }
}
