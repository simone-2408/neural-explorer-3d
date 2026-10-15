/**
 * Compare mode of the Training tab: two networks (A and B) trained side by
 * side on the same data with the same mini-batches, so the only difference is
 * the one under study (depth, activation, optimizer, capacity, regularization).
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import type { AnimationHandle } from '../../animation/AnimationController';
import { DisposeBag } from '../../core/DisposeBag';
import { ACTIVATIONS, type ActivationId } from '../../math/activations';
import { NetworkModel } from '../../neural/NetworkModel';
import { NetworkView } from '../../three/NetworkView';
import { COMPARISONS, type Comparison, type Contestant } from '../../training/comparisons';
import { DATASETS, datasetInfo, makeDataset, type Dataset, type DatasetId } from '../../training/datasets';
import { OPTIMIZERS, type OptimizerId } from '../../training/optimizers';
import { DEFAULT_TRAINING, sessionSpec, type TrainingSpeed } from '../../training/session';
import { Trainer } from '../../training/Trainer';
import { LineChart } from '../../ui/charts/lineChart';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { button, kv, section, segmented, select, type Segmented } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { toViewOptions } from '../network/NetworkScene';
import { block, inspectorHeader, paragraph } from '../network/inspectors/common';
import { BoundaryView } from './BoundaryView';

interface Run {
  label: 'A' | 'B';
  contestant: Contestant;
  model: NetworkModel;
  trainer: Trainer;
  /** Wall-clock time spent training (ms) and steps done, for "training speed". */
  time: number;
  view: NetworkView | null;
  boundary: BoundaryView | null;
  tag: CSS2DObject | null;
}

const ACTIVATION_CHOICES: ActivationId[] = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu'];

export class CompareWorkspace implements Workspace {
  readonly id = 'training' as const;
  comparison: Comparison = COMPARISONS[0];
  dataset: Dataset;
  datasetId: DatasetId;
  batchSize: number;
  runs: [Run, Run];
  speed: TrainingSpeed = 'fast';
  private running = false;
  private loop: AnimationHandle | null = null;
  private host: WorkspaceHost | null = null;
  private bag = new DisposeBag();
  private elements: HTMLElement[] = [];
  private palette: ScenePalette = scenePalette('dark', 'blue-orange');
  private inspector: InspectorPanel | null = null;
  private readonly lossChart = new LineChart({ title: 'Loss (A solid, B dashed; train)', log: true, xLabel: 'epoch', format: (v) => formatValue(v, 4) });
  private readonly accuracyChart = new LineChart({ title: 'Test accuracy (A solid, B dashed)', yMin: 0, yMax: 1, xLabel: 'epoch', format: (v) => `${formatValue(v * 100, 1)}%` });
  private readonly stats = h('div', { class: 'compare__stats' });
  private readonly editors = h('div', { class: 'compare__editors' });
  private readonly presetText = h('div', { class: 'compare__question' });
  private readonly counters = h('div', { class: 'training-bar__counters' });
  private playButton: HTMLButtonElement | null = null;
  private speedControl: Segmented<TrainingSpeed> | null = null;
  private datasetSelect: HTMLSelectElement | null = null;
  private lastBoundary = 0;

  constructor() {
    this.datasetId = this.comparison.dataset;
    this.batchSize = this.comparison.batchSize;
    this.dataset = makeDataset(this.datasetId, { ...DEFAULT_TRAINING.datasetOptions, noise: datasetInfo(this.datasetId).noise, ...this.comparison.datasetOptions });
    this.runs = [this.makeRun('A', this.comparison.a), this.makeRun('B', this.comparison.b)];
  }

  // ------------------------------------------------------------- lifecycle

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.bag = new DisposeBag();
    host.setSplit(0.5);
    const view = host.store.state.view;
    this.palette = scenePalette(view.theme, view.weightPalette);
    host.viewport.setBackground(this.palette.background);
    this.buildViews();

    // Left panel.
    const presetSelect = select<string>({
      options: COMPARISONS.map((c) => ({ value: c.id, label: c.title })),
      value: this.comparison.id,
      ariaLabel: 'Comparison',
      onChange: (id) => this.applyComparison(COMPARISONS.find((c) => c.id === id)!),
    });
    this.datasetSelect = select<DatasetId>({
      options: DATASETS.map((d) => ({ value: d.id, label: d.label })),
      value: this.datasetId,
      ariaLabel: 'Dataset',
      onChange: (id) => {
        this.datasetId = id;
        this.rebuild();
      },
    });
    const left = [
      section({ id: 'compare-preset', title: 'Comparison', children: [presetSelect, this.presetText, h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Dataset (shared)'), this.datasetSelect)] }),
      section({ id: 'compare-networks', title: 'Networks A and B', children: [this.editors] }),
    ];
    host.slots.left.append(...left);

    // Side: boundaries + charts + stats.
    const side = h(
      'div',
      { class: 'side' },
      h('div', { class: 'compare__boundaries' }, this.runs.map((run) => h('div', { class: 'compare__boundary' }, h('div', { class: 'compare__label' }, `Network ${run.label}`), run.boundary!.el))),
      h('div', { class: 'side__charts' }, this.lossChart.el, this.accuracyChart.el),
      this.stats,
    );
    host.slots.side.append(side);

    // Bottom bar.
    this.playButton = button({ icon: 'play', label: 'Train both', small: true, variant: 'primary', onClick: () => this.toggle() });
    this.speedControl = segmented<TrainingSpeed>({
      options: [
        { value: 'normal', label: '1×' },
        { value: 'fast', label: '12×' },
        { value: 'max', label: 'Max' },
      ],
      value: this.speed,
      ariaLabel: 'Training speed',
      onChange: (speed) => {
        this.speed = speed;
        this.speedControl?.set(speed);
      },
    });
    const bar = h(
      'div',
      { class: 'training-bar' },
      h(
        'div',
        { class: 'training-bar__transport' },
        this.playButton,
        button({ icon: 'stepForward', label: 'Epoch', small: true, onClick: () => this.steps(this.runs[0].trainer.stepsPerEpoch) }),
        button({ icon: 'reset', label: 'Reset', small: true, onClick: () => this.rebuild() }),
      ),
      h('div', { class: 'training-bar__speed' }, h('span', { class: 'field__label' }, 'Speed'), this.speedControl.el),
      this.counters,
    );
    host.slots.bottom.append(bar);

    this.inspector = new InspectorPanel(host.slots.right);
    this.elements = [...left, side, bar];
    this.bag.add(
      host.store.watch(
        (s) => s.view,
        (v) => {
          this.palette = scenePalette(v.theme, v.weightPalette);
          host.viewport.setBackground(this.palette.background);
          for (const run of this.runs) run.view?.setPalette(this.palette);
          this.refresh(true);
        },
      ),
    );
    this.renderEditors();
    this.refresh(true);
    this.fitCamera(false);
  }

  unmount(): void {
    this.pause();
    this.bag.dispose();
    for (const run of this.runs) this.disposeRunViews(run);
    for (const el of this.elements) el.remove();
    this.elements = [];
    this.inspector?.dispose();
    this.inspector = null;
    this.host?.setSplit(null);
    this.host = null;
  }

  // -------------------------------------------------------------- training

  toggle(): void {
    if (this.running) this.pause();
    else this.play();
  }

  play(): void {
    if (this.running || !this.host) return;
    this.running = true;
    this.loop = this.host.viewport.animations.add({ update: () => this.tick() }, { channel: 'compare-loop' });
    this.refresh(false);
  }

  pause(): void {
    if (!this.running) return;
    this.running = false;
    this.loop?.cancel();
    this.loop = null;
    this.refresh(true);
  }

  private tick(): boolean {
    if (!this.running) return true;
    if (this.speed === 'normal') this.steps(1, false);
    else if (this.speed === 'fast') this.steps(12, false);
    else {
      const start = performance.now();
      while (performance.now() - start < 9) this.steps(4, false);
      this.refresh(false);
    }
    return false;
  }

  /** The same number of steps for A and B (with their own timing). */
  steps(count: number, refresh = true): void {
    for (const run of this.runs) {
      const start = performance.now();
      run.model.batch(() => {
        for (let k = 0; k < count; k++) {
          if (!Number.isFinite(run.trainer.lastBatchLoss) && run.trainer.stepCount > 0) break;
          run.trainer.step();
        }
      });
      run.time += performance.now() - start;
      const last = run.trainer.history[run.trainer.history.length - 1];
      if (!last || run.trainer.epochCount > last.epoch) run.trainer.record();
    }
    if (refresh || this.speed !== 'max') this.refresh(false);
  }

  // ---------------------------------------------------------------- setup

  private makeRun(label: 'A' | 'B', contestant: Contestant): Run {
    const settings = { ...DEFAULT_TRAINING, dataset: this.datasetId, hidden: contestant.hidden, init: contestant.init, optimizer: contestant.optimizer, batchSize: this.batchSize };
    const model = new NetworkModel(sessionSpec(settings), { init: contestant.init, input: [this.dataset.x[0], this.dataset.x[1]] });
    const trainer = new Trainer(model, this.dataset, { optimizer: contestant.optimizer, batchSize: this.batchSize, seed: this.dataset.options.seed + 101 });
    trainer.record();
    return { label, contestant: structuredClone(contestant), model, trainer, time: 0, view: null, boundary: null, tag: null };
  }

  applyComparison(comparison: Comparison): void {
    this.comparison = comparison;
    this.datasetId = comparison.dataset;
    this.batchSize = comparison.batchSize;
    if (this.datasetSelect) this.datasetSelect.value = this.datasetId;
    this.runs[0].contestant = structuredClone(comparison.a);
    this.runs[1].contestant = structuredClone(comparison.b);
    this.rebuild();
  }

  /** Rebuilds data, models and views from the current contestants. */
  rebuild(): void {
    this.pause();
    this.dataset = makeDataset(this.datasetId, { ...DEFAULT_TRAINING.datasetOptions, noise: datasetInfo(this.datasetId).noise, ...(this.datasetId === this.comparison.dataset ? this.comparison.datasetOptions : {}) });
    const contestants = this.runs.map((r) => r.contestant);
    for (const run of this.runs) this.disposeRunViews(run);
    this.runs = [this.makeRun('A', contestants[0]), this.makeRun('B', contestants[1])];
    if (this.host) {
      this.buildViews();
      const boundaries = this.host.slots.side.querySelector('.compare__boundaries');
      if (boundaries) replaceChildren(boundaries, this.runs.map((run) => h('div', { class: 'compare__boundary' }, h('div', { class: 'compare__label' }, `Network ${run.label}`), run.boundary!.el)));
      this.renderEditors();
      this.refresh(true);
      this.fitCamera(true);
    }
  }

  private buildViews(): void {
    const host = this.host!;
    const options = toViewOptions(host.store.state.view);
    for (const run of this.runs) {
      run.view = new NetworkView({ viewport: host.viewport, model: run.model, palette: this.palette, options: { ...options, showFloor: false } });
      run.view.build();
      run.boundary = new BoundaryView({ model: run.model, dataset: () => this.dataset, palette: () => this.palette, onProbe: () => undefined });
    }
    // Side by side along z, with a gap.
    const [a, b] = this.runs.map((r) => r.view!.getLayout());
    const gap = 2.5;
    const offsetA = -(a.boundsMax[2] + gap / 2);
    const offsetB = -b.boundsMin[2] + gap / 2;
    this.runs[0].view!.root.position.z = offsetA;
    this.runs[1].view!.root.position.z = offsetB;
    for (const run of this.runs) {
      const tag = h('div', { class: 'compare__tag' }, `Network ${run.label}`);
      run.tag = new CSS2DObject(tag);
      const layout = run.view!.getLayout();
      run.tag.position.set(layout.boundsMin[0] - 0.8, layout.labelY, 0);
      run.view!.root.add(run.tag);
    }
  }

  private disposeRunViews(run: Run): void {
    run.tag?.element.remove();
    run.tag?.removeFromParent();
    run.tag = null;
    run.view?.dispose();
    run.view = null;
    run.boundary?.dispose();
    run.boundary = null;
  }

  private fitCamera(animate: boolean): void {
    const host = this.host;
    if (!host) return;
    const boxes = this.runs.map((r) => {
      const l = r.view!.getLayout();
      const z = r.view!.root.position.z;
      return { min: [l.boundsMin[0], l.boundsMin[1], l.boundsMin[2] + z], max: [l.boundsMax[0], l.boundsMax[1], l.boundsMax[2] + z] };
    });
    const min: [number, number, number] = [0, 1, 2].map((k) => Math.min(boxes[0].min[k], boxes[1].min[k])) as [number, number, number];
    const max: [number, number, number] = [0, 1, 2].map((k) => Math.max(boxes[0].max[k], boxes[1].max[k])) as [number, number, number];
    host.camera.setBounds(min, max);
    host.camera.reset(animate);
  }

  // -------------------------------------------------------------- editors

  private renderEditors(): void {
    replaceChildren(
      this.editors,
      this.runs.map((run, index) => {
        const c = run.contestant;
        const update = (patch: Partial<Contestant>) => {
          this.runs[index].contestant = { ...c, ...patch };
          this.rebuild();
        };
        const count = c.hidden.length;
        const width = c.hidden[0]?.size ?? 4;
        const activation = c.hidden[0]?.activation ?? 'relu';
        const setLayers = (n: number, size: number, act: ActivationId) => update({ hidden: Array.from({ length: Math.max(0, Math.min(8, n)) }, () => ({ size: Math.max(1, Math.min(32, size)), activation: act })) });
        const stepper = (value: number, onChange: (v: number) => void, labelText: string) =>
          h(
            'div',
            { class: 'stepper', title: labelText },
            h('button', { type: 'button', class: 'stepper__btn', 'aria-label': `Decrease ${labelText}`, onClick: () => onChange(value - 1) }, icon('minus', 12)),
            h('span', { class: 'stepper__value', style: 'display:inline-block;padding:3px 0' }, String(value)),
            h('button', { type: 'button', class: 'stepper__btn', 'aria-label': `Increase ${labelText}`, onClick: () => onChange(value + 1) }, icon('plus', 12)),
          );
        return h(
          'div',
          { class: 'compare__editor' },
          h('div', { class: 'compare__label' }, `Network ${run.label}`, h('span', { class: 'muted' }, ` · ${formatCount(run.model.totalParameters())} params`)),
          h('div', { class: 'field field--inline' }, h('span', { class: 'field__label' }, 'Hidden layers'), stepper(count, (n) => setLayers(n, width, activation), 'hidden layers')),
          h('div', { class: 'field field--inline' }, h('span', { class: 'field__label' }, 'Neurons each'), stepper(width, (s) => setLayers(count, s, activation), 'neurons per layer')),
          h(
            'label',
            { class: 'field field--inline' },
            h('span', { class: 'field__label' }, 'Activation'),
            select<ActivationId>({ options: ACTIVATION_CHOICES.map((id) => ({ value: id, label: ACTIVATIONS[id].label })), value: activation, ariaLabel: 'Activation', onChange: (act) => setLayers(count, width, act) }),
          ),
          h(
            'label',
            { class: 'field field--inline' },
            h('span', { class: 'field__label' }, 'Optimizer'),
            select<OptimizerId>({
              options: (Object.keys(OPTIMIZERS) as OptimizerId[]).map((id) => ({ value: id, label: OPTIMIZERS[id].label })),
              value: c.optimizer.id,
              ariaLabel: 'Optimizer',
              onChange: (id) => update({ optimizer: { ...c.optimizer, id } }),
            }),
          ),
          h(
            'label',
            { class: 'field field--inline' },
            h('span', { class: 'field__label' }, 'Learning rate'),
            select<string>({
              options: ['0.001', '0.003', '0.01', '0.02', '0.03', '0.05', '0.1', '0.2', '0.5', '1'].map((v) => ({ value: v, label: v })),
              value: String(c.optimizer.learningRate),
              ariaLabel: 'Learning rate',
              onChange: (v) => update({ optimizer: { ...c.optimizer, learningRate: Number(v) } }),
            }),
          ),
          h(
            'label',
            { class: 'field field--inline' },
            h('span', { class: 'field__label' }, 'Weight decay'),
            select<string>({
              options: ['0', '0.001', '0.01', '0.03', '0.1'].map((v) => ({ value: v, label: v })),
              value: String(c.optimizer.weightDecay),
              ariaLabel: 'Weight decay',
              onChange: (v) => update({ optimizer: { ...c.optimizer, weightDecay: Number(v) } }),
            }),
          ),
        );
      }),
    );
    replaceChildren(this.presetText, h('strong', null, this.comparison.question), h('p', { class: 'panel-text' }, this.comparison.insight));
  }

  // --------------------------------------------------------------- refresh

  private refresh(full: boolean): void {
    if (!this.host) return;
    const palette = this.palette;
    const [a, b] = this.runs;
    const xs = (run: Run) => run.trainer.history.map((p) => p.epoch);
    this.lossChart.draw([
      { label: 'A', color: palette.positive, values: a.trainer.history.map((p) => p.train.loss), xs: xs(a) },
      { label: 'B', color: palette.negative, values: b.trainer.history.map((p) => p.train.loss), xs: xs(b), dash: [5, 4] },
    ]);
    this.accuracyChart.draw([
      { label: 'A', color: palette.positive, values: a.trainer.history.map((p) => p.test?.accuracy ?? p.train.accuracy), xs: xs(a) },
      { label: 'B', color: palette.negative, values: b.trainer.history.map((p) => p.test?.accuracy ?? p.train.accuracy), xs: xs(b), dash: [5, 4] },
    ]);
    const now = performance.now();
    if (full || now - this.lastBoundary > 150) {
      this.lastBoundary = now;
      for (const run of this.runs) run.boundary?.render();
    }
    if (this.playButton) replaceChildren(this.playButton, icon(this.running ? 'pause' : 'play'), h('span', { class: 'btn__label' }, this.running ? 'Pause' : 'Train both'));
    replaceChildren(this.counters, h('span', null, 'Epoch ', h('strong', null, formatCount(a.trainer.epochCount))), h('span', null, 'Steps ', h('strong', null, formatCount(a.trainer.stepCount))));
    this.renderStats();
    if (full) this.inspector?.show(this.summaryView(), true);
  }

  private renderStats(): void {
    const rows: [string, (run: Run) => string][] = [
      ['Parameters', (r) => formatCount(r.model.totalParameters())],
      ['Hidden layers', (r) => `${r.contestant.hidden.length} × ${r.contestant.hidden[0]?.size ?? 0} ${r.contestant.hidden[0] ? ACTIVATIONS[r.contestant.hidden[0].activation].label : ''}`],
      ['Optimizer', (r) => `${OPTIMIZERS[r.contestant.optimizer.id].label} η=${r.contestant.optimizer.learningRate}${r.contestant.optimizer.weightDecay ? ` λ=${r.contestant.optimizer.weightDecay}` : ''}`],
      ['Train loss', (r) => formatValue(r.trainer.history.at(-1)!.train.loss, 4)],
      ['Train accuracy', (r) => `${formatValue(r.trainer.history.at(-1)!.train.accuracy * 100, 1)}%`],
      ['Test accuracy', (r) => (r.trainer.history.at(-1)!.test ? `${formatValue(r.trainer.history.at(-1)!.test!.accuracy * 100, 1)}%` : '—')],
      ['Gap train − test', (r) => {
        const last = r.trainer.history.at(-1)!;
        return last.test ? `${formatValue((last.train.accuracy - last.test.accuracy) * 100, 1)} pts` : '—';
      }],
      ['‖∂L/∂W‖ first / last layer', (r) => {
        const n = r.trainer.gradientNorms();
        return n.length > 1 && n[n.length - 1] > 0 ? (n[1] / n[n.length - 1]).toExponential(1) : '—';
      }],
      ['Time per 100 steps', (r) => (r.trainer.stepCount ? `${formatValue((r.time / r.trainer.stepCount) * 100, 1)} ms` : '—')],
    ];
    replaceChildren(
      this.stats,
      h(
        'table',
        { class: 'data-table compare__table' },
        h('thead', null, h('tr', null, h('th', null, ''), h('th', null, 'A'), h('th', null, 'B'))),
        h('tbody', null, rows.map(([label, value]) => h('tr', null, h('td', null, label), h('td', null, value(this.runs[0])), h('td', null, value(this.runs[1]))))),
      ),
    );
  }

  private summaryView(): InspectorView {
    const header = inspectorHeader({ kind: 'Comparison', title: this.comparison.title, subtitle: `${datasetInfo(this.datasetId).label} · same data, same mini-batches` });
    const el = h(
      'div',
      { class: 'inspector-view' },
      header,
      block(1, 'What you see', paragraph('Left: what each network predicts over the plane (A left, B right). Right: both networks in 3D (A in front, B behind). The curves overlay A (solid) and B (dashed).')),
      block(2, 'Question', paragraph(this.comparison.question), paragraph(this.comparison.insight)),
      block(
        3,
        'How to read it',
        kv('Underfitting', 'both train and test accuracy low'),
        kv('Overfitting', 'train ≫ test accuracy'),
        kv('Vanishing gradients', 'first/last gradient ratio ≪ 1'),
        paragraph('Training speed is wall-clock time in this browser: it grows with the number of parameters.'),
      ),
    );
    return { el, update: () => undefined };
  }
}
