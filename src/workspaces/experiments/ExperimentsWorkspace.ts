/**
 * ExperimentsWorkspace — the "Experiments" tab: controlled
 * interventions on a Transformer, compared with the untouched model.
 *
 * Two models: the toy Transformer, computed in the browser (every slider
 * move is recomputed exactly), and GPT-2 small, computed by the Python
 * backend (or shown from recorded presets when the backend is not running).
 * Each model keeps its own experiment, so switching back and forth loses
 * nothing.
 *
 * Kept separate from the guided tabs on purpose: here the user changes the
 * model and measures what happens; the inspectors report numbers and their
 * limits rather than explanations.
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { activeSpecs, gpt2Attention, gpt2Curve, gpt2Experiment, gpt2NeuronScan, gpt2Sweep } from '../../experiments/gpt2Engine';
import { METRICS, metricValue, support } from '../../experiments/metrics';
import { loadRecordedExperiments, toyPresets, type ExperimentPreset, type RecordedExperiment } from '../../experiments/presets';
import { newId, parseExperiment, serializeExperiment } from '../../experiments/serialization';
import { runToyExperiment, sameComponent, toyCurve, toyNeuronScan, toySweep, type ToyExperimentRun } from '../../experiments/toyEngine';
import type { AttentionPair, ComponentRef, Experiment, ExperimentModel, ExperimentResult, Intervention, InterventionSpec, MetricId, NeuronScan, ScaleCurve, SweepResult, TensorId, TokenRef } from '../../experiments/types';
import { BACKEND_HINT, fetchStatus, showToken } from '../../llm/client';
import type { LlmStatus } from '../../llm/types';
import type { CameraView } from '../../three/CameraController';
import { ExperimentView, type ComponentDisplay, type ExperimentPick } from '../../three/ExperimentView';
import { loadToyTransformer } from '../../transformer/loader';
import type { TransformerModel } from '../../transformer/model';
import { h, isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, toast, Tooltip } from '../../ui/overlays';
import { button } from '../../ui/widgets';
import { downloadText, pickFile } from '../../utils/download';
import { formatProbability, formatSigned } from '../../utils/format';
import { debounce, frameThrottle } from '../../utils/schedule';
import type { Vec3 } from '../../visualization/layout';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import type { Panel } from '../network/context';
import { ExperimentBar } from './ExperimentBar';
import { ExperimentToolbar } from './ExperimentToolbar';
import { OutputCard } from './OutputCard';
import { componentInspector } from './inspectors/componentInspector';
import { neuronInspector } from './inspectors/neuronInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { tokenInspector } from './inspectors/tokenInspector';
import { weightInspector } from './inspectors/weightInspector';
import { describeIntervention } from './describe';
import { InterventionsPanel, MeasurePanel, ModelPanel, PresetsPanel, PromptPanel } from './panels';

export type ExperimentSelection =
  | { kind: 'head'; layer: number; head: number }
  | { kind: 'mlp'; layer: number }
  | { kind: 'neuron'; layer: number; unit: number }
  | { kind: 'token'; position: number; stage: number | null }
  | { kind: 'weight'; id: string };

export interface ExperimentSettings {
  metric: MetricId;
  /** What the bars of the components wall compare against. */
  baseline: 'original' | 'experiment';
}

export type ExperimentEvents = {
  /** Model kind, model loading or backend status changed. */
  model: undefined;
  /** Text, target or interventions changed. */
  experiment: undefined;
  /** A new comparison (and possibly sweep) is shown. */
  result: undefined;
  selection: undefined;
  settings: undefined;
  busy: undefined;
};

/** An analysis computed on demand by an inspector (neuron scans, curves, attention). */
export interface Lazy<T> {
  value: T | null;
  error: string | null;
  pending: boolean;
}

const VIEWS: Record<CameraView, Vec3> = {
  // from the front and above: the bars (along z) show their length as a vertical offset
  iso: [-0.22, 0.48, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.25, 0.55],
};

/** Floor of the full-length bar, per metric, so that negligible effects stay short. */
const SCALE_FLOOR: Record<MetricId, number> = { logp: 0.25, logit: 0.25, p: 0.02, kl: 0.05 };

const DEFAULT_TEXT: Record<ExperimentModel, string> = {
  toy: "the cat didn't cross the river because it was",
  gpt2: 'When Mary and John went to the store, John gave a drink to',
};

export class ExperimentsWorkspace implements Workspace {
  readonly id = 'experiments' as const;
  readonly events = new Emitter<ExperimentEvents>();
  modelKind: ExperimentModel = 'toy';
  readonly experiments: Record<ExperimentModel, Experiment> = {
    toy: { model: 'toy', text: DEFAULT_TEXT.toy, target: null, contrast: null, interventions: [] },
    gpt2: { model: 'gpt2', text: DEFAULT_TEXT.gpt2, target: null, contrast: null, interventions: [] },
  };
  settings: ExperimentSettings = { metric: 'logp', baseline: 'original' };
  selection: ExperimentSelection | null = null;
  toy: TransformerModel | null = null;
  toyRun: ToyExperimentRun | null = null;
  result: ExperimentResult | null = null;
  sweep: SweepResult | null = null;
  /** Preset loaded last, per model (its description is shown until the experiment is edited). */
  preset: Record<ExperimentModel, ExperimentPreset | null> = { toy: null, gpt2: null };
  recorded: RecordedExperiment[] = [];
  status: LlmStatus | null = null;
  online = false;
  checked = false;
  busy = false;
  error: string | null = null;

  private readonly history: Record<ExperimentModel, Experiment[]> = { toy: [], gpt2: [] };
  private lastEdit = { key: '', time: 0 };
  private host: WorkspaceHost | null = null;
  private view: ExperimentView | null = null;
  private inspector: InspectorPanel | null = null;
  private tooltip: Tooltip | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private paletteValue: ScenePalette = scenePalette('dark', 'blue-orange');
  private statusTimer = 0;
  private request: AbortController | null = null;
  private readonly runSoon = debounce(() => void this.recompute(), 220);
  private readonly lazy = new Map<string, Lazy<unknown>>();
  private readonly sweepCache = new Map<string, SweepResult>();
  private placeholder: HTMLElement | null = null;

  get experiment(): Experiment {
    return this.experiments[this.modelKind];
  }

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  /** GPT-2 runs live (backend up and model loaded). */
  get live(): boolean {
    return this.online && !!this.status?.ready;
  }

  /** The current experiment can be edited (always for the toy model; GPT-2 needs the backend). */
  get editable(): boolean {
    return this.modelKind === 'toy' || this.live;
  }

  get canUndo(): boolean {
    return this.history[this.modelKind].length > 0;
  }

  get layers(): number {
    return this.modelKind === 'toy' ? (this.toy?.config.nLayers ?? 2) : (this.status?.config?.layers ?? 12);
  }

  get heads(): number {
    return this.modelKind === 'toy' ? (this.toy?.config.nHeads ?? 4) : (this.status?.config?.heads ?? 12);
  }

  get units(): number {
    return this.modelKind === 'toy' ? (this.toy?.config.dMlp ?? 128) : (this.status?.config?.dMlp ?? 3072);
  }

  /** Token text as displayed (GPT-2: leading space as “·”). */
  format(text: string): string {
    return this.modelKind === 'gpt2' ? showToken(text) : text;
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

    this.view = new ExperimentView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.onPick(pick, false) });
    this.inspector = new InspectorPanel(slots.right);
    this.tooltip = new Tooltip(slots.stage);
    this.panels = [new ModelPanel(this), new PromptPanel(this), new MeasurePanel(this), new InterventionsPanel(this), new PresetsPanel(this)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new ExperimentBar(this);
    slots.bottom.append(bar.el);
    const toolbar = new ExperimentToolbar(this);
    const card = new OutputCard(this);
    slots.stage.append(toolbar.el, card.el);
    this.panels.push(bar, toolbar, card);

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
    this.bag.add(() => {
      this.runSoon.cancel();
      this.request?.abort();
    });
    if (this.result) {
      this.render();
      this.fitView('iso', false);
    }
    this.showInspector();
    void this.start();
  }

  unmount(): void {
    window.clearTimeout(this.statusTimer);
    this.statusTimer = 0;
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
    this.placeholder?.remove();
    this.placeholder = null;
    this.host = null;
  }

  private async start(): Promise<void> {
    const host = this.host;
    if (!this.toy) {
      this.placeholder = h('div', { class: 'placeholder-card' }, h('h2', null, 'Loading the toy Transformer…'), h('p', { class: 'muted' }, 'Weights ≈ 300 KB.'));
      host?.slots.stage.append(this.placeholder);
      try {
        this.toy = await loadToyTransformer();
      } catch (error) {
        this.placeholder?.replaceChildren(h('h2', null, 'The model could not be loaded'), h('p', null, String(error instanceof Error ? error.message : error)), button({ label: 'Retry', onClick: () => void this.start() }));
        return;
      }
      this.placeholder?.remove();
      this.placeholder = null;
      if (!this.preset.toy && !this.experiment.interventions.length) {
        const first = toyPresets(this.toy)[0];
        this.experiments.toy = structuredClone(first.experiment);
        this.preset.toy = first;
      }
      this.events.emit('model', undefined);
    }
    if (!this.recorded.length) this.recorded = await loadRecordedExperiments();
    await this.checkStatus();
    if (this.host !== host) return;
    if (!this.result) {
      await this.recompute();
      this.fitView('iso', false);
    }
  }

  async checkStatus(): Promise<void> {
    window.clearTimeout(this.statusTimer);
    const wasLive = this.live;
    const status = await fetchStatus();
    this.status = status;
    this.online = status !== null;
    this.checked = true;
    this.events.emit('model', undefined);
    if (this.live !== wasLive && this.modelKind === 'gpt2') {
      if (this.live) toast('GPT-2 is running locally: experiments now run live.', 'success', 3500);
      void this.recompute();
    }
    if (this.host) this.statusTimer = window.setTimeout(() => void this.checkStatus(), this.live ? 15000 : 4000);
  }

  // -------------------------------------------------------------- the model

  setModel(kind: ExperimentModel): void {
    if (kind === this.modelKind) return;
    this.request?.abort();
    this.modelKind = kind;
    this.selection = null;
    this.result = null;
    this.sweep = null;
    this.toyRun = null;
    if (kind === 'gpt2' && !this.live && !this.preset.gpt2 && this.recorded.length) this.applyPreset(this.recorded[0], false);
    this.events.emit('model', undefined);
    this.events.emit('experiment', undefined);
    this.events.emit('selection', undefined);
    void this.recompute().then(() => this.fitView('iso', true));
  }

  /** Opens a recorded GPT-2 experiment by id (links from the LLM tab). */
  async openRecorded(id: string): Promise<void> {
    if (!this.recorded.length) this.recorded = await loadRecordedExperiments();
    const preset = this.recorded.find((r) => r.id === id);
    if (!preset) return;
    if (this.modelKind !== 'gpt2') this.setModel('gpt2');
    this.loadPreset(preset);
  }

  presets(): ExperimentPreset[] {
    if (this.modelKind === 'toy') return this.toy ? toyPresets(this.toy) : [];
    return this.recorded;
  }

  loadPreset(preset: ExperimentPreset): void {
    this.history[this.modelKind].push(structuredClone(this.experiment));
    this.applyPreset(preset, true);
  }

  private applyPreset(preset: ExperimentPreset, run: boolean): void {
    const kind = preset.experiment.model;
    this.experiments[kind] = structuredClone(preset.experiment);
    this.preset[kind] = preset;
    if (kind !== this.modelKind) return;
    const first = preset.experiment.interventions.find((i) => i.kind === 'head' || i.kind === 'mlp' || i.kind === 'neuron' || i.kind === 'weight');
    this.selection = !first ? null : first.kind === 'weight' ? { kind: 'weight', id: first.id } : first.kind === 'neuron' ? { kind: 'neuron', layer: first.layer, unit: first.unit } : first.kind === 'head' ? { kind: 'head', layer: first.layer, head: first.head } : first.kind === 'mlp' ? { kind: 'mlp', layer: first.layer } : null;
    this.events.emit('experiment', undefined);
    this.events.emit('selection', undefined);
    if (run) void this.recompute();
  }

  /** Recorded GPT-2 preset matching the current experiment exactly (offline mode). */
  private recordedMatch(): RecordedExperiment | null {
    const e = this.experiments.gpt2;
    const key = (x: Experiment) => JSON.stringify([x.text, x.target?.id ?? null, x.contrast?.id ?? null, activeSpecs(x)]);
    const k = key(e);
    return this.recorded.find((r) => key(r.experiment) === k) ?? null;
  }

  // ------------------------------------------------------------- computing

  /** Recomputes the comparison and the sweep of the current experiment. */
  async recompute(): Promise<void> {
    this.runSoon.cancel();
    this.request?.abort();
    this.lazy.clear();
    const kind = this.modelKind;
    const experiment = this.experiment;
    if (kind === 'toy') {
      const model = this.toy;
      if (!model) return;
      this.validInterventions(experiment);
      const run = runToyExperiment(model, experiment);
      this.toyRun = run;
      this.result = run.result;
      this.sweep = toySweep(model, experiment, run.result, this.settings.baseline);
      this.error = null;
      this.afterResult();
      return;
    }
    this.toyRun = null;
    if (!this.live) {
      const match = this.recordedMatch();
      this.result = match ? { ...match.result, model: 'gpt2', source: 'recorded' } : null;
      this.sweep = match && this.settings.baseline === 'original' ? { ...match.sweep, baseline: 'original', source: 'recorded' } : null;
      this.error = match ? null : this.checked ? 'offline' : null;
      this.afterResult();
      return;
    }
    const controller = new AbortController();
    this.request = controller;
    this.setBusy(true);
    try {
      const result = await gpt2Experiment(experiment, controller.signal);
      if (controller.signal.aborted) return;
      this.result = result;
      // Keep the measured tokens explicit, so that later edits keep measuring the same thing.
      if (!experiment.target) experiment.target = result.target;
      this.error = null;
      const key = this.sweepKey(experiment, result);
      const cached = this.sweepCache.get(key);
      this.sweep = cached ?? null;
      this.afterResult();
      if (!cached) {
        const sweep = await gpt2Sweep(experiment, result, this.settings.baseline, controller.signal);
        if (controller.signal.aborted) return;
        this.sweepCache.set(key, sweep);
        if (this.sweepCache.size > 24) this.sweepCache.delete(this.sweepCache.keys().next().value!);
        this.sweep = sweep;
        this.afterResult();
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      this.error = error instanceof Error ? error.message : String(error);
      toast(`Experiment failed: ${this.error}`, 'error', 5000);
      this.events.emit('result', undefined);
    } finally {
      if (this.request === controller) {
        this.request = null;
        this.setBusy(false);
      }
    }
  }

  private sweepKey(e: Experiment, r: ExperimentResult): string {
    return JSON.stringify([e.text, r.target.id, r.contrast?.id ?? null, this.settings.baseline, this.settings.baseline === 'experiment' ? activeSpecs(e) : []]);
  }

  /** Drops interventions that do not fit the model (e.g. after loading a file made for the other model). */
  private validInterventions(e: Experiment): void {
    const before = e.interventions.length;
    const L = this.layers;
    const H = this.heads;
    const U = this.units;
    e.interventions = e.interventions.filter((i) => {
      switch (i.kind) {
        case 'head':
          return i.layer < L && i.head < H;
        case 'mlp':
          return i.layer < L;
        case 'neuron':
          return i.layer < L && i.unit < U;
        case 'weight':
          return e.model === 'toy' && i.layer < L && i.head < H;
        default:
          return true;
      }
    });
    if (e.interventions.length !== before) {
      toast(`${before - e.interventions.length} intervention(s) did not fit this model and were dropped.`, 'warning', 4500);
      this.events.emit('experiment', undefined);
    }
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    this.events.emit('busy', undefined);
  }

  private afterResult(): void {
    this.selection = this.validSelection(this.selection);
    this.render();
    this.showInspector(true);
    this.events.emit('result', undefined);
  }

  private validSelection(s: ExperimentSelection | null): ExperimentSelection | null {
    if (!s) return s;
    if ((s.kind === 'head' || s.kind === 'mlp' || s.kind === 'neuron') && s.layer >= this.layers) return null;
    if (s.kind === 'head' && s.head >= this.heads) return null;
    if (s.kind === 'token' && this.result && s.position >= this.result.tokens.length) return null;
    if (s.kind === 'weight' && !this.experiment.interventions.some((i) => i.id === s.id)) return null;
    return s;
  }

  /** Scene from the current result, sweep and settings. */
  private render(): void {
    const view = this.view;
    if (!view) return;
    const L = this.layers;
    const H = this.heads;
    const metric = this.metric;
    const values: (number | null)[] = [];
    const sweep = this.sweep;
    const display = (c: ComponentRef & { kind: 'head' | 'mlp' }): ComponentDisplay => {
      const effect = !sweep ? null : c.kind === 'head' ? sweep.heads[c.layer]?.[c.head] : sweep.mlps[c.layer];
      const value = effect ? support(effect, metric) : null;
      values.push(value);
      const items = this.experiment.interventions.filter((i) => i.enabled && sameComponent(i, c));
      const scale = items.reduce((s, i) => s * (i.kind === 'head' || i.kind === 'mlp' ? i.scale : 1), 1);
      const state = !items.length || scale === 1 ? 'normal' : scale === 0 ? 'off' : 'scaled';
      return { value, state, badge: state === 'off' ? 'off' : state === 'scaled' ? `×${+scale.toFixed(2)}` : null };
    };
    const heads = Array.from({ length: L }, (_, layer) => Array.from({ length: H }, (_, head) => display({ kind: 'head', layer, head })));
    const mlps = Array.from({ length: L }, (_, layer) => display({ kind: 'mlp', layer }));
    const max = Math.max(SCALE_FLOOR[metric], ...values.map((v) => Math.abs(v ?? 0)));
    view.setComponents(L, H, { heads, mlps }, { signed: METRICS[metric].signed, scale: max });
    if (this.result) view.setResult(this.result, (t) => this.format(t));
    this.updateHighlight();
    this.resetCameraBounds();
  }

  /** The metric shown (logit difference falls back to log p without a contrast token). */
  get metric(): MetricId {
    const m = this.settings.metric;
    return m === 'logit' && !this.result?.contrast ? 'logp' : m;
  }

  // -------------------------------------------------------------- editing

  /** Applies a change to the current experiment; consecutive changes of one control are undone together. */
  edit(change: (e: Experiment) => void, o: { key?: string; immediate?: boolean } = {}): void {
    if (!this.editable) {
      toast(`Recorded presets are read-only: to change this experiment, ${BACKEND_HINT}.`, 'info', 4500);
      return;
    }
    const now = performance.now();
    const key = o.key ?? Math.random().toString();
    if (key !== this.lastEdit.key || now - this.lastEdit.time > 1200) this.history[this.modelKind].push(structuredClone(this.experiment));
    if (this.history[this.modelKind].length > 100) this.history[this.modelKind].shift();
    this.lastEdit = { key, time: now };
    change(this.experiment);
    this.preset[this.modelKind] = this.preset[this.modelKind] && JSON.stringify(activeSpecs(this.experiment)) === JSON.stringify(activeSpecs(this.preset[this.modelKind]!.experiment)) ? this.preset[this.modelKind] : null;
    this.events.emit('experiment', undefined);
    if (this.modelKind === 'toy' || o.immediate) void this.recompute();
    else this.runSoon();
  }

  undo(): void {
    const previous = this.history[this.modelKind].pop();
    if (!previous) return;
    this.experiments[this.modelKind] = previous;
    this.lastEdit = { key: '', time: 0 };
    this.events.emit('experiment', undefined);
    void this.recompute();
  }

  setText(text: string): void {
    const t = text.trim();
    if (!t || t === this.experiment.text) return;
    this.edit((e) => {
      const positional = e.interventions.filter((i) => i.kind === 'remove' || i.kind === 'embedding').length;
      e.text = t;
      e.interventions = e.interventions.filter((i) => i.kind !== 'remove' && i.kind !== 'embedding');
      // A new text gets its own most likely next token as the target.
      e.target = e.contrast = null;
      if (positional) toast('Token edits were removed: they referred to positions of the previous text.', 'info', 4000);
    }, { immediate: true });
  }

  setTarget(target: TokenRef | null): void {
    this.edit((e) => {
      e.target = target;
      if (target && e.contrast?.id === target.id) e.contrast = null;
    }, { immediate: true });
  }

  setContrast(contrast: TokenRef | null): void {
    this.edit((e) => (e.contrast = contrast), { immediate: true });
    if (contrast && this.settings.metric === 'logp') this.update({ metric: 'logit' });
  }

  update(patch: Partial<ExperimentSettings>): void {
    const before = this.settings;
    this.settings = { ...before, ...patch };
    this.events.emit('settings', undefined);
    if (patch.baseline && patch.baseline !== before.baseline) void this.recomputeSweep();
    else {
      this.render();
      this.showInspector(true);
    }
  }

  private async recomputeSweep(): Promise<void> {
    if (this.modelKind === 'toy' && this.toy && this.result) {
      this.sweep = toySweep(this.toy, this.experiment, this.result, this.settings.baseline);
      this.lazy.clear();
      this.afterResult();
      return;
    }
    await this.recompute();
  }

  // ---------------------------------------------------- interventions API

  addIntervention(spec: InterventionSpec, select = true): string {
    const id = newId();
    this.edit((e) => e.interventions.push({ ...spec, id, enabled: true } as Intervention), { immediate: true });
    if (select && spec.kind === 'weight') this.select({ kind: 'weight', id });
    return id;
  }

  updateIntervention(id: string, patch: Partial<InterventionSpec>): void {
    this.edit((e) => {
      const item = e.interventions.find((i) => i.id === id);
      if (item) Object.assign(item, patch);
    }, { key: `update:${id}` });
  }

  toggleIntervention(id: string): void {
    this.edit((e) => {
      const item = e.interventions.find((i) => i.id === id);
      if (item) item.enabled = !item.enabled;
    }, { immediate: true });
  }

  removeIntervention(id: string): void {
    this.edit((e) => (e.interventions = e.interventions.filter((i) => i.id !== id)), { immediate: true });
  }

  clearInterventions(): void {
    if (!this.experiment.interventions.length) return;
    this.edit((e) => (e.interventions = []), { immediate: true });
  }

  /** The enabled intervention acting on a component, if any. */
  interventionFor(c: ComponentRef): Intervention | undefined {
    return this.experiment.interventions.find((i) => i.enabled && sameComponent(i, c));
  }

  /** Current scale of a head or MLP layer (1 = untouched). */
  componentScale(c: ComponentRef & { kind: 'head' | 'mlp' }): number {
    const item = this.interventionFor(c);
    return item && (item.kind === 'head' || item.kind === 'mlp') ? item.scale : 1;
  }

  /** Sets the scale of a head or MLP layer (creating, updating or — at 1 — removing its intervention). */
  setComponentScale(c: ComponentRef & { kind: 'head' | 'mlp' }, scale: number, key?: string): void {
    this.edit((e) => {
      const index = e.interventions.findIndex((i) => sameComponent(i, c));
      if (index < 0) {
        if (scale !== 1) e.interventions.push({ ...c, scale, id: newId(), enabled: true } as Intervention);
      } else if (scale === 1 && !key) e.interventions.splice(index, 1);
      else Object.assign(e.interventions[index], { scale, enabled: true });
    }, { key });
  }

  toggleComponent(c: ComponentRef & { kind: 'head' | 'mlp' }): void {
    this.setComponentScale(c, this.componentScale(c) === 0 ? 1 : 0);
  }

  setNeuron(layer: number, unit: number, mode: 'scale' | 'set' | null, value: number, key?: string): void {
    const c: ComponentRef = { kind: 'neuron', layer, unit };
    this.edit((e) => {
      const index = e.interventions.findIndex((i) => sameComponent(i, c));
      if (mode === null) {
        if (index >= 0) e.interventions.splice(index, 1);
        return;
      }
      if (index < 0) e.interventions.push({ kind: 'neuron', layer, unit, mode, value, id: newId(), enabled: true });
      else Object.assign(e.interventions[index], { mode, value, enabled: true });
    }, { key });
  }

  tokenEdits(position: number): { removed: Intervention | undefined; embedding: Extract<Intervention, { kind: 'embedding' }> | undefined } {
    const items = this.experiment.interventions.filter((i) => i.enabled);
    return {
      removed: items.find((i) => i.kind === 'remove' && i.position === position),
      embedding: items.find((i): i is Extract<Intervention, { kind: 'embedding' }> => i.kind === 'embedding' && i.position === position),
    };
  }

  toggleRemoved(position: number): void {
    if (this.modelKind === 'toy' && position === 0) {
      toast('<s> starts every sentence the toy model was trained on: it cannot be removed.', 'info');
      return;
    }
    const existing = this.tokenEdits(position).removed;
    if (existing) this.removeIntervention(existing.id);
    else {
      if ((this.result?.tokensAfter.length ?? 2) <= 1) return;
      this.addIntervention({ kind: 'remove', position }, false);
    }
  }

  setReplacement(position: number, token: TokenRef | null, mix = 1, key?: string): void {
    this.edit((e) => {
      const index = e.interventions.findIndex((i) => i.kind === 'embedding' && i.position === position);
      if (!token) {
        if (index >= 0) e.interventions.splice(index, 1);
        return;
      }
      if (index < 0) e.interventions.push({ kind: 'embedding', position, token, mix, id: newId(), enabled: true });
      else Object.assign(e.interventions[index], { token, mix, enabled: true });
    }, { key });
  }

  addWeightEdit(tensor: TensorId, layer: number, head: number): void {
    this.addIntervention({ kind: 'weight', tensor, layer, head, scale: 1, cell: null });
  }

  // ------------------------------------------------------------ save / load

  save(): void {
    const e = this.experiment;
    const slug = e.text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'experiment';
    downloadText(`experiment-${e.model}-${slug}.json`, serializeExperiment(e));
  }

  async load(): Promise<void> {
    const file = await pickFile('application/json,.json');
    if (!file) return;
    try {
      const experiment = parseExperiment(JSON.parse(await file.text()));
      if (experiment.model === 'gpt2' && !this.live) {
        toast(`This experiment needs GPT-2: ${BACKEND_HINT}, then load it again.`, 'warning', 5000);
        return;
      }
      this.history[experiment.model].push(structuredClone(this.experiments[experiment.model]));
      this.experiments[experiment.model] = experiment;
      this.preset[experiment.model] = null;
      if (experiment.model !== this.modelKind) this.setModel(experiment.model);
      else {
        this.events.emit('experiment', undefined);
        await this.recompute();
      }
      toast(`Loaded ${file.name}`, 'success');
    } catch (error) {
      toast(`Could not load the experiment: ${error instanceof Error ? error.message : String(error)}`, 'error', 6000);
    }
  }

  // --------------------------------------------------- on-demand analyses

  /**
   * Cached on-demand analysis. `defer` (ms) postpones a heavy synchronous
   * computation, so that the inspector shows "Computing…" first and slider
   * drags that invalidate it in the meantime never pay for it.
   */
  private lazyValue<T>(key: string, compute: () => T | Promise<T>, defer = 0): Lazy<T> {
    const full = `${this.modelKind}|${key}`;
    let entry = this.lazy.get(full) as Lazy<T> | undefined;
    if (entry) return entry;
    entry = { value: null, error: null, pending: true };
    this.lazy.set(full, entry);
    const done = entry;
    const finish = () => {
      if (this.lazy.get(full) === done) this.showInspector(true);
    };
    if (defer > 0) {
      window.setTimeout(() => {
        if (this.lazy.get(full) !== done) return; // superseded by a newer experiment
        try {
          done.value = (compute() as T);
        } catch (error) {
          done.error = error instanceof Error ? error.message : String(error);
        }
        done.pending = false;
        finish();
      }, defer);
      return done;
    }
    try {
      const value = compute();
      if (value instanceof Promise) {
        value.then(
          (v) => {
            done.value = v;
            done.pending = false;
            finish();
          },
          (error: unknown) => {
            done.error = error instanceof Error ? error.message : String(error);
            done.pending = false;
            finish();
          },
        );
      } else {
        done.value = value;
        done.pending = false;
      }
    } catch (error) {
      done.error = error instanceof Error ? error.message : String(error);
      done.pending = false;
    }
    return done;
  }

  /** Neuron scan of one MLP layer (cached until the experiment changes); GPT-2 also measures `extra`. */
  neuronScan(layer: number, extra?: number): Lazy<NeuronScan> {
    const { result } = this;
    const baseline = this.settings.baseline;
    const toy = this.modelKind === 'toy';
    return this.lazyValue(`neurons|${layer}|${baseline}|${toy ? '' : (extra ?? '')}`, () => {
      if (!result) throw new Error('No result yet');
      if (toy) return toyNeuronScan(this.toy!, this.experiment, result, baseline, layer);
      if (!this.live) throw new Error('Neuron scans need the GPT-2 backend.');
      return gpt2NeuronScan(this.experiment, result, baseline, layer, extra === undefined ? [] : [extra]);
    }, toy ? 120 : 0);
  }

  static readonly CURVE_SCALES = [-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3];

  curve(component: ComponentRef): Lazy<ScaleCurve> {
    const { result } = this;
    return this.lazyValue(`curve|${JSON.stringify(component)}`, () => {
      if (!result) throw new Error('No result yet');
      if (this.modelKind === 'toy') return toyCurve(this.toy!, this.experiment, result, component, ExperimentsWorkspace.CURVE_SCALES);
      if (!this.live) throw new Error('Curves need the GPT-2 backend.');
      return gpt2Curve(this.experiment, result, component, ExperimentsWorkspace.CURVE_SCALES);
    });
  }

  /** Attention of one head before / after the interventions (GPT-2 only; the toy model reads its traces). */
  attention(layer: number, head: number): Lazy<AttentionPair> {
    return this.lazyValue(`attention|${layer}|${head}`, () => {
      if (!this.live) throw new Error('Attention patterns need the GPT-2 backend.');
      return gpt2Attention(this.experiment, layer, head);
    });
  }

  // ------------------------------------------------------------- selection

  select(selection: ExperimentSelection | null): void {
    this.selection = this.validSelection(selection);
    this.updateHighlight();
    this.showInspector();
    this.events.emit('selection', undefined);
  }

  private onPick(pick: ExperimentPick | null, toggle: boolean): void {
    if (!pick) return this.select(null);
    switch (pick.kind) {
      case 'head':
      case 'mlp':
        if (toggle && this.editable) this.toggleComponent(pick);
        return this.select(pick);
      case 'cell':
        return this.select({ kind: 'token', position: pick.position, stage: pick.stage });
      case 'token':
        return this.select({ kind: 'token', position: pick.position, stage: null });
    }
  }

  private updateHighlight(): void {
    const s = this.selection;
    let component: { kind: 'head'; layer: number; head: number } | { kind: 'mlp'; layer: number } | null = null;
    if (s?.kind === 'head') component = { kind: 'head', layer: s.layer, head: s.head };
    else if (s?.kind === 'mlp' || s?.kind === 'neuron') component = { kind: 'mlp', layer: s.layer };
    this.view?.setHighlight({ component, position: s?.kind === 'token' ? s.position : null, stage: s?.kind === 'token' ? s.stage : null });
  }

  showInspector(keepScroll = false): void {
    if (!this.inspector) return;
    this.inspector.show(this.inspect(), keepScroll);
  }

  private inspect(): InspectorView {
    const s = this.selection;
    if (!s || !this.result) return overviewInspector(this);
    switch (s.kind) {
      case 'head':
      case 'mlp':
        return componentInspector(this, s);
      case 'neuron':
        return neuronInspector(this, s.layer, s.unit);
      case 'token':
        return tokenInspector(this, s.position, s.stage);
      case 'weight': {
        const item = this.experiment.interventions.find((i) => i.id === s.id);
        return item?.kind === 'weight' && this.toy ? weightInspector(this, item) : overviewInspector(this);
      }
    }
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
    if (s.kind === 'head') this.host.camera.focus(view.componentCenter(s), 2.4);
    else if (s.kind === 'mlp' || s.kind === 'neuron') this.host.camera.focus(view.componentCenter({ kind: 'mlp', layer: s.layer }), 2.4);
    else if (s.kind === 'token') {
      const [x, y, z] = view.cellCenter(s.stage ?? Math.floor((this.result?.delta.length ?? 1) / 2), s.position);
      this.host.camera.focus([x, y, z], s.stage === null ? 4 : 2);
    }
  }

  // ----------------------------------------------------------------- input

  private describe(pick: ExperimentPick): HTMLElement | null {
    const r = this.result;
    const metric = this.metric;
    switch (pick.kind) {
      case 'head':
      case 'mlp': {
        const effect = !this.sweep ? null : pick.kind === 'head' ? this.sweep.heads[pick.layer]?.[pick.head] : this.sweep.mlps[pick.layer];
        const scale = this.componentScale(pick);
        const name = pick.kind === 'head' ? `Layer ${pick.layer + 1} · head ${pick.head + 1}` : `MLP of layer ${pick.layer + 1}`;
        const value = effect ? support(effect, metric) : null;
        const delta = effect ? metricValue(effect, metric) : null;
        const reading = value === null || !METRICS[metric].signed ? 'how much the whole prediction changes' : value > 0.05 ? 'the target relies on it' : value < -0.05 ? 'it works against the target' : 'little effect';
        return h(
          'div',
          null,
          h('strong', null, name),
          scale !== 1 ? h('span', { class: 'tooltip__meta' }, scale === 0 ? ' · off' : ` · ×${scale}`) : null,
          effect && r && delta !== null
            ? h('div', null, `Switched off alone: ${METRICS[metric].symbol} ${metric === 'kl' ? `${delta.toFixed(3)} bits` : formatSigned(delta, 3)}`, h('div', { class: 'tooltip__meta' }, reading))
            : h('div', { class: 'tooltip__meta' }, this.busy ? 'measuring…' : 'not measured'),
          h('div', { class: 'tooltip__meta' }, 'click: inspect · shift-click: switch off / on'),
        );
      }
      case 'cell': {
        if (!r) return null;
        const v = r.delta[pick.stage]?.[pick.position];
        const stage = pick.stage === 0 ? 'embeddings' : `after the ${(pick.stage - 1) % 2 === 0 ? 'attention' : 'MLP'} of layer ${Math.floor((pick.stage - 1) / 2) + 1}`;
        return h(
          'div',
          null,
          h('strong', null, `“${this.format(r.tokens[pick.position].text)}”`),
          ` · ${stage}`,
          h('div', null, v === null || v === undefined ? 'removed token' : `changed by ${(v * 100).toFixed(v < 0.01 ? 2 : 1)}% (‖Δx‖ / ‖x‖)`),
        );
      }
      default:
        return null;
    }
  }

  private bindPointer(): void {
    const { viewport } = this.host!;
    const canvas = viewport.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    let last: { clientX: number; clientY: number } | null = null;
    const hoverAt = frameThrottle(() => {
      if (!last || down) return;
      const { x, y } = viewport.toCanvas(last.clientX, last.clientY);
      const hit = this.view?.pick(x, y) ?? null;
      this.view?.setHover(hit);
      canvas.style.cursor = hit ? 'pointer' : '';
      const content = hit ? this.describe(hit) : null;
      if (content) this.tooltip?.show(x, y, content);
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
        this.onPick(this.view?.pick(x, y) ?? null, event.shiftKey);
      }
      hoverAt();
    });
    this.bag.listen(canvas, 'pointercancel', () => (down = null));
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
      if (e.defaultPrevented || e.altKey || isModalOpen() || isTypingTarget(e.target)) return;
      if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
        this.undo();
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey) return;
      if (e.key.startsWith('Arrow') && e.target instanceof HTMLInputElement && e.target.type === 'range') return;
      if (this.handleKey(e)) e.preventDefault();
    });
  }

  private handleKey(e: KeyboardEvent): boolean {
    const s = this.selection;
    const L = this.layers;
    const H = this.heads;
    switch (e.key) {
      case 'Escape':
        if (!s) return false;
        this.select(null);
        return true;
      case 'x':
      case 'X':
        if (s?.kind === 'head' || s?.kind === 'mlp') this.toggleComponent(s);
        else if (s?.kind === 'neuron') this.setNeuron(s.layer, s.unit, this.interventionFor({ kind: 'neuron', layer: s.layer, unit: s.unit }) ? null : 'scale', 0);
        else if (s?.kind === 'token') this.toggleRemoved(s.position);
        else return false;
        return true;
      case 'ArrowLeft':
      case 'ArrowRight': {
        const d = e.key === 'ArrowLeft' ? -1 : 1;
        if (s?.kind === 'head') {
          const next = s.head + d;
          this.select(next >= H ? { kind: 'mlp', layer: s.layer } : { ...s, head: Math.max(0, next) });
        } else if (s?.kind === 'mlp' && d < 0) this.select({ kind: 'head', layer: s.layer, head: H - 1 });
        else if (s?.kind === 'token' && this.result) this.select({ ...s, position: Math.max(0, Math.min(this.result.tokens.length - 1, s.position + d)) });
        else return false;
        return true;
      }
      case 'ArrowUp':
      case 'ArrowDown': {
        const d = e.key === 'ArrowUp' ? 1 : -1;
        if (s?.kind === 'head' || s?.kind === 'mlp') this.select({ ...s, layer: Math.max(0, Math.min(L - 1, s.layer + d)) });
        else if (s?.kind === 'token' && this.result) this.select({ ...s, stage: Math.max(0, Math.min(this.result.delta.length - 1, (s.stage ?? 0) + d)) });
        else return false;
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

  /** One-line summary of the measured change (bottom bar, overview). */
  summary(): string {
    const r = this.result;
    if (!r) return '';
    const t = `p(${this.format(r.target.text)})`;
    const parts = [`${t} ${formatProbability(r.before.pTarget)} → ${formatProbability(r.after.pTarget)}`, `Δ log p ${formatSigned(r.effect.dlogp, 2)}`];
    if (r.effect.dlogit !== null && r.contrast) parts.push(`Δ logit(${this.format(r.target.text)} − ${this.format(r.contrast.text)}) ${formatSigned(r.effect.dlogit, 2)}`);
    parts.push(`KL ${r.effect.kl.toFixed(r.effect.kl < 0.01 ? 4 : 2)} bits`);
    return parts.join(' · ');
  }

  describeIntervention(i: Intervention): string {
    return describeIntervention(i, this.result, (t) => this.format(t), this.toy);
  }
}
