/**
 * LlmWorkspace — the "LLM" tab: GPT-2 small, a real language model, run by
 * the local Python backend (python/server) or, when the backend is not
 * running, read from recorded analyses (public/llm-demo).
 *
 * 3-D views: the logit-lens tower (tokens × layers) with the attention arcs of
 * one head, and the token journey (each token's vector, layer after layer,
 * projected to 3-D). A bottom slider moves through the layers.
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { analyze, fetchStatus, generate, loadDemo, loadDemoIndex, ONLINE_DEMO, showToken } from '../../llm/client';
import type { DemoEntry, GenerateResult, LlmAnalysis, LlmStatus } from '../../llm/types';
import type { CameraView } from '../../three/CameraController';
import { EmbeddingView, type EmbeddingPick, type EmbeddingPoint } from '../../three/EmbeddingView';
import { LlmView, type LlmPick } from '../../three/LlmView';
import { DEFAULT_SAMPLING, type SamplingOptions } from '../../transformer/sampling';
import { h, isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, toast, Tooltip } from '../../ui/overlays';
import { formatPercent, formatProbability } from '../../utils/format';
import { frameThrottle } from '../../utils/schedule';
import type { Vec3 } from '../../visualization/layout';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import type { Panel } from '../network/context';
import { ExperimentsWorkspace } from '../experiments/ExperimentsWorkspace';
import { cellInspector } from './inspectors/cellInspector';
import { headInspector } from './inspectors/headInspector';
import { layerInspector } from './inspectors/layerInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { predictionInspector } from './inspectors/predictionInspector';
import { tokenInspector } from './inspectors/tokenInspector';
import { LayerBar } from './LayerBar';
import { BackendPanel, LlmDisplayPanel, LlmExamplesPanel, PromptPanel } from './panels';
import { LlmToolbar } from './LlmToolbar';

export type LlmSelection =
  | { kind: 'cell'; layer: number; position: number }
  | { kind: 'token'; position: number }
  | { kind: 'head'; layer: number; head: number }
  | { kind: 'layer'; layer: number }
  | { kind: 'prediction' };

export interface LlmSettings {
  view: 'tower' | 'journey';
  /** Head whose arcs are drawn (the block is the one that produced the current layer). */
  arcHead: number;
  arcsOn: boolean;
  arcQueries: 'focus' | 'all';
  showWords: boolean;
  sampling: SamplingOptions;
  maxNewTokens: number;
}

export type LlmEvents = {
  status: undefined;
  analysis: undefined;
  selection: undefined;
  settings: undefined;
  layer: undefined;
  busy: undefined;
};

const VIEWS: Record<CameraView, Vec3> = {
  iso: [-0.35, 0.35, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.3, 0.4],
};

export const DEFAULT_PROMPT = 'When Mary and John went to the store, John gave a drink to';

export class LlmWorkspace implements Workspace {
  readonly id = 'llm' as const;
  readonly events = new Emitter<LlmEvents>();
  status: LlmStatus | null = null;
  /** True when the backend answered the last status request. */
  online = false;
  checked = false;
  demos: DemoEntry[] = [];
  demo: DemoEntry | null = null;
  analysis: LlmAnalysis | null = null;
  text = DEFAULT_PROMPT;
  busy = false;
  error: string | null = null;
  selection: LlmSelection | null = null;
  focus = 0;
  /** Residual stage shown by the slider: 0 = embeddings … 12 = after the last block. */
  layer = 0;
  generation: GenerateResult | null = null;
  seed = 1;
  settings: LlmSettings = {
    view: 'tower',
    arcHead: 0,
    arcsOn: true,
    arcQueries: 'focus',
    showWords: false,
    sampling: { ...DEFAULT_SAMPLING },
    maxNewTokens: 8,
  };
  private readonly history: string[] = [];

  private host: WorkspaceHost | null = null;
  private view: LlmView | null = null;
  private journey: EmbeddingView | null = null;
  private inspector: InspectorPanel | null = null;
  private tooltip: Tooltip | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private paletteValue: ScenePalette = scenePalette('dark', 'blue-orange');
  private statusTimer = 0;
  private playTimer = 0;

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  get canUndo(): boolean {
    return this.history.length > 0;
  }

  get stages(): number {
    return this.analysis?.lens.length ?? 13;
  }

  get playing(): boolean {
    return this.playTimer !== 0;
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

    this.view = new LlmView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.onPick(pick) });
    this.journey = new EmbeddingView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.onJourneyPick(pick) });
    this.inspector = new InspectorPanel(slots.right);
    this.tooltip = new Tooltip(slots.stage);
    this.panels = [new BackendPanel(this), new PromptPanel(this), new LlmExamplesPanel(this), new LlmDisplayPanel(this)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new LayerBar(this);
    slots.bottom.append(bar.el);
    const toolbar = new LlmToolbar(this);
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
          this.journey?.setPalette(this.paletteValue);
          this.refreshScene();
          this.showInspector(true);
        },
      ),
    );
    this.bindPointer();
    this.bindKeyboard();

    if (this.analysis) {
      this.refreshScene();
      this.applyView(false);
    }
    this.showInspector();
    void this.start();
  }

  unmount(): void {
    this.stopPlaying();
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
    this.journey?.dispose();
    this.journey = null;
    this.host = null;
  }

  /** First mount: find out whether the backend runs; show a live analysis or a recorded one. */
  private async start(): Promise<void> {
    if (!this.demos.length) this.demos = await loadDemoIndex();
    await this.checkStatus();
    if (this.analysis || !this.host) return;
    if (this.online && this.status?.ready) await this.run(this.text);
    else if (this.demos.length) await this.openDemo(this.demos.find((d) => d.text === this.text) ?? this.demos[0]);
  }

  async checkStatus(): Promise<void> {
    window.clearTimeout(this.statusTimer);
    const status = await fetchStatus();
    const wasReady = this.online && !!this.status?.ready;
    this.status = status;
    this.online = status !== null;
    this.checked = true;
    this.events.emit('status', undefined);
    const ready = this.online && !!status?.ready;
    if (ready && !wasReady && this.analysis?.source === 'demo') toast('GPT-2 is running locally: prompts are now analyzed live.', 'success', 3500);
    // Keep polling while mounted: the user may start (or stop) the backend at any time.
    if (this.host) this.statusTimer = window.setTimeout(() => void this.checkStatus(), ready ? 15000 : 4000);
  }

  get live(): boolean {
    return this.online && !!this.status?.ready;
  }

  // -------------------------------------------------------------- analyses

  /** Analyzes `text` with the backend. */
  async run(text: string, keepHistory = false): Promise<void> {
    if (!this.live) {
      toast(
        ONLINE_DEMO
          ? 'This online demo has no GPT-2 backend: pick a recorded example, or run the app on your computer (see the panel on the left).'
          : 'The GPT-2 backend is not running: start it with “npm run llm” (see the panel on the left), or pick a recorded example.',
        'info',
        5000,
      );
      return;
    }
    if (!keepHistory) this.history.length = 0;
    this.text = text;
    this.setBusy(true);
    try {
      const analysis = await analyze({ text });
      this.demo = null;
      this.setAnalysis(analysis);
      this.error = null;
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
      toast(`Analysis failed: ${this.error}`, 'error', 5000);
    } finally {
      this.setBusy(false);
    }
  }

  async openDemo(entry: DemoEntry): Promise<void> {
    if (this.live) {
      await this.run(entry.text);
      return;
    }
    this.setBusy(true);
    try {
      const analysis = await loadDemo(entry);
      this.demo = entry;
      this.text = entry.text;
      this.history.length = 0;
      this.setAnalysis(analysis);
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error), 'error');
    } finally {
      this.setBusy(false);
    }
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    this.events.emit('busy', undefined);
  }

  private setAnalysis(analysis: LlmAnalysis): void {
    const first = !this.analysis;
    this.analysis = analysis;
    this.generation = null;
    this.focus = analysis.T - 1;
    this.layer = analysis.lens.length - 1;
    this.selection = this.selection?.kind === 'head' || this.selection?.kind === 'layer' ? this.selection : null;
    this.refreshScene();
    if (first) this.applyView(false);
    else this.resetCameraBounds();
    this.showInspector();
    this.events.emit('analysis', undefined);
    this.events.emit('layer', undefined);
    this.events.emit('selection', undefined);
  }

  async generate(mode: 'sample' | 'greedy'): Promise<void> {
    if (!this.live || !this.analysis) return;
    const s = this.settings.sampling;
    this.setBusy(true);
    try {
      const result = await generate({
        text: this.text,
        temperature: mode === 'greedy' ? 0 : s.temperature,
        topK: s.topK,
        topP: s.topP,
        maxNewTokens: this.settings.maxNewTokens,
        seed: this.seed++,
      });
      this.history.push(this.text);
      const previous = this.text;
      await this.run(result.text, true);
      this.generation = { ...result, text: result.text.slice(previous.length) };
      this.showInspector(true);
    } catch (error) {
      toast(`Generation failed: ${error instanceof Error ? error.message : String(error)}`, 'error', 5000);
      this.setBusy(false);
    }
  }

  async undo(): Promise<void> {
    const previous = this.history.pop();
    if (previous !== undefined) await this.run(previous, true);
  }

  update(patch: Partial<LlmSettings>): void {
    const before = this.settings;
    this.settings = { ...before, ...patch };
    if (patch.view && patch.view !== before.view) this.applyView(true);
    this.refreshScene();
    this.events.emit('settings', undefined);
  }

  setSampling(patch: Partial<SamplingOptions>): void {
    this.settings = { ...this.settings, sampling: { ...this.settings.sampling, ...patch } };
  }

  setLayer(layer: number): void {
    const l = Math.max(0, Math.min(this.stages - 1, layer));
    if (l === this.layer) return;
    this.layer = l;
    this.refreshArcs();
    this.updateHighlight();
    this.journey?.setStage(l);
    this.events.emit('layer', undefined);
  }

  togglePlay(): void {
    if (this.playing) {
      this.stopPlaying();
      return;
    }
    if (this.layer >= this.stages - 1) this.setLayer(0);
    const step = () => {
      if (this.layer >= this.stages - 1) {
        this.stopPlaying();
        return;
      }
      this.setLayer(this.layer + 1);
      this.playTimer = window.setTimeout(step, 750);
    };
    this.playTimer = window.setTimeout(step, 500);
    this.events.emit('layer', undefined);
  }

  private stopPlaying(): void {
    window.clearTimeout(this.playTimer);
    this.playTimer = 0;
    this.events.emit('layer', undefined);
  }

  setFocus(position: number): void {
    if (!this.analysis) return;
    this.focus = Math.max(0, Math.min(this.analysis.T - 1, position));
    this.refreshArcs();
    this.updateHighlight();
    this.events.emit('layer', undefined);
  }

  // ------------------------------------------------------------- selection

  select(selection: LlmSelection | null): void {
    this.selection = selection;
    if (selection?.kind === 'cell') {
      this.focus = selection.position;
      this.layer = selection.layer;
      this.journey?.setStage(this.layer);
      this.events.emit('layer', undefined);
    } else if (selection?.kind === 'token') this.focus = selection.position;
    else if (selection?.kind === 'head') {
      this.settings = { ...this.settings, arcHead: selection.head, arcsOn: true };
      this.layer = selection.layer + 1;
      this.journey?.setStage(this.layer);
      this.events.emit('layer', undefined);
      this.events.emit('settings', undefined);
    } else if (selection?.kind === 'layer') {
      this.layer = selection.layer;
      this.journey?.setStage(this.layer);
      this.events.emit('layer', undefined);
    }
    this.refreshArcs();
    this.updateHighlight();
    this.journey?.setSelection(selection?.kind === 'token' || selection?.kind === 'cell' ? { kind: 'point', position: selection.position } : null);
    this.showInspector();
    this.events.emit('selection', undefined);
  }

  private onPick(pick: LlmPick | null): void {
    if (!pick) return this.select(null);
    switch (pick.kind) {
      case 'cell':
        return this.select({ kind: 'cell', layer: pick.layer, position: pick.position });
      case 'token':
        return this.select({ kind: 'token', position: pick.position });
      case 'layer':
        return this.select({ kind: 'layer', layer: pick.layer });
      case 'arc':
        this.focus = pick.query;
        return this.select({ kind: 'head', layer: pick.layer, head: pick.head });
      case 'prediction':
        return this.select({ kind: 'prediction' });
    }
  }

  private onJourneyPick(pick: EmbeddingPick | null): void {
    if (!pick || pick.kind !== 'point') return this.select(null);
    this.select({ kind: 'cell', layer: this.layer, position: pick.position });
  }

  private updateHighlight(): void {
    const s = this.selection;
    this.view?.setHighlight({
      layer: this.layer,
      cell: s?.kind === 'cell' ? { layer: s.layer, position: s.position } : null,
      token: s?.kind === 'token' ? s.position : null,
      focus: this.focus,
    });
  }

  // ----------------------------------------------------------------- scene

  private refreshScene(): void {
    const a = this.analysis;
    if (!a || !this.view || !this.journey) return;
    this.view.setAnalysis(a);
    this.view.setShowWords(this.settings.showWords);
    this.refreshArcs();
    this.updateHighlight();
    // token journey: positions ≥ 1 (the first token was excluded from the PCA fit)
    const points: EmbeddingPoint[] = [];
    for (let p = 1; p < a.T; p++) {
      const t = (p - 1) / Math.max(1, a.T - 2);
      points.push({
        path: a.journey.coords.map((stage) => stage[p] as Vec3),
        label: `${showToken(a.tokens[p].text)}·${p}`,
        color: `hsl(${Math.round(200 + 130 * t)} 70% ${this.paletteValue.theme === 'dark' ? 66 : 45}%)`,
        pick: { kind: 'point', position: p },
      });
    }
    this.journey.setPoints(points, a.journey.explained.map((v, k) => `PC${k + 1} · ${formatPercent(v, 0)}`), { trails: true });
    this.journey.setStage(this.layer);
  }

  private refreshArcs(): void {
    const a = this.analysis;
    if (!a || !this.view) return;
    const block = this.layer - 1; // the block that produced the current layer
    const st = this.settings;
    if (!st.arcsOn || block < 0) {
      this.view.setArcs(null);
      return;
    }
    const queries = st.arcQueries === 'all' ? Array.from({ length: a.T }, (_, i) => i) : [this.focus];
    this.view.setArcs({ layer: block, head: st.arcHead, queries, min: 0.04, color: this.paletteValue.theme === 'dark' ? '#c9b8ff' : '#6a4fd6' });
  }

  private applyView(animate: boolean): void {
    const journey = this.settings.view === 'journey';
    this.view?.setVisible(!journey);
    this.journey?.setVisible(journey);
    this.tooltip?.hide();
    this.resetCameraBounds();
    this.fitView('iso', animate);
  }

  private resetCameraBounds(): void {
    const host = this.host;
    if (!host || !this.view || !this.journey || !this.analysis) return;
    const b = this.settings.view === 'journey' ? this.journey.bounds() : this.view.bounds();
    host.camera.setBounds(b.min, b.max);
  }

  private fitView(view: CameraView, animate: boolean): void {
    if (this.settings.view === 'journey') this.host?.camera.setView(view, animate);
    else this.host?.camera.fitDirection(VIEWS[view], animate);
  }

  switchTo(id: 'transformer' | 'experiments'): void {
    this.host?.switchWorkspace(id);
  }

  /** Opens a recorded GPT-2 experiment (public/llm-demo/experiments.json) in the Experiments tab. */
  openExperiment(id: string): void {
    const workspace = this.host?.switchWorkspace('experiments');
    if (workspace instanceof ExperimentsWorkspace) void workspace.openRecorded(id);
  }

  setCameraView(view: CameraView): void {
    this.resetCameraBounds();
    this.fitView(view, true);
  }

  focusSelection(): void {
    const s = this.selection;
    if (!s || !this.view || !this.host || this.settings.view !== 'tower') return;
    if (s.kind === 'cell') this.host.camera.focus(this.view.cellCenter(s.layer, s.position), 2.2);
    else if (s.kind === 'token') this.host.camera.focus(this.view.cellCenter(6, s.position), 7);
  }

  // ------------------------------------------------------------- inspector

  showInspector(keepScroll = false): void {
    this.inspector?.show(this.inspect(), keepScroll);
  }

  private inspect(): InspectorView {
    const s = this.selection;
    if (!this.analysis || !s) return overviewInspector(this);
    switch (s.kind) {
      case 'cell':
        return cellInspector(this, s.layer, s.position);
      case 'token':
        return tokenInspector(this, s.position);
      case 'head':
        return headInspector(this, s.layer, s.head);
      case 'layer':
        return layerInspector(this, s.layer);
      case 'prediction':
        return predictionInspector(this);
    }
  }

  // ----------------------------------------------------------------- input

  private describe(pick: LlmPick | EmbeddingPick): HTMLElement | null {
    const a = this.analysis;
    if (!a) return null;
    const word = (id: number) => showToken(a.words[String(id)] ?? `#${id}`);
    switch (pick.kind) {
      case 'cell': {
        const [id, p] = a.lens[pick.layer].top[pick.position][0];
        return h(
          'div',
          null,
          h('strong', null, `“${showToken(a.tokens[pick.position].text)}” · ${pick.layer === 0 ? 'embeddings' : `after layer ${pick.layer}`}`),
          h('div', { class: 'tooltip__meta' }, 'logit lens (a rough probe)'),
          h('div', null, `top word ${word(id)} ${formatProbability(p)} · final answer ${word(a.predictions[pick.position][0])} ranks #${a.lens[pick.layer].finalRank[pick.position] + 1}`),
        );
      }
      case 'arc':
        return h('div', null, h('strong', null, `Layer ${pick.layer + 1} · head ${pick.head + 1}`), h('div', null, `“${showToken(a.tokens[pick.query].text)}” reads “${showToken(a.tokens[pick.key].text)}” ${formatPercent(pick.weight, 1)}`));
      case 'point':
        return h('div', null, h('strong', null, `“${showToken(a.tokens[pick.position].text)}”`), ` · position ${pick.position}`, h('div', { class: 'tooltip__meta' }, `its vector at layer ${this.layer} (PCA projection)`));
      default:
        return null;
    }
  }

  private pickAt(x: number, y: number): LlmPick | EmbeddingPick | null {
    return this.settings.view === 'journey' ? (this.journey?.pick(x, y) ?? null) : (this.view?.pick(x, y) ?? null);
  }

  private bindPointer(): void {
    const { viewport } = this.host!;
    const canvas = viewport.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    let last: { clientX: number; clientY: number } | null = null;
    const isJourney = (p: LlmPick | EmbeddingPick | null): p is EmbeddingPick => !!p && (p.kind === 'point' || p.kind === 'word');
    const hoverAt = frameThrottle(() => {
      if (!last || down) return;
      const { x, y } = viewport.toCanvas(last.clientX, last.clientY);
      const hit = this.pickAt(x, y);
      if (this.settings.view === 'journey') this.journey?.setHover(isJourney(hit) ? hit : null);
      else this.view?.setHover(hit && !isJourney(hit) ? hit : null);
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
        const hit = this.pickAt(x, y);
        if (this.settings.view === 'journey') this.onJourneyPick(isJourney(hit) ? hit : null);
        else this.onPick(hit && !isJourney(hit) ? hit : null);
      }
    });
    this.bag.listen(canvas, 'pointerleave', () => {
      last = null;
      this.view?.setHover(null);
      this.journey?.setHover(null);
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
      const a = this.analysis;
      if (!a) return;
      let handled = true;
      const s = this.selection;
      switch (e.key) {
        case ' ':
          this.togglePlay();
          break;
        case 'ArrowUp':
        case 'ArrowDown': {
          const d = e.key === 'ArrowUp' ? 1 : -1;
          if (s?.kind === 'cell') this.select({ ...s, layer: Math.max(0, Math.min(this.stages - 1, s.layer + d)) });
          else this.setLayer(this.layer + d);
          break;
        }
        case 'ArrowLeft':
        case 'ArrowRight': {
          const d = e.key === 'ArrowLeft' ? -1 : 1;
          if (s?.kind === 'cell') this.select({ ...s, position: Math.max(0, Math.min(a.T - 1, s.position + d)) });
          else if (s?.kind === 'token') this.select({ kind: 'token', position: Math.max(0, Math.min(a.T - 1, s.position + d)) });
          else if (s?.kind === 'head') this.select({ ...s, head: (s.head + d + a.heads) % a.heads });
          else this.setFocus(this.focus + d);
          break;
        }
        case 'Escape':
          if (!s) return;
          this.select(null);
          break;
        case 'r':
        case 'R':
          this.setCameraView('iso');
          break;
        case 'f':
        case 'F':
          this.focusSelection();
          break;
        case 'j':
        case 'J':
          this.update({ view: this.settings.view === 'tower' ? 'journey' : 'tower' });
          break;
        case '1':
        case '2':
        case '3':
        case '4':
          this.setCameraView((['iso', 'side', 'top', 'front'] as const)[Number(e.key) - 1]);
          break;
        default:
          handled = false;
      }
      if (handled) e.preventDefault();
    });
  }
}
