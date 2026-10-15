/**
 * TransformerWorkspace — the "Transformer" tab: a tiny GPT-style model
 * (trained in PyTorch, python/toy_transformer) computed in the browser, step
 * by step, on any text written with its 83-word vocabulary.
 *
 * Two 3-D views share the stage: the model (residual stream, attention arcs,
 * MLP blocks, predictions) and the embedding space (PCA of the token
 * embeddings, or of this sentence's vectors moving layer after layer).
 * Everything shown is read from one TransformerTrace, recomputed whenever the
 * text changes (≈ 1 ms).
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import { Emitter } from '../../core/Emitter';
import { createRng } from '../../math/random';
import type { CameraView } from '../../three/CameraController';
import { EmbeddingView, type EmbeddingPick, type EmbeddingPoint } from '../../three/EmbeddingView';
import { TransformerView, type ArcSpec, type TransformerHighlight, type TransformerPick } from '../../three/TransformerView';
import { isSpecial, project3, residualProjection, residualVectors, topTokens, vocabularyProjection, type ResidualSpace } from '../../transformer/analysis';
import { runTransformer, stageCount, type TransformerTrace } from '../../transformer/forward';
import { loadToyTransformer } from '../../transformer/loader';
import type { TransformerModel } from '../../transformer/model';
import { distribution, sample, DEFAULT_SAMPLING, type SamplingOptions } from '../../transformer/sampling';
import { tokenize } from '../../transformer/tokenizer';
import { h, isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, toast, Tooltip } from '../../ui/overlays';
import { button } from '../../ui/widgets';
import { formatPercent, formatSigned, formatValue } from '../../utils/format';
import { frameThrottle } from '../../utils/schedule';
import type { Vec3 } from '../../visualization/layout';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import type { Panel } from '../network/context';
import { groupColor, headColor, headColors } from './colors';
import { attentionInspector } from './inspectors/attentionInspector';
import { headInspector } from './inspectors/headInspector';
import { mlpInspector } from './inspectors/mlpInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { predictionInspector } from './inspectors/predictionInspector';
import { tokenInspector } from './inspectors/tokenInspector';
import { wordInspector } from './inspectors/wordInspector';
import { AttentionPanel, DisplayPanel, ExamplesPanel, TextPanel } from './panels';
import { PipelineBar } from './PipelineBar';
import { PipelinePlayer, pipelineSteps } from './pipeline';
import { TransformerToolbar } from './TransformerToolbar';

export type TransformerSelection =
  | { kind: 'token'; position: number; stage: number }
  | { kind: 'head'; layer: number; head: number; key: number | null }
  | { kind: 'attention'; layer: number }
  | { kind: 'mlp'; layer: number; position: number }
  | { kind: 'prediction'; position: number }
  | { kind: 'word'; id: number };

export interface TransformerSettings {
  /** Attention arcs drawn: layer and head (or all). */
  arcLayer: number | 'all';
  arcHead: number | 'all';
  /** Arcs of the focused token only, or of every token. */
  arcQueries: 'focus' | 'all';
  /** Hide arcs whose weight is below 5 %. */
  hideWeak: boolean;
  predictionsEverywhere: boolean;
  view: 'model' | 'embeddings';
  embedding: 'vocabulary' | 'sentence';
  residualSpace: ResidualSpace;
  sampling: SamplingOptions;
}

export const DEFAULT_TEXT = "the cat didn't cross the river because it was";

export interface Example {
  label: string;
  text: string;
  description: string;
}

export const EXAMPLES: Example[] = [
  { label: 'Ambiguous “it”', text: DEFAULT_TEXT, description: 'The next word depends on both the animal (hungry) and the place (deep).' },
  { label: 'A fact', text: 'the capital of france is', description: 'Recalled from the weights: the model was trained on these facts.' },
  { label: 'Reverse fact', text: 'rome is the capital of', description: 'The same fact, asked the other way round.' },
  { label: 'Royal', text: 'the queen is a', description: 'A word pair learned from the training sentences.' },
  { label: 'Counting', text: 'one two three four', description: 'Continues a sequence: an easy pattern for attention.' },
  { label: 'Predator', text: 'the lion chased the', description: 'Who chases whom in this tiny world.' },
  { label: 'Unknown word', text: 'the zebra is an', description: 'A word outside the vocabulary becomes <unk>.' },
];

export type TransformerEvents = {
  /** Text, tokens or trace changed. */
  trace: undefined;
  selection: undefined;
  settings: undefined;
  /** The focused token (whose attention arcs are shown) changed. */
  focus: undefined;
};

const CONTEXT_FULL = 'The context is full: this model reads at most 16 tokens.';

/** Camera directions of the model view (tokens along x, layers along y, tiles facing +z). */
const MODEL_VIEWS: Record<CameraView, Vec3> = {
  iso: [-0.3, 0.5, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.35, 0.3],
};

export class TransformerWorkspace implements Workspace {
  readonly id = 'transformer' as const;
  readonly events = new Emitter<TransformerEvents>();
  model: TransformerModel | null = null;
  text = DEFAULT_TEXT;
  trace: TransformerTrace | null = null;
  settings: TransformerSettings = {
    arcLayer: 'all',
    arcHead: 'all',
    arcQueries: 'focus',
    hideWeak: true,
    predictionsEverywhere: false,
    view: 'model',
    embedding: 'vocabulary',
    residualSpace: 'normalized',
    sampling: { ...DEFAULT_SAMPLING },
  };
  selection: TransformerSelection | null = null;
  /** Token whose attention arcs are drawn (the "query"). */
  focus = 0;
  /** Step of the Q/K/V walkthrough in the head inspector. */
  qkvStep = 0;
  player: PipelinePlayer | null = null;
  seed = 7;
  private rng = createRng(this.seed);
  private readonly history: string[] = [];

  private host: WorkspaceHost | null = null;
  private view: TransformerView | null = null;
  private embeddingView: EmbeddingView | null = null;
  private inspector: InspectorPanel | null = null;
  private tooltip: Tooltip | null = null;
  private panels: Panel[] = [];
  private bag = new DisposeBag();
  private placeholder: HTMLElement | null = null;
  private vocabulary: ReturnType<typeof vocabularyProjection> | null = null;
  private paletteValue: ScenePalette = scenePalette('dark', 'blue-orange');

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  get canUndo(): boolean {
    return this.history.length > 0;
  }

  get stages(): number {
    return this.model ? stageCount(this.model) : 0;
  }

  headColor(head: number): string {
    return headColor(head, this.paletteValue.theme);
  }

  // ------------------------------------------------------------- lifecycle

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.bag = new DisposeBag();
    host.setSplit(null);
    const view = host.store.state.view;
    this.paletteValue = scenePalette(view.theme, view.weightPalette);
    host.viewport.setBackground(this.paletteValue.background);
    if (this.model) {
      this.mountScene();
      return;
    }
    this.placeholder = h('div', { class: 'placeholder-card' }, h('h2', null, 'Loading the Transformer…'), h('p', { class: 'muted' }, 'Downloading the weights of the toy model (≈ 300 KB).'));
    host.slots.stage.append(this.placeholder);
    loadToyTransformer()
      .then((model) => {
        this.model = model;
        this.run();
        if (this.host !== host) return;
        this.placeholder?.remove();
        this.placeholder = null;
        this.mountScene();
      })
      .catch((error: unknown) => {
        console.error(error);
        if (this.host !== host || !this.placeholder) return;
        this.placeholder.replaceChildren(
          h('h2', null, 'The model could not be loaded'),
          h('p', null, String(error instanceof Error ? error.message : error)),
          button({
            label: 'Retry',
            onClick: () => {
              this.unmount();
              this.mount(host);
            },
          }),
        );
      });
  }

  unmount(): void {
    this.player?.stop();
    this.player = null;
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
    this.embeddingView?.dispose();
    this.embeddingView = null;
    this.placeholder?.remove();
    this.placeholder = null;
    this.host = null;
  }

  private mountScene(): void {
    const host = this.host!;
    const model = this.model!;
    const { viewport, slots, store } = host;
    this.player = new PipelinePlayer(pipelineSteps(model), viewport.animations, (stage) => this.onStage(stage));
    this.view = new TransformerView({
      viewport,
      palette: this.paletteValue,
      onLabelClick: (pick) => this.onPick(pick),
      onLabelHover: (pick) => this.view?.setHover(pick),
    });
    this.embeddingView = new EmbeddingView({ viewport, palette: this.paletteValue, onLabelClick: (pick) => this.onEmbeddingPick(pick) });
    this.inspector = new InspectorPanel(slots.right);
    this.tooltip = new Tooltip(slots.stage);

    this.panels = [new TextPanel(this), new ExamplesPanel(this), new AttentionPanel(this), new DisplayPanel(this)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bar = new PipelineBar(this, this.player);
    slots.bottom.append(bar.el);
    const toolbar = new TransformerToolbar(this);
    slots.stage.append(toolbar.el);
    this.panels.push(bar, toolbar);

    this.refreshScene();
    this.applyViewMode(false);
    this.showInspector();

    this.bag.add(
      store.watch(
        (s) => s.view.theme + s.view.weightPalette,
        () => {
          const v = store.state.view;
          this.paletteValue = scenePalette(v.theme, v.weightPalette);
          viewport.setBackground(this.paletteValue.background);
          this.view?.setPalette(this.paletteValue);
          this.embeddingView?.setPalette(this.paletteValue);
          this.refreshScene();
          this.showInspector(true);
          this.events.emit('settings', undefined);
        },
      ),
    );
    this.bindPointer();
    this.bindKeyboard();
  }

  // ---------------------------------------------------------------- model

  /** Tokenizes the text and runs the model. */
  private run(): void {
    const model = this.model;
    if (!model) return;
    const previousT = this.trace?.tokens.length ?? 0;
    const tokens = tokenize(model, this.text);
    this.trace = runTransformer(model, tokens);
    const T = tokens.length;
    if (previousT === 0 || this.focus >= T || this.focus === previousT - 1) this.focus = T - 1;
    // The prediction of the last token follows the end of the text.
    const s = this.selection;
    if (s?.kind === 'prediction' && s.position === previousT - 1) this.selection = { kind: 'prediction', position: T - 1 };
    this.selection = this.validSelection(this.selection);
  }

  private validSelection(s: TransformerSelection | null): TransformerSelection | null {
    if (!s || !this.trace) return s;
    const T = this.trace.tokens.length;
    if ((s.kind === 'token' || s.kind === 'mlp') && s.position >= T) return null;
    if (s.kind === 'prediction' && s.position >= T) return { kind: 'prediction', position: T - 1 };
    if (s.kind === 'head' && s.key !== null && s.key >= T) return { ...s, key: null };
    return s;
  }

  setText(text: string): void {
    if (text === this.text) return;
    this.text = text;
    this.run();
    this.refreshScene();
    this.showInspector(true);
    this.events.emit('trace', undefined);
  }

  /** Loads an example (and clears the generation history). */
  loadExample(text: string): void {
    this.history.length = 0;
    this.selection = null;
    this.text = text;
    this.run();
    this.focus = (this.trace?.tokens.length ?? 1) - 1;
    this.refreshScene();
    this.showInspector();
    this.events.emit('trace', undefined);
    this.events.emit('focus', undefined);
    this.events.emit('selection', undefined);
  }

  update(patch: Partial<TransformerSettings>): void {
    const before = this.settings;
    this.settings = { ...before, ...patch };
    if (patch.view !== undefined && patch.view !== before.view) this.applyViewMode(true);
    else if (patch.embedding !== undefined && patch.embedding !== before.embedding && this.settings.view === 'embeddings') this.applyViewMode(true);
    this.refreshScene();
    this.events.emit('settings', undefined);
  }

  /** Sampling options change without rebuilding the inspector (its sliders stay under the pointer). */
  setSampling(patch: Partial<SamplingOptions>): void {
    this.settings = { ...this.settings, sampling: { ...this.settings.sampling, ...patch } };
  }

  setFocus(position: number): void {
    const T = this.trace?.tokens.length ?? 0;
    const p = Math.max(0, Math.min(T - 1, position));
    if (p === this.focus) return;
    this.focus = p;
    if (this.selection?.kind === 'head') this.selection = { ...this.selection, key: null };
    this.refreshArcs();
    this.updateHighlight();
    this.showInspector(true);
    this.events.emit('focus', undefined);
  }

  setQkvStep(step: number): void {
    this.qkvStep = step;
    this.showInspector(true);
  }

  // ------------------------------------------------------------ generation

  /** Appends `count` tokens chosen by sampling (or greedily) from the next-token distribution. */
  generate(mode: 'sample' | 'greedy', count = 1): void {
    const model = this.model;
    if (!model) return;
    for (let k = 0; k < count; k++) {
      const trace = this.trace!;
      const T = trace.tokens.length;
      const lastText = trace.tokens[T - 1].text;
      if (lastText === '.') {
        if (k === 0) toast('The sentence is finished: the model was trained on single sentences, so after “.” its predictions mean nothing.', 'info', 4500);
        break;
      }
      if (T >= model.config.context) {
        if (k === 0) toast(CONTEXT_FULL, 'warning');
        break;
      }
      // Special tokens are never sampled (they cannot be written as text).
      const logits = Float64Array.from(trace.logits[T - 1], (v, id) => (isSpecial(model, id) ? -Infinity : v));
      const options = mode === 'greedy' ? { temperature: 1, topK: 1, topP: 1 } : this.settings.sampling;
      const id = mode === 'greedy' ? distribution(logits, options).order[0] : sample(distribution(logits, options), this.rng);
      const word = model.config.vocab[id];
      this.history.push(this.text);
      const base = this.text.replace(/\s+$/, '');
      this.text = /^[.,!?;:]$/.test(word) ? `${base}${word}` : `${base} ${word}`;
      const followLast = this.focus === T - 1;
      this.run();
      if (followLast) this.focus = this.trace!.tokens.length - 1;
    }
    this.refreshScene();
    this.showInspector(true);
    this.events.emit('trace', undefined);
    this.events.emit('focus', undefined);
  }

  undo(): void {
    const previous = this.history.pop();
    if (previous === undefined) return;
    this.text = previous;
    this.run();
    this.refreshScene();
    this.showInspector(true);
    this.events.emit('trace', undefined);
  }

  reseed(): void {
    this.seed = Math.floor(Math.random() * 100000);
    this.rng = createRng(this.seed);
    this.showInspector(true);
  }

  // ------------------------------------------------------------- selection

  select(selection: TransformerSelection | null): void {
    this.selection = this.validSelection(selection);
    if (selection?.kind === 'word' && this.settings.view !== 'embeddings') this.update({ view: 'embeddings', embedding: 'vocabulary' });
    this.updateHighlight();
    this.refreshArcs();
    this.embeddingView?.setSelection(this.embeddingSelection());
    this.showInspector();
    this.events.emit('selection', undefined);
  }

  private onPick(pick: TransformerPick | null): void {
    if (!pick || !this.trace) {
      this.select(null);
      return;
    }
    const S = this.stages;
    switch (pick.kind) {
      case 'cell':
        this.setFocus(pick.position);
        this.select({ kind: 'token', position: pick.position, stage: pick.stage });
        break;
      case 'arc':
        this.setFocus(pick.query);
        this.select({ kind: 'head', layer: pick.layer, head: pick.head, key: pick.key });
        break;
      case 'mlp':
        this.setFocus(pick.position);
        this.select({ kind: 'mlp', layer: pick.layer, position: pick.position });
        break;
      case 'sublayer':
        this.select(pick.part === 'attention' ? { kind: 'attention', layer: pick.layer } : { kind: 'mlp', layer: pick.layer, position: this.focus });
        break;
      case 'prediction':
        this.select({ kind: 'prediction', position: pick.position });
        break;
      case 'token':
        this.setFocus(pick.position);
        this.select({ kind: 'token', position: pick.position, stage: S - 1 });
        break;
      case 'stage':
        this.select({ kind: 'token', position: this.focus, stage: pick.stage });
        break;
    }
  }

  private onEmbeddingPick(pick: EmbeddingPick | null): void {
    if (!pick) {
      this.select(null);
      return;
    }
    if (pick.kind === 'word') this.select({ kind: 'word', id: pick.id });
    else {
      this.setFocus(pick.position);
      this.select({ kind: 'token', position: pick.position, stage: Math.round(this.embeddingStage()) });
    }
  }

  private embeddingSelection(): EmbeddingPick | null {
    const s = this.selection;
    if (s?.kind === 'word') return { kind: 'word', id: s.id };
    if (s?.kind === 'token') return { kind: 'point', position: s.position };
    return null;
  }

  /** Highlights a dimension of every vector in the 3-D view (from the inspector's vector strips). */
  markDimension(dim: number | null): void {
    if (!this.view) return;
    this.view.setHighlight({ ...this.highlight(), dim });
  }

  private highlight(): TransformerHighlight {
    const s = this.selection;
    const S = this.stages;
    const none: TransformerHighlight = { slab: null, mlp: null, sublayer: null, focus: null, dim: null };
    if (!s) return none;
    switch (s.kind) {
      case 'token':
        return { ...none, slab: { position: s.position, stage: s.stage } };
      case 'head':
        return { ...none, slab: { position: this.focus, stage: 2 * s.layer + 1 }, sublayer: { layer: s.layer, part: 'attention' } };
      case 'attention':
        return { ...none, sublayer: { layer: s.layer, part: 'attention' } };
      case 'mlp':
        return { ...none, mlp: { layer: s.layer, position: s.position }, sublayer: { layer: s.layer, part: 'mlp' } };
      case 'prediction':
        return { ...none, slab: { position: s.position, stage: S - 1 } };
      default:
        return none;
    }
  }

  private updateHighlight(): void {
    this.view?.setHighlight(this.highlight());
  }

  // ----------------------------------------------------------------- scene

  private refreshScene(): void {
    const { model, trace, view } = this;
    if (!model || !trace || !view) return;
    view.setData(model, trace);
    this.refreshArcs();
    this.refreshPredictions();
    this.updateHighlight();
    if (this.settings.view === 'embeddings') this.refreshEmbeddings();
  }

  /** Attention arcs to draw, from the settings, the focus and the selection. */
  arcs(): ArcSpec[] {
    const { model, trace } = this;
    if (!model || !trace) return [];
    const { nLayers, nHeads } = model.config;
    const T = trace.tokens.length;
    const st = this.settings;
    const layers = st.arcLayer === 'all' ? Array.from({ length: nLayers }, (_, l) => l) : [st.arcLayer];
    const heads = st.arcHead === 'all' ? Array.from({ length: nHeads }, (_, k) => k) : [st.arcHead];
    const queries = st.arcQueries === 'all' ? Array.from({ length: T }, (_, i) => i) : [this.focus];
    const min = st.hideWeak ? 0.05 : 0.004;
    const sel = this.selection;
    const arcs: ArcSpec[] = [];
    for (const layer of layers) {
      for (const head of heads) {
        const a = trace.blocks[layer].attention[head];
        for (const query of queries) {
          for (let key = 0; key <= query; key++) {
            const weight = a[query * T + key];
            if (weight < min) continue;
            let emphasis = 1;
            if (sel?.kind === 'head') {
              if (sel.layer !== layer || sel.head !== head) emphasis = 0;
              else if (sel.key === key && query === this.focus) emphasis = 2;
            } else if (sel?.kind === 'attention' && sel.layer !== layer) emphasis = 0;
            else if (sel?.kind === 'token' && st.arcQueries === 'all' && query !== sel.position) emphasis = 0;
            arcs.push({ layer, head, query, key, weight, emphasis });
          }
        }
      }
    }
    return arcs;
  }

  private refreshArcs(): void {
    this.view?.setArcs(this.arcs(), headColors(this.model?.config.nHeads ?? 0, this.paletteValue.theme));
  }

  private refreshPredictions(): void {
    const { model, trace, view } = this;
    if (!model || !trace || !view) return;
    const T = trace.tokens.length;
    const positions = this.settings.predictionsEverywhere ? Array.from({ length: T }, (_, p) => p) : [T - 1];
    view.setPredictions(positions.map((position) => ({ position, top: topTokens(model, trace.logits[position], position === T - 1 ? 4 : 1) })));
  }

  private refreshEmbeddings(): void {
    const { model, trace, embeddingView } = this;
    if (!model || !trace || !embeddingView) return;
    const theme = this.paletteValue.theme;
    const pct = (v: number) => formatPercent(v, 0);
    if (this.settings.embedding === 'vocabulary') {
      const proj = (this.vocabulary ??= vocabularyProjection(model));
      const d = model.config.dModel;
      const points: EmbeddingPoint[] = proj.ids.map((id) => {
        const word = model.config.vocab[id];
        return { path: [project3(proj, model.tokenEmbedding.subarray(id * d, (id + 1) * d))], label: word, color: groupColor(model.groups[word], theme), pick: { kind: 'word', id } };
      });
      embeddingView.setPoints(points, proj.explained.map((v, k) => `PC${k + 1} · ${pct(v)}`), { trails: false });
    } else {
      const vectors = residualVectors(trace, model, this.settings.residualSpace);
      const proj = residualProjection(vectors, 1);
      const points: EmbeddingPoint[] = [];
      for (let p = 1; p < trace.tokens.length; p++) {
        const token = trace.tokens[p];
        points.push({
          path: vectors.map((stage) => project3(proj, stage[p])),
          label: `${token.text}·${p}`,
          color: groupColor(model.groups[token.text], theme),
          pick: { kind: 'point', position: p },
        });
      }
      embeddingView.setPoints(points, proj.explained.map((v, k) => `PC${k + 1} · ${pct(v)}`), { trails: true });
      embeddingView.setStage(this.embeddingStage());
    }
    embeddingView.setSelection(this.embeddingSelection());
  }

  /** Stage of the sentence vectors shown in the embedding view (follows the pipeline). */
  embeddingStage(): number {
    const stage = this.player?.stage ?? this.stages;
    return Math.max(0, Math.min(this.stages - 1, stage - 1));
  }

  private onStage(stage: number): void {
    this.view?.setStage(stage);
    this.view?.setPredictionsVisible(!this.player || this.player.step === this.player.last);
    if (this.settings.view === 'embeddings' && this.settings.embedding === 'sentence') this.embeddingView?.setStage(this.embeddingStage());
  }

  private applyViewMode(animate: boolean): void {
    const host = this.host;
    if (!host || !this.view || !this.embeddingView) return;
    const embeddings = this.settings.view === 'embeddings';
    this.view.setVisible(!embeddings);
    this.embeddingView.setVisible(embeddings);
    if (embeddings) this.refreshEmbeddings();
    this.tooltip?.hide();
    this.resetCameraBounds();
    this.fitView('iso', animate);
  }

  private fitView(view: CameraView, animate: boolean): void {
    if (this.settings.view === 'embeddings') this.host?.camera.setView(view, animate);
    else this.host?.camera.fitDirection(MODEL_VIEWS[view], animate);
  }

  // ---------------------------------------------------------------- camera

  resetCamera(): void {
    this.setCameraView('iso');
  }

  setCameraView(view: CameraView): void {
    this.resetCameraBounds();
    this.fitView(view, true);
  }

  private resetCameraBounds(): void {
    const host = this.host;
    if (!host || !this.view || !this.embeddingView) return;
    const bounds = this.settings.view === 'embeddings' ? this.embeddingView.bounds() : this.view.bounds();
    host.camera.setBounds(bounds.min, bounds.max);
  }

  focusSelection(): void {
    const host = this.host;
    const s = this.selection;
    if (!host || !s || !this.view) return;
    if (this.settings.view === 'embeddings') {
      const pick = this.embeddingSelection();
      const position = pick ? this.embeddingView?.positionOf(pick) : null;
      if (position) host.camera.focus(position, 1.6);
      return;
    }
    switch (s.kind) {
      case 'token':
        host.camera.focus(this.view.slabCenter(s.position, s.stage), 2.4);
        break;
      case 'head':
      case 'attention': {
        const [x, y, z] = this.view.slabCenter(this.focus, 2 * s.layer + 0.5);
        host.camera.focus([x, y, z], 4.5);
        break;
      }
      case 'mlp':
        host.camera.focus(this.view.mlpCenter(s.layer, s.position), 2);
        break;
      case 'prediction':
        host.camera.focus(this.view.slabCenter(s.position, this.stages - 0.6), 2.6);
        break;
    }
  }

  // ------------------------------------------------------------- inspector

  showInspector(keepScroll = false): void {
    if (!this.inspector || !this.model || !this.trace) return;
    this.inspector.show(this.inspect(), keepScroll);
  }

  private inspect(): InspectorView {
    const s = this.selection;
    if (!s) return overviewInspector(this);
    switch (s.kind) {
      case 'token':
        return tokenInspector(this, s.position, s.stage);
      case 'head':
        return headInspector(this, s.layer, s.head, s.key);
      case 'attention':
        return attentionInspector(this, s.layer);
      case 'mlp':
        return mlpInspector(this, s.layer, s.position);
      case 'prediction':
        return predictionInspector(this, s.position);
      case 'word':
        return wordInspector(this, s.id);
    }
  }

  // ----------------------------------------------------------------- input

  private describe(pick: TransformerPick | EmbeddingPick): HTMLElement | null {
    const { model, trace } = this;
    if (!model || !trace) return null;
    const tokenName = (p: number) => `“${trace.tokens[p].text}”`;
    const stageName = (s: number) => (s === 0 ? 'embedding + position' : (s - 1) % 2 === 0 ? `after attention ${Math.floor((s - 1) / 2) + 1}` : `after MLP ${Math.floor((s - 1) / 2) + 1}`);
    switch (pick.kind) {
      case 'cell': {
        const vector = pick.stage === 0 ? trace.x0[pick.position] : (pick.stage - 1) % 2 === 0 ? trace.blocks[Math.floor((pick.stage - 1) / 2)].mid[pick.position] : trace.blocks[Math.floor((pick.stage - 1) / 2)].output[pick.position];
        return h(
          'div',
          null,
          h('strong', null, tokenName(pick.position)),
          ` · position ${pick.position}`,
          h('div', { class: 'tooltip__meta' }, `x ${stageName(pick.stage)} · dimension ${pick.dim + 1}`),
          h('div', null, `value ${formatSigned(vector[pick.dim])} · ‖x‖ = ${formatValue(Math.hypot(...vector), 2)}`),
        );
      }
      case 'arc':
        return h(
          'div',
          null,
          h('strong', null, `Layer ${pick.layer + 1} · head ${pick.head + 1}`),
          h('div', { class: 'tooltip__meta' }, `${tokenName(pick.query)} reads ${tokenName(pick.key)}`),
          h('div', null, `attention weight ${formatPercent(pick.weight, 1)}`),
        );
      case 'mlp': {
        const out = trace.blocks[pick.layer].mlpOut[pick.position];
        return h('div', null, h('strong', null, `MLP ${pick.layer + 1} · ${tokenName(pick.position)}`), h('div', { class: 'tooltip__meta' }, '32 → 128 → 32, applied to this position only'), h('div', null, `‖output‖ = ${formatValue(Math.hypot(...out), 2)}`));
      }
      case 'word': {
        const word = model.config.vocab[pick.id];
        return h('div', null, h('strong', null, word), h('div', { class: 'tooltip__meta' }, model.groups[word] ?? 'other word'), h('div', null, 'click: nearest words'));
      }
      case 'point':
        return h('div', null, h('strong', null, tokenName(pick.position)), ` · position ${pick.position}`, h('div', { class: 'tooltip__meta' }, 'its vector, layer after layer (projection)'));
      default:
        return null;
    }
  }

  private pickAt(x: number, y: number): TransformerPick | EmbeddingPick | null {
    return this.settings.view === 'embeddings' ? (this.embeddingView?.pick(x, y) ?? null) : (this.view?.pick(x, y) ?? null);
  }

  private bindPointer(): void {
    const host = this.host!;
    const { viewport } = host;
    const canvas = viewport.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    let last: { clientX: number; clientY: number } | null = null;

    const hoverAt = frameThrottle(() => {
      if (!last || down) return;
      const { x, y } = viewport.toCanvas(last.clientX, last.clientY);
      const hit = this.pickAt(x, y);
      if (this.settings.view === 'embeddings') this.embeddingView?.setHover(hit && (hit.kind === 'word' || hit.kind === 'point') ? hit : null);
      else this.view?.setHover(hit && hit.kind !== 'word' && hit.kind !== 'point' ? hit : null);
      canvas.style.cursor = hit ? 'pointer' : '';
      const content = hit ? this.describe(hit) : null;
      if (content) this.tooltip?.show(x, y, content);
      else this.tooltip?.hide();
    });
    this.bag.add(() => hoverAt.cancel());
    this.bag.listen(canvas, 'pointermove', (event) => {
      last = { clientX: event.clientX, clientY: event.clientY };
      if (down) {
        this.tooltip?.hide();
        return;
      }
      hoverAt();
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
        if (this.settings.view === 'embeddings') this.onEmbeddingPick(hit && (hit.kind === 'word' || hit.kind === 'point') ? hit : null);
        else this.onPick(hit && hit.kind !== 'word' && hit.kind !== 'point' ? hit : null);
      }
      hoverAt();
    });
    this.bag.listen(canvas, 'pointercancel', () => (down = null));
    this.bag.listen(canvas, 'pointerleave', () => {
      last = null;
      this.view?.setHover(null);
      this.embeddingView?.setHover(null);
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
    const player = this.player;
    const trace = this.trace;
    if (!player || !trace) return false;
    const T = trace.tokens.length;
    const S = this.stages;
    const s = this.selection;
    switch (e.key) {
      case ' ':
        player.toggle();
        return true;
      case '.':
        player.pause();
        player.next();
        return true;
      case ',':
        player.previous();
        return true;
      case 'Home':
        player.pause();
        player.restart();
        return true;
      case 'End':
        player.stop();
        player.events.emit('change', undefined);
        return true;
      case 'Escape':
        if (!s) return false;
        this.select(null);
        return true;
      case 'ArrowLeft':
      case 'ArrowRight': {
        const delta = e.key === 'ArrowLeft' ? -1 : 1;
        if (s?.kind === 'token') this.select({ ...s, position: clampInt(s.position + delta, 0, T - 1) });
        else if (s?.kind === 'mlp') this.select({ ...s, position: clampInt(s.position + delta, 0, T - 1) });
        else if (s?.kind === 'head') this.select({ ...s, head: clampInt(s.head + delta, 0, this.model!.config.nHeads - 1), key: null });
        else {
          this.setFocus(this.focus + delta);
          return true;
        }
        if (s.kind === 'token' || s.kind === 'mlp') this.setFocus((this.selection as { position: number }).position);
        return true;
      }
      case 'ArrowUp':
      case 'ArrowDown': {
        const delta = e.key === 'ArrowUp' ? 1 : -1;
        if (s?.kind === 'token') this.select({ ...s, stage: clampInt(s.stage + delta, 0, S - 1) });
        else if (s?.kind === 'head' || s?.kind === 'attention') this.select({ ...s, layer: clampInt(s.layer + delta, 0, this.model!.config.nLayers - 1) } as TransformerSelection);
        else return false;
        return true;
      }
      case 'r':
      case 'R':
        this.resetCamera();
        return true;
      case 'f':
      case 'F':
        this.focusSelection();
        return true;
      case 'e':
      case 'E':
        this.update({ view: this.settings.view === 'model' ? 'embeddings' : 'model' });
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

function clampInt(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
