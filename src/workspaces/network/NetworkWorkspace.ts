/**
 * NetworkWorkspace — the "Neural Network" tab: free exploration of a
 * feed-forward network, its animated forward pass, "Follow the
 * Signal" and the activation function explorer.
 *
 * Composition only: NetworkScene (3D + picking + inspector) + panels + the
 * forward player + follow-the-signal; architecture actions edit the model
 * through validated configurations.
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import type { ActivationId } from '../../math/activations';
import { formatConfig, LIMITS, specFromConfig, specToConfig, validateConfig, type NetworkConfig } from '../../neural/config';
import { DEFAULT_INIT } from '../../neural/init';
import type { ChangeReason, NetworkModel } from '../../neural/NetworkModel';
import { findPreset } from '../../neural/presets';
import { applyLoadedModel, deserializeModel, serializeModel, stringifyModel } from '../../neural/serialization';
import { openActivationExplorer } from '../../ui/ActivationExplorer';
import type { InspectorView } from '../../ui/InspectorPanel';
import { toast } from '../../ui/overlays';
import { button } from '../../ui/widgets';
import { downloadText, pickFile, slugify } from '../../utils/download';
import type { NetworkSelection } from '../../visualization/selection';
import { BottomBar } from './BottomBar';
import type { ArchitectureActions, NetworkWorkspaceContext, Panel, SignalActions } from './context';
import { FollowSignal } from './FollowSignal';
import { ForwardBar } from './ForwardBar';
import { ForwardPlayer } from './ForwardPlayer';
import { connectionInspector } from './inspectors/connectionInspector';
import { layerInspector } from './inspectors/layerInspector';
import { neuronInspector } from './inspectors/neuronInspector';
import { overviewInspector } from './inspectors/overviewInspector';
import { NetworkScene } from './NetworkScene';
import { ArchitecturePanel } from './panels/ArchitecturePanel';
import { InputPanel } from './panels/InputPanel';
import { ParametersPanel } from './panels/ParametersPanel';
import { ViewPanel } from './panels/ViewPanel';
import { ViewportToolbar } from './ViewportToolbar';

export class NetworkWorkspace implements Workspace, ArchitectureActions, SignalActions {
  readonly id = 'network' as const;
  readonly model: NetworkModel;
  private host!: WorkspaceHost;
  private scene!: NetworkScene;
  private ctx!: NetworkWorkspaceContext;
  private player!: ForwardPlayer;
  private follow!: FollowSignal;
  private forwardBar!: ForwardBar;
  private panels: Panel[] = [];
  private bag = new DisposeBag();

  constructor(model: NetworkModel) {
    this.model = model;
  }

  /** Scene of the mounted workspace (lessons drive it). */
  get networkScene(): NetworkScene {
    return this.scene;
  }

  get forwardPlayer(): ForwardPlayer {
    return this.player;
  }

  // ------------------------------------------------------------- lifecycle

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.bag = new DisposeBag();
    const { store, viewport, slots } = host;
    const model = this.model;

    this.scene = new NetworkScene({
      host,
      model,
      inspect: (selection) => this.inspect(selection),
      onKey: (event) => this.onKey(event),
      onRebuilt: () => {
        this.player.refresh();
        this.follow.update();
      },
    });
    const scene = this.scene;
    this.ctx = {
      model,
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
        toggleForward: () => this.toggleForward(),
        toggleFollow: () => this.toggleFollow(),
        openActivationExplorer: (id, z) => this.openActivationExplorer(id, z),
        applyPreset: (id) => this.applyPreset(id),
        applyConfig: (c, r) => this.applyConfig(c, r),
        setLayerSize: (l, s) => this.setLayerSize(l, s),
        setLayerActivation: (l, a) => this.setLayerActivation(l, a),
        addHiddenLayer: () => this.addHiddenLayer(),
        removeLayer: (l) => this.removeLayer(l),
        exportModel: () => this.exportModel(),
        importModel: () => this.importModel(),
        copyConfig: () => this.copyConfig(),
      },
    };

    this.player = new ForwardPlayer(scene.view, viewport.animations, () => model.layerCount);
    this.follow = new FollowSignal(scene.view, model, viewport.animations);
    this.forwardBar = new ForwardBar(this.ctx, this.player);

    const tools = [
      button({ icon: 'play', label: 'Forward', title: 'Animate the forward pass, layer by layer (P)', small: true, variant: 'ghost', onClick: () => this.toggleForward() }),
      button({ icon: 'signal', label: 'Follow', title: 'Follow the signal of the selected neuron or connection (S)', small: true, variant: 'ghost', onClick: () => this.toggleFollow() }),
      button({ icon: 'fx', label: 'f(z)', title: 'Explore the activation functions', small: true, variant: 'ghost', onClick: () => this.openActivationExplorer() }),
    ];
    const [forwardButton, followButton] = tools;

    this.panels = [new ArchitecturePanel(this.ctx), new InputPanel(this.ctx), new ParametersPanel(this.ctx), new ViewPanel(this.ctx)];
    slots.left.append(...this.panels.map((p) => p.el));
    const bottom = new BottomBar(this.ctx, viewport);
    slots.bottom.append(this.forwardBar.el, bottom.el);
    const toolbar = new ViewportToolbar(this.ctx, tools);
    slots.stage.append(toolbar.el);
    this.panels.push(bottom, toolbar, this.forwardBar);
    scene.showInspector();

    // Signal modes.
    this.bag.add(
      store.watch(
        (s) => s.signalMode,
        (mode, previous) => {
          if (previous === 'forward' && mode !== 'forward') this.player.stop();
          if (previous === 'follow' && mode !== 'follow') this.follow.stop();
          if (mode === 'forward' && !this.player.isActive) this.player.start();
          if (mode === 'follow' && !this.follow.follow(store.state.selection)) {
            toast('Select a neuron or a connection to follow its signal', 'info');
            store.set({ signalMode: 'off' });
          }
          forwardButton.setAttribute('aria-pressed', String(mode === 'forward'));
          followButton.setAttribute('aria-pressed', String(mode === 'follow'));
          scene.showInspector(true);
        },
      ),
    );
    this.bag.add(
      store.watch(
        (s) => s.selection,
        (selection) => {
          if (store.state.signalMode === 'follow' && !this.follow.follow(selection)) store.set({ signalMode: 'off' });
        },
      ),
    );
    this.bag.add(model.events.on('activations', () => this.follow.update()));
    this.bag.add(
      this.player.events.on('stepStart', ({ layer }) => {
        if (!this.forwardBar.followCamera) return;
        const { center, radius } = scene.view.focusTarget({ kind: 'layer', layer });
        host.camera.focus(center, radius * 1.6);
      }),
    );
  }

  unmount(): void {
    this.host.store.set({ signalMode: 'off' });
    this.bag.dispose();
    this.player.dispose();
    this.follow.stop();
    for (const panel of this.panels) {
      panel.dispose();
      panel.el.remove();
    }
    this.panels = [];
    this.scene.dispose();
  }

  private inspect(selection: NetworkSelection | null): InspectorView {
    if (!selection) return overviewInspector(this.ctx);
    if (selection.kind === 'neuron') return neuronInspector(this.ctx, selection.layer, selection.index);
    if (selection.kind === 'connection') return connectionInspector(this.ctx, selection.layer, selection.target, selection.source);
    return layerInspector(this.ctx, selection.layer);
  }

  private onKey(event: KeyboardEvent): boolean {
    const mode = this.host.store.state.signalMode;
    switch (event.key) {
      case 'p':
      case 'P':
        this.toggleForward();
        return true;
      case 's':
      case 'S':
        this.toggleFollow();
        return true;
      case ' ':
        if (mode !== 'forward') this.toggleForward();
        this.player.toggle();
        return true;
      case '.':
        if (mode !== 'forward') return false;
        this.player.next();
        return true;
      case ',':
        if (mode !== 'forward') return false;
        this.player.previous();
        return true;
      case 'Home':
        if (mode !== 'forward') return false;
        this.player.restart();
        return true;
      case 'Escape':
        if (mode === 'off') return false;
        this.host.store.set({ signalMode: 'off' });
        return true;
      default:
        return false;
    }
  }

  // ---------------------------------------------------------- signal modes

  toggleForward(): void {
    const { store } = this.host;
    store.set({ signalMode: store.state.signalMode === 'forward' ? 'off' : 'forward' });
  }

  toggleFollow(): void {
    const { store } = this.host;
    store.set({ signalMode: store.state.signalMode === 'follow' ? 'off' : 'follow' });
  }

  openActivationExplorer(activation?: ActivationId, z?: number): void {
    openActivationExplorer({ initial: activation, z });
  }

  // -------------------------------------------------------- architecture

  applyPreset(id: string): void {
    const preset = findPreset(id);
    if (!preset) return;
    const spec = specFromConfig(preset.config);
    this.host.store.set({ selection: null, hover: null, presetId: id });
    this.model.batch(() => {
      this.model.setArchitecture(spec, { preserveParams: false, input: preset.input?.() ?? new Float64Array(spec.layers[0].size), reason: 'preset' });
      this.model.reinitialize({ ...DEFAULT_INIT, ...preset.init });
    });
  }

  applyConfig(config: NetworkConfig, reason: ChangeReason = 'config'): string[] {
    const result = validateConfig(config);
    if (!result.ok) return result.errors.map((e) => `${e.path}: ${e.message}`);
    this.model.setArchitecture(result.spec, { preserveParams: true, reason });
    this.host.store.set({ presetId: null });
    return [];
  }

  /** Edits the current configuration and applies it, reporting validation errors as toasts. */
  private editConfig(edit: (config: NetworkConfig) => void): void {
    const config = specToConfig(this.model.spec);
    edit(config);
    const errors = this.applyConfig(config, 'edit');
    if (errors.length) toast(errors[0], 'error', 4500);
  }

  setLayerSize(layer: number, size: number): void {
    if (size < 1 || size > LIMITS.maxLayerSize) return;
    this.editConfig((config) => {
      const entry = config.layers[layer];
      entry.size = size;
      // A size change invalidates per-neuron labels and an image shape.
      if (entry.shape && entry.shape[0] * entry.shape[1] !== size) delete entry.shape;
      if (entry.labels && entry.labels.length !== size) delete entry.labels;
    });
  }

  setLayerActivation(layer: number, activation: ActivationId): void {
    this.editConfig((config) => {
      config.layers[layer].activation = activation;
    });
  }

  addHiddenLayer(): void {
    this.editConfig((config) => {
      const hidden = config.layers.filter((l) => l.type === 'dense');
      const template = hidden[hidden.length - 1];
      config.layers.splice(config.layers.length - 1, 0, { type: 'dense', size: template?.size ?? 4, activation: template?.activation ?? 'relu' });
    });
  }

  removeLayer(layer: number): void {
    if (this.model.spec.layers[layer]?.role !== 'hidden') return;
    const { store } = this.host;
    if (store.state.selection) store.set({ selection: null });
    this.editConfig((config) => {
      config.layers.splice(layer, 1);
    });
  }

  exportModel(): void {
    const data = serializeModel(this.model);
    downloadText(`${slugify(this.model.spec.name)}.model.json`, stringifyModel(data));
    toast('Model exported (architecture, weights and input)', 'success');
  }

  async importModel(): Promise<void> {
    const file = await pickFile('.json,application/json');
    if (!file) return;
    let raw: unknown;
    try {
      raw = JSON.parse(await file.text());
    } catch (error) {
      toast(`Not valid JSON: ${(error as Error).message}`, 'error', 5000);
      return;
    }
    const result = deserializeModel(raw);
    if (!result.ok) {
      console.error('[import]', result.errors);
      toast(`Cannot load ${file.name}: ${result.errors[0]}`, 'error', 6000);
      return;
    }
    this.host.store.set({ selection: null, hover: null, presetId: null });
    applyLoadedModel(this.model, result.model);
    toast(result.model.weights ? `Loaded ${file.name} (with weights)` : `Loaded configuration ${file.name} (fresh weights)`, 'success');
  }

  async copyConfig(): Promise<void> {
    try {
      await navigator.clipboard.writeText(formatConfig(specToConfig(this.model.spec)));
      toast('Configuration copied to the clipboard', 'success');
    } catch {
      toast('Clipboard not available in this context', 'warning');
    }
  }
}
