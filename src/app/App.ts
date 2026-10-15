/**
 * Composition root: creates the store, the shell, the shared viewport and
 * camera, and switches between workspaces (tabs). Workspace instances are
 * created lazily and kept alive, so each tab keeps its model and state; the
 * camera position is remembered per tab.
 */
import * as THREE from 'three';
import { Store } from '../core/Store';
import { NetworkModel } from '../neural/NetworkModel';
import { DEFAULT_PRESET_ID, findPreset } from '../neural/presets';
import { CameraController } from '../three/CameraController';
import { Viewport } from '../three/Viewport';
import { isTypingTarget } from '../ui/dom';
import { isModalOpen } from '../ui/overlays';
import { scenePalette } from '../visualization/palette';
import { NetworkWorkspace } from '../workspaces/network/NetworkWorkspace';
import { TrainingTab } from '../workspaces/training/TrainingTab';
import { TransformerWorkspace } from '../workspaces/transformer/TransformerWorkspace';
import { LlmWorkspace } from '../workspaces/llm/LlmWorkspace';
import { ExperimentsWorkspace } from '../workspaces/experiments/ExperimentsWorkspace';
import { CnnWorkspace } from '../workspaces/cnn/CnnWorkspace';
import { RnnWorkspace } from '../workspaces/rnn/RnnWorkspace';
import { openHelp } from './help';
import { buildShell } from './shell';
import { createInitialState, isWorkspaceId, saveViewPreferences, type AppState, type WorkspaceId } from './state';
import type { Workspace, WorkspaceHost } from './workspace';

interface CameraMemory {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

export class App {
  readonly store: Store<AppState>;
  private host: WorkspaceHost | null = null;
  private readonly instances = new Map<WorkspaceId, Workspace>();
  private active: Workspace | null = null;
  private readonly cameraMemory = new Map<WorkspaceId, CameraMemory>();
  private networkModel: NetworkModel | null = null;

  constructor(root: HTMLElement) {
    const hash = location.hash.slice(1);
    const initial: WorkspaceId = isWorkspaceId(hash) ? hash : 'network';
    this.store = new Store(createInitialState(DEFAULT_PRESET_ID, initial));
    const shell = buildShell(root, this.store, {
      onHelp: () => openHelp(this.store.state.workspace),
      onToggleTheme: () => this.toggleTheme(),
      onToggleEffects: () => this.toggleEffects(),
      onSelectWorkspace: (id) => this.switchTo(id),
      onOpenLessons: () => this.openLessons(),
    });
    this.applyTheme();
    this.store.watch(
      (s) => s.view,
      (view, previous) => {
        if (view.theme !== previous.theme || view.weightPalette !== previous.weightPalette || view.effects !== previous.effects) this.applyTheme();
        saveViewPreferences(view);
      },
    );

    let viewport: Viewport;
    try {
      viewport = new Viewport(shell.viewport, { atmosphere: true });
    } catch (error) {
      console.error(error);
      shell.showFatal('WebGL is not available', 'Neural Explorer 3D needs WebGL 2. Try a recent version of Chrome, Edge, Firefox or Safari, and check that hardware acceleration is enabled.');
      return;
    }
    const camera = new CameraController(viewport);
    this.host = {
      store: this.store,
      viewport,
      camera,
      slots: shell.slots,
      setSplit: (fraction) => shell.setSplit(fraction),
      switchWorkspace: (id) => this.switchTo(id),
    };
    this.applyTheme();
    this.mountWorkspace(initial);

    window.addEventListener('keydown', (event) => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || isModalOpen() || isTypingTarget(event.target)) return;
      if (event.key === '?') openHelp(this.store.state.workspace);
      else if (event.key === 't' || event.key === 'T') this.toggleTheme();
      else if (event.key === 'm' || event.key === 'M') this.toggleEffects();
      else if (event.key === '[') this.store.set((s) => ({ leftPanelOpen: !s.leftPanelOpen }));
      else if (event.key === ']') this.store.set((s) => ({ rightPanelOpen: !s.rightPanelOpen }));
      else return;
      event.preventDefault();
    });
    window.addEventListener('hashchange', () => {
      const id = location.hash.slice(1);
      if (isWorkspaceId(id)) this.switchTo(id);
    });

    // Debug handle for the browser console: models can be inspected and driven without the UI.
    const app = this;
    (window as unknown as Record<string, unknown>).neuralExplorer = {
      app,
      store: this.store,
      viewport,
      camera,
      get model() {
        return app.getNetworkModel();
      },
      get workspace() {
        return app.active;
      },
    };
  }

  /** The model of the Neural Network tab (created on first use). */
  getNetworkModel(): NetworkModel {
    if (!this.networkModel) {
      const preset = findPreset(DEFAULT_PRESET_ID)!;
      this.networkModel = NetworkModel.fromConfig(preset.config, { input: preset.input?.(), init: preset.init });
    }
    return this.networkModel;
  }

  /** Workspace instance for a tab (created on first use, not mounted). */
  getWorkspace(id: WorkspaceId): Workspace {
    let workspace = this.instances.get(id);
    if (!workspace) {
      workspace = this.create(id);
      this.instances.set(id, workspace);
    }
    return workspace;
  }

  switchTo(id: WorkspaceId): Workspace {
    if (this.active?.id === id) return this.active;
    if (!this.host) throw new Error('No viewport');
    if (this.active) {
      const { camera, controls } = this.host.viewport;
      this.cameraMemory.set(this.active.id, { position: camera.position.clone(), target: controls.target.clone() });
      this.active.unmount();
      this.host.setSplit(null);
      for (const slot of Object.values(this.host.slots)) slot.replaceChildren();
    }
    this.store.set({ workspace: id, selection: null, hover: null, signalMode: 'off' });
    if (location.hash.slice(1) !== id) history.replaceState(null, '', `#${id}`);
    return this.mountWorkspace(id);
  }

  private mountWorkspace(id: WorkspaceId): Workspace {
    const host = this.host!;
    const workspace = this.getWorkspace(id);
    workspace.mount(host);
    this.active = workspace;
    const memory = this.cameraMemory.get(id);
    if (memory) {
      const { camera, controls } = host.viewport;
      host.viewport.animations.cancelChannel('camera');
      camera.position.copy(memory.position);
      controls.target.copy(memory.target);
      camera.lookAt(memory.target);
      controls.update();
      host.viewport.requestRender();
    }
    return workspace;
  }

  private create(id: WorkspaceId): Workspace {
    switch (id) {
      case 'network':
        return new NetworkWorkspace(this.getNetworkModel());
      case 'training':
        return new TrainingTab();
      case 'cnn':
        return new CnnWorkspace();
      case 'rnn':
        return new RnnWorkspace();
      case 'transformer':
        return new TransformerWorkspace();
      case 'llm':
        return new LlmWorkspace();
      case 'experiments':
        return new ExperimentsWorkspace();
    }
  }

  private openLessons(): void {
    void import('../lessons/LessonCatalog').then(({ openLessonCatalog }) => openLessonCatalog(this));
  }

  private toggleEffects(): void {
    this.store.set((s) => ({ view: { ...s.view, effects: !s.view.effects } }));
  }

  private toggleTheme(): void {
    this.store.set((s) => ({ view: { ...s.view, theme: s.view.theme === 'dark' ? 'light' : 'dark' } }));
  }

  /** Theme and sign colours are shared by CSS (panels, charts) and the 3D scenes. */
  private applyTheme(): void {
    const { theme, weightPalette } = this.store.state.view;
    const palette = scenePalette(theme, weightPalette);
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.style.setProperty('--pos', palette.positive);
    root.style.setProperty('--neg', palette.negative);
    root.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#0b0f15' : '#f3f5f8');
    this.host?.viewport.setEffects(this.store.state.view.effects, {
      theme,
      background: palette.background,
      grid: palette.floorGrid,
    });
  }
}
