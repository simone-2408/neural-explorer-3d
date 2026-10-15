/**
 * Workspaces are the top-level tabs (Neural Network, Training, Transformer…).
 * Each one plugs its own scene objects, panels and inspectors into the shared
 * shell (viewport, camera, store, panel slots) when mounted, and removes them
 * when unmounted. Workspace instances are kept alive between switches, so
 * their models and state survive.
 */
import type { Store } from '../core/Store';
import type { CameraController } from '../three/CameraController';
import type { Viewport } from '../three/Viewport';
import type { AppState, WorkspaceId } from './state';

export interface WorkspaceSlots {
  /** Left panel: controls, architecture, parameters. */
  left: HTMLElement;
  /** Right panel: inspector. */
  right: HTMLElement;
  /** Bottom bar: layer selector, legend, timeline. */
  bottom: HTMLElement;
  /** Overlay on top of the 3D viewport (toolbars, tooltips). */
  stage: HTMLElement;
  /** 2D area on the left of the stage, visible when the stage is split (see WorkspaceHost.setSplit). */
  side: HTMLElement;
}

/** Bounds in CSS pixels for the width of the 2D side of a split stage. */
export interface SplitLimits {
  min?: number;
  max?: number;
}

export interface WorkspaceHost {
  store: Store<AppState>;
  viewport: Viewport;
  camera: CameraController;
  slots: WorkspaceSlots;
  /**
   * Splits the stage: `fraction` of the width goes to `slots.side`, the rest to the 3D view (null = no split).
   * `limits` clamp the side width, so the 3D view keeps the extra room on wide stages.
   */
  setSplit(fraction: number | null, limits?: SplitLimits): void;
  /** Switches to another workspace (used by lessons and cross-links); resolves with the mounted workspace. */
  switchWorkspace(id: WorkspaceId): Workspace;
}

export interface Workspace {
  readonly id: WorkspaceId;
  mount(host: WorkspaceHost): void;
  unmount(): void;
}

export interface WorkspaceEntry {
  id: WorkspaceId | 'learn';
  label: string;
  summary: string;
}

export const WORKSPACES: WorkspaceEntry[] = [
  { id: 'network', label: 'Neural Network', summary: '3D explorer of a feed-forward network: forward pass, follow the signal, activation functions.' },
  { id: 'training', label: 'Training', summary: 'Train a network live: datasets, loss, backpropagation, gradients, decision boundary, loss landscape, comparisons.' },
  { id: 'cnn', label: 'CNN', summary: 'A small convolutional network trained live on images: filters, feature maps, pooling, receptive fields.' },
  { id: 'rnn', label: 'RNN', summary: 'A recurrent network (vanilla or GRU) unrolled through time: hidden state, shared weights, backpropagation through time, vanishing gradients.' },
  { id: 'transformer', label: 'Transformer', summary: 'A tiny trained Transformer computed step by step: tokens, embeddings, attention, residual stream, MLP.' },
  { id: 'llm', label: 'LLM', summary: 'Inspect a small real language model (GPT-2 small) through a local Python backend.' },
  { id: 'experiments', label: 'Experiments', summary: 'Controlled interventions: ablate heads and neurons, edit activations and embeddings, compare outputs.' },
  { id: 'learn', label: 'Learn', summary: 'Interactive lessons that drive the live scenes.' },
];
