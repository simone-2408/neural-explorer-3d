/**
 * Application/UI state (immutable, held in a Store). The numeric models live
 * in their domain objects (NetworkModel, Trainer, TransformerModel…); this is
 * only *how the user is looking at them*.
 */
import type { ActivationScaleMode } from '../visualization/encoding';
import type { LayoutMode } from '../visualization/layout';
import type { ThemeId, WeightPaletteId } from '../visualization/palette';
import type { NetworkSelection } from '../visualization/selection';

export type WorkspaceId = 'network' | 'training' | 'cnn' | 'rnn' | 'transformer' | 'llm' | 'experiments';

/** Animated signal modes of the network views. */
export type SignalMode = 'off' | 'forward' | 'follow';

export interface ViewOptions {
  layout: LayoutMode;
  showPlates: boolean;
  showNeuronNames: boolean;
  showValues: boolean;
  colorByActivation: boolean;
  activationScale: ActivationScaleMode;
  dimUnrelated: boolean;
  thickness: number;
  showFloor: boolean;
  autoRotate: boolean;
  weightPalette: WeightPaletteId;
  theme: ThemeId;
}

export interface AppState {
  workspace: WorkspaceId;
  /** Selection in the network-based workspaces (Neural Network, Training). */
  selection: NetworkSelection | null;
  hover: NetworkSelection | null;
  view: ViewOptions;
  /** Preset currently loaded in the Neural Network tab, or null once the architecture has been edited. */
  presetId: string | null;
  signalMode: SignalMode;
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
}

export const DEFAULT_VIEW: ViewOptions = {
  layout: 'grid',
  showPlates: true,
  showNeuronNames: false,
  showValues: true,
  colorByActivation: true,
  activationScale: 'global',
  dimUnrelated: true,
  thickness: 1,
  showFloor: true,
  autoRotate: false,
  weightPalette: 'blue-orange',
  theme: 'dark',
};

const VIEW_KEY = 'ne3d.view';

/** View preferences are remembered per browser (best effort: storage may be unavailable). */
export function loadViewPreferences(): ViewOptions {
  const view = { ...DEFAULT_VIEW };
  try {
    const stored = JSON.parse(localStorage.getItem(VIEW_KEY) ?? 'null') as Partial<ViewOptions> | null;
    if (stored && typeof stored === 'object') {
      for (const key of Object.keys(DEFAULT_VIEW) as (keyof ViewOptions)[]) {
        if (typeof stored[key] === typeof DEFAULT_VIEW[key]) (view as Record<string, unknown>)[key] = stored[key];
      }
    } else if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
      view.theme = 'light';
    }
  } catch {
    /* ignore */
  }
  view.autoRotate = false;
  return view;
}

export function saveViewPreferences(view: ViewOptions): void {
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify(view));
  } catch {
    /* ignore */
  }
}

const WORKSPACE_IDS: WorkspaceId[] = ['network', 'training', 'cnn', 'rnn', 'transformer', 'llm', 'experiments'];

export function isWorkspaceId(value: unknown): value is WorkspaceId {
  return typeof value === 'string' && (WORKSPACE_IDS as string[]).includes(value);
}

export function createInitialState(presetId: string, workspace: WorkspaceId = 'network'): AppState {
  return {
    workspace,
    selection: null,
    hover: null,
    view: loadViewPreferences(),
    presetId,
    signalMode: 'off',
    leftPanelOpen: true,
    rightPanelOpen: true,
  };
}
