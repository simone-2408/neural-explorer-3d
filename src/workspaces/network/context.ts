/**
 * Contracts between the network-based workspaces (Neural Network, Training)
 * and their panels/inspectors. Panels never talk to three.js: they read the
 * model/store and call actions.
 */
import type { AppState, ViewOptions } from '../../app/state';
import type { Store } from '../../core/Store';
import type { ActivationId } from '../../math/activations';
import type { NetworkConfig } from '../../neural/config';
import type { ChangeReason, NetworkModel } from '../../neural/NetworkModel';
import type { CameraView } from '../../three/CameraController';
import type { ScenePalette } from '../../visualization/palette';
import type { NetworkSelection } from '../../visualization/selection';

export type NavigateDirection = 'left' | 'right' | 'up' | 'down';

/** Selection and camera — provided by NetworkScene in every network-based workspace. */
export interface SceneActions {
  select(selection: NetworkSelection | null): void;
  setHover(selection: NetworkSelection | null): void;
  focusSelection(): void;
  resetCamera(): void;
  setCameraView(view: CameraView): void;
  /** Face-on view of a layer (default: the layer of the current selection). */
  faceLayer(layer?: number): void;
  navigate(direction: NavigateDirection): void;
  setView(patch: Partial<ViewOptions>): void;
}

/** Animated signal features (Neural Network workspace). */
export interface SignalActions {
  /** Enters forward-pass mode (step 0) or leaves it. */
  toggleForward(): void;
  /** Follow the signal from the current selection (or stop following). */
  toggleFollow(): void;
  openActivationExplorer(activation?: ActivationId, z?: number): void;
}

/** Architecture editing (Neural Network workspace). */
export interface ArchitectureActions {
  applyPreset(id: string): void;
  /** Validates and applies a configuration; returns error messages (empty when applied). */
  applyConfig(config: NetworkConfig, reason?: ChangeReason): string[];
  setLayerSize(layer: number, size: number): void;
  setLayerActivation(layer: number, activation: ActivationId): void;
  addHiddenLayer(): void;
  removeLayer(layer: number): void;
  exportModel(): void;
  importModel(): Promise<void>;
  copyConfig(): Promise<void>;
}

/** Training information shown by the inspectors in the Training workspace. */
export interface TrainingInspectorInfo {
  /** ∂L/∂W and ∂L/∂b of the last step (batch average), or null before the first step. */
  gradients(): { dW: readonly Float64Array[]; db: readonly Float64Array[]; delta: readonly Float64Array[] } | null;
  /** Weights before the last update (same layout as the model weights), or null. */
  previousWeights(): readonly Float64Array[] | null;
  previousBiases(): readonly Float64Array[] | null;
  learningRate(): number;
  optimizerLabel(): string;
  /** TeX of the update rule of the optimizer. */
  updateRuleTex(): string;
  /** Loss of the current sample/batch and its formula. */
  lossInfo(): { value: number; tex: string; label: string } | null;
}

export interface NetworkContext {
  model: NetworkModel;
  store: Store<AppState>;
  actions: SceneActions & Partial<SignalActions>;
  /** Current palette, so 2D charts use the same colours as the 3D scene. */
  palette(): ScenePalette;
  training?: TrainingInspectorInfo;
}

/** Context of the Neural Network workspace panels. */
export interface NetworkWorkspaceContext extends NetworkContext {
  actions: SceneActions & SignalActions & ArchitectureActions;
}

/** Common shape of left-panel sections and bars owned by a workspace. */
export interface Panel {
  readonly el: HTMLElement;
  dispose(): void;
}
