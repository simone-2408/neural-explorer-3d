/**
 * NetworkScene — the interactive 3D network shared by the network-based
 * workspaces (Neural Network, Training): NetworkView + picking + hover tooltip
 * + keyboard navigation + camera actions + inspector switching.
 *
 * Wiring only: the maths lives in src/neural, the drawing in src/three.
 *
 *   model events ──► view.update*() + inspector.refresh()
 *   store changes ─► view.setSelection / setHover / setOptions / setPalette
 *   pointer/keys ──► actions ──► store
 */
import type { ViewOptions } from '../../app/state';
import type { WorkspaceHost } from '../../app/workspace';
import { DisposeBag } from '../../core/DisposeBag';
import type { ChangeReason, NetworkModel } from '../../neural/NetworkModel';
import type { CameraView } from '../../three/CameraController';
import { NetworkView, type NetworkViewOptions } from '../../three/NetworkView';
import { isTypingTarget } from '../../ui/dom';
import { InspectorPanel, type InspectorView } from '../../ui/InspectorPanel';
import { isModalOpen, Tooltip } from '../../ui/overlays';
import { frameThrottle } from '../../utils/schedule';
import { scenePalette, type ScenePalette } from '../../visualization/palette';
import { isSelectionValid, selectionEquals, type NetworkSelection } from '../../visualization/selection';
import type { NavigateDirection, SceneActions } from './context';
import { describeTooltip } from './describe';

export function toViewOptions(view: ViewOptions): NetworkViewOptions {
  return {
    layout: view.layout,
    showPlates: view.showPlates,
    showNeuronNames: view.showNeuronNames,
    showValues: view.showValues,
    colorByActivation: view.colorByActivation,
    activationScale: view.activationScale,
    dimUnrelated: view.dimUnrelated,
    thickness: view.thickness,
    showFloor: view.showFloor,
  };
}

export interface NetworkSceneOptions {
  host: WorkspaceHost;
  model: NetworkModel;
  /** Builds the inspector view for a selection (null = overview). */
  inspect(selection: NetworkSelection | null): InspectorView;
  /** Workspace-specific keys, checked before the scene's own shortcuts. Return true if handled. */
  onKey?(event: KeyboardEvent): boolean;
  /** Called after the view has been rebuilt for a new architecture. */
  onRebuilt?(reason: ChangeReason): void;
}

export class NetworkScene implements SceneActions {
  readonly view: NetworkView;
  readonly inspector: InspectorPanel;
  readonly model: NetworkModel;
  private readonly host: WorkspaceHost;
  private readonly options: NetworkSceneOptions;
  private readonly tooltip: Tooltip;
  private readonly bag = new DisposeBag();
  private paletteValue: ScenePalette;
  /** Pointer/keyboard interaction can be suspended (e.g. while a lesson drives the scene). */
  interactive = true;

  constructor(options: NetworkSceneOptions) {
    this.options = options;
    this.host = options.host;
    this.model = options.model;
    const { store, viewport, camera, slots } = this.host;
    const viewState = store.state.view;
    this.paletteValue = scenePalette(viewState.theme, viewState.weightPalette);

    this.view = new NetworkView({
      viewport,
      model: this.model,
      palette: this.paletteValue,
      options: toViewOptions(viewState),
      onLayerLabelClick: (layer) => this.select({ kind: 'layer', layer }),
    });
    this.view.build();
    viewport.setBackground(this.paletteValue.background);
    viewport.controls.autoRotate = viewState.autoRotate;
    this.updateCameraBounds();
    camera.reset(false);

    this.tooltip = new Tooltip(slots.stage);
    this.inspector = new InspectorPanel(slots.right);

    this.bag.add(this.model.events.on('structure', ({ reason }) => this.onStructure(reason)));
    this.bag.add(this.model.events.on('params', () => this.view.updateParams()));
    this.bag.add(
      this.model.events.on('activations', () => {
        this.view.updateActivations();
        this.inspector.refresh();
      }),
    );
    this.bag.add(
      store.watch(
        (s) => s.selection,
        (selection) => {
          this.view.setSelection(selection);
          this.showInspector();
        },
        selectionEquals,
      ),
    );
    this.bag.add(store.watch((s) => s.hover, (hover) => this.view.setHover(hover), selectionEquals));
    this.bag.add(store.watch((s) => s.view, (next, prev) => this.onViewChange(next, prev)));

    this.bindPointer();
    this.bindKeyboard();
  }

  get palette(): ScenePalette {
    return this.paletteValue;
  }

  dispose(): void {
    this.bag.dispose();
    this.inspector.dispose();
    this.tooltip.el.remove();
    this.view.dispose();
  }

  /** Rebuilds the inspector for the current selection. */
  showInspector(keepScroll = false): void {
    this.inspector.show(this.options.inspect(this.host.store.state.selection), keepScroll);
  }

  updateCameraBounds(): void {
    const layout = this.view.getLayout();
    this.host.camera.setBounds(layout.boundsMin, layout.boundsMax);
  }

  // ------------------------------------------------------------ reactions

  private onStructure(reason: ChangeReason): void {
    const before = this.view.getLayout().radius;
    this.view.build();
    const { store, camera } = this.host;
    const selection = store.state.selection;
    if (selection && !isSelectionValid(selection, this.model.spec)) store.set({ selection: null });
    else this.view.setSelection(selection);
    store.set({ hover: null });
    this.view.setHover(null);
    this.tooltip.hide();
    this.updateCameraBounds();
    const after = this.view.getLayout().radius;
    if (reason === 'preset' || reason === 'import' || Math.abs(after - before) / before > 0.3) camera.reset(true);
    this.showInspector(true);
    this.options.onRebuilt?.(reason);
  }

  private onViewChange(next: ViewOptions, prev: ViewOptions): void {
    const { viewport } = this.host;
    this.view.setOptions(toViewOptions(next));
    if (next.theme !== prev.theme || next.weightPalette !== prev.weightPalette) {
      this.paletteValue = scenePalette(next.theme, next.weightPalette);
      this.view.setPalette(this.paletteValue);
      viewport.setBackground(this.paletteValue.background);
      this.showInspector(true); // charts use palette colours
    }
    if (next.autoRotate !== prev.autoRotate) viewport.controls.autoRotate = next.autoRotate;
    if (next.layout !== prev.layout) this.updateCameraBounds();
  }

  // ---------------------------------------------------------------- input

  private bindPointer(): void {
    const { viewport } = this.host;
    const canvas = viewport.renderer.domElement;
    let down: { x: number; y: number } | null = null;
    let last: { clientX: number; clientY: number } | null = null;

    const hoverAt = frameThrottle(() => {
      if (!last || down || !this.interactive) return;
      const { x, y } = viewport.toCanvas(last.clientX, last.clientY);
      const hit = this.view.pick(x, y);
      this.setHover(hit);
      canvas.style.cursor = hit ? 'pointer' : '';
      if (hit) this.tooltip.show(x + this.stageOffset(), y, describeTooltip(this.model, hit));
      else this.tooltip.hide();
    });
    this.bag.add(() => hoverAt.cancel());

    this.bag.listen(canvas, 'pointermove', (event) => {
      last = { clientX: event.clientX, clientY: event.clientY };
      if (down) {
        this.tooltip.hide();
        return;
      }
      hoverAt();
    });
    this.bag.listen(canvas, 'pointerdown', (event) => {
      down = { x: event.clientX, y: event.clientY };
      this.tooltip.hide();
    });
    this.bag.listen(canvas, 'pointerup', (event) => {
      if (!down) return;
      const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
      down = null;
      if (moved < 5 && event.button === 0 && this.interactive) {
        const { x, y } = viewport.toCanvas(event.clientX, event.clientY);
        this.select(this.view.pick(x, y));
      }
      hoverAt();
    });
    this.bag.listen(canvas, 'pointercancel', () => (down = null));
    this.bag.listen(canvas, 'pointerleave', () => {
      last = null;
      this.setHover(null);
      this.tooltip.hide();
      canvas.style.cursor = '';
    });
    this.bag.listen(canvas, 'dblclick', () => {
      if (this.host.store.state.selection) this.focusSelection();
    });
  }

  /** The canvas may start to the right of the stage overlay when the stage is split. */
  private stageOffset(): number {
    const canvas = this.host.viewport.renderer.domElement.getBoundingClientRect();
    const stage = this.host.slots.stage.getBoundingClientRect();
    return canvas.left - stage.left;
  }

  private bindKeyboard(): void {
    this.bag.listen(window, 'keydown', (event) => {
      const e = event as KeyboardEvent;
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isModalOpen() || isTypingTarget(e.target)) return;
      if (e.key.startsWith('Arrow') && e.target instanceof HTMLInputElement && e.target.type === 'range') return;
      if (this.options.onKey?.(e)) {
        e.preventDefault();
        return;
      }
      const view = this.host.store.state.view;
      switch (e.key) {
        case 'Escape':
          if (!this.host.store.state.selection) return;
          this.select(null);
          break;
        case 'r':
        case 'R':
          this.resetCamera();
          break;
        case 'f':
        case 'F':
          this.focusSelection();
          break;
        case 'l':
        case 'L':
          this.faceLayer();
          break;
        case '1':
        case '2':
        case '3':
        case '4':
          this.setCameraView((['iso', 'side', 'top', 'front'] as const)[Number(e.key) - 1]);
          break;
        case 'a':
        case 'A':
          this.setView({ autoRotate: !view.autoRotate });
          break;
        case 'v':
        case 'V':
          this.setView({ showValues: !view.showValues });
          break;
        case 'n':
        case 'N':
          this.setView({ showNeuronNames: !view.showNeuronNames });
          break;
        case 'g':
        case 'G': {
          const order = ['grid', 'column', 'ring'] as const;
          this.setView({ layout: order[(order.indexOf(view.layout) + 1) % order.length] });
          break;
        }
        case 'ArrowLeft':
          this.navigate('left');
          break;
        case 'ArrowRight':
          this.navigate('right');
          break;
        case 'ArrowUp':
          this.navigate('up');
          break;
        case 'ArrowDown':
          this.navigate('down');
          break;
        default:
          return;
      }
      e.preventDefault();
    });
  }

  // -------------------------------------------------------------- actions

  select(selection: NetworkSelection | null): void {
    this.host.store.set({ selection });
  }

  setHover(selection: NetworkSelection | null): void {
    this.host.store.set({ hover: selection });
  }

  focusSelection(): void {
    const selection = this.host.store.state.selection;
    if (!selection) {
      this.resetCamera();
      return;
    }
    const { center, radius } = this.view.focusTarget(selection);
    this.host.camera.focus(center, radius);
  }

  resetCamera(): void {
    this.host.camera.reset(true);
  }

  setCameraView(view: CameraView): void {
    this.host.camera.setView(view, true);
  }

  faceLayer(layer?: number): void {
    const selection = this.host.store.state.selection;
    const target = layer ?? selection?.layer ?? 0;
    const frame = this.view.frame(target);
    const o = this.view.root.position;
    this.host.camera.faceLayer([frame.center[0] + o.x, frame.center[1] + o.y, frame.center[2] + o.z], frame.halfWidth, frame.halfHeight);
  }

  navigate(direction: NavigateDirection): void {
    const sizes = this.model.topology.sizes;
    const selection = this.host.store.state.selection;
    const last = sizes.length - 1;
    // Map an index proportionally between layers of different sizes.
    const mapIndex = (index: number, from: number, to: number) => (sizes[from] <= 1 ? 0 : Math.round((index / (sizes[from] - 1)) * (sizes[to] - 1)));
    if (!selection) {
      this.select({ kind: 'neuron', layer: 0, index: 0 });
      return;
    }
    switch (selection.kind) {
      case 'neuron': {
        const { layer, index } = selection;
        const n = sizes[layer];
        if (direction === 'up') this.select({ kind: 'neuron', layer, index: (index - 1 + n) % n });
        else if (direction === 'down') this.select({ kind: 'neuron', layer, index: (index + 1) % n });
        else {
          const to = direction === 'left' ? Math.max(0, layer - 1) : Math.min(last, layer + 1);
          this.select({ kind: 'neuron', layer: to, index: mapIndex(index, layer, to) });
        }
        return;
      }
      case 'layer': {
        const { layer } = selection;
        if (direction === 'left') this.select({ kind: 'layer', layer: Math.max(0, layer - 1) });
        else if (direction === 'right') this.select({ kind: 'layer', layer: Math.min(last, layer + 1) });
        else this.select({ kind: 'neuron', layer, index: direction === 'down' ? 0 : sizes[layer] - 1 });
        return;
      }
      case 'connection': {
        const { layer, target, source } = selection;
        const fanIn = sizes[layer - 1];
        if (direction === 'left') this.select({ kind: 'neuron', layer: layer - 1, index: source });
        else if (direction === 'right') this.select({ kind: 'neuron', layer, index: target });
        else this.select({ kind: 'connection', layer, target, source: (source + (direction === 'down' ? 1 : fanIn - 1)) % fanIn });
        return;
      }
    }
  }

  setView(patch: Partial<ViewOptions>): void {
    this.host.store.set((s) => ({ view: { ...s.view, ...patch } }));
  }
}
