/**
 * NetworkView — draws a NetworkModel with three.js. It *reads* the model and
 * never modifies it: every visual property is derived from model state via
 * the encodings in src/visualization.
 *
 * Scene structure (one draw call per element type, whatever the network size):
 *   root
 *   ├─ connections  InstancedMesh  (1 instance per weight, unit cylinder)
 *   ├─ neurons      InstancedMesh  (1 instance per neuron, unit sphere)
 *   ├─ plates       one translucent rounded frame per layer (clickable)
 *   ├─ floor        reference grid + "signal flow" arrow
 *   ├─ labels       CSS2D HTML labels (layer titles, optional neuron values)
 *   └─ markers      selection ring (neuron) / direction cone (connection)
 *
 * Update paths are granular so nothing is rebuilt per frame:
 *   build()             structure changed (new meshes)
 *   updateParams()      weights changed   (connection colours + thickness)
 *   updateActivations() forward changed   (neuron colours + size, labels, pulse strengths)
 *   setSelection()      selection changed (emphasis attributes)
 *   setHover()          hover changed     (one uniform per mesh)
 *   setSignalState()    forward/backward animation (uniforms only)
 *   setFlow()           follow-the-signal scores (one attribute upload)
 *
 * What the lines and spheres encode can be switched from weights/activations
 * to other per-element values (e.g. gradients and δ during training) with
 * setConnectionValues() / setNeuronValues().
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { easeInOutCubic, easeOutBack } from '../animation/easing';
import { ACTIVATIONS } from '../math/activations';
import { clamp, maxAbs } from '../math/stats';
import type { NetworkModel } from '../neural/NetworkModel';
import type { FlowScores } from '../neural/signalFlow';
import { neuronName } from '../neural/topology';
import { formatValue } from '../utils/format';
import { computeEmphasis, createEmphasis, EMPHASIS, type Emphasis } from '../visualization/emphasis';
import {
  activationScales,
  connectionRadius,
  magnitude,
  visibleIntensity,
  type ActivationScaleMode,
} from '../visualization/encoding';
import { computeLayout, type LayerFrame, type LayoutMode, type NetworkLayout, type Vec3 } from '../visualization/layout';
import type { ScenePalette } from '../visualization/palette';
import type { NetworkSelection } from '../visualization/selection';
import { chooseLabelDetail, LABEL_DETAILS, type LabelDetail, type LabelSize } from './labelDetail';
import { createConnectionMaterial, createNeuronMaterial, REVEAL_ALL } from './materials';
import { pointSegmentDistance, projectPoints } from './picking';
import type { Viewport } from './Viewport';

export interface NetworkViewOptions {
  layout: LayoutMode;
  showPlates: boolean;
  showNeuronNames: boolean;
  showValues: boolean;
  colorByActivation: boolean;
  activationScale: ActivationScaleMode;
  dimUnrelated: boolean;
  /** Visual multiplier for connection thickness. */
  thickness: number;
  showFloor: boolean;
}

/** Which layers already show their value during a forward (direction 1) or backward (−1) animation. */
export interface RevealState {
  layer: number;
  /** Progress of `layer` itself (0 = empty, 1 = shown). */
  t: number;
  direction: 1 | -1;
}

/** A pulse travelling along the connections that enter layer `layer`. */
export interface PulseState {
  layer: number;
  t: number;
  /** 1: source → target (forward), −1: target → source (backward). */
  direction: 1 | -1;
  /** 0 = invisible … 1 = full. */
  intensity: number;
}

/** Custom values encoded by line thickness/colour instead of the weights (e.g. ∂L/∂W). */
export interface ConnectionValues {
  values: readonly ArrayLike<number>[];
  /** Logarithmic magnitude scale (shows values spanning many orders of magnitude, e.g. vanishing gradients). */
  log?: boolean;
}

/** Per-neuron HTML labels are only created up to this many neurons. */
export const MAX_NEURON_LABELS = 160;

const UP = new THREE.Vector3(0, 1, 0);
const LOG_DECADES = 5;

interface NeuronLabel {
  object: CSS2DObject;
  element: HTMLElement;
  name: HTMLElement;
  value: HTMLElement;
  layer: number;
}

interface Plate {
  group: THREE.Group;
  fill: THREE.Mesh<THREE.ShapeGeometry, THREE.MeshBasicMaterial>;
  border: THREE.LineLoop<THREE.BufferGeometry, THREE.LineBasicMaterial>;
}

function roundedRect(halfWidth: number, halfHeight: number, radius: number): THREE.Shape {
  const r = Math.min(radius, halfWidth, halfHeight);
  const s = new THREE.Shape();
  s.moveTo(-halfWidth + r, -halfHeight);
  s.lineTo(halfWidth - r, -halfHeight);
  s.absarc(halfWidth - r, -halfHeight + r, r, -Math.PI / 2, 0, false);
  s.lineTo(halfWidth, halfHeight - r);
  s.absarc(halfWidth - r, halfHeight - r, r, 0, Math.PI / 2, false);
  s.lineTo(-halfWidth + r, halfHeight);
  s.absarc(-halfWidth + r, halfHeight - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(-halfWidth, -halfHeight + r);
  s.absarc(-halfWidth + r, -halfHeight + r, r, Math.PI, Math.PI * 1.5, false);
  return s;
}

export class NetworkView {
  readonly root = new THREE.Group();

  private readonly viewport: Viewport;
  private readonly model: NetworkModel;
  private readonly onLayerLabelClick: (layer: number) => void;
  private palette: ScenePalette;
  private options: NetworkViewOptions;

  private layout!: NetworkLayout;
  /** Neuron positions as currently drawn (they differ from `layout` during a layout transition). */
  private positions = new Float32Array(0);
  /** Base neuron radii as currently drawn. */
  private baseRadii = new Float32Array(0);
  /** Size multiplier from the activation encoding. */
  private boost = new Float32Array(0);
  private connectionRadii = new Float32Array(0);
  private projected = new Float32Array(0);
  private emphasis: Emphasis = { neurons: new Uint8Array(0), connections: new Uint8Array(0) };

  private readonly neuronMaterial = createNeuronMaterial();
  private readonly connectionMaterial = createConnectionMaterial();
  private neuronMesh: THREE.InstancedMesh | null = null;
  private connectionMesh: THREE.InstancedMesh | null = null;
  private neuronColor = new THREE.InstancedBufferAttribute(new Float32Array(3), 3);
  private neuronState = new THREE.InstancedBufferAttribute(new Float32Array(3), 3);
  private connectionColor = new THREE.InstancedBufferAttribute(new Float32Array(3), 3);
  private connectionStyle = new THREE.InstancedBufferAttribute(new Float32Array(4), 4);
  private connectionSignal = new THREE.InstancedBufferAttribute(new Float32Array(3), 3);

  private connectionValues: ConnectionValues | null = null;
  private neuronValues: readonly ArrayLike<number>[] | null = null;
  private pulseValues: readonly ArrayLike<number>[] | null = null;
  private grid: THREE.GridHelper | null = null;
  private readonly removeEffectsListener: () => void;
  private flow: FlowScores | null = null;
  private reveal: RevealState | null = null;

  private readonly plateGroup = new THREE.Group();
  private plates: Plate[] = [];
  private plateFade = 1;
  private readonly floor = new THREE.Group();
  private readonly labelGroup = new THREE.Group();
  private layerLabels: CSS2DObject[] = [];
  private neuronLabels: NeuronLabel[] = [];
  /** Level of detail of the layer titles (see labelDetail.ts); only changes with `compactLabels`. */
  private labelDetail: LabelDetail = 0;
  /** Largest layer title at each level of detail, measured once the titles are on the page. */
  private labelSizes: Record<LabelDetail, LabelSize> | null = null;
  private labelAnchors = new Float32Array(0);
  private labelPoints = new Float32Array(0);

  private readonly ring: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  private readonly arrow: THREE.Mesh<THREE.ConeGeometry, THREE.MeshBasicMaterial>;
  private ringPop = 1;

  private selection: NetworkSelection | null = null;
  private hover: NetworkSelection | null = null;
  private readonly removeBeforeRender: () => void;
  private readonly removeAfterRender: () => void;

  // Scratch objects (avoid allocations in hot loops).
  private readonly m4 = new THREE.Matrix4();
  private readonly vp = new THREE.Matrix4();
  private readonly va = new THREE.Vector3();
  private readonly vb = new THREE.Vector3();
  private readonly vc = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly identity = new THREE.Quaternion();
  private readonly raycaster = new THREE.Raycaster();
  private readonly colorScratch = { pos: new THREE.Color(), neg: new THREE.Color(), off: new THREE.Color(), tmp: new THREE.Color() };

  constructor(params: {
    viewport: Viewport;
    model: NetworkModel;
    palette: ScenePalette;
    options: NetworkViewOptions;
    onLayerLabelClick?: (layer: number) => void;
    /** Translation of the whole network (used to show two networks side by side). */
    offset?: Vec3;
    /** Shorten the layer titles when neighbouring layers are too close on screen to show them in full. */
    compactLabels?: boolean;
  }) {
    this.viewport = params.viewport;
    this.model = params.model;
    this.palette = params.palette;
    this.options = { ...params.options };
    this.onLayerLabelClick = params.onLayerLabelClick ?? (() => undefined);

    this.ring = new THREE.Mesh(
      new THREE.RingGeometry(1.32, 1.58, 64),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthWrite: false }),
    );
    this.ring.renderOrder = 3;
    this.ring.visible = false;
    this.arrow = new THREE.Mesh(new THREE.ConeGeometry(1, 2.4, 20), new THREE.MeshBasicMaterial());
    this.arrow.visible = false;

    this.root.name = 'NetworkView';
    if (params.offset) this.root.position.set(...params.offset);
    this.root.add(this.plateGroup, this.floor, this.labelGroup, this.ring, this.arrow);
    this.viewport.scene.add(this.root);
    this.removeBeforeRender = this.viewport.onBeforeRender((camera) => {
      if (this.ring.visible) this.ring.quaternion.copy(camera.quaternion);
    });
    // After the label pass: the titles are on the page (measurable) and the world matrices are current.
    this.removeAfterRender = params.compactLabels ? this.viewport.onAfterRender(() => this.updateLabelDetail()) : () => undefined;
    // With the visual effects the atmosphere draws the floor grid.
    this.removeEffectsListener = this.viewport.onEffectsChange((on) => {
      if (this.grid) this.grid.visible = !on;
    });
    this.applyPaletteUniforms();
  }

  // ---------------------------------------------------------------- queries

  getLayout(): NetworkLayout {
    return this.layout;
  }

  frame(layer: number): LayerFrame {
    return this.layout.frames[layer];
  }

  /** World position of a neuron (includes the view offset). */
  neuronWorldPosition(layer: number, index: number): Vec3 {
    const p = this.positionOf(this.model.topology.neuronIndex(layer, index));
    return [p[0] + this.root.position.x, p[1] + this.root.position.y, p[2] + this.root.position.z];
  }

  /** World-space bounding sphere to frame when focusing on a selection (or the whole network). */
  focusTarget(selection: NetworkSelection | null): { center: Vec3; radius: number } {
    const { topology } = this.model;
    const o = this.root.position;
    const shift = (v: Vec3): Vec3 => [v[0] + o.x, v[1] + o.y, v[2] + o.z];
    if (!selection) return { center: shift(this.layout.center), radius: this.layout.radius };
    switch (selection.kind) {
      case 'neuron': {
        const g = topology.neuronIndex(selection.layer, selection.index);
        return { center: shift(this.positionOf(g)), radius: Math.max(this.layout.layerGap * 0.85, 1.4) };
      }
      case 'connection': {
        const a = this.positionOf(topology.neuronIndex(selection.layer - 1, selection.source));
        const b = this.positionOf(topology.neuronIndex(selection.layer, selection.target));
        const center: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
        return { center: shift(center), radius: Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / 2 + 0.8 };
      }
      case 'layer': {
        const f = this.layout.frames[selection.layer];
        return { center: shift(f.center), radius: Math.hypot(f.halfWidth, f.halfHeight) * 1.05 };
      }
    }
  }

  // --------------------------------------------------------------- building

  /** Full rebuild — call when the architecture changes. */
  build(): void {
    this.viewport.animations.cancelChannel('layout');
    const { topology } = this.model;
    this.layout = computeLayout(this.model.spec, topology, { mode: this.options.layout });
    this.positions = this.layout.positions.slice();
    this.baseRadii = this.layout.radii.slice();
    this.boost = new Float32Array(topology.neuronCount).fill(1);
    this.connectionRadii = new Float32Array(topology.connectionCount);
    this.projected = new Float32Array(topology.neuronCount * 3);
    this.emphasis = createEmphasis(topology);
    this.plateFade = 1;
    // Value overrides refer to the old shapes.
    this.connectionValues = null;
    this.neuronValues = null;
    this.pulseValues = null;
    this.flow = null;

    this.disposeMeshes();

    const neurons = topology.neuronCount;
    const [widthSegments, heightSegments] = neurons <= 150 ? [32, 20] : neurons <= 600 ? [16, 12] : [10, 7];
    const sphere = new THREE.SphereGeometry(1, widthSegments, heightSegments);
    this.neuronColor = new THREE.InstancedBufferAttribute(new Float32Array(neurons * 3), 3);
    this.neuronState = new THREE.InstancedBufferAttribute(new Float32Array(neurons * 3), 3);
    sphere.setAttribute('aColor', this.neuronColor);
    sphere.setAttribute('aState', this.neuronState);
    this.neuronMesh = new THREE.InstancedMesh(sphere, this.neuronMaterial, neurons);
    this.neuronMesh.name = 'neurons';
    this.neuronMesh.frustumCulled = false;
    this.neuronMesh.renderOrder = 2;

    const connections = topology.connectionCount;
    const radialSegments = connections <= 2_000 ? 8 : connections <= 15_000 ? 5 : 3;
    const cylinder = new THREE.CylinderGeometry(1, 1, 1, radialSegments, 1, true);
    this.connectionColor = new THREE.InstancedBufferAttribute(new Float32Array(connections * 3), 3);
    this.connectionStyle = new THREE.InstancedBufferAttribute(new Float32Array(connections * 4), 4);
    this.connectionSignal = new THREE.InstancedBufferAttribute(new Float32Array(connections * 3), 3);
    cylinder.setAttribute('aColor', this.connectionColor);
    cylinder.setAttribute('aStyle', this.connectionStyle);
    cylinder.setAttribute('aSignal', this.connectionSignal);
    this.connectionMesh = new THREE.InstancedMesh(cylinder, this.connectionMaterial, connections);
    this.connectionMesh.name = 'connections';
    this.connectionMesh.frustumCulled = false;
    this.connectionMesh.renderOrder = 1;
    this.connectionMaterial.uniforms.uAlphaScale.value = clamp(1.25 - Math.log10(Math.max(connections, 1)) / 4.5, 0.28, 1);
    this.connectionMaterial.uniforms.uDashSize.value = clamp(this.layout.layerGap * 0.05, 0.08, 0.3);

    // Static per-instance data: layer index of every neuron / connection, flow = 1.
    const state = this.neuronState.array as Float32Array;
    for (let g = 0; g < neurons; g++) {
      state[3 * g + 1] = topology.neuronAt(g).layer;
      state[3 * g + 2] = 1;
    }
    const signal = this.connectionSignal.array as Float32Array;
    for (let l = 1; l < topology.sizes.length; l++) {
      for (let c = topology.connectionOffsets[l]; c < topology.connectionOffsets[l + 1]; c++) {
        signal[3 * c] = l;
        signal[3 * c + 2] = 1;
      }
    }

    this.root.add(this.connectionMesh, this.neuronMesh);

    this.buildPlates();
    this.buildFloor();
    this.buildLabels();
    this.updateParams();
    this.updateActivations();
    this.applyEmphasis();
    this.applyFlow();
    this.applyReveal();
    this.viewport.requestRender();
  }

  /** Weights changed (or the connection values): recompute connection colour, dash and thickness. */
  updateParams(): void {
    if (!this.connectionMesh) return;
    const { spec } = this.model;
    const source = this.connectionValues;
    const valuesOf = (l: number): ArrayLike<number> => (source ? source.values[l] : this.model.weights(l));
    let maxV = 0;
    for (let l = 1; l < spec.layers.length; l++) maxV = Math.max(maxV, maxAbs(valuesOf(l)));
    const logMax = Math.log10(maxV || 1);

    const { pos, neg, off } = this.colorScratch;
    const colors = this.connectionColor.array as Float32Array;
    const styles = this.connectionStyle.array as Float32Array;
    let c = 0;
    for (let l = 1; l < spec.layers.length; l++) {
      const W = valuesOf(l);
      const layerRadius = Math.min(this.layout.frames[l - 1].neuronRadius, this.layout.frames[l].neuronRadius);
      const max = Math.min(0.07, layerRadius * 0.3) * this.options.thickness;
      const range = { min: max * 0.12, max };
      for (let k = 0; k < W.length; k++, c++) {
        const w = W[k];
        let t: number;
        if (source?.log) {
          const a = Math.abs(w);
          t = a > 0 && maxV > 0 ? clamp((Math.log10(a) - logMax + LOG_DECADES) / LOG_DECADES, 0, 1) : 0;
        } else {
          t = magnitude(w, maxV);
        }
        this.connectionRadii[c] = source?.log ? range.min + (range.max - range.min) * t : connectionRadius(w, maxV, range);
        const sign = w < 0 ? neg : pos;
        const mix = w === 0 ? 0 : 0.35 + 0.65 * t;
        colors[3 * c] = off.r + (sign.r - off.r) * mix;
        colors[3 * c + 1] = off.g + (sign.g - off.g) * mix;
        colors[3 * c + 2] = off.b + (sign.b - off.b) * mix;
        styles[4 * c + 2] = w < 0 ? 1 : 0;
        styles[4 * c + 3] = 0.18 + 0.72 * t;
      }
    }
    this.connectionColor.needsUpdate = true;
    this.writeConnectionMatrices();
    this.updatePulseStrengths();
    this.updateMarkers();
    this.viewport.requestRender();
  }

  /** Forward pass changed (or the neuron values): recompute neuron colour/size and value labels. */
  updateActivations(): void {
    if (!this.neuronMesh) return;
    const { spec, trace, topology } = this.model;
    const values = this.neuronValues ?? trace.a;
    const scales = this.neuronValues ? this.customScales(values) : activationScales(trace, this.options.activationScale);
    const { pos, neg, off, tmp } = this.colorScratch;
    const colors = this.neuronColor.array as Float32Array;
    const structural = {
      input: new THREE.Color(this.palette.neuronInput),
      hidden: new THREE.Color(this.palette.neuronHidden),
      output: new THREE.Color(this.palette.neuronOutput),
    };
    for (let l = 0; l < spec.layers.length; l++) {
      const a = values[l];
      const offset = topology.neuronOffsets[l];
      for (let j = 0; j < spec.layers[l].size; j++) {
        const g = offset + j;
        if (this.options.colorByActivation || this.neuronValues) {
          const t = visibleIntensity(magnitude(a[j], scales[l]));
          tmp.copy(off).lerp(a[j] < 0 ? neg : pos, t);
          this.boost[g] = 0.82 + 0.38 * t;
        } else {
          tmp.copy(structural[spec.layers[l].role]);
          this.boost[g] = 1;
        }
        colors[3 * g] = tmp.r;
        colors[3 * g + 1] = tmp.g;
        colors[3 * g + 2] = tmp.b;
      }
    }
    this.neuronColor.needsUpdate = true;
    this.writeNeuronMatrices();
    this.updateNeuronLabels();
    this.updatePulseStrengths();
    this.updateMarkers();
    this.viewport.requestRender();
  }

  /** Encode lines by custom values (e.g. gradients) instead of the weights; null restores the weights. */
  setConnectionValues(values: ConnectionValues | null): void {
    this.connectionValues = values;
    this.updateParams();
  }

  /** Encode neurons by custom values (e.g. δ = ∂L/∂z) instead of the activations; null restores the activations. */
  setNeuronValues(values: readonly ArrayLike<number>[] | null): void {
    this.neuronValues = values;
    this.updateActivations();
  }

  /** Signed per-connection strengths shown by travelling pulses; null = forward contributions w·a. */
  setPulseValues(values: readonly ArrayLike<number>[] | null): void {
    this.pulseValues = values;
    // The flowing particles of the effects show forward contributions only.
    this.connectionMaterial.uniforms.uAmbient.value = values ? 0 : 1;
    this.updatePulseStrengths();
  }

  /** Forward/backward animation state (null = everything visible, no pulse). */
  setSignalState(reveal: RevealState | null, pulse: PulseState | null): void {
    this.reveal = reveal;
    this.applyReveal();
    const uniform = this.connectionMaterial.uniforms.uPulse.value as THREE.Vector4;
    if (pulse) uniform.set(pulse.layer, pulse.t, pulse.direction, pulse.intensity);
    else uniform.set(-1, 0, 1, 0);
    this.viewport.requestRender();
  }

  /** Follow-the-signal scores (null = normal rendering). */
  setFlow(flow: FlowScores | null): void {
    this.flow = flow;
    this.applyFlow();
  }

  /** Clock for continuous effects (flow pulses), in seconds. */
  setTime(seconds: number): void {
    this.connectionMaterial.uniforms.uTime.value = seconds;
    this.viewport.requestRender();
  }

  setSelection(selection: NetworkSelection | null): void {
    const previous = this.selection;
    this.selection = selection;
    this.applyEmphasis();
    if (selection?.kind === 'neuron' && !(previous?.kind === 'neuron' && previous.layer === selection.layer && previous.index === selection.index)) {
      this.ringPop = 0.55;
      this.viewport.animations.tween({
        channel: 'selection-ring',
        duration: 0.35,
        easing: easeOutBack,
        onUpdate: (t) => {
          this.ringPop = 0.55 + 0.45 * t;
          this.updateMarkers();
          this.viewport.requestRender();
        },
      });
    }
    this.updateMarkers();
    this.viewport.requestRender();
  }

  setHover(hover: NetworkSelection | null): void {
    this.hover = hover;
    const { topology } = this.model;
    this.neuronMaterial.uniforms.uHover.value = hover?.kind === 'neuron' ? topology.neuronIndex(hover.layer, hover.index) : -1;
    this.connectionMaterial.uniforms.uHover.value =
      hover?.kind === 'connection' ? topology.connectionIndex(hover.layer, hover.target, hover.source) : -1;
    this.applyPlateStyles();
    this.layerLabels.forEach((label, l) => label.element.classList.toggle('is-hover', hover?.kind === 'layer' && hover.layer === l));
    this.viewport.requestRender();
  }

  setOptions(next: NetworkViewOptions): void {
    const prev = this.options;
    this.options = { ...next };
    if (!this.neuronMesh) return;
    if (prev.layout !== next.layout) this.transitionLayout();
    if (prev.showPlates !== next.showPlates) this.plateGroup.visible = next.showPlates;
    if (prev.showFloor !== next.showFloor) this.floor.visible = next.showFloor;
    if (prev.thickness !== next.thickness) this.updateParams();
    if (prev.colorByActivation !== next.colorByActivation || prev.activationScale !== next.activationScale) this.updateActivations();
    if (prev.dimUnrelated !== next.dimUnrelated) this.applyEmphasis();
    if (prev.showNeuronNames !== next.showNeuronNames || prev.showValues !== next.showValues) this.updateNeuronLabels();
    this.viewport.requestRender();
  }

  setPalette(palette: ScenePalette): void {
    this.palette = palette;
    this.applyPaletteUniforms();
    if (!this.neuronMesh) return;
    for (const plate of this.plates) plate.fill.material.color.set(palette.plateFill);
    this.applyPlateStyles();
    this.buildFloor();
    this.updateParams();
    this.updateActivations();
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible;
    this.viewport.requestRender();
  }

  // ---------------------------------------------------------------- picking

  /**
   * Returns the element under the canvas point (x, y) in CSS pixels.
   * Priority: neuron > connection > layer plate.
   */
  pick(x: number, y: number): NetworkSelection | null {
    if (!this.neuronMesh || !this.root.visible) return null;
    const { topology } = this.model;
    const { width, height } = this.viewport.size;
    const camera = this.viewport.camera;
    this.root.updateMatrixWorld();
    const m = this.viewport.viewProjection(this.vp).multiply(this.root.matrixWorld).elements;
    const projected = projectPoints(this.positions, m, width, height, this.projected);
    const focal = (height / 2) * camera.projectionMatrix.elements[5];

    // 1. Neurons: cursor inside the projected disc (+2 px), nearest to the camera wins.
    let bestNeuron = -1;
    let bestDepth = Infinity;
    for (let g = 0; g < topology.neuronCount; g++) {
      const w = projected[3 * g + 2];
      if (w <= 0) continue;
      const r = (this.baseRadii[g] * this.boost[g] * focal) / w;
      const d = Math.hypot(x - projected[3 * g], y - projected[3 * g + 1]);
      if (d <= r + 2 && w < bestDepth) {
        bestDepth = w;
        bestNeuron = g;
      }
    }
    if (bestNeuron >= 0) {
      const ref = topology.neuronAt(bestNeuron);
      return { kind: 'neuron', layer: ref.layer, index: ref.index };
    }

    // 2. Connections: closest projected segment within a pixel tolerance.
    let bestConnection = -1;
    let bestDistance = Infinity;
    const sizes = topology.sizes;
    let c = 0;
    for (let l = 1; l < sizes.length; l++) {
      const targetOffset = topology.neuronOffsets[l];
      const sourceOffset = topology.neuronOffsets[l - 1];
      for (let j = 0; j < sizes[l]; j++) {
        const gt = targetOffset + j;
        const wt = projected[3 * gt + 2];
        for (let i = 0; i < sizes[l - 1]; i++, c++) {
          const gs = sourceOffset + i;
          const ws = projected[3 * gs + 2];
          if (wt <= 0 || ws <= 0) continue;
          const { distance, t } = pointSegmentDistance(
            x,
            y,
            projected[3 * gs],
            projected[3 * gs + 1],
            projected[3 * gt],
            projected[3 * gt + 1],
          );
          const depth = ws + (wt - ws) * t;
          const tolerance = Math.max(5, (this.connectionRadii[c] * focal) / depth + 3);
          if (distance <= tolerance && distance < bestDistance) {
            bestDistance = distance;
            bestConnection = c;
          }
        }
      }
    }
    if (bestConnection >= 0) return { kind: 'connection', ...topology.connectionAt(bestConnection) };

    // 3. Layer plates: ray / plane intersection (plates are perpendicular to X), in the view's local space.
    if (!this.options.showPlates) return null;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, -(y / height) * 2 + 1), camera);
    const ray = this.raycaster.ray.clone().applyMatrix4(this.m4.copy(this.root.matrixWorld).invert());
    const { origin, direction } = ray;
    let bestLayer = -1;
    let bestT = Infinity;
    for (const frame of this.layout.frames) {
      if (Math.abs(direction.x) < 1e-9) continue;
      const t = (frame.center[0] - origin.x) / direction.x;
      if (t <= 0 || t >= bestT) continue;
      const py = origin.y + t * direction.y - frame.center[1];
      const pz = origin.z + t * direction.z - frame.center[2];
      if (Math.abs(py) <= frame.halfHeight && Math.abs(pz) <= frame.halfWidth) {
        bestT = t;
        bestLayer = frame.layer;
      }
    }
    return bestLayer >= 0 ? { kind: 'layer', layer: bestLayer } : null;
  }

  dispose(): void {
    this.viewport.animations.cancelChannel('layout');
    this.viewport.animations.cancelChannel('selection-ring');
    this.removeBeforeRender();
    this.removeAfterRender();
    this.removeEffectsListener();
    this.disposeMeshes();
    this.clearPlates();
    this.clearFloor();
    this.clearLabels();
    this.neuronMaterial.dispose();
    this.connectionMaterial.dispose();
    this.ring.geometry.dispose();
    this.ring.material.dispose();
    this.arrow.geometry.dispose();
    this.arrow.material.dispose();
    this.root.removeFromParent();
    this.viewport.requestRender();
  }

  // -------------------------------------------------------------- internals

  private positionOf(g: number): Vec3 {
    return [this.positions[3 * g], this.positions[3 * g + 1], this.positions[3 * g + 2]];
  }

  /** Per-layer (or global) scale for custom neuron values. */
  private customScales(values: readonly ArrayLike<number>[]): number[] {
    const perLayer = values.map((v) => maxAbs(v));
    if (this.options.activationScale === 'layer') return perLayer.map((m) => m || 1);
    const global = Math.max(...perLayer) || 1;
    return perLayer.map(() => global);
  }

  private applyPaletteUniforms(): void {
    const p = this.palette;
    this.colorScratch.pos.set(p.positive);
    this.colorScratch.neg.set(p.negative);
    this.colorScratch.off.set(p.neuronOff);
    const highlight = new THREE.Color(p.theme === 'dark' ? '#ffffff' : '#000000');
    this.neuronMaterial.uniforms.uBackground.value.set(p.background);
    this.neuronMaterial.uniforms.uHighlight.value.copy(highlight);
    this.neuronMaterial.uniforms.uEmpty.value.set(p.theme === 'dark' ? '#2a313b' : '#dde2e8');
    this.neuronMaterial.uniforms.uDim.value = p.theme === 'dark' ? 0.74 : 0.68;
    this.connectionMaterial.uniforms.uHighlight.value.copy(highlight);
    this.connectionMaterial.uniforms.uPositive.value.set(p.positive);
    this.connectionMaterial.uniforms.uNegative.value.set(p.negative);
    this.connectionMaterial.uniforms.uDimAlpha.value = p.theme === 'dark' ? 0.13 : 0.16;
    this.ring.material.color.set(p.selection);
    this.arrow.material.color.set(p.selection);
  }

  /** Pulse strength per connection: custom values, or the forward contribution w·a (normalized per layer). */
  private updatePulseStrengths(): void {
    if (!this.connectionMesh) return;
    const { spec, trace } = this.model;
    const signal = this.connectionSignal.array as Float32Array;
    let c = 0;
    for (let l = 1; l < spec.layers.length; l++) {
      const n = spec.layers[l].size * spec.layers[l - 1].size;
      const start = c;
      let max = 0;
      if (this.pulseValues) {
        const values = this.pulseValues[l];
        for (let k = 0; k < n; k++) max = Math.max(max, Math.abs(values[k]));
        for (let k = 0; k < n; k++, c++) signal[3 * c + 1] = max > 0 ? values[k] / max : 0;
      } else {
        const W = this.model.weights(l);
        const x = trace.a[l - 1];
        const fanIn = spec.layers[l - 1].size;
        for (let k = 0; k < n; k++) max = Math.max(max, Math.abs(W[k] * x[k % fanIn]));
        for (let k = 0; k < n; k++, c++) signal[3 * c + 1] = max > 0 ? (W[k] * x[k % fanIn]) / max : 0;
      }
      c = start + n;
    }
    this.connectionSignal.needsUpdate = true;
  }

  private applyFlow(): void {
    if (!this.neuronMesh) return;
    const state = this.neuronState.array as Float32Array;
    const signal = this.connectionSignal.array as Float32Array;
    const flow = this.flow;
    for (let g = 0; g < state.length / 3; g++) state[3 * g + 2] = flow ? flow.neurons[g] : 1;
    for (let c = 0; c < signal.length / 3; c++) signal[3 * c + 2] = flow ? flow.connections[c] : 1;
    this.neuronState.needsUpdate = true;
    this.connectionSignal.needsUpdate = true;
    this.neuronMaterial.uniforms.uFlowMode.value = flow ? 1 : 0;
    this.connectionMaterial.uniforms.uFlowMode.value = flow ? 1 : 0;
    this.neuronLabels.forEach((label, g) => label.element.classList.toggle('is-faded', !!flow && flow.neurons[g] < 0.15));
    this.viewport.requestRender();
  }

  private applyReveal(): void {
    const reveal = this.reveal;
    const value = reveal ? new THREE.Vector3(reveal.layer, reveal.t, reveal.direction) : REVEAL_ALL;
    (this.neuronMaterial.uniforms.uReveal.value as THREE.Vector3).copy(value);
    (this.connectionMaterial.uniforms.uReveal.value as THREE.Vector3).copy(value);
    for (const label of this.neuronLabels) {
      const pending = reveal ? (label.layer - reveal.layer) * reveal.direction > 0.5 || (label.layer === reveal.layer && reveal.t < 0.5) : false;
      label.element.classList.toggle('is-pending', pending);
    }
  }

  private writeNeuronMatrices(): void {
    const mesh = this.neuronMesh;
    if (!mesh) return;
    for (let g = 0; g < mesh.count; g++) {
      const r = this.baseRadii[g] * this.boost[g];
      this.va.set(this.positions[3 * g], this.positions[3 * g + 1], this.positions[3 * g + 2]);
      this.vb.set(r, r, r);
      mesh.setMatrixAt(g, this.m4.compose(this.va, this.identity, this.vb));
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.writeLabelPositions();
  }

  private writeConnectionMatrices(): void {
    const mesh = this.connectionMesh;
    if (!mesh) return;
    const { topology } = this.model;
    const sizes = topology.sizes;
    const p = this.positions;
    const styles = this.connectionStyle.array as Float32Array;
    let c = 0;
    for (let l = 1; l < sizes.length; l++) {
      const targetOffset = topology.neuronOffsets[l];
      const sourceOffset = topology.neuronOffsets[l - 1];
      for (let j = 0; j < sizes[l]; j++) {
        const gt = targetOffset + j;
        for (let i = 0; i < sizes[l - 1]; i++, c++) {
          const gs = sourceOffset + i;
          this.va.set(p[3 * gs], p[3 * gs + 1], p[3 * gs + 2]);
          this.vb.set(p[3 * gt], p[3 * gt + 1], p[3 * gt + 2]);
          this.vc.subVectors(this.vb, this.va);
          const length = this.vc.length();
          this.quat.setFromUnitVectors(UP, this.vc.divideScalar(length || 1));
          this.va.add(this.vb).multiplyScalar(0.5);
          const r = this.connectionRadii[c];
          this.vb.set(r, length, r);
          mesh.setMatrixAt(c, this.m4.compose(this.va, this.quat, this.vb));
          styles[4 * c + 1] = length;
        }
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.connectionStyle.needsUpdate = true;
  }

  private applyEmphasis(): void {
    if (!this.neuronMesh) return;
    computeEmphasis(this.model.topology, this.selection, this.options.dimUnrelated, this.emphasis);
    const state = this.neuronState.array as Float32Array;
    const levels = this.emphasis.neurons;
    for (let g = 0; g < levels.length; g++) state[3 * g] = levels[g];
    this.neuronState.needsUpdate = true;
    const styles = this.connectionStyle.array as Float32Array;
    const connectionLevels = this.emphasis.connections;
    for (let c = 0; c < connectionLevels.length; c++) styles[4 * c] = connectionLevels[c];
    this.connectionStyle.needsUpdate = true;

    this.applyPlateStyles();
    const selectedLayer = this.selection?.kind === 'layer' ? this.selection.layer : -1;
    this.layerLabels.forEach((label, l) => label.element.classList.toggle('is-selected', l === selectedLayer));
    this.neuronLabels.forEach((label, g) => label.element.classList.toggle('is-dimmed', this.emphasis.neurons[g] === EMPHASIS.dim));
  }

  private updateMarkers(): void {
    const selection = this.selection;
    const { topology } = this.model;
    this.ring.visible = selection?.kind === 'neuron';
    if (selection?.kind === 'neuron') {
      const g = topology.neuronIndex(selection.layer, selection.index);
      this.ring.position.set(this.positions[3 * g], this.positions[3 * g + 1], this.positions[3 * g + 2]);
      this.ring.scale.setScalar(this.baseRadii[g] * this.boost[g] * this.ringPop);
    }
    this.arrow.visible = selection?.kind === 'connection';
    if (selection?.kind === 'connection') {
      const gs = topology.neuronIndex(selection.layer - 1, selection.source);
      const gt = topology.neuronIndex(selection.layer, selection.target);
      this.va.set(this.positions[3 * gs], this.positions[3 * gs + 1], this.positions[3 * gs + 2]);
      this.vb.set(this.positions[3 * gt], this.positions[3 * gt + 1], this.positions[3 * gt + 2]);
      this.vc.subVectors(this.vb, this.va);
      const length = this.vc.length();
      const c = topology.connectionIndex(selection.layer, selection.target, selection.source);
      const size = Math.max(this.connectionRadii[c] * 3.2, 0.06, length * 0.012);
      this.arrow.position.copy(this.va).addScaledVector(this.vc, 0.6);
      this.arrow.quaternion.setFromUnitVectors(UP, this.vc.divideScalar(length || 1));
      this.arrow.scale.setScalar(size);
    }
  }

  // ----- plates

  private buildPlates(): void {
    this.clearPlates();
    for (const frame of this.layout.frames) {
      const shape = roundedRect(frame.halfWidth, frame.halfHeight, frame.cornerRadius);
      const fill = new THREE.Mesh(
        new THREE.ShapeGeometry(shape, 24),
        new THREE.MeshBasicMaterial({
          color: this.palette.plateFill,
          transparent: true,
          opacity: 0.3,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      fill.renderOrder = 0;
      const border = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(shape.getPoints(24)),
        new THREE.LineBasicMaterial({ color: this.palette.plateBorder, transparent: true, opacity: 0.85, depthWrite: false }),
      );
      border.renderOrder = 0;
      const group = new THREE.Group();
      group.add(fill, border);
      group.position.set(frame.center[0], frame.center[1], frame.center[2]);
      group.rotation.y = Math.PI / 2; // the shape is drawn in XY: rotate it into the YZ plane
      this.plateGroup.add(group);
      this.plates.push({ group, fill, border });
    }
    this.plateGroup.visible = this.options.showPlates;
    this.applyPlateStyles();
  }

  private applyPlateStyles(): void {
    const selected = this.selection?.kind === 'layer' ? this.selection.layer : -1;
    const hovered = this.hover?.kind === 'layer' ? this.hover.layer : -1;
    const dimOthers = !!this.selection && this.options.dimUnrelated;
    const light = this.palette.theme === 'light';
    this.plates.forEach((plate, l) => {
      const isSelected = l === selected;
      const isHovered = l === hovered;
      const base = light ? 0.42 : 0.3;
      const opacity = isSelected ? base + 0.3 : isHovered ? base + 0.14 : dimOthers ? base * 0.6 : base;
      plate.fill.material.opacity = opacity * this.plateFade;
      plate.border.material.color.set(isSelected || isHovered ? this.palette.plateFocus : this.palette.plateBorder);
      plate.border.material.opacity = (isSelected ? 1 : isHovered ? 0.9 : dimOthers ? 0.45 : 0.8) * this.plateFade;
    });
  }

  private clearPlates(): void {
    for (const plate of this.plates) {
      plate.fill.geometry.dispose();
      plate.fill.material.dispose();
      plate.border.geometry.dispose();
      plate.border.material.dispose();
      plate.group.removeFromParent();
    }
    this.plates = [];
  }

  // ----- floor

  private buildFloor(): void {
    this.clearFloor();
    const { boundsMin: min, boundsMax: max, center } = this.layout;
    const extent = Math.max(max[0] - min[0], max[2] - min[2]) * 1.5 + 4;
    const size = Math.ceil(extent / 2) * 2;
    const y = min[1] - 0.9;
    const grid = new THREE.GridHelper(size, size, this.palette.floorGrid, this.palette.floorGrid);
    const gridMaterial = grid.material as THREE.LineBasicMaterial;
    gridMaterial.transparent = true;
    gridMaterial.opacity = this.palette.theme === 'dark' ? 0.75 : 0.9;
    gridMaterial.depthWrite = false;
    grid.position.set(center[0], y, center[2]);
    grid.renderOrder = -1;
    grid.visible = !this.viewport.effectsActive;
    this.grid = grid;

    const length = Math.max(max[0] - min[0], 1);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(min[0], y + 0.02, max[2] + 0.7),
      length,
      this.palette.axis,
      Math.min(0.45, length * 0.12),
      0.22,
    );
    const label = document.createElement('div');
    label.className = 'axis-label';
    label.textContent = 'signal flow: input → output';
    const labelObject = new CSS2DObject(label);
    labelObject.position.set(center[0], y + 0.02, max[2] + 1.15);
    this.floor.add(grid, arrow, labelObject);
    this.floor.visible = this.options.showFloor;
  }

  private clearFloor(): void {
    for (const child of [...this.floor.children]) {
      if (child instanceof CSS2DObject) child.element.remove();
      child.traverse((object) => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
        const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(material)) material.forEach((m) => m.dispose());
        else material?.dispose();
      });
      child.removeFromParent();
    }
  }

  // ----- labels

  private buildLabels(): void {
    this.clearLabels();
    const { spec, topology } = this.model;
    spec.layers.forEach((layer, l) => {
      const element = document.createElement('button');
      element.type = 'button';
      element.className = 'layer-label';
      element.title = `Select ${layer.name}`;
      const name = document.createElement('span');
      name.className = 'layer-label__name';
      name.textContent = layer.name;
      const meta = document.createElement('span');
      meta.className = 'layer-label__meta';
      const sizeText = layer.shape ? `${layer.shape[0]}×${layer.shape[1]}` : `${layer.size}`;
      meta.textContent = layer.role === 'input' ? `${sizeText} inputs` : `${sizeText} · ${ACTIVATIONS[layer.activation].label}`;
      // Shown instead of the name when the layers are crowded (compactLabels).
      const short = document.createElement('span');
      short.className = 'layer-label__short';
      short.textContent = layer.role === 'input' ? 'In' : layer.role === 'output' ? 'Out' : layer.prefix;
      element.append(name, short, meta);
      element.addEventListener('pointerdown', (event) => event.stopPropagation());
      element.addEventListener('click', () => this.onLayerLabelClick(l));
      const object = new CSS2DObject(element);
      object.position.copy(this.layerLabelAnchor(l));
      this.labelGroup.add(object);
      this.layerLabels.push(object);
    });
    this.labelDetail = 0;
    this.labelSizes = null;
    this.labelAnchors = new Float32Array(this.layerLabels.length * 3);
    this.labelPoints = new Float32Array(this.layerLabels.length * 3);

    if (topology.neuronCount <= MAX_NEURON_LABELS) {
      for (let g = 0; g < topology.neuronCount; g++) {
        const ref = topology.neuronAt(g);
        const element = document.createElement('div');
        element.className = 'neuron-label';
        const nameEl = document.createElement('span');
        nameEl.className = 'neuron-label__name';
        nameEl.textContent = neuronName(spec, ref.layer, ref.index);
        const valueEl = document.createElement('span');
        valueEl.className = 'neuron-label__value';
        element.append(nameEl, valueEl);
        const object = new CSS2DObject(element);
        this.labelGroup.add(object);
        this.neuronLabels.push({ object, element, name: nameEl, value: valueEl, layer: ref.layer });
      }
    }
    this.writeLabelPositions();
  }

  /** Layer titles float above the layer, on the back edge of its plate (so the top view does not hide neurons). */
  private layerLabelAnchor(layer: number): THREE.Vector3 {
    const frame = this.layout.frames[layer];
    return new THREE.Vector3(frame.center[0], this.layout.labelY, frame.center[2] - frame.halfWidth);
  }

  /** Picks how much of the layer titles to show from the room between neighbouring layers on screen. */
  private updateLabelDetail(): void {
    const labels = this.layerLabels;
    if (labels.length < 2 || !this.root.visible) return;
    this.labelSizes ??= this.measureLabels();
    if (!this.labelSizes) return;
    labels.forEach((label, l) => {
      this.labelAnchors[3 * l] = label.position.x;
      this.labelAnchors[3 * l + 1] = label.position.y;
      this.labelAnchors[3 * l + 2] = label.position.z;
    });
    const { width, height } = this.viewport.size;
    const m = this.viewport.viewProjection(this.vp).multiply(this.root.matrixWorld).elements;
    projectPoints(this.labelAnchors, m, width, height, this.labelPoints);
    const detail = chooseLabelDetail(this.labelPoints, this.labelSizes, this.labelDetail);
    if (detail === this.labelDetail) return;
    // The titles are centred on their anchor with a percentage transform: no new frame needed.
    this.labelDetail = detail;
    this.setLabelClasses(detail);
  }

  /** Largest layer title at each level of detail (null while the titles are not laid out on the page). */
  private measureLabels(): Record<LabelDetail, LabelSize> | null {
    const elements = this.layerLabels.map((label) => label.element);
    const sizes = LABEL_DETAILS.map((detail) => {
      this.setLabelClasses(detail);
      let width = 0;
      let height = 0;
      for (const element of elements) {
        width = Math.max(width, element.offsetWidth);
        height = Math.max(height, element.offsetHeight);
      }
      return { width, height };
    });
    this.setLabelClasses(this.labelDetail);
    return sizes[0].width > 0 ? { 0: sizes[0], 1: sizes[1], 2: sizes[2] } : null;
  }

  private setLabelClasses(detail: LabelDetail): void {
    for (const label of this.layerLabels) {
      label.element.classList.toggle('layer-label--name', detail === 1);
      label.element.classList.toggle('layer-label--short', detail === 2);
    }
  }

  private updateNeuronLabels(): void {
    const { trace, topology } = this.model;
    const values = this.neuronValues ?? trace.a;
    const show = this.options.showNeuronNames || this.options.showValues;
    this.neuronLabels.forEach((label, g) => {
      label.object.visible = show;
      if (!show) return;
      const ref = topology.neuronAt(g);
      label.name.hidden = !this.options.showNeuronNames;
      label.value.hidden = !this.options.showValues;
      if (this.options.showValues) {
        const v = values[ref.layer][ref.index];
        label.value.textContent = formatValue(v, Math.abs(v) < 0.01 && v !== 0 ? 4 : 2);
        label.value.classList.toggle('is-negative', v < 0);
      }
    });
  }

  private writeLabelPositions(): void {
    this.neuronLabels.forEach((label, g) => {
      const r = this.baseRadii[g] * this.boost[g];
      label.object.position.set(this.positions[3 * g], this.positions[3 * g + 1] - r - 0.16, this.positions[3 * g + 2]);
    });
  }

  private clearLabels(): void {
    for (const object of [...this.layerLabels, ...this.neuronLabels.map((l) => l.object)]) {
      object.element.remove();
      object.removeFromParent();
    }
    this.layerLabels = [];
    this.neuronLabels = [];
  }

  // ----- layout transition

  private transitionLayout(): void {
    const { spec, topology } = this.model;
    const fromPositions = this.positions.slice();
    const fromRadii = this.baseRadii.slice();
    const fromLabels = this.layerLabels.map((label) => label.position.clone());
    const target = computeLayout(spec, topology, { mode: this.options.layout });
    this.layout = target;
    this.plateFade = 0; // new plates fade in while the neurons move
    this.buildPlates();
    this.buildFloor();
    const toLabels = target.frames.map((_, l) => this.layerLabelAnchor(l));

    this.viewport.animations.tween({
      channel: 'layout',
      duration: 0.8,
      easing: easeInOutCubic,
      onUpdate: (t) => {
        for (let k = 0; k < fromPositions.length; k++) this.positions[k] = fromPositions[k] + (target.positions[k] - fromPositions[k]) * t;
        for (let g = 0; g < fromRadii.length; g++) this.baseRadii[g] = fromRadii[g] + (target.radii[g] - fromRadii[g]) * t;
        this.layerLabels.forEach((label, l) => label.position.lerpVectors(fromLabels[l], toLabels[l], t));
        this.plateFade = t;
        this.applyPlateStyles();
        this.writeNeuronMatrices();
        this.writeConnectionMatrices();
        this.updateMarkers();
        this.viewport.requestRender();
      },
      onComplete: () => {
        this.plateFade = 1;
        this.positions.set(target.positions);
        this.baseRadii.set(target.radii);
        // Thickness ranges depend on the neuron size of the new layout.
        this.updateParams();
        this.writeNeuronMatrices();
      },
    });
  }

  private disposeMeshes(): void {
    for (const mesh of [this.neuronMesh, this.connectionMesh]) {
      if (!mesh) continue;
      mesh.geometry.dispose();
      mesh.dispose();
      mesh.removeFromParent();
    }
    this.neuronMesh = null;
    this.connectionMesh = null;
  }
}
