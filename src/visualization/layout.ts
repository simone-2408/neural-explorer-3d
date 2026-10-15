/**
 * 3D layout of a feed-forward network — pure geometry, no three.js.
 *
 * Layers are placed along the X axis (signal flows from −X to +X); the
 * neurons of each layer are arranged in the YZ plane as a grid, a column or a
 * ring. Big layers are shrunk so every layer stays within `maxExtent`.
 */
import type { NetworkSpec } from '../neural/config';
import type { Topology } from '../neural/topology';

export type Vec3 = [number, number, number];
export type LayoutMode = 'grid' | 'column' | 'ring';

export const LAYOUT_MODES: { id: LayoutMode; label: string; description: string }[] = [
  { id: 'grid', label: 'Grid', description: 'Each layer is a square grid of neurons (true 3D structure).' },
  { id: 'column', label: 'Flat', description: 'The classic textbook diagram: each layer is a vertical column.' },
  { id: 'ring', label: 'Ring', description: 'Each layer is a circle of neurons around the signal axis.' },
];

export interface LayoutOptions {
  mode: LayoutMode;
  /** Minimum distance between consecutive layers. */
  layerSpacing: number;
  /** Preferred distance between neighbouring neurons. */
  neuronSpacing: number;
  /** Maximum width/height of a layer (big layers are compressed). */
  maxExtent: number;
  /** Maximum neuron radius. */
  maxRadius: number;
}

export const DEFAULT_LAYOUT_OPTIONS: LayoutOptions = {
  mode: 'grid',
  layerSpacing: 4,
  neuronSpacing: 1.15,
  maxExtent: 9,
  maxRadius: 0.3,
};

export interface LayerFrame {
  layer: number;
  center: Vec3;
  /** Half size along Z (width) and Y (height) of the layer plate. */
  halfWidth: number;
  halfHeight: number;
  /** Corner radius of the plate (equal to the half size for rings → circle). */
  cornerRadius: number;
  neuronRadius: number;
  spacing: number;
}

export interface NetworkLayout {
  mode: LayoutMode;
  /** xyz per neuron, indexed by global neuron index. */
  positions: Float32Array;
  /** Base radius per neuron. */
  radii: Float32Array;
  frames: LayerFrame[];
  boundsMin: Vec3;
  boundsMax: Vec3;
  center: Vec3;
  /** Radius of a sphere that contains the whole network. */
  radius: number;
  /** Mean distance between consecutive layers. */
  layerGap: number;
  /** Height at which layer titles are anchored (aligned above the tallest layer). */
  labelY: number;
}

interface Arrangement {
  /** (y, z) pairs relative to the layer centre. */
  yz: Float32Array;
  spacing: number;
  halfY: number;
  halfZ: number;
  circle: boolean;
}

function arrange(size: number, shape: readonly [number, number] | null, mode: LayoutMode, o: LayoutOptions): Arrangement {
  const yz = new Float32Array(size * 2);
  if (mode === 'ring' && size > 1) {
    const idealRadius = Math.max((o.neuronSpacing * size) / (2 * Math.PI), o.neuronSpacing * 0.75);
    const radius = Math.min(idealRadius, o.maxExtent / 2);
    for (let i = 0; i < size; i++) {
      const theta = Math.PI / 2 - (2 * Math.PI * i) / size;
      yz[2 * i] = radius * Math.sin(theta);
      yz[2 * i + 1] = radius * Math.cos(theta);
    }
    const spacing = Math.min(o.neuronSpacing, (2 * Math.PI * radius) / size);
    return { yz, spacing, halfY: radius, halfZ: radius, circle: true };
  }

  let rows: number;
  let cols: number;
  if (mode === 'column') {
    rows = size;
    cols = 1;
  } else if (shape) {
    [rows, cols] = shape;
  } else {
    rows = Math.ceil(Math.sqrt(size));
    cols = Math.ceil(size / rows);
    rows = Math.ceil(size / cols);
  }
  const maxColumn = mode === 'column' ? o.maxExtent * 1.4 : o.maxExtent;
  let spacing = o.neuronSpacing;
  if (cols > 1) spacing = Math.min(spacing, o.maxExtent / (cols - 1));
  if (rows > 1) spacing = Math.min(spacing, maxColumn / (rows - 1));

  for (let i = 0; i < size; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    // Centre an incomplete last row (only for automatic grids).
    const inRow = !shape && r === rows - 1 ? size - r * cols : cols;
    const offset = (cols - inRow) / 2;
    yz[2 * i] = ((rows - 1) / 2 - r) * spacing;
    yz[2 * i + 1] = (c + offset - (cols - 1) / 2) * spacing;
  }
  return { yz, spacing, halfY: ((rows - 1) / 2) * spacing, halfZ: ((cols - 1) / 2) * spacing, circle: false };
}

export function computeLayout(spec: NetworkSpec, topology: Topology, options: Partial<LayoutOptions> = {}): NetworkLayout {
  const o: LayoutOptions = { ...DEFAULT_LAYOUT_OPTIONS, ...options };
  const layers = spec.layers;
  const arrangements = layers.map((l) => arrange(l.size, l.shape, o.mode, o));

  // X position of each layer: gaps grow a little for very large layers.
  const xs = [0];
  for (let l = 1; l < layers.length; l++) {
    const a = arrangements[l - 1];
    const b = arrangements[l];
    const extent = Math.max(a.halfY, a.halfZ, b.halfY, b.halfZ) * 2;
    xs.push(xs[l - 1] + Math.max(o.layerSpacing, extent * 0.5));
  }
  const shiftX = (xs[0] + xs[xs.length - 1]) / 2;

  const positions = new Float32Array(topology.neuronCount * 3);
  const radii = new Float32Array(topology.neuronCount);
  const frames: LayerFrame[] = [];
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];

  layers.forEach((layer, l) => {
    const arrangement = arrangements[l];
    const x = xs[l] - shiftX;
    const radius = Math.min(o.maxRadius, arrangement.spacing * 0.27);
    const offset = topology.neuronOffsets[l];
    for (let i = 0; i < layer.size; i++) {
      const g = offset + i;
      positions[3 * g] = x;
      positions[3 * g + 1] = arrangement.yz[2 * i];
      positions[3 * g + 2] = arrangement.yz[2 * i + 1];
      radii[g] = radius;
    }
    const pad = Math.max(0.3, radius * 1.2);
    const halfWidth = arrangement.halfZ + radius + pad;
    const halfHeight = arrangement.halfY + radius + pad;
    const cornerRadius = arrangement.circle ? Math.min(halfWidth, halfHeight) : Math.min(halfWidth, halfHeight) * 0.22;
    frames.push({ layer: l, center: [x, 0, 0], halfWidth, halfHeight, cornerRadius, neuronRadius: radius, spacing: arrangement.spacing });

    min[0] = Math.min(min[0], x - radius);
    max[0] = Math.max(max[0], x + radius);
    min[1] = Math.min(min[1], -halfHeight);
    max[1] = Math.max(max[1], halfHeight);
    min[2] = Math.min(min[2], -halfWidth);
    max[2] = Math.max(max[2], halfWidth);
  });

  // Keep room for the layer titles above the plates.
  const labelY = max[1] + 0.5;
  max[1] = labelY + 0.6;

  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2;
  const layerGap = layers.length > 1 ? (xs[xs.length - 1] - xs[0]) / (layers.length - 1) : o.layerSpacing;
  return { mode: o.mode, positions, radii, frames, boundsMin: min, boundsMax: max, center, radius, layerGap, labelY };
}
