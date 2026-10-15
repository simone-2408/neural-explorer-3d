/**
 * What the user can select in the network workspace. Shared by the 3D view,
 * the inspector, the layer chips and keyboard navigation.
 *
 * Later workspaces add their own kinds (token, attention head, …) — see
 * docs/ARCHITECTURE.md → "Extension points".
 */
import type { NetworkSpec } from '../neural/config';
import type { ConnectionRef } from '../neural/topology';

export type NetworkSelection =
  | { kind: 'neuron'; layer: number; index: number }
  | ({ kind: 'connection' } & ConnectionRef)
  | { kind: 'layer'; layer: number };

export function selectionEquals(a: NetworkSelection | null, b: NetworkSelection | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.kind !== b.kind || a.layer !== b.layer) return false;
  if (a.kind === 'neuron' && b.kind === 'neuron') return a.index === b.index;
  if (a.kind === 'connection' && b.kind === 'connection') return a.target === b.target && a.source === b.source;
  return a.kind === 'layer';
}

/** True if the selection still refers to an existing element of `spec`. */
export function isSelectionValid(selection: NetworkSelection, spec: NetworkSpec): boolean {
  const layers = spec.layers;
  if (selection.layer < 0 || selection.layer >= layers.length) return false;
  switch (selection.kind) {
    case 'layer':
      return true;
    case 'neuron':
      return selection.index >= 0 && selection.index < layers[selection.layer].size;
    case 'connection':
      return (
        selection.layer >= 1 &&
        selection.target >= 0 &&
        selection.target < layers[selection.layer].size &&
        selection.source >= 0 &&
        selection.source < layers[selection.layer - 1].size
      );
  }
}
