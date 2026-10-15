/** Short textual descriptions of network elements (hover tooltips). */
import { ACTIVATIONS } from '../../math/activations';
import type { NetworkModel } from '../../neural/NetworkModel';
import { connectionName, neuronTitle } from '../../neural/topology';
import { h } from '../../ui/dom';
import { num } from '../../ui/widgets';
import type { NetworkSelection } from '../../visualization/selection';

export function describeTooltip(model: NetworkModel, selection: NetworkSelection): HTMLElement {
  const { spec, trace } = model;
  switch (selection.kind) {
    case 'neuron': {
      const layer = spec.layers[selection.layer];
      const a = trace.a[selection.layer][selection.index];
      if (layer.role === 'input') {
        return h('div', null, h('strong', null, neuronTitle(spec, selection.layer, selection.index)), h('div', { class: 'tooltip__meta' }, 'input value'), h('div', null, 'x = ', num(a)));
      }
      const z = trace.z[selection.layer][selection.index];
      return h(
        'div',
        null,
        h('strong', null, neuronTitle(spec, selection.layer, selection.index)),
        h('div', { class: 'tooltip__meta' }, `${layer.name} · ${ACTIVATIONS[layer.activation].label}`),
        h('div', null, 'z = ', num(z), '   a = ', num(a)),
      );
    }
    case 'connection': {
      const w = model.weight(selection.layer, selection.target, selection.source);
      const x = trace.a[selection.layer - 1][selection.source];
      return h(
        'div',
        null,
        h('strong', null, connectionName(spec, selection)),
        h('div', { class: 'tooltip__meta' }, `weight W(${selection.layer})[${selection.target + 1},${selection.source + 1}]`),
        h('div', null, 'w = ', num(w, { signed: true }), '   w·x = ', num(w * x, { signed: true })),
      );
    }
    case 'layer': {
      const layer = spec.layers[selection.layer];
      const params = selection.layer === 0 ? 0 : layer.size * spec.layers[selection.layer - 1].size + layer.size;
      return h(
        'div',
        null,
        h('strong', null, layer.name),
        h(
          'div',
          { class: 'tooltip__meta' },
          layer.role === 'input' ? `${layer.size} inputs · no parameters` : `${layer.size} neurons · ${ACTIVATIONS[layer.activation].label} · ${params} parameters`,
        ),
      );
    }
  }
}
