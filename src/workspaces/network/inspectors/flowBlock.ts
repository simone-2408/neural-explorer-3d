/** Inspector block for "Follow the Signal": where the selected element's signal goes and comes from. */
import { signalFlow, type FlowSource } from '../../../neural/signalFlow';
import { neuronName, neuronTitle } from '../../../neural/topology';
import { h, type Child } from '../../../ui/dom';
import { button, note } from '../../../ui/widgets';
import { formatPercent } from '../../../utils/format';
import type { NetworkContext } from '../context';
import { block, entityLink, paragraph } from './common';

/** Static button shown when the mode is off. */
export function followButton(ctx: NetworkContext): HTMLElement | null {
  const toggle = ctx.actions.toggleFollow;
  if (!toggle) return null;
  const active = ctx.store.state.signalMode === 'follow';
  return button({
    icon: 'signal',
    label: active ? 'Stop following' : 'Follow the signal',
    title: 'Highlight where this signal goes and where it comes from (S)',
    small: true,
    variant: active ? 'default' : 'primary',
    pressed: active,
    onClick: () => toggle(),
  });
}

/** Live content of the block (re-rendered when values change). */
export function flowContent(ctx: NetworkContext, source: FlowSource): Child {
  const { model, actions } = ctx;
  const { spec, topology } = model;
  if (ctx.store.state.signalMode !== 'follow') {
    return paragraph('Follow the signal highlights the paths that carry most of this signal to the output, and the inputs it mostly comes from; the rest of the network fades.');
  }
  const flow = signalFlow(model, source);
  const sourceLayer = source.kind === 'neuron' ? source.layer : source.layer;
  const last = spec.layers.length - 1;
  const name = source.kind === 'neuron' ? neuronName(spec, source.layer, source.index) : `${neuronName(spec, source.layer - 1, source.source)} → ${neuronName(spec, source.layer, source.target)}`;

  const downstream: Child[] = [];
  if (sourceLayer < last || source.kind === 'connection') {
    const rows = Array.from({ length: spec.layers[last].size }, (_, j) => ({ j, share: flow.neurons[topology.neuronIndex(last, j)] }));
    downstream.push(
      h('h4', { class: 'insp-subheading' }, `Share of each output that comes from ${name}`),
      h(
        'table',
        { class: 'data-table' },
        h('tbody', null, rows.map((r) => h('tr', null, h('td', null, entityLink(neuronTitle(spec, last, r.j), () => actions.select({ kind: 'neuron', layer: last, index: r.j }))), h('td', null, formatPercent(r.share))))),
      ),
    );
  }

  const upstream: Child[] = [];
  const upLayer = source.kind === 'neuron' ? source.layer : source.layer - 1;
  if (upLayer > 0 || source.kind === 'connection') {
    const candidates: { layer: number; index: number; share: number }[] = [];
    for (let l = 0; l < upLayer; l++) {
      for (let i = 0; i < spec.layers[l].size; i++) candidates.push({ layer: l, index: i, share: flow.neurons[topology.neuronIndex(l, i)] });
    }
    if (source.kind === 'connection') candidates.push({ layer: upLayer, index: source.source, share: 1 });
    const top = candidates.filter((c) => c.share > 0).sort((a, b) => b.share - a.share).slice(0, 6);
    if (top.length) {
      upstream.push(
        h('h4', { class: 'insp-subheading' }, `Where the input of ${name} comes from (top ${top.length})`),
        h(
          'table',
          { class: 'data-table' },
          h('tbody', null, top.map((c) => h('tr', null, h('td', null, entityLink(neuronTitle(spec, c.layer, c.index), () => actions.select({ kind: 'neuron', layer: c.layer, index: c.index }))), h('td', null, formatPercent(c.share))))),
        ),
      );
    }
  }

  return [
    paragraph('Bright paths carry the largest shares of |w·a|. Inactive neurons (a = 0) block the paths that pass through them.'),
    downstream,
    upstream,
    note('These shares compare the magnitudes |w·a| of the terms of each weighted sum. They are a simple attribution heuristic, not a causal explanation: positive and negative terms are counted alike and non-linear interactions are ignored.'),
  ];
}

export function flowBlock(ctx: NetworkContext, body: HTMLElement): HTMLElement | null {
  if (!ctx.actions.toggleFollow) return null;
  return block(null, h('span', null, 'Follow the signal'), followButton(ctx), body);
}
