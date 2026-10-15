/** Bottom bar: layer selector strip, visual-encoding legend and size/performance stats. */
import { DisposeBag } from '../../core/DisposeBag';
import { ACTIVATIONS } from '../../math/activations';
import type { Viewport } from '../../three/Viewport';
import { h, replaceChildren, svg } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { formatCount } from '../../utils/format';
import type { NetworkContext, Panel } from './context';

function legendLine(className: string, dashed: boolean, width: number): SVGSVGElement {
  return svg(
    'svg',
    { class: `legend__glyph ${className}`, viewBox: '0 0 28 10', width: 28, height: 10, 'aria-hidden': 'true' },
    svg('line', { x1: 2, x2: 26, y1: 5, y2: 5, 'stroke-width': width, 'stroke-dasharray': dashed ? '4 3' : undefined, 'stroke-linecap': 'round' }),
  );
}

export class BottomBar implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkContext;
  private readonly bag = new DisposeBag();
  private readonly strip = h('nav', { class: 'layer-strip', 'aria-label': 'Layers' });
  private readonly stats = h('div', { class: 'stats' });
  private readonly perf = h('span', { class: 'stats__perf', title: 'GPU draw calls and triangles of the last frame' });

  constructor(ctx: NetworkContext, viewport: Viewport) {
    this.ctx = ctx;
    const legend = h(
      'div',
      { class: 'legend', 'aria-label': 'Legend' },
      h('span', { class: 'legend__item', title: 'Positive weight: solid line' }, legendLine('is-pos', false, 2.4), '+ weight'),
      h('span', { class: 'legend__item', title: 'Negative weight: dashed line' }, legendLine('is-neg', true, 2.4), '− weight'),
      h(
        'span',
        { class: 'legend__item', title: 'Line thickness grows with |w|' },
        svg('svg', { class: 'legend__glyph', viewBox: '0 0 28 10', width: 28, height: 10, 'aria-hidden': 'true' }, svg('path', { d: 'M2 5 L26 2.5 L26 7.5 Z', class: 'legend__wedge' })),
        'thickness = |w|',
      ),
      h('span', { class: 'legend__item', title: 'Neuron colour/size: |activation|' }, h('span', { class: 'legend__dots', 'aria-hidden': 'true' }, h('i'), h('i'), h('i')), 'brightness = |a|'),
    );
    this.el = h('div', { class: 'bottombar__content' }, this.strip, legend, h('div', { class: 'stats-wrap' }, this.stats, this.perf));

    this.bag.add(ctx.model.events.on('structure', () => this.render()));
    this.bag.add(ctx.store.watch((s) => s.selection, () => this.highlight()));
    let last = '';
    this.bag.add(
      viewport.onAfterRender(() => {
        const text = `${viewport.stats.calls} draw calls · ${formatCount(viewport.stats.triangles)} triangles`;
        if (text !== last) this.perf.textContent = last = text;
      }),
    );
    this.render();
  }

  dispose(): void {
    this.bag.dispose();
  }

  private render(): void {
    const { model, actions } = this.ctx;
    const { spec, topology } = model;
    const chips = spec.layers.flatMap((layer, l) => {
      const chip = h(
        'button',
        { type: 'button', class: 'layer-chip', dataset: { layer: String(l) }, title: `Select ${layer.name}`, onClick: () => actions.select({ kind: 'layer', layer: l }) },
        h('span', { class: 'layer-chip__name' }, layer.name),
        h('span', { class: 'layer-chip__meta' }, layer.role === 'input' ? `${layer.size}` : `${layer.size} · ${ACTIVATIONS[layer.activation].label}`),
      );
      return l === 0 ? [chip] : [h('span', { class: 'layer-strip__arrow', 'aria-hidden': 'true' }, icon('arrowRight', 12)), chip];
    });
    replaceChildren(this.strip, chips);
    replaceChildren(
      this.stats,
      h('span', null, `${formatCount(topology.neuronCount)} neurons`),
      h('span', null, `${formatCount(topology.connectionCount)} weights`),
      h('span', null, `${formatCount(model.totalParameters())} parameters`),
    );
    this.highlight();
  }

  private highlight(): void {
    const selection = this.ctx.store.state.selection;
    for (const chip of this.strip.querySelectorAll<HTMLElement>('.layer-chip')) {
      const l = Number(chip.dataset.layer);
      const selected = selection?.kind === 'layer' && selection.layer === l;
      const related =
        (selection?.kind === 'neuron' && selection.layer === l) || (selection?.kind === 'connection' && (selection.layer === l || selection.layer - 1 === l));
      chip.classList.toggle('is-selected', selected);
      chip.classList.toggle('is-related', !selected && related);
      chip.setAttribute('aria-pressed', String(selected));
    }
  }
}
