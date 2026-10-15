/** Bottom bar of the Experiments workspace: the measured change, the baseline of the bars, a legend. */
import { DisposeBag } from '../../core/DisposeBag';
import { METRICS } from '../../experiments/metrics';
import { h, replaceChildren } from '../../ui/dom';
import { segmented, type Segmented } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { ExperimentsWorkspace } from './ExperimentsWorkspace';

export class ExperimentBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly caption = h('div', { class: 'pipeline__caption ex-caption', 'aria-live': 'polite' });
  private readonly legend = h('div', { class: 'ex-legend' });
  private readonly baseline: Segmented<'original' | 'experiment'>;

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.baseline = segmented<'original' | 'experiment'>({
      options: [
        { value: 'original', label: 'Original model', title: 'Bars: each component switched off alone in the untouched model' },
        { value: 'experiment', label: 'With interventions', title: 'Bars: each component switched off on top of the current interventions (reveals components that take over)' },
      ],
      value: ws.settings.baseline,
      ariaLabel: 'Bars measured on',
      onChange: (v) => ws.update({ baseline: v }),
    });
    this.el = h(
      'div',
      { class: 'pipeline ex-bar' },
      this.caption,
      h('div', { class: 'ex-bar__baseline' }, h('span', { class: 'field__label' }, 'Bars measured on'), this.baseline.el),
      this.legend,
    );
    for (const e of ['result', 'busy', 'settings', 'model'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const r = ws.result;
    this.baseline.set(ws.settings.baseline);
    const source = !r ? '' : r.source === 'browser' ? 'computed in your browser' : r.source === 'live' ? `GPT-2, computed live${r.seconds !== undefined ? ` in ${r.seconds.toFixed(2)} s` : ''}` : 'recorded result';
    replaceChildren(
      this.caption,
      h('span', { class: 'pipeline__badge' }, ws.busy ? '…' : 'Δ'),
      r
        ? h('span', null, h('strong', null, ws.summary()), h('span', { class: 'muted' }, ` — ${ws.experiment.interventions.filter((i) => i.enabled).length || 'no'} intervention(s), ${source}${ws.busy ? ' · updating…' : ''}`))
        : h('span', null, ws.busy ? 'Running the experiment…' : ws.modelKind === 'gpt2' && !ws.live ? 'Start the backend (npm run llm) to run your own GPT-2 experiments, or pick a recorded preset.' : 'No result yet.'),
    );
    const signed = METRICS[ws.metric].signed;
    replaceChildren(
      this.legend,
      signed
        ? [
            h('span', { class: 'ex-legend__item' }, h('span', { class: 'ex-legend__swatch is-pos' }), 'toward you: the target relies on it'),
            h('span', { class: 'ex-legend__item' }, h('span', { class: 'ex-legend__swatch is-neg' }), 'behind the wall: it works against the target'),
          ]
        : h('span', { class: 'ex-legend__item' }, h('span', { class: 'ex-legend__swatch is-neutral' }), 'bar: how much the whole prediction changes (KL)'),
      h('span', { class: 'ex-legend__item' }, h('span', { class: 'ex-legend__swatch is-change' }), 'residual: √(relative change)'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
