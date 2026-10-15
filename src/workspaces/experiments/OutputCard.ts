/** The measured prediction, before → after, pinned to the top-right corner of the 3D view. */
import { DisposeBag } from '../../core/DisposeBag';
import { h, replaceChildren } from '../../ui/dom';
import { formatProbability } from '../../utils/format';
import type { Panel } from '../network/context';
import type { ExperimentsWorkspace } from './ExperimentsWorkspace';

export class OutputCard implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.el = h('button', { type: 'button', class: 'tt-pred tt-pred--main ex-hud', title: 'The measured prediction — click for the full comparison', onClick: () => ws.select(null) });
    for (const e of ['result', 'busy', 'model'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const r = this.ws.result;
    this.el.hidden = !r;
    if (!r) return;
    const f = (t: string) => this.ws.format(t);
    const row = (name: string, p: number, cls: string) =>
      h('span', { class: `tt-pred__row ex-output__row ${cls}` }, h('span', { class: 'tt-pred__word' }, name), h('span', { class: 'tt-pred__bar', style: `--p: ${p}` }), h('span', { class: 'tt-pred__value' }, formatProbability(p)));
    replaceChildren(
      this.el,
      h('span', { class: 'ex-output__title' }, `p(${f(r.target.text)})`, this.ws.busy ? h('span', { class: 'ex-pending' }) : null),
      row('original', r.before.pTarget, 'is-before'),
      row('now', r.after.pTarget, 'is-after'),
      h('span', { class: 'ex-output__top' }, `most likely now: ${r.after.top.slice(0, 3).map((t) => `${f(t.text)} ${formatProbability(t.p)}`).join(' · ')}`),
    );
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }
}
