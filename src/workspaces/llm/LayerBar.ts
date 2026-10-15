/** Bottom bar of the LLM workspace: a slider through the layers, with play and a caption. */
import { DisposeBag } from '../../core/DisposeBag';
import { showToken } from '../../llm/client';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button } from '../../ui/widgets';
import { formatProbability } from '../../utils/format';
import type { Panel } from '../network/context';
import type { LlmWorkspace } from './LlmWorkspace';

export class LayerBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly range: HTMLInputElement;
  private readonly playButton: HTMLButtonElement;
  private readonly value = h('span', { class: 'layerbar__value' });
  private readonly caption = h('div', { class: 'pipeline__caption', 'aria-live': 'polite' });

  private readonly ws: LlmWorkspace;

  constructor(ws: LlmWorkspace) {
    this.ws = ws;
    this.range = h('input', { type: 'range', class: 'layerbar__range', min: 0, max: 12, step: 1, 'aria-label': 'Layer' });
    this.range.addEventListener('input', () => ws.setLayer(Number(this.range.value)));
    this.playButton = button({ icon: 'play', title: 'Move through the layers (Space)', small: true, variant: 'primary', onClick: () => ws.togglePlay() });
    this.el = h(
      'div',
      { class: 'pipeline layerbar' },
      h('div', { class: 'pipeline__transport' }, this.playButton),
      h('label', { class: 'layerbar__slider' }, h('span', { class: 'field__label' }, 'Layer'), this.range, this.value),
      this.caption,
    );
    for (const e of ['layer', 'analysis', 'selection'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const a = ws.analysis;
    const max = (a?.lens.length ?? 13) - 1;
    this.range.max = String(max);
    if (document.activeElement !== this.range) this.range.value = String(ws.layer);
    this.value.textContent = ws.layer === 0 ? 'embed' : `${ws.layer} / ${max}`;
    replaceChildren(this.playButton, icon(ws.playing ? 'pause' : 'play'));
    if (!a) {
      replaceChildren(this.caption, 'Waiting for an analysis…');
      return;
    }
    const p = ws.focus;
    const lens = a.lens[ws.layer];
    const [id, prob] = lens.top[p][0];
    const word = (i: number) => showToken(a.words[String(i)] ?? `#${i}`);
    const final = word(a.predictions[p][0]);
    const where = ws.layer === 0 ? 'the embeddings' : `the residual stream after block ${ws.layer}`;
    replaceChildren(
      this.caption,
      h('span', { class: 'pipeline__badge' }, ws.layer === 0 ? 'embed' : `L${ws.layer}`),
      h(
        'span',
        null,
        `Decoding ${where} at “${showToken(a.tokens[p].text)}” gives ${word(id)} (${formatProbability(prob)}). The model’s final answer ${final} ranks #${lens.finalRank[p] + 1} here.`,
        ws.layer > 0 && ws.settings.arcsOn && ws.settings.view === 'tower' ? ` Arcs: head ${ws.settings.arcHead + 1} of block ${ws.layer}.` : '',
      ),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
