/**
 * Bottom bar of the Transformer workspace: the computation as a pipeline
 * (Tokens → Embedding → Attention/MLP per layer → Prediction) with transport
 * controls and a caption for the current step.
 */
import { DisposeBag } from '../../core/DisposeBag';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button } from '../../ui/widgets';
import type { Panel } from '../network/context';
import { PIPELINE_SPEEDS, type PipelinePlayer } from './pipeline';
import type { TransformerWorkspace } from './TransformerWorkspace';

export class PipelineBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly player: PipelinePlayer;
  private readonly playButton: HTMLButtonElement;
  private readonly steps = h('ol', { class: 'step-list pipeline__steps', 'aria-label': 'Pipeline steps' });
  private readonly caption = h('div', { class: 'pipeline__caption', 'aria-live': 'polite' });
  private readonly speed = h('div', { class: 'segmented segmented--compact', role: 'group', 'aria-label': 'Animation speed' });

  constructor(ws: TransformerWorkspace, player: PipelinePlayer) {
    this.player = player;
    this.playButton = button({ icon: 'play', title: 'Play the computation step by step (Space)', small: true, variant: 'primary', onClick: () => player.toggle() });
    for (const s of PIPELINE_SPEEDS) {
      this.speed.append(h('button', { type: 'button', class: 'segmented__item', dataset: { speed: String(s) }, onClick: () => player.setSpeed(s) }, `${s}×`));
    }
    this.el = h(
      'div',
      { class: 'pipeline' },
      h(
        'div',
        { class: 'pipeline__transport' },
        button({ icon: 'restart', title: 'Back to the tokens (Home)', small: true, variant: 'ghost', onClick: () => player.restart() }),
        button({ icon: 'stepBack', title: 'Previous step (,)', small: true, variant: 'ghost', onClick: () => player.previous() }),
        this.playButton,
        button({
          icon: 'stepForward',
          title: 'Next step (.)',
          small: true,
          variant: 'ghost',
          onClick: () => {
            player.pause();
            player.next();
          },
        }),
      ),
      this.steps,
      this.speed,
      this.caption,
    );
    this.bag.add(player.events.on('change', () => this.render()));
    this.bag.add(ws.events.on('trace', () => this.render()));
    this.render();
  }

  private render(): void {
    const { player } = this;
    replaceChildren(this.playButton, icon(player.playing ? 'pause' : 'play'));
    this.playButton.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
    for (const b of this.speed.querySelectorAll<HTMLElement>('button')) b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === player.speed));
    replaceChildren(
      this.steps,
      player.steps.map((step, k) => {
        const status = k < player.step || (k === player.step && !player.animating) ? 'done' : k === player.step ? 'current' : 'pending';
        return h(
          'li',
          { class: `step-list__item is-${status}${k === player.step ? ' is-active' : ''}` },
          h(
            'button',
            {
              type: 'button',
              title: step.caption,
              onClick: () => {
                player.pause();
                player.goTo(k, k === player.step + 1);
              },
            },
            step.label,
          ),
        );
      }),
    );
    const step = player.current;
    replaceChildren(
      this.caption,
      h('span', { class: 'pipeline__badge' }, `${player.step + 1}/${player.steps.length}`),
      h('span', null, player.complete && !player.playing ? `${step.caption} Press ▶ to watch the whole computation step by step.` : step.caption),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
