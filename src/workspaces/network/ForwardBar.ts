/**
 * Forward-pass transport bar (bottom of the Neural Network workspace):
 * Restart · Previous · Play/Pause · Next · Speed, the list of steps and a
 * caption that explains the current step with its formula and values.
 */
import { DisposeBag } from '../../core/DisposeBag';
import { ACTIVATIONS } from '../../math/activations';
import { layerVectorSymbol } from '../../neural/topology';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { tex } from '../../ui/tex';
import { button } from '../../ui/widgets';
import { texNumber } from '../../utils/format';
import type { NetworkWorkspaceContext, Panel } from './context';
import { PLAYER_SPEEDS, type ForwardPlayer, type PlayerState } from './ForwardPlayer';

const MAX_VECTOR = 6;

function vectorTex(values: ArrayLike<number>): string {
  if (values.length > MAX_VECTOR) return `\\in \\mathbb{R}^{${values.length}}`;
  return `= \\left(${Array.from(values, (v) => texNumber(v, 2)).join(',\\ ')}\\right)`;
}

export class ForwardBar implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkWorkspaceContext;
  private readonly bag = new DisposeBag();
  private readonly playButton: HTMLButtonElement;
  private readonly steps = h('ol', { class: 'step-list', 'aria-label': 'Forward pass steps' });
  private readonly caption = h('div', { class: 'forward-caption', 'aria-live': 'polite' });
  private readonly speed = h('div', { class: 'segmented segmented--compact', role: 'group', 'aria-label': 'Animation speed' });
  private readonly cameraToggle: HTMLButtonElement;
  followCamera = false;

  constructor(ctx: NetworkWorkspaceContext, player: ForwardPlayer) {
    this.ctx = ctx;
    this.playButton = button({ icon: 'play', title: 'Play / pause (Space)', small: true, variant: 'primary', onClick: () => player.toggle() });
    this.cameraToggle = button({
      icon: 'camera',
      title: 'Signal view: the camera follows the layer being computed',
      small: true,
      variant: 'ghost',
      pressed: false,
      onClick: () => {
        this.followCamera = !this.followCamera;
        this.cameraToggle.setAttribute('aria-pressed', String(this.followCamera));
      },
    });
    for (const s of PLAYER_SPEEDS) {
      this.speed.append(h('button', { type: 'button', class: 'segmented__item', dataset: { speed: String(s) }, onClick: () => player.setSpeed(s) }, `${s}×`));
    }

    this.el = h(
      'div',
      { class: 'forward-bar', hidden: true },
      h('div', { class: 'forward-bar__badge' }, icon('play', 12), 'Forward pass'),
      h(
        'div',
        { class: 'forward-bar__transport' },
        button({ icon: 'restart', title: 'Restart (Home)', small: true, variant: 'ghost', onClick: () => player.restart() }),
        button({ icon: 'stepBack', title: 'Previous step (,)', small: true, variant: 'ghost', onClick: () => player.previous() }),
        this.playButton,
        button({ icon: 'stepForward', title: 'Next step (.)', small: true, variant: 'ghost', onClick: () => player.next() }),
      ),
      this.speed,
      this.steps,
      this.caption,
      h('div', { class: 'forward-bar__end' }, this.cameraToggle, button({ icon: 'close', title: 'Exit the forward pass (Esc)', small: true, variant: 'ghost', onClick: () => ctx.actions.toggleForward() })),
    );

    this.bag.add(player.events.on('change', (state) => this.render(state)));
    this.bag.add(ctx.model.events.on('activations', () => this.render(player.state)));
    this.render(player.state);
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }

  private render(state: PlayerState): void {
    this.el.hidden = !state.active;
    if (!state.active) return;
    const { spec, trace } = this.ctx.model;
    const playing = state.playing;
    replaceChildren(this.playButton, icon(playing ? 'pause' : 'play'));
    this.playButton.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    for (const b of this.speed.querySelectorAll<HTMLElement>('button')) b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === state.speed));

    replaceChildren(
      this.steps,
      spec.layers.map((layer, l) => {
        const status = l <= state.step ? 'done' : l === state.animating ? 'current' : 'pending';
        return h(
          'li',
          { class: `step-list__item is-${status}`, title: layer.name },
          h('button', { type: 'button', onClick: () => this.ctx.actions.select({ kind: 'layer', layer: l }) }, layer.role === 'hidden' ? layer.prefix : layer.role === 'input' ? 'In' : 'Out'),
        );
      }),
    );

    // Caption: what happens in the current step, with the formula and the values.
    const l = state.animating ?? state.step;
    const layer = spec.layers[l];
    const vec = layerVectorSymbol(spec, l);
    let text: string;
    let formula: string;
    if (l === 0) {
      text = `Step 1/${spec.layers.length} · ${layer.name}: the data point enters the network.`;
      formula = `x ${vectorTex(trace.a[0])}`;
    } else {
      const activation = ACTIVATIONS[layer.activation];
      const prev = layerVectorSymbol(spec, l - 1);
      const f = activation.id === 'linear' ? '' : activation.tex;
      const affine = `W^{(${l})}${prev} + b^{(${l})}`;
      if (state.phase === 'travel') {
        text = `Step ${l + 1}/${spec.layers.length} · ${layer.name}: every connection carries w·a (pulse brightness ∝ |w·a|).`;
        formula = `z^{(${l})} = ${affine}`;
      } else if (state.phase === 'activate') {
        text = `Step ${l + 1}/${spec.layers.length} · ${layer.name}: each neuron applies ${activation.label} to its weighted sum.`;
        formula = f ? `${vec} = ${f}\\!\\left(z^{(${l})}\\right)` : `${vec} = z^{(${l})}`;
      } else {
        text = l === spec.layers.length - 1 ? `Done · ${layer.name}: the network output.` : `Step ${l + 1}/${spec.layers.length} · ${layer.name} computed.`;
        formula = `${vec} ${vectorTex(trace.a[l])}`;
      }
    }
    replaceChildren(this.caption, h('span', { class: 'forward-caption__text' }, text), tex(formula));
  }
}
