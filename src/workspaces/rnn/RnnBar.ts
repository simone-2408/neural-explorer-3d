/** Bottom bar of the RNN workspace: training transport, speed, counters and metrics. */
import { DisposeBag } from '../../core/DisposeBag';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button, segmented, type Segmented } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { Panel } from '../network/context';
import type { RnnSpeed, RnnWorkspace } from './RnnWorkspace';

export class RnnBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: RnnWorkspace;
  private readonly playButton: HTMLButtonElement;
  private readonly speed: Segmented<RnnSpeed>;
  private readonly counters = h('div', { class: 'training-bar__counters' });
  private readonly metrics = h('div', { class: 'training-bar__metrics' });

  constructor(ws: RnnWorkspace) {
    this.ws = ws;
    this.playButton = button({ icon: 'play', label: 'Train', title: 'Start / pause training (Space)', small: true, variant: 'primary', onClick: () => ws.toggle() });
    this.speed = segmented<RnnSpeed>({
      options: [
        { value: 'slow', label: 'Slow', title: 'About 5 steps per second' },
        { value: 'normal', label: '1×', title: 'One step per frame' },
        { value: 'fast', label: '4×', title: 'Four steps per frame' },
      ],
      value: ws.settings.speed,
      ariaLabel: 'Training speed',
      onChange: (speed) => ws.update({ speed }),
    });
    this.el = h(
      'div',
      { class: 'training-bar' },
      h(
        'div',
        { class: 'training-bar__transport' },
        this.playButton,
        button({ icon: 'stepForward', label: 'Step', title: 'One mini-batch step', small: true, onClick: () => ws.stepOnce() }),
        button({ label: '+100', title: '100 steps at once', small: true, onClick: () => ws.stepMany(100) }),
        button({ icon: 'reset', label: 'Reset', title: 'New random weights', small: true, onClick: () => ws.reset() }),
      ),
      h('div', { class: 'training-bar__speed' }, h('span', { class: 'field__label' }, 'Speed'), this.speed.el),
      this.counters,
      this.metrics,
    );
    for (const e of ['progress', 'running', 'settings', 'model'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    replaceChildren(this.playButton, icon(ws.running ? 'pause' : 'play'), h('span', { class: 'btn__label' }, ws.running ? 'Pause' : 'Train'));
    this.speed.set(ws.settings.speed);
    const t = ws.trainer;
    replaceChildren(this.counters, h('span', null, 'Step ', h('strong', null, formatCount(t.step))), h('span', null, '‖grad‖ ', h('strong', null, formatValue(t.lastGradNorm, 3))));
    const last = t.history[t.history.length - 1];
    replaceChildren(
      this.metrics,
      last ? h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Train'), ` loss ${formatValue(last.train.loss, 3)} · acc ${formatValue(last.train.accuracy * 100, 1)}%`) : null,
      last ? h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Test'), ` loss ${formatValue(last.test.loss, 3)} · acc ${formatValue(last.test.accuracy * 100, 1)}%`) : null,
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
