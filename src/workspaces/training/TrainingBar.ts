/** Bottom bar of the Training workspace: transport, speed and live metrics. */
import { DisposeBag } from '../../core/DisposeBag';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button, segmented, type Segmented } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { Panel } from '../network/context';
import type { TrainingSpeed } from '../../training/session';
import type { TrainingWorkspace } from './TrainingWorkspace';

export class TrainingBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly playButton: HTMLButtonElement;
  private readonly speed: Segmented<TrainingSpeed>;
  private readonly counters = h('div', { class: 'training-bar__counters' });
  private readonly metrics = h('div', { class: 'training-bar__metrics' });

  constructor(private readonly ws: TrainingWorkspace) {
    this.playButton = button({ icon: 'play', label: 'Train', title: 'Start / pause training (Space)', small: true, variant: 'primary', onClick: () => ws.toggle() });
    this.speed = segmented<TrainingSpeed>({
      options: [
        { value: 'slow', label: 'Slow', title: 'About 8 steps per second: watch every update' },
        { value: 'normal', label: '1×', title: 'One step per frame' },
        { value: 'fast', label: '12×', title: '12 steps per frame' },
        { value: 'max', label: 'Max', title: 'As many steps as fit in each frame' },
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
        button({ label: 'Epoch', title: 'Train until one more pass over the training set is done', small: true, onClick: () => ws.stepEpoch() }),
        button({ icon: 'reset', label: 'Reset', title: 'Re-initialize the weights and restart', small: true, onClick: () => ws.reset() }),
        button({ icon: 'bolt', label: 'Backprop step', title: 'Animate one training step on the probe point (B)', small: true, onClick: () => ws.animateBackprop() }),
      ),
      h('div', { class: 'training-bar__speed' }, h('span', { class: 'field__label' }, 'Speed'), this.speed.el),
      this.counters,
      this.metrics,
      h(
        'div',
        { class: 'training-bar__files' },
        button({ icon: 'download', title: 'Export the session (settings, weights, history)', small: true, variant: 'ghost', onClick: () => ws.exportSession() }),
        button({ icon: 'upload', title: 'Import a session', small: true, variant: 'ghost', onClick: () => void ws.importSession() }),
      ),
    );
    this.bag.add(ws.events.on('progress', () => this.render()));
    this.bag.add(ws.events.on('running', () => this.render()));
    this.bag.add(ws.events.on('settings', () => this.render()));
    this.render();
  }

  private render(): void {
    const ws = this.ws;
    replaceChildren(this.playButton, icon(ws.running ? 'pause' : 'play'), h('span', { class: 'btn__label' }, ws.running ? 'Pause' : 'Train'));
    this.speed.set(ws.settings.speed);
    const t = ws.trainer;
    replaceChildren(this.counters, h('span', null, 'Epoch ', h('strong', null, formatCount(t.epochCount))), h('span', null, 'Step ', h('strong', null, formatCount(t.stepCount))));
    const last = t.history[t.history.length - 1];
    if (!last) {
      this.metrics.replaceChildren();
      return;
    }
    replaceChildren(
      this.metrics,
      h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Train'), ` loss ${formatValue(last.train.loss, 4)} · acc ${formatValue(last.train.accuracy * 100, 1)}%`),
      last.test ? h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Test'), ` loss ${formatValue(last.test.loss, 4)} · acc ${formatValue(last.test.accuracy * 100, 1)}%`) : null,
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
