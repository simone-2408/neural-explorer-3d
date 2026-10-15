/** Bottom bar of the CNN workspace: training transport, speed, counters, metrics, and the kernel slide. */
import { DisposeBag } from '../../core/DisposeBag';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button, segmented, type Segmented } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { Panel } from '../network/context';
import type { CnnSpeed, CnnWorkspace } from './CnnWorkspace';

export class CnnBar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: CnnWorkspace;
  private readonly playButton: HTMLButtonElement;
  private readonly speed: Segmented<CnnSpeed>;
  private readonly counters = h('div', { class: 'training-bar__counters' });
  private readonly metrics = h('div', { class: 'training-bar__metrics' });
  private readonly slide = h('div', { class: 'cnn-slide' });

  constructor(ws: CnnWorkspace) {
    this.ws = ws;
    this.playButton = button({ icon: 'play', label: 'Train', title: 'Start / pause training (Space)', small: true, variant: 'primary', onClick: () => ws.toggle() });
    this.speed = segmented<CnnSpeed>({
      options: [
        { value: 'slow', label: 'Slow', title: 'About 6 steps per second: watch the filters change' },
        { value: 'normal', label: '1×', title: 'One step per frame' },
        { value: 'fast', label: '6×', title: 'Six steps per frame' },
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
        button({ label: 'Epoch', title: 'One pass over the training images', small: true, onClick: () => ws.stepEpoch() }),
        button({ icon: 'reset', label: 'Reset', title: 'New random weights', small: true, onClick: () => ws.reset() }),
      ),
      h('div', { class: 'training-bar__speed' }, h('span', { class: 'field__label' }, 'Speed'), this.speed.el),
      this.counters,
      this.metrics,
      this.slide,
    );
    for (const e of ['progress', 'running', 'settings', 'model', 'slide', 'selection'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    replaceChildren(this.playButton, icon(ws.running ? 'pause' : 'play'), h('span', { class: 'btn__label' }, ws.running ? 'Pause' : 'Train'));
    this.speed.set(ws.settings.speed);
    const t = ws.trainer;
    replaceChildren(this.counters, h('span', null, 'Epoch ', h('strong', null, formatCount(t.epoch))), h('span', null, 'Step ', h('strong', null, formatCount(t.step))));
    const last = t.history[t.history.length - 1];
    replaceChildren(
      this.metrics,
      last ? h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Train'), ` loss ${formatValue(last.train.loss, 3)} · acc ${formatValue(last.train.accuracy * 100, 1)}%`) : null,
      last ? h('span', { class: 'metric' }, h('span', { class: 'metric__label' }, 'Test'), ` loss ${formatValue(last.test.loss, 3)} · acc ${formatValue(last.test.accuracy * 100, 1)}%`) : null,
    );
    const s = ws.selection;
    const stage = s && s.kind !== 'output' ? ws.stages[s.stage] : null;
    if (ws.slide) {
      const st = ws.stages[ws.slide.stage];
      replaceChildren(
        this.slide,
        h('span', { class: 'muted' }, `Sliding filter ${ws.slide.map + 1}: cell ${Math.min(ws.slide.count + 1, st.size * st.size)} / ${st.size * st.size}`),
        button({ icon: 'close', label: 'Stop', small: true, variant: 'ghost', title: 'Stop the animation (Esc)', onClick: () => ws.stopSlide() }),
      );
    } else if (stage?.kind === 'conv' && s && s.kind !== 'output') {
      replaceChildren(this.slide, button({ icon: 'play', label: `Slide filter ${s.map + 1}`, small: true, title: 'Animate this filter moving over its input, cell by cell', onClick: () => ws.startSlide(s.stage, s.map) }));
    } else replaceChildren(this.slide);
  }

  dispose(): void {
    this.bag.dispose();
  }
}
