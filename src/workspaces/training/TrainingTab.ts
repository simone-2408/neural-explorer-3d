/**
 * The "Training" tab: three modes on top of the same shell.
 *   Train     — train one network, inspect gradients, landscape, backprop
 *   Break it  — the same, with the Break-the-Network scenarios
 *   Compare   — two networks side by side
 */
import type { Workspace, WorkspaceHost } from '../../app/workspace';
import { h } from '../../ui/dom';
import { segmented, type Segmented } from '../../ui/widgets';
import { CompareWorkspace } from './CompareWorkspace';
import { TrainingWorkspace } from './TrainingWorkspace';

export type TrainingMode = 'train' | 'break' | 'compare';

export class TrainingTab implements Workspace {
  readonly id = 'training' as const;
  readonly train = new TrainingWorkspace();
  private compareValue: CompareWorkspace | null = null;
  mode: TrainingMode = 'train';
  private host: WorkspaceHost | null = null;
  private active: Workspace | null = null;
  private readonly switcher: Segmented<TrainingMode>;
  private readonly bar: HTMLElement;

  constructor() {
    this.switcher = segmented<TrainingMode>({
      options: [
        { value: 'train', label: 'Train', title: 'Train one network and look inside it' },
        { value: 'break', label: 'Break it', title: 'Classic mistakes: what goes wrong and why' },
        { value: 'compare', label: 'Compare', title: 'Two networks side by side' },
      ],
      value: this.mode,
      ariaLabel: 'Training mode',
      onChange: (mode) => this.setMode(mode),
    });
    this.bar = h('div', { class: 'mode-switch' }, this.switcher.el);
  }

  get compare(): CompareWorkspace {
    if (!this.compareValue) this.compareValue = new CompareWorkspace();
    return this.compareValue;
  }

  mount(host: WorkspaceHost): void {
    this.host = host;
    this.mountMode();
  }

  unmount(): void {
    this.active?.unmount();
    this.active = null;
    this.bar.remove();
    this.host = null;
  }

  setMode(mode: TrainingMode): void {
    if (mode === this.mode && this.active) return;
    this.mode = mode;
    this.switcher.set(mode);
    if (!this.host) return;
    this.active?.unmount();
    this.active = null;
    for (const slot of Object.values(this.host.slots)) slot.replaceChildren();
    this.host.store.set({ selection: null, hover: null });
    this.mountMode();
  }

  private mountMode(): void {
    const host = this.host!;
    if (this.mode === 'compare') this.active = this.compare;
    else {
      this.train.mode = this.mode;
      if (this.mode === 'train') this.train.clearScenario();
      this.active = this.train;
    }
    this.active.mount(host);
    host.slots.left.prepend(this.bar);
  }
}
