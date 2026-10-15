/** Camera toolbar of the RNN workspace: views, fold / unfold, gradients. */
import { DisposeBag } from '../../core/DisposeBag';
import { h } from '../../ui/dom';
import { button, toggle } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { RnnWorkspace } from './RnnWorkspace';

export class RnnToolbar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();

  constructor(ws: RnnWorkspace) {
    const views = [
      { id: 'iso', label: '3D', key: '1', title: 'Three-quarter view' },
      { id: 'side', label: 'Front', key: '2', title: 'Front view' },
      { id: 'top', label: 'Top', key: '3', title: 'Top view: values as depth' },
    ] as const;
    const fold = button({ label: 'Fold', title: 'Fold the steps into one cell with a loop (U)', small: true, variant: 'ghost', onClick: () => ws.setUnfold(!ws.unfolded) });
    const grads = toggle({ label: 'Gradients', title: '‖∂L/∂h_t‖ under each step, from backpropagation through time (G)', checked: ws.settings.showGradients, onChange: (v) => ws.update({ showGradients: v }) });
    const sync = () => {
      fold.querySelector('.btn__label')!.textContent = ws.unfolded ? 'Fold' : 'Unfold';
      grads.set(ws.settings.showGradients);
    };
    const toolbar = h(
      'div',
      { class: 'viewport-toolbar', role: 'toolbar', 'aria-label': 'Camera and view' },
      h(
        'div',
        { class: 'toolbar-group' },
        button({ icon: 'reset', label: 'Reset', title: 'Reset camera (R)', small: true, variant: 'ghost', onClick: () => ws.setCameraView('iso') }),
        button({ icon: 'focus', label: 'Focus', title: 'Focus the selection (F or double-click)', small: true, variant: 'ghost', onClick: () => ws.focusSelection() }),
      ),
      h('div', { class: 'toolbar-group' }, views.map((v) => button({ label: v.label, title: `${v.title} (${v.key})`, small: true, variant: 'ghost', onClick: () => ws.setCameraView(v.id) }))),
      h('div', { class: 'toolbar-group' }, fold, grads.el),
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'click'), ' inspect a step or an arrow'),
      h('span', null, h('kbd', null, 'Space'), ' train'),
      h('span', null, h('kbd', null, '←→'), ' step'),
      h('span', null, h('kbd', null, 'U'), ' fold'),
      h('span', null, h('kbd', null, 'N'), ' new example'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);
    this.bag.add(ws.events.on('settings', sync));
    sync();
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }
}
