/** Camera toolbar of the Transformer workspace, overlaid on the 3D view. */
import { DisposeBag } from '../../core/DisposeBag';
import { h } from '../../ui/dom';
import { button, segmented } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { TransformerWorkspace } from './TransformerWorkspace';

export class TransformerToolbar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();

  constructor(ws: TransformerWorkspace) {
    const mode = segmented<'model' | 'embeddings'>({
      options: [
        { value: 'model', label: 'Model', title: 'The computation in 3D (E)' },
        { value: 'embeddings', label: 'Embeddings', title: 'The embedding space, projected to 3D (E)' },
      ],
      value: ws.settings.view,
      ariaLabel: 'View',
      onChange: (v) => ws.update({ view: v }),
    });
    const views = [
      { id: 'iso', label: '3D', key: '1', title: 'Three-quarter view' },
      { id: 'side', label: 'Front', key: '2', title: 'Front view: tokens left → right, layers bottom → top' },
      { id: 'top', label: 'Top', key: '3', title: 'Top view' },
      { id: 'front', label: 'Side', key: '4', title: 'Side view: the 32 numbers of each vector along the depth' },
    ] as const;
    const toolbar = h(
      'div',
      { class: 'viewport-toolbar', role: 'toolbar', 'aria-label': 'Camera and view' },
      h(
        'div',
        { class: 'toolbar-group' },
        button({ icon: 'reset', label: 'Reset', title: 'Reset camera (R)', small: true, variant: 'ghost', onClick: () => ws.resetCamera() }),
        button({ icon: 'focus', label: 'Focus', title: 'Focus the selection (F or double-click)', small: true, variant: 'ghost', onClick: () => ws.focusSelection() }),
      ),
      h(
        'div',
        { class: 'toolbar-group' },
        views.map((v) => button({ label: v.label, title: `${v.title} (${v.key})`, small: true, variant: 'ghost', onClick: () => ws.setCameraView(v.id) })),
      ),
      h('div', { class: 'toolbar-group toolbar-group--segmented' }, mode.el),
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'drag'), ' rotate'),
      h('span', null, h('kbd', null, 'right-drag'), ' pan'),
      h('span', null, h('kbd', null, 'scroll'), ' zoom'),
      h('span', null, h('kbd', null, 'click'), ' inspect'),
      h('span', null, h('kbd', null, 'Space'), ' play'),
      h('span', null, h('kbd', null, '←→↑↓'), ' move'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);
    this.bag.add(ws.events.on('settings', () => mode.set(ws.settings.view)));
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }
}
