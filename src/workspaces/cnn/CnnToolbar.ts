/** Camera toolbar of the CNN workspace, overlaid on the 3D view. */
import { h } from '../../ui/dom';
import { button, toggle } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { CnnWorkspace } from './CnnWorkspace';

export class CnnToolbar implements Panel {
  readonly el: HTMLElement;

  constructor(ws: CnnWorkspace) {
    const views = [
      { id: 'iso', label: '3D', key: '1', title: 'Three-quarter view' },
      { id: 'side', label: 'Front', key: '2', title: 'Front view: the maps as flat images' },
      { id: 'top', label: 'Top', key: '3', title: 'Top view' },
      { id: 'front', label: 'Side', key: '4', title: 'Side view: values as depth' },
    ] as const;
    const values = toggle({ label: 'Heights', title: 'Show values as heights (otherwise colour only)', checked: ws.settings.showValues, onChange: (v) => ws.update({ showValues: v }) });
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
      h('div', { class: 'toolbar-group' }, values.el),
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'click'), ' inspect a cell or a filter'),
      h('span', null, h('kbd', null, 'Space'), ' train'),
      h('span', null, h('kbd', null, 'N'), ' next image'),
      h('span', null, h('kbd', null, '←→↑↓'), ' move'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);
  }

  dispose(): void {
    this.el.remove();
  }
}
