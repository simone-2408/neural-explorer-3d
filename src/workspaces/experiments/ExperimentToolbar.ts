/** Camera toolbar of the Experiments workspace, overlaid on the 3D view. */
import { h } from '../../ui/dom';
import { button } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { ExperimentsWorkspace } from './ExperimentsWorkspace';

export class ExperimentToolbar implements Panel {
  readonly el: HTMLElement;

  constructor(ws: ExperimentsWorkspace) {
    const views = [
      { id: 'iso', label: '3D', key: '1', title: 'Three-quarter view: bar lengths' },
      { id: 'side', label: 'Front', key: '2', title: 'Front view: the two walls as flat maps' },
      { id: 'top', label: 'Top', key: '3', title: 'Top view: bars toward you and behind the wall' },
      { id: 'front', label: 'Side', key: '4', title: 'Side view' },
    ] as const;
    const toolbar = h(
      'div',
      { class: 'viewport-toolbar', role: 'toolbar', 'aria-label': 'Camera' },
      h(
        'div',
        { class: 'toolbar-group' },
        button({ icon: 'reset', label: 'Reset', title: 'Reset camera (R)', small: true, variant: 'ghost', onClick: () => ws.setCameraView('iso') }),
        button({ icon: 'focus', label: 'Focus', title: 'Focus the selection (F or double-click)', small: true, variant: 'ghost', onClick: () => ws.focusSelection() }),
      ),
      h('div', { class: 'toolbar-group' }, views.map((v) => button({ label: v.label, title: `${v.title} (${v.key})`, small: true, variant: 'ghost', onClick: () => ws.setCameraView(v.id) }))),
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'click'), ' inspect'),
      h('span', null, h('kbd', null, 'shift-click'), ' switch off / on'),
      h('span', null, h('kbd', null, 'X'), ' toggle selected'),
      h('span', null, h('kbd', null, '⌘Z'), ' undo'),
      h('span', null, h('kbd', null, 'drag'), ' rotate'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);
  }

  dispose(): void {
    this.el.remove();
  }
}
