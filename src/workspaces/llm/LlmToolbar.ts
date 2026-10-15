/** Camera toolbar of the LLM workspace. */
import { DisposeBag } from '../../core/DisposeBag';
import { h } from '../../ui/dom';
import { button, segmented } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { LlmWorkspace } from './LlmWorkspace';

export class LlmToolbar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();

  constructor(ws: LlmWorkspace) {
    const mode = segmented<'tower' | 'journey'>({
      options: [
        { value: 'tower', label: 'Layers', title: 'Logit-lens tower (J)' },
        { value: 'journey', label: 'Token journey', title: 'Vectors through the layers, projected to 3-D (J)' },
      ],
      value: ws.settings.view,
      ariaLabel: 'View',
      onChange: (v) => ws.update({ view: v }),
    });
    const views = [
      { id: 'iso', label: '3D', key: '1' },
      { id: 'side', label: 'Front', key: '2' },
      { id: 'top', label: 'Top', key: '3' },
      { id: 'front', label: 'Side', key: '4' },
    ] as const;
    const toolbar = h(
      'div',
      { class: 'viewport-toolbar', role: 'toolbar', 'aria-label': 'Camera and view' },
      h(
        'div',
        { class: 'toolbar-group' },
        button({ icon: 'reset', label: 'Reset', title: 'Reset camera (R)', small: true, variant: 'ghost', onClick: () => ws.setCameraView('iso') }),
        button({ icon: 'focus', label: 'Focus', title: 'Focus the selection (F or double-click)', small: true, variant: 'ghost', onClick: () => ws.focusSelection() }),
      ),
      h(
        'div',
        { class: 'toolbar-group' },
        views.map((v) => button({ label: v.label, title: `${v.label} view (${v.key})`, small: true, variant: 'ghost', onClick: () => ws.setCameraView(v.id) })),
      ),
      h('div', { class: 'toolbar-group toolbar-group--segmented' }, mode.el),
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'drag'), ' rotate'),
      h('span', null, h('kbd', null, 'scroll'), ' zoom'),
      h('span', null, h('kbd', null, 'click'), ' inspect'),
      h('span', null, h('kbd', null, '↑↓'), ' layers'),
      h('span', null, h('kbd', null, 'Space'), ' play'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);
    this.bag.add(ws.events.on('settings', () => mode.set(ws.settings.view)));
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }
}
