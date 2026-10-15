/** Camera toolbar and navigation hint, overlaid on the 3D viewport. */
import { DisposeBag } from '../../core/DisposeBag';
import { CAMERA_VIEWS } from '../../three/CameraController';
import { h } from '../../ui/dom';
import { button } from '../../ui/widgets';
import type { NetworkContext, Panel } from './context';

export class ViewportToolbar implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();

  /** `extra` buttons are appended as an additional group (workspace-specific tools). */
  constructor(ctx: NetworkContext, extra: HTMLElement[] = []) {
    const { actions, store } = ctx;
    const autoRotate = button({
      icon: 'rotate',
      title: 'Auto-rotate (A)',
      small: true,
      variant: 'ghost',
      pressed: store.state.view.autoRotate,
      onClick: () => actions.setView({ autoRotate: !store.state.view.autoRotate }),
    });
    const faceLayer = button({ icon: 'face', label: 'Layer', title: 'Look at the selected layer face-on (L)', small: true, variant: 'ghost', onClick: () => actions.faceLayer() });

    const toolbar = h(
      'div',
      { class: 'viewport-toolbar', role: 'toolbar', 'aria-label': 'Camera and tools' },
      h(
        'div',
        { class: 'toolbar-group' },
        button({ icon: 'reset', label: 'Reset', title: 'Reset camera (R)', small: true, variant: 'ghost', onClick: () => actions.resetCamera() }),
        button({ icon: 'focus', label: 'Focus', title: 'Focus the selection (F or double-click)', small: true, variant: 'ghost', onClick: () => actions.focusSelection() }),
      ),
      h(
        'div',
        { class: 'toolbar-group' },
        CAMERA_VIEWS.map((v) => button({ label: v.label, title: `${v.description} (${v.key})`, small: true, variant: 'ghost', onClick: () => actions.setCameraView(v.id) })),
      ),
      h('div', { class: 'toolbar-group' }, faceLayer, autoRotate),
      extra.length ? h('div', { class: 'toolbar-group toolbar-group--tools' }, extra) : null,
    );
    const hint = h(
      'div',
      { class: 'viewport-hint', 'aria-hidden': 'true' },
      h('span', null, h('kbd', null, 'drag'), ' rotate'),
      h('span', null, h('kbd', null, 'right-drag'), ' / ', h('kbd', null, '⇧ drag'), ' pan'),
      h('span', null, h('kbd', null, 'scroll'), ' zoom'),
      h('span', null, h('kbd', null, 'click'), ' inspect'),
      h('span', null, h('kbd', null, '?'), ' help'),
    );
    this.el = h('div', { class: 'viewport-ui' }, toolbar, hint);

    this.bag.add(store.watch((s) => s.view.autoRotate, (on) => autoRotate.setAttribute('aria-pressed', String(on))));
  }

  dispose(): void {
    this.bag.dispose();
    this.el.remove();
  }
}
