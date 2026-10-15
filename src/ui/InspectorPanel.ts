/**
 * Host for the right-hand Inspector. Workspaces create an InspectorView for
 * the current selection; the panel swaps views and coalesces value refreshes
 * to one per animation frame (sliders can fire many events per frame).
 */
import { frameThrottle } from '../utils/schedule';
import { h, replaceChildren, type Child } from './dom';

export interface InspectorView {
  readonly el: HTMLElement;
  /** Re-renders the live values (called when the model changes). */
  update(): void;
  dispose?(): void;
}

export class InspectorPanel {
  readonly el: HTMLElement;
  private current: InspectorView | null = null;
  private readonly refreshSoon = frameThrottle(() => this.current?.update());

  constructor(container: HTMLElement) {
    this.el = h('div', { class: 'inspector' });
    container.appendChild(this.el);
  }

  show(view: InspectorView, keepScroll = false): void {
    const scroll = this.el.parentElement?.scrollTop ?? 0;
    this.refreshSoon.cancel();
    this.current?.dispose?.();
    this.current = view;
    this.el.replaceChildren(view.el);
    if (this.el.parentElement) this.el.parentElement.scrollTop = keepScroll ? scroll : 0;
  }

  refresh(): void {
    this.refreshSoon();
  }

  dispose(): void {
    this.refreshSoon.cancel();
    this.current?.dispose?.();
    this.current = null;
    this.el.remove();
  }
}

/** A region whose content is fully re-rendered by `update()`. */
export interface LiveRegion {
  el: HTMLElement;
  update(): void;
}

export function live(render: () => Child, className = ''): LiveRegion {
  const el = h('div', { class: `live ${className}`.trim() });
  const update = () => replaceChildren(el, render());
  update();
  return { el, update };
}
