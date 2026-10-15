/** Tooltip, toasts and modal dialogs. */
import { h, type Child } from './dom';
import { icon } from './icons';

// ------------------------------------------------------------------ tooltip

export class Tooltip {
  readonly el: HTMLElement;
  private readonly container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.el = h('div', { class: 'tooltip', role: 'tooltip', hidden: true });
    container.appendChild(this.el);
  }

  /** Shows `content` near (x, y), given in container coordinates. */
  show(x: number, y: number, content: Child): void {
    this.el.replaceChildren();
    if (content instanceof Node) this.el.appendChild(content);
    else this.el.textContent = String(content ?? '');
    this.el.hidden = false;
    this.move(x, y);
  }

  move(x: number, y: number): void {
    if (this.el.hidden) return;
    const pad = 14;
    const { clientWidth: cw, clientHeight: ch } = this.container;
    const { offsetWidth: w, offsetHeight: tipH } = this.el;
    const left = x + pad + w > cw ? x - pad - w : x + pad;
    const top = y + pad + tipH > ch ? y - pad - tipH : y + pad;
    this.el.style.transform = `translate(${Math.max(4, left)}px, ${Math.max(4, top)}px)`;
  }

  hide(): void {
    this.el.hidden = true;
  }
}

// ------------------------------------------------------------------- toasts

let toastHost: HTMLElement | null = null;

export function toast(message: string, kind: 'info' | 'success' | 'warning' | 'error' = 'info', duration = 2800): void {
  if (!toastHost) {
    toastHost = h('div', { class: 'toasts', 'aria-live': 'polite', 'aria-atomic': 'false' });
    document.body.appendChild(toastHost);
  }
  const iconName = kind === 'success' ? 'check' : kind === 'info' ? 'info' : 'warning';
  const el = h('div', { class: `toast toast--${kind}`, role: kind === 'error' ? 'alert' : 'status' }, icon(iconName, 14), h('span', null, message));
  toastHost.appendChild(el);
  requestAnimationFrame(() => el.classList.add('is-visible'));
  setTimeout(() => {
    el.classList.remove('is-visible');
    setTimeout(() => el.remove(), 250);
  }, duration);
}

// ------------------------------------------------------------------- modals

let openModals = 0;

export function isModalOpen(): boolean {
  return openModals > 0;
}

export function openModal(o: { title: string; content: Child; wide?: boolean; onClose?(): void }): { close(): void } {
  const titleId = `modal-title-${Math.random().toString(36).slice(2, 8)}`;
  const previousFocus = document.activeElement as HTMLElement | null;
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    openModals--;
    document.removeEventListener('keydown', onKey, true);
    backdrop.classList.remove('is-visible');
    setTimeout(() => backdrop.remove(), 180);
    previousFocus?.focus?.();
    o.onClose?.();
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      close();
    }
  };

  const closeButton = h('button', { type: 'button', class: 'btn btn--ghost btn--icon', 'aria-label': 'Close', onClick: close }, icon('close'));
  const dialog = h(
    'div',
    { class: `modal${o.wide ? ' modal--wide' : ''}`, role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': titleId },
    h('header', { class: 'modal__header' }, h('h2', { id: titleId, class: 'modal__title' }, o.title), closeButton),
    h('div', { class: 'modal__body' }, o.content),
  );
  const backdrop = h('div', { class: 'modal-backdrop' }, dialog);
  backdrop.addEventListener('pointerdown', (event) => {
    if (event.target === backdrop) close();
  });
  document.body.appendChild(backdrop);
  document.addEventListener('keydown', onKey, true);
  openModals++;
  requestAnimationFrame(() => backdrop.classList.add('is-visible'));
  closeButton.focus();
  return { close };
}
