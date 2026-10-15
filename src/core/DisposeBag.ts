/** Collects cleanup callbacks (unsubscribe, removeEventListener, dispose…) and runs them once. */
export class DisposeBag {
  private readonly items: (() => void)[] = [];

  add(dispose: () => void): void {
    this.items.push(dispose);
  }

  /** Convenience for DOM listeners that are removed on dispose. */
  listen<K extends keyof HTMLElementEventMap>(
    target: HTMLElement,
    type: K,
    handler: (event: HTMLElementEventMap[K]) => void,
    options?: AddEventListenerOptions,
  ): void;
  listen(target: EventTarget, type: string, handler: (event: Event) => void, options?: AddEventListenerOptions): void;
  listen(target: EventTarget, type: string, handler: (event: never) => void, options?: AddEventListenerOptions): void {
    const fn = handler as EventListener;
    target.addEventListener(type, fn, options);
    this.items.push(() => target.removeEventListener(type, fn, options));
  }

  dispose(): void {
    while (this.items.length) {
      const item = this.items.pop()!;
      try {
        item();
      } catch (error) {
        console.error('[DisposeBag] cleanup failed', error);
      }
    }
  }
}
