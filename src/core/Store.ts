/**
 * Small observable store for *application / UI state* (selection, hover,
 * view options…). State objects are treated as immutable: every update
 * produces a new top-level object, so subscribers can compare by reference.
 *
 * Large numeric data (weights, activations, gradients) deliberately does NOT
 * live here: it lives in mutable domain models that emit their own events
 * (see `NetworkModel`). See docs/ARCHITECTURE.md → "State management".
 */
export type Updater<S> = Partial<S> | ((state: Readonly<S>) => Partial<S>);
export type StoreListener<S> = (state: Readonly<S>, previous: Readonly<S>) => void;

export class Store<S extends object> {
  private current: S;
  private readonly listeners = new Set<StoreListener<S>>();

  constructor(initial: S) {
    this.current = initial;
  }

  get state(): Readonly<S> {
    return this.current;
  }

  /** Shallow-merge an update. Listeners run only if at least one key changed. */
  set(update: Updater<S>): void {
    const patch = typeof update === 'function' ? update(this.current) : update;
    let changed = false;
    for (const key of Object.keys(patch) as (keyof S)[]) {
      if (!Object.is(patch[key], this.current[key])) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    const previous = this.current;
    this.current = { ...previous, ...patch };
    for (const listener of [...this.listeners]) listener(this.current, previous);
  }

  subscribe(listener: StoreListener<S>): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Subscribe to a derived slice of the state. The callback runs only when
   * the selected value changes according to `equals` (default: Object.is).
   */
  watch<T>(
    selector: (state: Readonly<S>) => T,
    callback: (value: T, previous: T) => void,
    equals: (a: T, b: T) => boolean = Object.is,
  ): () => void {
    let last = selector(this.current);
    return this.subscribe((state) => {
      const next = selector(state);
      if (equals(next, last)) return;
      const previous = last;
      last = next;
      callback(next, previous);
    });
  }
}
