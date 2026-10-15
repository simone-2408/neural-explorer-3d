import { describe, expect, it, vi } from 'vitest';
import { Store } from './Store';

describe('Store', () => {
  it('notifies only when a value actually changes', () => {
    const store = new Store({ a: 1, b: 'x' });
    const listener = vi.fn();
    store.subscribe(listener);
    store.set({ a: 1 });
    expect(listener).not.toHaveBeenCalled();
    store.set((s) => ({ a: s.a + 1 }));
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.state).toEqual({ a: 2, b: 'x' });
  });

  it('watches derived slices with a custom equality', () => {
    const store = new Store({ selection: { id: 1 } as { id: number } | null, other: 0 });
    const callback = vi.fn();
    store.watch((s) => s.selection, callback, (x, y) => x?.id === y?.id);
    store.set({ other: 1 });
    store.set({ selection: { id: 1 } }); // new object, same id
    expect(callback).not.toHaveBeenCalled();
    store.set({ selection: null });
    expect(callback).toHaveBeenCalledWith(null, { id: 1 });
  });
});
