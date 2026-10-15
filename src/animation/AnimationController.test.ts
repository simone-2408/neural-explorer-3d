import { describe, expect, it, vi } from 'vitest';
import { linear } from './easing';
import { AnimationController } from './AnimationController';

describe('AnimationController', () => {
  it('runs tweens to completion', () => {
    const controller = new AnimationController();
    const values: number[] = [];
    const done = vi.fn();
    controller.tween({ duration: 1, easing: linear, onUpdate: (t) => values.push(t), onComplete: done });
    controller.update(0.25);
    controller.update(0.5);
    controller.update(0.5);
    expect(values).toEqual([0.25, 0.75, 1]);
    expect(done).toHaveBeenCalledTimes(1);
    expect(controller.busy).toBe(false);
    expect(controller.update(0.1)).toBe(false); // nothing left to animate → no redraw needed
  });

  it('cancels the previous animation of the same channel', () => {
    const controller = new AnimationController();
    const cancelled = vi.fn();
    controller.tween({ channel: 'camera', duration: 1, onUpdate: () => {}, onCancel: cancelled });
    controller.tween({ channel: 'camera', duration: 1, onUpdate: () => {} });
    expect(cancelled).toHaveBeenCalledTimes(1);
  });

  it('applies timeScale and pause only to scaled (simulation) animations', () => {
    const controller = new AnimationController();
    controller.timeScale = 2;
    controller.paused = true;
    const ui: number[] = [];
    const sim: number[] = [];
    controller.tween({ duration: 1, easing: linear, onUpdate: (t) => ui.push(t) });
    controller.tween({ duration: 1, easing: linear, scaled: true, onUpdate: (t) => sim.push(t) });
    controller.update(0.25);
    expect(ui).toEqual([0.25]);
    expect(sim).toEqual([]);
    controller.paused = false;
    controller.update(0.25);
    expect(sim).toEqual([0.5]);
  });
});
