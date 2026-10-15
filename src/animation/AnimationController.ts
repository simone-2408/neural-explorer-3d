/**
 * Centralized animation system. Every time-based change in the scene goes
 * through here instead of ad-hoc requestAnimationFrame loops:
 *
 *  - the Viewport ticks it once per frame and keeps rendering while it is busy
 *    (render-on-demand: when nothing animates, nothing is drawn);
 *  - "channels" guarantee a single animation per concern (e.g. a new camera
 *    flight cancels the previous one);
 *  - "scaled" animations obey `timeScale` / `paused` — the hook for the
 *    forward-pass player (Play / Pause / Speed / Step), while UI
 *    animations (camera, layout) keep running at real speed.
 */
import { easeInOutCubic, type Easing } from './easing';

export interface Animation {
  /** Advance by `dt` seconds. Return true when finished. */
  update(dt: number): boolean;
  /** Called if the animation is cancelled before finishing. */
  cancel?(): void;
}

export interface AnimationHandle {
  cancel(): void;
  readonly finished: boolean;
}

export interface AnimationOptions {
  /** Starting an animation on a channel cancels the one already running there. */
  channel?: string;
  /** Obey timeScale / paused (simulation animations). Default false (UI animations). */
  scaled?: boolean;
}

export interface TweenOptions extends AnimationOptions {
  /** Duration in seconds. */
  duration: number;
  easing?: Easing;
  /** Receives the eased progress in [0, 1]. */
  onUpdate(progress: number): void;
  onComplete?(): void;
  onCancel?(): void;
}

interface Entry {
  animation: Animation;
  channel?: string;
  scaled: boolean;
  finished: boolean;
}

export class AnimationController {
  /** Speed multiplier for scaled animations. */
  timeScale = 1;
  /** Pauses scaled animations. */
  paused = false;
  /** Honour the OS "reduce motion" preference: tweens complete instantly. */
  reducedMotion = false;

  private readonly entries = new Set<Entry>();
  private readonly channels = new Map<string, Entry>();

  get busy(): boolean {
    return this.entries.size > 0;
  }

  add(animation: Animation, options: AnimationOptions = {}): AnimationHandle {
    if (options.channel) this.cancelChannel(options.channel);
    const entry: Entry = { animation, channel: options.channel, scaled: !!options.scaled, finished: false };
    this.entries.add(entry);
    if (entry.channel) this.channels.set(entry.channel, entry);
    return {
      cancel: () => this.cancelEntry(entry),
      get finished() {
        return entry.finished;
      },
    };
  }

  tween(options: TweenOptions): AnimationHandle {
    const duration = this.reducedMotion ? 0 : Math.max(0, options.duration);
    const easing = options.easing ?? easeInOutCubic;
    let elapsed = 0;
    return this.add(
      {
        update: (dt) => {
          elapsed += dt;
          const t = duration === 0 ? 1 : Math.min(1, elapsed / duration);
          options.onUpdate(easing(t));
          if (t >= 1) {
            options.onComplete?.();
            return true;
          }
          return false;
        },
        cancel: options.onCancel,
      },
      options,
    );
  }

  cancelChannel(channel: string): void {
    const entry = this.channels.get(channel);
    if (entry) this.cancelEntry(entry);
  }

  cancelAll(): void {
    for (const entry of [...this.entries]) this.cancelEntry(entry);
  }

  /** Advances all animations. Returns true if anything was updated (the scene must be redrawn). */
  update(dt: number): boolean {
    if (this.entries.size === 0) return false;
    let progressed = false;
    for (const entry of [...this.entries]) {
      if (entry.scaled && this.paused) continue;
      const done = entry.animation.update(entry.scaled ? dt * this.timeScale : dt);
      progressed = true;
      if (done) this.finish(entry);
    }
    return progressed;
  }

  private cancelEntry(entry: Entry): void {
    if (entry.finished) return;
    this.finish(entry);
    entry.animation.cancel?.();
  }

  private finish(entry: Entry): void {
    entry.finished = true;
    this.entries.delete(entry);
    if (entry.channel && this.channels.get(entry.channel) === entry) this.channels.delete(entry.channel);
  }
}
