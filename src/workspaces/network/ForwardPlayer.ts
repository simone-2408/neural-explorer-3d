/**
 * Forward-pass player: shows the information flowing through the network one
 * layer at a time.
 *
 *   step 0            the input layer shows x
 *   step l (l ≥ 1)    1. travel   — pulses run along W^(l) (brightness ∝ |w·a|)
 *                     2. activate — the neurons of layer l fill with a = f(z)
 *
 * It runs as a *scaled* animation of the central AnimationController, so
 * Pause and Speed are the simulation clock (`paused`, `timeScale`). The
 * player only produces a visual state for NetworkView; the numbers always
 * come from the model's ForwardTrace.
 */
import type { AnimationController, AnimationHandle } from '../../animation/AnimationController';
import { easeInOutCubic } from '../../animation/easing';
import { Emitter } from '../../core/Emitter';
import type { NetworkView } from '../../three/NetworkView';

export interface PlayerState {
  active: boolean;
  /** Last fully computed layer (0 = only the input is shown). */
  step: number;
  /** Number of layers (including input and output). */
  layers: number;
  /** Layer currently being animated, or null. */
  animating: number | null;
  phase: 'travel' | 'activate' | null;
  playing: boolean;
  speed: number;
}

const TRAVEL = 1.15;
const ACTIVATE = 0.6;
const HOLD = 0.45;

export const PLAYER_SPEEDS = [0.25, 0.5, 1, 2, 4];

export class ForwardPlayer {
  readonly events = new Emitter<{ change: PlayerState; stepStart: { layer: number } }>();
  private readonly view: NetworkView;
  private readonly animations: AnimationController;
  private readonly layerCount: () => number;
  private active = false;
  private step = 0;
  private animating: number | null = null;
  private phase: 'travel' | 'activate' | 'hold' | null = null;
  private time = 0;
  private playing = false;
  private loop: AnimationHandle | null = null;

  constructor(view: NetworkView, animations: AnimationController, layerCount: () => number) {
    this.view = view;
    this.animations = animations;
    this.layerCount = layerCount;
  }

  get state(): PlayerState {
    return {
      active: this.active,
      step: this.step,
      layers: this.layerCount(),
      animating: this.animating,
      phase: this.phase === 'hold' ? null : this.phase,
      playing: this.playing,
      speed: this.animations.timeScale,
    };
  }

  get isActive(): boolean {
    return this.active;
  }

  /** Enters forward mode: only the input layer is shown. */
  start(): void {
    this.active = true;
    this.goTo(0);
  }

  /** Leaves forward mode: every layer is shown again. */
  stop(): void {
    this.active = false;
    this.playing = false;
    this.animating = null;
    this.phase = null;
    this.animations.paused = false;
    this.loop?.cancel();
    this.loop = null;
    this.view.setSignalState(null, null);
    this.emit();
  }

  play(): void {
    if (!this.active) this.start();
    if (this.animating === null && this.step >= this.lastLayer) this.goTo(0);
    this.playing = true;
    this.animations.paused = false;
    if (this.animating === null) this.beginStep(this.step + 1);
    this.ensureLoop();
    this.emit();
  }

  pause(): void {
    this.playing = false;
    if (this.animating !== null) this.animations.paused = true;
    this.emit();
  }

  toggle(): void {
    if (this.playing && !this.animations.paused) this.pause();
    else this.play();
  }

  /** Completes the step in progress, or animates the next one. */
  next(): void {
    if (!this.active) this.start();
    if (this.animating !== null) {
      this.finishStep();
    } else if (this.step < this.lastLayer) {
      this.animations.paused = false;
      this.beginStep(this.step + 1);
      this.ensureLoop();
    }
    this.emit();
  }

  previous(): void {
    if (!this.active) return;
    const target = this.animating !== null ? this.step : Math.max(0, this.step - 1);
    this.playing = false;
    this.goTo(target);
  }

  restart(): void {
    if (!this.active) this.start();
    this.playing = false;
    this.goTo(0);
  }

  setSpeed(speed: number): void {
    this.animations.timeScale = speed;
    this.emit();
  }

  /** Re-applies the visual state (e.g. after the view was rebuilt). */
  refresh(): void {
    if (!this.active) return;
    if (this.step > this.lastLayer) this.step = this.lastLayer;
    if (this.animating !== null && this.animating > this.lastLayer) this.animating = null;
    this.apply();
    this.emit();
  }

  dispose(): void {
    this.stop();
    this.animations.timeScale = 1;
  }

  // -------------------------------------------------------------- internals

  private get lastLayer(): number {
    return this.layerCount() - 1;
  }

  private goTo(step: number): void {
    this.step = step;
    this.animating = null;
    this.phase = null;
    this.animations.paused = false;
    this.apply();
    this.emit();
  }

  private beginStep(layer: number): void {
    this.animating = layer;
    this.phase = 'travel';
    this.time = 0;
    this.apply();
    this.events.emit('stepStart', { layer });
  }

  private finishStep(): void {
    if (this.animating === null) return;
    this.step = this.animating;
    this.animating = null;
    this.phase = null;
    this.animations.paused = false;
    if (this.step >= this.lastLayer) this.playing = false;
    this.apply();
  }

  private ensureLoop(): void {
    if (this.loop && !this.loop.finished) return;
    this.loop = this.animations.add(
      {
        update: (dt) => {
          if (!this.active) return true;
          this.time += dt;
          if (this.phase === 'travel' && this.time >= TRAVEL) {
            this.phase = 'activate';
            this.time = 0;
            this.emit();
          } else if (this.phase === 'activate' && this.time >= ACTIVATE) {
            this.finishStep();
            if (this.playing && this.step < this.lastLayer) {
              this.phase = 'hold';
              this.time = 0;
            }
            this.emit();
          } else if (this.phase === 'hold' && this.time >= HOLD) {
            this.phase = null;
            if (this.playing) this.beginStep(this.step + 1);
            this.emit();
          }
          this.apply();
          // Keep running only while something moves.
          return this.animating === null && this.phase !== 'hold';
        },
      },
      { channel: 'forward-player', scaled: true },
    );
  }

  private apply(): void {
    if (!this.active) {
      this.view.setSignalState(null, null);
      return;
    }
    const layer = this.animating;
    if (layer === null || this.phase === 'hold' || this.phase === null) {
      this.view.setSignalState({ layer: this.step, t: 1, direction: 1 }, null);
      return;
    }
    if (this.phase === 'travel') {
      const t = easeInOutCubic(Math.min(1, this.time / TRAVEL));
      this.view.setSignalState({ layer: layer - 1, t: 1, direction: 1 }, { layer, t, direction: 1, intensity: 1 });
    } else {
      const t = Math.min(1, this.time / ACTIVATE);
      this.view.setSignalState({ layer, t: easeInOutCubic(t), direction: 1 }, { layer, t: 1, direction: 1, intensity: 1 - t });
    }
  }

  private emit(): void {
    this.events.emit('change', this.state);
  }
}
