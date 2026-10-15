/**
 * The Transformer computation as a sequence of steps (tokens → embedding →
 * attention/MLP per layer → prediction) and the player that animates it.
 *
 * The player animates a single number, `stage` (see TransformerView): rows
 * below `stage` are computed, the row being computed first shows a copy of
 * the row below (the residual stream carries the vector up) and then morphs
 * into its own value (the sub-layer adds its output).
 */
import type { AnimationController, AnimationHandle } from '../../animation/AnimationController';
import { easeInOutCubic } from '../../animation/easing';
import { Emitter } from '../../core/Emitter';
import type { TransformerModel } from '../../transformer/model';

export type PipelineStepId = 'tokens' | 'embed' | 'attention' | 'mlp' | 'predict';

export interface PipelineStep {
  id: PipelineStepId;
  layer: number | null;
  label: string;
  /** Value of the view's stage once this step is done. */
  stage: number;
  /** Seconds (at 1×) of the animation that leads to this step. */
  duration: number;
  caption: string;
}

export function pipelineSteps(model: TransformerModel): PipelineStep[] {
  const { nLayers, vocab, dModel, dMlp, nHeads } = model.config;
  const steps: PipelineStep[] = [
    {
      id: 'tokens',
      layer: null,
      label: 'Tokens',
      stage: 0,
      duration: 0.4,
      caption: `The text is split into words and each word is replaced by its index in the ${vocab.length}-word vocabulary; <s> marks the start of the sequence.`,
    },
    {
      id: 'embed',
      layer: null,
      label: 'Embedding',
      stage: 1,
      duration: 1.3,
      caption: `Each token selects its row of the embedding matrix E (${dModel} numbers) and its position adds a vector P: x₀ = E[token] + P[position]. Bars up = positive, down = negative.`,
    },
  ];
  for (let l = 0; l < nLayers; l++) {
    steps.push({
      id: 'attention',
      layer: l,
      label: `Attention ${l + 1}`,
      stage: 2 + 2 * l,
      duration: 2.2,
      caption: `Attention ${l + 1}: every token compares its query with the keys of itself and of the tokens before it, then adds a weighted mix of their values to its vector (${nHeads} heads in parallel). Arcs show where each head reads from.`,
    });
    steps.push({
      id: 'mlp',
      layer: l,
      label: `MLP ${l + 1}`,
      stage: 3 + 2 * l,
      duration: 1.5,
      caption: `MLP ${l + 1}: the same ${dModel} → ${dMlp} → ${dModel} network processes every position separately and adds its output to the vector. Positions exchange no information here.`,
    });
  }
  steps.push({
    id: 'predict',
    layer: null,
    label: 'Prediction',
    stage: 1 + 2 * nLayers,
    duration: 0.5,
    caption: 'The last vector is normalized and compared with every word’s embedding (logits = LN(x)·Eᵀ); a softmax turns these scores into the probabilities of the next word.',
  });
  return steps;
}

export const PIPELINE_SPEEDS = [0.5, 1, 2] as const;

const HOLD = 0.8;

export class PipelinePlayer {
  readonly events = new Emitter<{ change: undefined }>();
  readonly steps: PipelineStep[];
  /** Index of the current step (the target, while animating). */
  step: number;
  /** Animated stage value shown by the views. */
  stage: number;
  playing = false;
  animating = false;
  speed = 1;
  private readonly animations: AnimationController;
  private readonly onStage: (stage: number) => void;
  private handle: AnimationHandle | null = null;

  constructor(steps: PipelineStep[], animations: AnimationController, onStage: (stage: number) => void) {
    this.steps = steps;
    this.animations = animations;
    this.onStage = onStage;
    this.step = steps.length - 1;
    this.stage = steps[this.step].stage;
  }

  get last(): number {
    return this.steps.length - 1;
  }

  get current(): PipelineStep {
    return this.steps[this.step];
  }

  /** True when the whole computation is shown (default state, outside the walkthrough). */
  get complete(): boolean {
    return this.step === this.last && !this.animating;
  }

  goTo(step: number, animate = true): void {
    const target = Math.max(0, Math.min(this.last, step));
    this.cancelAnimation();
    const forwardOne = animate && target === this.step + 1;
    this.step = target;
    if (forwardOne) this.animateTo(target);
    else this.setStage(this.steps[target].stage);
    this.events.emit('change', undefined);
  }

  next(): void {
    if (this.step < this.last) this.goTo(this.step + 1);
  }

  previous(): void {
    this.pause();
    if (this.step > 0) this.goTo(this.step - 1, false);
  }

  restart(): void {
    this.goTo(0, false);
  }

  play(): void {
    if (this.playing) return;
    if (this.step === this.last && !this.animating) this.goTo(0, false);
    this.playing = true;
    this.events.emit('change', undefined);
    if (!this.animating) this.hold();
  }

  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    if (!this.animating) this.cancelAnimation();
    this.events.emit('change', undefined);
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    this.events.emit('change', undefined);
  }

  /** Stops any animation and shows the complete computation. */
  stop(): void {
    this.playing = false;
    this.cancelAnimation();
    this.step = this.last;
    this.setStage(this.steps[this.last].stage);
  }

  private animateTo(target: number): void {
    const from = this.stage;
    const to = this.steps[target].stage;
    const duration = this.steps[target].duration / this.speed;
    this.animating = true;
    let elapsed = 0;
    this.handle = this.animations.add(
      {
        update: (dt) => {
          elapsed += dt;
          const t = Math.min(1, elapsed / Math.max(duration, 1e-3));
          this.setStage(from + (to - from) * easeInOutCubic(t));
          if (t < 1) return false;
          this.animating = false;
          this.handle = null;
          this.events.emit('change', undefined);
          if (this.playing) queueMicrotask(() => this.hold());
          return true;
        },
      },
      { channel: 'transformer-pipeline' },
    );
  }

  /** Waits a moment on the current step, then advances (while playing). */
  private hold(): void {
    if (!this.playing) return;
    if (this.step >= this.last) {
      this.playing = false;
      this.events.emit('change', undefined);
      return;
    }
    let elapsed = 0;
    const wait = HOLD / this.speed;
    this.handle = this.animations.add(
      {
        update: (dt) => {
          elapsed += dt;
          if (elapsed < wait) return false;
          this.handle = null;
          if (this.playing) queueMicrotask(() => this.goTo(this.step + 1));
          return true;
        },
      },
      { channel: 'transformer-pipeline' },
    );
  }

  private cancelAnimation(): void {
    this.handle?.cancel();
    this.handle = null;
    this.animating = false;
  }

  private setStage(stage: number): void {
    this.stage = stage;
    this.onStage(stage);
  }
}
