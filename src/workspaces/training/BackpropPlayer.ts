/**
 * Backpropagation, animated: ONE real training step on the probe point,
 * shown in four phases.
 *
 *   1. forward   pulses run input → output, layers fill with their activations
 *   2. loss      the output is compared with the label
 *   3. backward  δ = ∂L/∂z flows output → input; lines show |∂L/∂w| (log scale)
 *   4. update    the optimizer changes the weights; lines show |Δw|
 *
 * The numbers are those of the real step (Trainer.stepOn with this point).
 */
import type { AnimationController, AnimationHandle } from '../../animation/AnimationController';
import { easeInOutCubic } from '../../animation/easing';
import { createTrace, forwardInto } from '../../neural/forward';
import type { NetworkView } from '../../three/NetworkView';
import { backpropSample, createGradients, createScratch, type Gradients } from '../../training/backprop';
import { LOSSES, lossFor, sampleLoss, targetValue } from '../../training/loss';
import { OPTIMIZERS } from '../../training/optimizers';
import { h, replaceChildren } from '../../ui/dom';
import { tex } from '../../ui/tex';
import { button } from '../../ui/widgets';
import { formatValue, texNumber } from '../../utils/format';
import type { TrainingWorkspace } from './TrainingWorkspace';

type Phase = { kind: 'forward'; layer: number } | { kind: 'loss' } | { kind: 'backward'; layer: number } | { kind: 'update' };

const DURATION = { forward: 0.85, loss: 1.6, backward: 1.2, update: 2.2 } as const;

export class BackpropPlayer {
  private readonly ws: TrainingWorkspace;
  private readonly view: NetworkView;
  private readonly animations: AnimationController;
  private handle: AnimationHandle | null = null;
  private phases: Phase[] = [];
  private index = 0;
  private time = 0;
  private sample = 0;
  private grads: Gradients | null = null;
  private loss = 0;
  private before: Float64Array[] = [];
  private delta: Float64Array[] = [];
  private stepped = false;
  readonly caption: HTMLElement;
  private readonly captionBody = h('div', { class: 'backprop-caption__body' });

  constructor(ws: TrainingWorkspace, view: NetworkView, animations: AnimationController) {
    this.ws = ws;
    this.view = view;
    this.animations = animations;
    this.caption = h(
      'div',
      { class: 'backprop-caption', hidden: true, role: 'status' },
      this.captionBody,
      button({ icon: 'close', title: 'Stop the animation', small: true, variant: 'ghost', onClick: () => this.stop() }),
    );
  }

  get active(): boolean {
    return this.handle !== null && !this.handle.finished;
  }

  /** Animates a training step on sample `index` of the dataset. */
  start(index: number): void {
    this.stop();
    const ws = this.ws;
    const { spec } = ws.model;
    const L = spec.layers.length - 1;
    this.sample = index;
    this.stepped = false;
    ws.setProbe({ kind: 'sample', index });

    // Per-sample gradients (exactly what this point contributes).
    const trace = createTrace(spec.layers.map((l) => l.size));
    const weights = Array.from({ length: spec.layers.length }, (_, l) => ws.model.weights(l) as Float64Array);
    const biases = Array.from({ length: spec.layers.length }, (_, l) => ws.model.biases(l) as Float64Array);
    const x = [ws.dataset.x[2 * index], ws.dataset.x[2 * index + 1]];
    forwardInto(spec, weights, biases, x, trace);
    const lossId = lossFor(spec);
    this.loss = sampleLoss(lossId, spec, trace.a[L], ws.dataset.y[index]);
    this.grads = createGradients(spec);
    backpropSample(spec, weights, trace, lossId, ws.dataset.y[index], this.grads, createScratch(spec));
    this.delta = this.grads.delta.map((d) => d.slice());
    this.delta[0] = new Float64Array(spec.layers[0].size);
    this.before = weights.map((w) => w.slice());

    this.phases = [];
    for (let l = 1; l <= L; l++) this.phases.push({ kind: 'forward', layer: l });
    this.phases.push({ kind: 'loss' });
    for (let l = L; l >= 1; l--) this.phases.push({ kind: 'backward', layer: l });
    this.phases.push({ kind: 'update' });
    this.index = 0;
    this.time = 0;

    this.view.setNeuronValues(null);
    this.view.setConnectionValues(null);
    this.view.setPulseValues(null);
    this.mountCaption();
    this.handle = this.animations.add({ update: (dt) => this.update(dt) }, { channel: 'backprop' });
    this.enter();
  }

  stop(): void {
    if (this.handle && !this.handle.finished) this.handle.cancel();
    this.handle = null;
    this.finish();
  }

  // -------------------------------------------------------------- internals

  private mountCaption(): void {
    const stage = document.querySelector('.stage__overlay');
    if (stage && !this.caption.isConnected) stage.append(this.caption);
    this.caption.hidden = false;
  }

  private finish(): void {
    this.caption.hidden = true;
    this.caption.remove();
    this.view.setSignalState(null, null);
    this.view.setPulseValues(null);
    this.ws.applyEncoding();
  }

  private update(dt: number): boolean {
    const phase = this.phases[this.index];
    if (!phase) {
      this.handle = null;
      this.finish();
      return true;
    }
    this.time += dt;
    const duration = DURATION[phase.kind];
    const t = Math.min(1, this.time / duration);
    this.apply(phase, t);
    if (t >= 1) {
      this.index++;
      this.time = 0;
      this.enter();
    }
    return false;
  }

  /** Side effects when a phase starts, then the caption. */
  private enter(): void {
    const phase = this.phases[this.index];
    if (!phase) return;
    const view = this.view;
    if (phase.kind === 'backward' && phase.layer === this.ws.model.layerCount - 1) {
      // Entering the backward pass: neurons show δ, lines show |∂L/∂w|.
      view.setNeuronValues(this.delta);
      view.setConnectionValues({ values: this.grads!.dW, log: true });
      view.setPulseValues(this.grads!.dW);
    }
    if (phase.kind === 'update' && !this.stepped) {
      // The real optimizer step on this single point.
      this.stepped = true;
      this.ws.trainer.stepOn([this.sample]);
      this.ws.noteExternalStep();
      view.setNeuronValues(null);
      view.setPulseValues(null);
      view.setSignalState(null, null);
    }
    this.render();
  }

  private apply(phase: Phase, t: number): void {
    const view = this.view;
    const e = easeInOutCubic(t);
    switch (phase.kind) {
      case 'forward': {
        if (t < 0.65) view.setSignalState({ layer: phase.layer - 1, t: 1, direction: 1 }, { layer: phase.layer, t: easeInOutCubic(t / 0.65), direction: 1, intensity: 1 });
        else view.setSignalState({ layer: phase.layer, t: (t - 0.65) / 0.35, direction: 1 }, { layer: phase.layer, t: 1, direction: 1, intensity: 1 - (t - 0.65) / 0.35 });
        break;
      }
      case 'loss':
        view.setSignalState({ layer: this.ws.model.layerCount - 1, t: 0.5 + 0.5 * Math.cos(t * Math.PI * 4), direction: 1 }, null);
        break;
      case 'backward': {
        const L = this.ws.model.layerCount - 1;
        if (phase.layer === L && t < 0.35) {
          view.setSignalState({ layer: L, t: t / 0.35, direction: -1 }, null);
        } else {
          const p = phase.layer === L ? (t - 0.35) / 0.65 : t;
          view.setSignalState({ layer: phase.layer, t: 1, direction: -1 }, { layer: phase.layer, t: easeInOutCubic(p), direction: -1, intensity: 1 });
          if (p > 0.8) view.setSignalState({ layer: phase.layer - 1, t: (p - 0.8) / 0.2, direction: -1 }, { layer: phase.layer, t: 1, direction: -1, intensity: 1 - (p - 0.8) / 0.2 });
        }
        break;
      }
      case 'update': {
        const after = Array.from({ length: this.ws.model.layerCount }, (_, l) => this.ws.model.weights(l));
        const change = after.map((w, l) => {
          const d = new Float64Array(w.length);
          for (let k = 0; k < w.length; k++) d[k] = (w[k] - (this.before[l]?.[k] ?? w[k])) * (0.2 + 0.8 * e);
          return d;
        });
        view.setConnectionValues({ values: change, log: true });
        break;
      }
    }
  }

  private render(): void {
    const phase = this.phases[this.index];
    if (!phase) return;
    const ws = this.ws;
    const { spec } = ws.model;
    const L = spec.layers.length - 1;
    const label = ws.dataset.y[this.sample];
    const lossId = lossFor(spec);
    const total = this.phases.length;
    const step = `${this.index + 1}/${total}`;
    let title: string;
    let text: string;
    let formula: string;
    switch (phase.kind) {
      case 'forward':
        title = `Forward pass · ${spec.layers[phase.layer].name}`;
        text = `The point (${formatValue(ws.dataset.x[2 * this.sample], 2)}, ${formatValue(ws.dataset.x[2 * this.sample + 1], 2)}) flows through the network.`;
        formula = `a^{(${phase.layer})} = f\\!\\left(W^{(${phase.layer})} a^{(${phase.layer - 1})} + b^{(${phase.layer})}\\right)`;
        break;
      case 'loss': {
        const out = ws.model.output;
        title = 'Loss';
        text = `The prediction is compared with the true class y = ${label}: this number is what training tries to make smaller.`;
        formula =
          out.length > 1
            ? `${LOSSES[lossId].tex} = -\\log ${texNumber(out[label], 3)} = ${texNumber(this.loss, 4)}\\quad (c = ${label})`
            : `${LOSSES[lossId].tex} = ${texNumber(this.loss, 4)}\\quad \\left(\\hat{y} = ${texNumber(out[0], 3)},\\; y = ${texNumber(targetValue(spec, label, 0), 0)}\\right)`;
        break;
      }
      case 'backward':
        title = `Backward pass · ${spec.layers[phase.layer].name}`;
        text =
          phase.layer === L
            ? 'The error signal δ = ∂L/∂z starts at the output. Lines now show |∂L/∂w| (log scale), neurons show δ.'
            : 'δ travels back through the same weights, multiplied by the slope f′(z) of each neuron. Where the slope is 0 the signal stops.';
        formula =
          phase.layer === L
            ? `\\delta^{(${L})} = \\hat{y} - y,\\qquad \\frac{\\partial L}{\\partial W^{(${L})}} = \\delta^{(${L})}\\, a^{(${L - 1})\\top}`
            : `\\delta^{(${phase.layer})} = \\left(W^{(${phase.layer + 1})\\top} \\delta^{(${phase.layer + 1})}\\right) \\odot f'\\!\\left(z^{(${phase.layer})}\\right),\\quad \\frac{\\partial L}{\\partial W^{(${phase.layer})}} = \\delta^{(${phase.layer})} a^{(${phase.layer - 1})\\top}`;
        break;
      case 'update': {
        const o = ws.settings.optimizer;
        title = `Update · ${OPTIMIZERS[o.id].label}`;
        text = `Every weight moves against its gradient (η = ${formatValue(o.learningRate, 4)}). Lines now show |Δw|: one step changes the weights only a little — training repeats this thousands of times.`;
        formula = OPTIMIZERS[o.id].tex;
        break;
      }
    }
    replaceChildren(this.captionBody, h('div', { class: 'backprop-caption__title' }, h('span', { class: 'backprop-caption__step' }, step), title), h('div', { class: 'backprop-caption__text' }, text), tex(formula));
  }
}
