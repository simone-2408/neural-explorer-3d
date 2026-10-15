/**
 * "Follow the Signal" mode: highlights where the signal of the selected
 * neuron/connection goes (downstream) and where it comes from (upstream),
 * fading everything else, with pulses running along the strongest paths.
 * Scores come from src/neural/signalFlow (|w·a| shares); this class only
 * keeps them up to date and drives the time uniform.
 */
import type { AnimationController, AnimationHandle } from '../../animation/AnimationController';
import type { NetworkModel } from '../../neural/NetworkModel';
import { normalizeFlowPerLayer, signalFlow, type FlowSource } from '../../neural/signalFlow';
import type { NetworkView } from '../../three/NetworkView';
import type { NetworkSelection } from '../../visualization/selection';

export function flowSourceOf(selection: NetworkSelection | null): FlowSource | null {
  if (!selection || selection.kind === 'layer') return null;
  return selection;
}

export class FollowSignal {
  private readonly view: NetworkView;
  private readonly model: NetworkModel;
  private readonly animations: AnimationController;
  private source: FlowSource | null = null;
  private clock: AnimationHandle | null = null;
  private time = 0;

  constructor(view: NetworkView, model: NetworkModel, animations: AnimationController) {
    this.view = view;
    this.model = model;
    this.animations = animations;
  }

  get active(): boolean {
    return this.source !== null;
  }

  /** Starts (or retargets) the mode. Returns false if the selection cannot be followed (nothing or a layer). */
  follow(selection: NetworkSelection | null): boolean {
    this.source = flowSourceOf(selection);
    if (!this.source) {
      this.stop();
      return false;
    }
    this.update();
    if (!this.clock || this.clock.finished) {
      this.clock = this.animations.add({
        update: (dt) => {
          if (!this.source) return true;
          if (!this.animations.reducedMotion) {
            this.time += dt;
            this.view.setTime(this.time);
          }
          return false;
        },
      });
    }
    return true;
  }

  /** Recomputes the scores (call after weights/inputs changed). */
  update(): void {
    if (!this.source) return;
    this.view.setFlow(normalizeFlowPerLayer(this.model, signalFlow(this.model, this.source)));
  }

  stop(): void {
    this.source = null;
    this.clock?.cancel();
    this.clock = null;
    this.view.setFlow(null);
  }
}
