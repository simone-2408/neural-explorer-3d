/**
 * Inspector blocks that explain backpropagation and the parameter update.
 * They are filled when the context carries training information (Training
 * workspace); otherwise they point to the Training tab.
 */
import { ACTIVATIONS } from '../../../math/activations';
import { biasSymbol, neuronName, preActivationSymbol, weightSymbol } from '../../../neural/topology';
import { h } from '../../../ui/dom';
import { live, type LiveRegion } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note, num } from '../../../ui/widgets';
import { formatValue, texNumber, texParen } from '../../../utils/format';
import type { NetworkContext } from '../context';
import { block, paragraph } from './common';

export interface TrainingSection {
  el: HTMLElement;
  update(): void;
}

const NO_TRAINING = 'Gradients are computed in the Training tab: there the network learns from a dataset, and this block shows ∂L/∂z, ∂L/∂w and the update of each parameter.';

/** δ = ∂L/∂z of a neuron and how it is obtained. */
export function neuronGradientSection(ctx: NetworkContext, layer: number, index: number): TrainingSection {
  const training = ctx.training;
  if (!training) {
    const el = block(null, 'Gradient', note(NO_TRAINING));
    return { el, update: () => undefined };
  }
  const { spec } = ctx.model;
  const region: LiveRegion = live(() => {
    const grads = training.gradients();
    if (layer === 0) {
      return paragraph('Input neurons have no parameters: backpropagation stops at the first weight matrix.');
    }
    if (!grads) return note('Run at least one training step to see the gradients of this neuron.');
    const delta = grads.delta[layer][index];
    const db = grads.db[layer][index];
    const activation = ACTIVATIONS[spec.layers[layer].activation];
    const isOutput = layer === spec.layers.length - 1;
    const loss = training.lossInfo();
    const z = preActivationSymbol(layer, index);
    return [
      kv(h('span', null, 'δ = ∂L/∂z ', h('span', { class: 'muted' }, '(batch average)')), num(delta, { signed: true, digits: 5 }), 'kv--strong'),
      kv('∂L/∂b = δ', num(db, { signed: true, digits: 5 })),
      isOutput
        ? tex(
            activation.id === 'sigmoid' || activation.id === 'softmax'
              ? `\\delta^{(${layer})}_{${index + 1}} = \\frac{\\partial L}{\\partial ${z}} = \\hat{y}_{${index + 1}} - y_{${index + 1}}`
              : `\\delta^{(${layer})}_{${index + 1}} = (\\hat{y}_{${index + 1}} - y_{${index + 1}})\\, f'(${z})`,
            { display: true },
          )
        : tex(`\\delta^{(${layer})}_{${index + 1}} = ${activation.kind === 'elementwise' ? `f'(${z})` : ''}\\sum_k W^{(${layer + 1})}_{k,${index + 1}}\\, \\delta^{(${layer + 1})}_k`, { display: true }),
      paragraph(
        isOutput
          ? `For an output neuron δ compares the prediction with the target${loss ? ` (${loss.label})` : ''}.`
          : `δ arrives from the next layer through the same weights used in the forward pass (traversed backwards), and is multiplied by the slope of ${activation.label} at z. Where the slope is 0 (inactive ReLU, saturated sigmoid) the gradient stops.`,
      ),
      Math.abs(delta) < 1e-6 ? note('This gradient is (almost) zero: the parameters feeding this neuron barely change in this step.', 'warning') : null,
    ];
  });
  return { el: block(null, 'Gradient (backpropagation)', region.el), update: region.update };
}

/** Weight update of a connection: w_old, ∂L/∂w, η, Δw, w_new. */
export function connectionUpdateSection(ctx: NetworkContext, layer: number, target: number, source: number): TrainingSection {
  const training = ctx.training;
  if (!training) {
    const el = block(null, 'Training', note(`During training this weight is updated by gradient descent: w ← w − η · ∂L/∂w. ${NO_TRAINING}`));
    return { el, update: () => undefined };
  }
  const { spec } = ctx.model;
  const fanIn = spec.layers[layer - 1].size;
  const k = target * fanIn + source;
  const ref = { layer, target, source };
  const region = live(() => {
    const grads = training.gradients();
    const previous = training.previousWeights();
    const wNew = ctx.model.weight(layer, target, source);
    if (!grads || !previous) return note('Run at least one training step to see how this weight is updated.');
    const g = grads.dW[layer][k];
    const wOld = previous[layer][k];
    const delta = wNew - wOld;
    const lr = training.learningRate();
    const x = `${neuronName(spec, layer - 1, source)}`;
    return [
      kv('Weight before the step  w_old', num(wOld, { signed: true, digits: 5 })),
      kv(h('span', null, 'Gradient ∂L/∂w ', h('span', { class: 'muted' }, '(batch avg)')), num(g, { signed: true, digits: 5 })),
      kv('Learning rate η', formatValue(lr, lr < 0.01 ? 4 : 3)),
      kv('Update Δw = w_new − w_old', num(delta, { signed: true, digits: 5 })),
      kv('Weight after the step  w_new', num(wNew, { signed: true, digits: 5 }), 'kv--strong'),
      tex(training.updateRuleTex(), { display: true }),
      training.optimizerLabel() === 'SGD'
        ? tex(`${texNumber(wNew, 5)} = ${texNumber(wOld, 5)} - ${texNumber(lr, 4)} \\times ${texParen(g, 5)}`, { display: true })
        : paragraph(`With ${training.optimizerLabel()} the step also depends on the history of past gradients, so Δw is not simply −η·∂L/∂w.`),
      tex(`\\frac{\\partial L}{\\partial ${weightSymbol(ref)}} = \\delta^{(${layer})}_{${target + 1}} \\cdot a^{(${layer - 1})}_{${source + 1}}\\quad(\\text{averaged over the batch})`, { display: true }),
      paragraph(`The gradient is the error signal δ of the target neuron times the activation that flowed through this connection (from ${x}): a connection that carried no signal (a = 0) gets no gradient.`),
    ];
  });
  return { el: block(null, 'Training update', region.el), update: region.update };
}

/** Bias update of a neuron (used by the neuron inspector in training). */
export function biasUpdateSection(ctx: NetworkContext, layer: number, index: number): TrainingSection | null {
  const training = ctx.training;
  if (!training || layer === 0) return null;
  const region = live(() => {
    const grads = training.gradients();
    const previous = training.previousBiases();
    if (!grads || !previous) return null;
    const bOld = previous[layer][index];
    const bNew = ctx.model.bias(layer, index);
    return [
      kv(h('span', null, 'Bias ', tex(biasSymbol(layer, index)), ' before → after'), h('span', null, num(bOld, { signed: true, digits: 4 }), ' → ', num(bNew, { signed: true, digits: 4 }))),
    ];
  });
  return { el: region.el, update: region.update };
}
