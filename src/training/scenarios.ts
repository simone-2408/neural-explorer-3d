/**
 * "Break the Network": deliberately bad configurations, each with what to
 * observe, why it happens and how to fix it. The claims are checked by
 * src/training/scenarios.test.ts, which trains every scenario (and its fix)
 * headless and verifies that the problem really appears — and disappears.
 */
import { NetworkModel } from '../neural/NetworkModel';
import { makeDataset, type Dataset } from './datasets';
import { diagnose, type Diagnosis } from './diagnostics';
import { DEFAULT_OPTIMIZER } from './optimizers';
import { DEFAULT_TRAINING, sessionSpec, type TrainingSettings } from './session';
import { Trainer } from './Trainer';

export interface Scenario {
  id: string;
  title: string;
  /** One-line summary of the mistake. */
  mistake: string;
  /** Where to look in the app. */
  watch: string;
  /** The explanation, with the mechanism. */
  why: string;
  fix: string;
  settings: TrainingSettings;
  fixed: TrainingSettings;
  /** Diagnostic expected to fire (if any). */
  expected?: Diagnosis['id'];
}

function settings(patch: Partial<TrainingSettings>): TrainingSettings {
  return {
    ...structuredClone(DEFAULT_TRAINING),
    ...patch,
    datasetOptions: { ...DEFAULT_TRAINING.datasetOptions, ...patch.datasetOptions },
    optimizer: { ...DEFAULT_OPTIMIZER, ...patch.optimizer },
  };
}

const relu = (size: number, count: number) => Array.from({ length: count }, () => ({ size, activation: 'relu' as const }));

export const SCENARIOS: readonly Scenario[] = [
  {
    id: 'huge-lr',
    title: 'Learning rate far too large',
    mistake: 'Plain SGD with η = 30.',
    watch: 'The loss chart jumps around or explodes; the boundary flips at every step.',
    why: 'Each step moves the weights by η·∂L/∂w. With a huge η the step overshoots the minimum and lands where the loss is even higher; the gradients there are larger, so the next step is bigger still.',
    fix: 'Use η around 0.1 for SGD (or Adam with η ≈ 0.03).',
    settings: settings({ dataset: 'moons', optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 30 }, init: { scheme: 'pytorch', seed: 4 } }),
    fixed: settings({ dataset: 'moons', optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.3 }, init: { scheme: 'pytorch', seed: 4 } }),
  },
  {
    id: 'tiny-lr',
    title: 'Learning rate far too small',
    mistake: 'Plain SGD with η = 0.0001.',
    watch: 'The loss barely moves; after hundreds of steps the boundary has hardly changed.',
    why: 'The direction of every step is right, but each step changes the weights by only 0.0001 · ∂L/∂w: reaching a good solution would take millions of steps.',
    fix: 'Increase η by about three orders of magnitude (≈ 0.1–0.5 for SGD), or use Adam.',
    settings: settings({ dataset: 'xor', optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.0001 }, init: { scheme: 'pytorch', seed: 2 } }),
    fixed: settings({ dataset: 'xor', optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.5 }, init: { scheme: 'pytorch', seed: 2 } }),
    expected: 'slow',
  },
  {
    id: 'too-small',
    title: 'Network too small (underfitting)',
    mistake: 'No hidden layer for the circles dataset.',
    watch: 'The boundary stays a straight line; accuracy plateaus far from 100%.',
    why: 'Without hidden layers the network is σ(w·x + b): its decision boundary w·x + b = 0 is always a straight line, and no line separates a disc from the ring around it.',
    fix: 'Add a hidden layer (e.g. 2 → 8 → 8 → 1).',
    settings: settings({ dataset: 'circles', hidden: [], init: { scheme: 'pytorch', seed: 1 } }),
    fixed: settings({ dataset: 'circles', hidden: relu(8, 2), init: { scheme: 'pytorch', seed: 1 } }),
    expected: 'plateau',
  },
  {
    id: 'deep-sigmoid',
    title: 'Too deep, with sigmoids (vanishing gradients)',
    mistake: 'Six sigmoid hidden layers trained with plain SGD.',
    watch: 'Open the Gradients tab: the bars shrink by ~10× per layer towards the input. Accuracy stays at 50%.',
    why: 'Backpropagation multiplies δ by f′(z) at every layer and the sigmoid slope is at most 0.25: after six layers the error signal reaching the first weights is thousands of times smaller.',
    fix: 'Use ReLU (slope 1 where active) with He initialization, fewer layers, or Adam (which rescales small gradients).',
    settings: settings({
      dataset: 'xor',
      hidden: Array.from({ length: 6 }, () => ({ size: 6, activation: 'sigmoid' as const })),
      optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.1 },
      init: { scheme: 'xavier', seed: 3 },
    }),
    fixed: settings({ dataset: 'xor', hidden: relu(6, 3), optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.02 }, init: { scheme: 'he', seed: 3 } }),
    expected: 'vanishing',
  },
  {
    id: 'zero-init',
    title: 'All weights initialized to zero',
    mistake: 'Every weight and bias starts at 0.',
    watch: 'Select two neurons of the same hidden layer: their weights stay identical forever. Accuracy stays near 50%.',
    why: 'Identical neurons compute identical outputs and receive identical gradients, so every update keeps them identical: the layer behaves like a single neuron. Random initialization is what breaks this symmetry.',
    fix: 'Use a random initialization (Xavier, He or the PyTorch default).',
    settings: settings({ dataset: 'xor', hidden: [{ size: 6, activation: 'sigmoid' }], optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.05 }, init: { scheme: 'zeros', seed: 1 } }),
    fixed: settings({ dataset: 'xor', hidden: [{ size: 6, activation: 'sigmoid' }], optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.05 }, init: { scheme: 'xavier', seed: 1 } }),
    expected: 'symmetric',
  },
  {
    id: 'large-init',
    title: 'Initial weights far too large (saturation)',
    mistake: 'Weights drawn from N(0, 3²) with tanh neurons.',
    watch: 'Neurons sit at ±1 (fully saturated); the Gradients tab shows tiny gradients and training crawls.',
    why: 'With large weights the weighted sums z are huge, where tanh is flat: f′(z) ≈ 0, so almost no gradient flows back through these neurons.',
    fix: 'Scale the initial weights to the layer size (Xavier for tanh/sigmoid).',
    settings: settings({ dataset: 'circles', hidden: [{ size: 8, activation: 'tanh' }, { size: 8, activation: 'tanh' }], optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.1 }, init: { scheme: 'large', seed: 5 } }),
    fixed: settings({ dataset: 'circles', hidden: [{ size: 8, activation: 'tanh' }, { size: 8, activation: 'tanh' }], optimizer: { ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.1 }, init: { scheme: 'xavier', seed: 5 } }),
    expected: 'saturated',
  },
  {
    id: 'dead-relu',
    title: 'Dead ReLU neurons',
    mistake: 'All biases start at −2.',
    watch: 'Every hidden neuron shows 0 for every point; the boundary never forms.',
    why: 'A ReLU neuron with z ≤ 0 outputs 0 and has slope 0: no gradient reaches its weights, so nothing can bring it back to life. Here every neuron starts (and stays) in that region.',
    fix: 'Initialize biases near 0 (PyTorch default or He); use Leaky ReLU, whose slope is never exactly 0.',
    settings: settings({ dataset: 'moons', hidden: relu(8, 2), optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.03 }, init: { scheme: 'negative-bias', seed: 2 } }),
    fixed: settings({ dataset: 'moons', hidden: relu(8, 2), optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.03 }, init: { scheme: 'he', seed: 2 } }),
    expected: 'dead-relu',
  },
  {
    id: 'overfitting',
    title: 'Overfitting',
    mistake: 'A big network trained for long on few, noisy points.',
    watch: 'Training accuracy reaches ~100% while test accuracy (dashed) stays lower; the boundary wraps around individual noisy points.',
    why: 'With many more parameters than points, the network can memorize the noise of the training set instead of the shape that generalizes to new points.',
    fix: 'More data, a smaller network, weight decay (L2), or stopping earlier.',
    settings: settings({
      dataset: 'moons',
      datasetOptions: { samples: 60, noise: 0.45, testFraction: 0.5, seed: 3 },
      hidden: relu(32, 2),
      optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.02 },
      batchSize: 100000,
      init: { scheme: 'he', seed: 1 },
    }),
    fixed: settings({
      dataset: 'moons',
      datasetOptions: { samples: 400, noise: 0.45, testFraction: 0.5, seed: 3 },
      hidden: relu(8, 1),
      optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.02, weightDecay: 0.01 },
      batchSize: 100000,
      init: { scheme: 'he', seed: 1 },
    }),
    expected: 'overfitting',
  },
];

export interface HeadlessRun {
  model: NetworkModel;
  dataset: Dataset;
  trainer: Trainer;
  diagnosis: Diagnosis | null;
}

/** Trains a session without any UI (tests, precomputation). */
export function runHeadless(s: TrainingSettings, epochs: number): HeadlessRun {
  const dataset = makeDataset(s.dataset, s.datasetOptions);
  const model = new NetworkModel(sessionSpec(s), { init: s.init });
  const trainer = new Trainer(model, dataset, { optimizer: s.optimizer, batchSize: s.batchSize, seed: s.datasetOptions.seed + 101 });
  trainer.record();
  for (let e = 0; e < epochs; e++) {
    trainer.runSteps(trainer.stepsPerEpoch);
    trainer.record();
    if (!Number.isFinite(trainer.lastBatchLoss)) break;
  }
  return { model, dataset, trainer, diagnosis: diagnose({ model, dataset, trainer }) };
}
