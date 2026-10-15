/**
 * Ready-made A/B comparisons: two networks trained on the same data, with
 * the same batches, so the only difference is the one being studied.
 */
import type { ActivationId } from '../math/activations';
import type { InitSettings } from '../neural/init';
import type { DatasetId, DatasetOptions } from './datasets';
import { DEFAULT_OPTIMIZER, type OptimizerSettings } from './optimizers';
import type { HiddenLayer } from './session';

export interface Contestant {
  hidden: HiddenLayer[];
  init: InitSettings;
  optimizer: OptimizerSettings;
}

export interface Comparison {
  id: string;
  title: string;
  question: string;
  /** What usually happens and why (shown next to the results). */
  insight: string;
  dataset: DatasetId;
  datasetOptions?: Partial<DatasetOptions>;
  batchSize: number;
  a: Contestant;
  b: Contestant;
}

const layers = (count: number, size: number, activation: ActivationId): HiddenLayer[] => Array.from({ length: count }, () => ({ size, activation }));
const adam = (learningRate = 0.02): OptimizerSettings => ({ ...DEFAULT_OPTIMIZER, id: 'adam', learningRate });
const sgd = (learningRate = 0.2): OptimizerSettings => ({ ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate });

export const COMPARISONS: readonly Comparison[] = [
  {
    id: 'depth',
    title: '2 hidden layers vs 6 hidden layers',
    question: 'Does a deeper network learn the spiral better?',
    insight:
      'Depth adds expressive power (more ways to bend the boundary) but also makes optimization harder and slower per step. Compare how fast each loss falls and the shape of the two boundaries.',
    dataset: 'spiral',
    batchSize: 32,
    a: { hidden: layers(2, 12, 'relu'), init: { scheme: 'he', seed: 1 }, optimizer: adam(0.02) },
    b: { hidden: layers(6, 12, 'relu'), init: { scheme: 'he', seed: 1 }, optimizer: adam(0.02) },
  },
  {
    id: 'activation',
    title: 'ReLU vs sigmoid (same architecture)',
    question: 'How much does the activation function matter?',
    insight: 'With several layers the sigmoid’s small slope (≤ 0.25) shrinks the gradients at every layer: the sigmoid network starts learning much later. Open the Gradients tab of the Train mode to see why.',
    dataset: 'moons',
    batchSize: 16,
    a: { hidden: layers(3, 8, 'relu'), init: { scheme: 'he', seed: 2 }, optimizer: sgd(0.2) },
    b: { hidden: layers(3, 8, 'sigmoid'), init: { scheme: 'xavier', seed: 2 }, optimizer: sgd(0.2) },
  },
  {
    id: 'optimizer',
    title: 'SGD vs Adam',
    question: 'Same network, same data: which optimizer gets there first?',
    insight: 'Adam rescales every parameter’s step by the history of its gradients, so it is less sensitive to the scale of each gradient. SGD with a good learning rate can still reach a similar solution, often more slowly.',
    dataset: 'circles',
    batchSize: 16,
    a: { hidden: layers(2, 8, 'tanh'), init: { scheme: 'xavier', seed: 3 }, optimizer: sgd(0.1) },
    b: { hidden: layers(2, 8, 'tanh'), init: { scheme: 'xavier', seed: 3 }, optimizer: adam(0.02) },
  },
  {
    id: 'momentum',
    title: 'SGD vs SGD with momentum',
    question: 'Does remembering past gradients help?',
    insight: 'Momentum accumulates a velocity: steps grow along directions where gradients agree and cancel where they oscillate.',
    dataset: 'xor',
    batchSize: 8,
    a: { hidden: layers(1, 6, 'tanh'), init: { scheme: 'xavier', seed: 4 }, optimizer: sgd(0.05) },
    b: { hidden: layers(1, 6, 'tanh'), init: { scheme: 'xavier', seed: 4 }, optimizer: { ...sgd(0.05), id: 'momentum', momentum: 0.9 } },
  },
  {
    id: 'capacity',
    title: 'Small vs large network (capacity)',
    question: 'How big must a network be for the circles?',
    insight: 'Too few neurons cannot represent a closed boundary (underfitting): accuracy plateaus. A larger network can, at the cost of more parameters.',
    dataset: 'circles',
    batchSize: 16,
    a: { hidden: layers(1, 2, 'relu'), init: { scheme: 'he', seed: 5 }, optimizer: adam(0.03) },
    b: { hidden: layers(2, 16, 'relu'), init: { scheme: 'he', seed: 5 }, optimizer: adam(0.03) },
  },
  {
    id: 'regularization',
    title: 'Overfitting: no regularization vs weight decay',
    question: 'Few noisy points and a big network: what does weight decay change?',
    insight: 'Without regularization the big network memorizes the noise (train ≫ test accuracy). Weight decay keeps the weights small and the boundary smoother: training accuracy drops a little, test accuracy usually improves.',
    dataset: 'moons',
    datasetOptions: { samples: 60, noise: 0.45, testFraction: 0.5, seed: 3 },
    batchSize: 100000,
    a: { hidden: layers(2, 32, 'relu'), init: { scheme: 'he', seed: 1 }, optimizer: adam(0.02) },
    b: { hidden: layers(2, 32, 'relu'), init: { scheme: 'he', seed: 1 }, optimizer: { ...adam(0.02), weightDecay: 0.03 } },
  },
];
