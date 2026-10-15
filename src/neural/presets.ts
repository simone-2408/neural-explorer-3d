/**
 * Ready-made architectures. Every preset is just a NetworkConfig (+ optional
 * default input): the scene is always generated from configuration.
 */
import type { NetworkConfig } from './config';
import type { InitSettings } from './init';
import { makePattern } from './inputPatterns';

export interface NetworkPreset {
  id: string;
  label: string;
  description: string;
  config: NetworkConfig;
  /** Default input vector (must match the input size). */
  input?: () => number[];
  /** Initialization (defaults to DEFAULT_INIT). */
  init?: Partial<InitSettings>;
}

export const DEFAULT_PRESET_ID = 'default';

export const PRESETS: readonly NetworkPreset[] = [
  {
    id: 'default',
    label: 'Default MLP · 2-4-4-1',
    description: 'The reference network: two inputs, two hidden ReLU layers, one sigmoid output.',
    config: {
      name: 'Default MLP',
      layers: [
        { type: 'input', size: 2 },
        { type: 'dense', size: 4, activation: 'relu' },
        { type: 'dense', size: 4, activation: 'relu' },
        { type: 'output', size: 1, activation: 'sigmoid' },
      ],
    },
    input: () => [0.7, 0.2],
    // Seed chosen so the untrained network shows both active and inactive ReLU units.
    init: { scheme: 'pytorch', seed: 261 },
  },
  {
    id: 'classifier',
    label: 'Classifier · 3-5-4-2 (softmax)',
    description: 'Three inputs and a softmax output over two classes: the outputs are probabilities that sum to 1.',
    config: {
      name: 'Softmax classifier',
      layers: [
        { type: 'input', size: 3 },
        { type: 'dense', size: 5, activation: 'relu' },
        { type: 'dense', size: 4, activation: 'relu' },
        { type: 'output', size: 2, activation: 'softmax', labels: ['class A', 'class B'] },
      ],
    },
    input: () => [0.5, -0.3, 0.8],
  },
  {
    id: 'perceptron',
    label: 'Single neuron · 2-1',
    description: 'A single-layer perceptron: one weighted sum followed by a sigmoid. The smallest possible network.',
    config: {
      name: 'Single-layer perceptron',
      layers: [
        { type: 'input', size: 2 },
        { type: 'output', size: 1, activation: 'sigmoid' },
      ],
    },
    input: () => [0.7, 0.2],
  },
  {
    id: 'xor',
    label: 'Tiny MLP · 2-2-1 (tanh)',
    description: 'Two hidden tanh neurons: the smallest MLP that can represent XOR once trained (train it in the Training tab).',
    config: {
      name: 'Tiny MLP (XOR-sized)',
      layers: [
        { type: 'input', size: 2 },
        { type: 'dense', size: 2, activation: 'tanh' },
        { type: 'output', size: 1, activation: 'sigmoid' },
      ],
    },
    input: () => [1, -1],
  },
  {
    id: 'deep-sigmoid',
    label: 'Deep & narrow · 2-6-6-6-6-6-1 (sigmoid)',
    description: 'Five sigmoid hidden layers. Watch how activations drift towards the same values deep in the network.',
    config: {
      name: 'Deep sigmoid MLP',
      layers: [
        { type: 'input', size: 2 },
        { type: 'dense', size: 6, activation: 'sigmoid' },
        { type: 'dense', size: 6, activation: 'sigmoid' },
        { type: 'dense', size: 6, activation: 'sigmoid' },
        { type: 'dense', size: 6, activation: 'sigmoid' },
        { type: 'dense', size: 6, activation: 'sigmoid' },
        { type: 'output', size: 1, activation: 'sigmoid' },
      ],
    },
    input: () => [0.9, -0.6],
  },
  {
    id: 'transformer-mlp',
    label: 'Transformer-style MLP · 8-32-8 (GELU)',
    description:
      'Same shape as the feed-forward block of a Transformer layer: expand ×4 with GELU, then project back to the model width (linear output).',
    config: {
      name: 'Transformer MLP block',
      layers: [
        { type: 'input', size: 8, name: 'Residual in' },
        { type: 'dense', size: 32, activation: 'gelu', name: 'Expand ×4' },
        { type: 'output', size: 8, activation: 'linear', name: 'Project back' },
      ],
    },
    input: () => [0.8, -0.4, 0.1, 0.6, -0.9, 0.3, -0.2, 0.5],
  },
  {
    id: 'wide',
    label: 'Wide · 4-16-16-3 (softmax)',
    description: 'Wider hidden layers (16 neurons) and three output classes.',
    config: {
      name: 'Wide MLP',
      layers: [
        { type: 'input', size: 4 },
        { type: 'dense', size: 16, activation: 'relu' },
        { type: 'dense', size: 16, activation: 'relu' },
        { type: 'output', size: 3, activation: 'softmax' },
      ],
    },
    input: () => [0.6, -0.2, 0.9, -0.7],
  },
  {
    id: 'mnist',
    label: 'Image classifier · 784-32-16-10',
    description:
      'MNIST-sized network: a 28×28 grayscale image (784 inputs) and 10 output classes. Weights are random, so the prediction is meaningless: this one shows the shape of a real classifier, it is not trained.',
    config: {
      name: 'Image classifier (MNIST-sized)',
      layers: [
        { type: 'input', size: 784, shape: [28, 28], range: [0, 1], name: 'Pixels' },
        { type: 'dense', size: 32, activation: 'relu' },
        { type: 'dense', size: 16, activation: 'relu' },
        { type: 'output', size: 10, activation: 'softmax', labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] },
      ],
    },
    input: () => Array.from(makePattern('ring', 28, 28)),
  },
];

export function findPreset(id: string): NetworkPreset | undefined {
  return PRESETS.find((p) => p.id === id);
}
