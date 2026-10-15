/**
 * Training session settings and the network they describe. A session is a
 * dataset + an architecture (2 inputs, configurable hidden layers, output
 * chosen from the number of classes) + initialization + optimizer.
 */
import type { ActivationId } from '../math/activations';
import { specFromConfig, type NetworkConfig, type NetworkSpec } from '../neural/config';
import { DEFAULT_INIT, type InitSettings } from '../neural/init';
import { DEFAULT_DATASET_OPTIONS, datasetInfo, type DatasetId, type DatasetOptions } from './datasets';
import { DEFAULT_OPTIMIZER, type OptimizerSettings } from './optimizers';

export interface HiddenLayer {
  size: number;
  activation: ActivationId;
}

export type TrainingSpeed = 'slow' | 'normal' | 'fast' | 'max';

export interface TrainingSettings {
  dataset: DatasetId;
  datasetOptions: DatasetOptions;
  hidden: HiddenLayer[];
  init: InitSettings;
  optimizer: OptimizerSettings;
  batchSize: number;
  /** Stop automatically after this many epochs (0 = never). */
  stopAtEpoch: number;
  speed: TrainingSpeed;
}

export const DEFAULT_TRAINING: TrainingSettings = {
  dataset: 'xor',
  datasetOptions: { ...DEFAULT_DATASET_OPTIONS, noise: 0.15 },
  hidden: [
    { size: 4, activation: 'relu' },
    { size: 4, activation: 'relu' },
  ],
  init: { ...DEFAULT_INIT, scheme: 'pytorch', seed: 7 },
  optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.03 },
  batchSize: 16,
  stopAtEpoch: 0,
  speed: 'normal',
};

export const NETWORK_TEMPLATES: { id: string; label: string; hidden: HiddenLayer[] }[] = [
  { id: 'none', label: 'No hidden layer (logistic regression)', hidden: [] },
  { id: 'small', label: '2 → 4 → 1', hidden: [{ size: 4, activation: 'relu' }] },
  { id: 'default', label: '2 → 4 → 4 → 1', hidden: [{ size: 4, activation: 'relu' }, { size: 4, activation: 'relu' }] },
  { id: 'wide', label: '2 → 8 → 8 → 1', hidden: [{ size: 8, activation: 'relu' }, { size: 8, activation: 'relu' }] },
  { id: 'spiral', label: '2 → 16 → 16 → 16 → 1 (for the spiral)', hidden: [{ size: 16, activation: 'tanh' }, { size: 16, activation: 'tanh' }, { size: 16, activation: 'tanh' }] },
  {
    id: 'deep-sigmoid',
    label: 'Deep sigmoid 2 → 6×6 → 1',
    hidden: Array.from({ length: 6 }, () => ({ size: 6, activation: 'sigmoid' as ActivationId })),
  },
];

export const MAX_HIDDEN_LAYERS = 8;
export const MAX_HIDDEN_SIZE = 32;

/** Network configuration of a session: 2 inputs, the hidden layers, an output that matches the classes. */
export function sessionConfig(settings: TrainingSettings): NetworkConfig {
  const classes = datasetInfo(settings.dataset).classes;
  return {
    name: `Classifier for ${datasetInfo(settings.dataset).label}`,
    layers: [
      { type: 'input', size: 2, labels: ['x₁', 'x₂'], range: [-1.3, 1.3] },
      ...settings.hidden.map((h) => ({ type: 'dense' as const, size: h.size, activation: h.activation })),
      classes > 2
        ? { type: 'output' as const, size: classes, activation: 'softmax' as const, labels: Array.from({ length: classes }, (_, k) => `class ${k}`) }
        : { type: 'output' as const, size: 1, activation: 'sigmoid' as const, labels: ['p(class 1)'] },
    ],
  };
}

export function sessionSpec(settings: TrainingSettings): NetworkSpec {
  return specFromConfig(sessionConfig(settings));
}
