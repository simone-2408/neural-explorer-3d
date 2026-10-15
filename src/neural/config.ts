/**
 * Network configuration: the JSON format users write, its validation and the
 * normalized `NetworkSpec` used everywhere else in the app.
 *
 *   NetworkConfig (lenient, user-facing JSON)
 *        │ validateConfig()
 *        ▼
 *   NetworkSpec (validated, normalized, immutable)
 *
 * The layer `type` field is a discriminator that leaves room for more layer kinds
 * (e.g. "conv2d", "rnn", "attention"); this format supports input / dense / output
 * (the CNN and RNN tabs have their own models).
 * A dense layer maps 1:1 to PyTorch `nn.Linear(in, out)` + activation.
 */
import { ACTIVATION_IDS, isActivationId, type ActivationId } from '../math/activations';

export type LayerType = 'input' | 'dense' | 'output';

/** One layer as written in the JSON configuration. */
export interface LayerConfig {
  type: LayerType;
  size: number;
  /** Dense/output layers only. Default: "relu" for hidden layers, "linear" for the output layer. */
  activation?: ActivationId;
  /** Optional display name, e.g. "Pixels". */
  name?: string;
  /** Optional per-neuron labels, e.g. ["cat", "dog"]. Must have `size` entries. */
  labels?: string[];
  /** Input layer only: 2-D arrangement [rows, cols] with rows × cols = size (e.g. [28, 28] for images). */
  shape?: [number, number];
  /** Input layer only: range of the input controls. Default [-1, 1]. */
  range?: [number, number];
}

export interface NetworkConfig {
  name?: string;
  description?: string;
  layers: LayerConfig[];
}

export type LayerRole = 'input' | 'hidden' | 'output';

/** Normalized, validated description of one layer. */
export interface LayerSpec {
  readonly index: number;
  readonly role: LayerRole;
  readonly size: number;
  /** "linear" for the input layer (it performs no computation). */
  readonly activation: ActivationId;
  /** Display name: custom name or "Input" / "Hidden 1" / "Output". */
  readonly name: string;
  readonly customName: string | null;
  /** Prefix for neuron names: "I", "H1", "O". */
  readonly prefix: string;
  readonly labels: readonly string[] | null;
  readonly shape: readonly [number, number] | null;
  readonly range: readonly [number, number];
}

export interface NetworkSpec {
  readonly name: string;
  readonly description: string;
  readonly layers: readonly LayerSpec[];
}

export interface ConfigIssue {
  /** Location of the problem, e.g. "layers[2].size". */
  path: string;
  message: string;
}

export type ConfigResult =
  | { ok: true; spec: NetworkSpec; warnings: ConfigIssue[] }
  | { ok: false; errors: ConfigIssue[]; warnings: ConfigIssue[] };

/** Limits that keep the 3D scene interactive on modest hardware. */
export const LIMITS = {
  maxLayers: 16,
  maxLayerSize: 1024,
  maxNeurons: 4096,
  maxConnections: 120_000,
} as const;

export const DEFAULT_INPUT_RANGE: readonly [number, number] = [-1, 1];

const LAYER_KEYS = new Set(['type', 'size', 'activation', 'name', 'labels', 'shape', 'range']);
const NETWORK_KEYS = new Set(['name', 'description', 'layers']);

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPositiveInt = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 1;

export function validateConfig(raw: unknown): ConfigResult {
  const errors: ConfigIssue[] = [];
  const warnings: ConfigIssue[] = [];
  const fail = (): ConfigResult => ({ ok: false, errors, warnings });

  if (!isObject(raw)) {
    errors.push({ path: '(root)', message: 'The configuration must be a JSON object with a "layers" array.' });
    return fail();
  }
  for (const key of Object.keys(raw)) {
    if (!NETWORK_KEYS.has(key)) warnings.push({ path: key, message: `Unknown property "${key}" is ignored.` });
  }
  if (raw.name !== undefined && typeof raw.name !== 'string') errors.push({ path: 'name', message: '"name" must be a string.' });
  if (raw.description !== undefined && typeof raw.description !== 'string')
    errors.push({ path: 'description', message: '"description" must be a string.' });

  const layers = raw.layers;
  if (!Array.isArray(layers)) {
    errors.push({ path: 'layers', message: '"layers" must be an array of layer objects.' });
    return fail();
  }
  if (layers.length < 2) {
    errors.push({ path: 'layers', message: 'A network needs at least an input layer and an output layer.' });
    return fail();
  }
  if (layers.length > LIMITS.maxLayers) {
    errors.push({ path: 'layers', message: `At most ${LIMITS.maxLayers} layers are supported (got ${layers.length}).` });
    return fail();
  }

  const last = layers.length - 1;
  const specs: LayerSpec[] = [];
  let hiddenCounter = 0;

  layers.forEach((layer, index) => {
    const path = `layers[${index}]`;
    if (!isObject(layer)) {
      errors.push({ path, message: 'Each layer must be an object like {"type": "dense", "size": 4}.' });
      return;
    }
    for (const key of Object.keys(layer)) {
      if (!LAYER_KEYS.has(key)) warnings.push({ path: `${path}.${key}`, message: `Unknown property "${key}" is ignored.` });
    }

    const type = layer.type;
    if (type !== 'input' && type !== 'dense' && type !== 'output') {
      errors.push({
        path: `${path}.type`,
        message:
          typeof type === 'string'
            ? `Layer type "${type}" is not supported here. Supported types: "input", "dense", "output" (convolutional and recurrent networks have their own tabs, CNN and RNN; attention is in the Transformer tab).`
            : 'Missing layer "type" ("input", "dense" or "output").',
      });
      return;
    }
    if (index === 0 && type !== 'input') errors.push({ path: `${path}.type`, message: 'The first layer must have type "input".' });
    if (index !== 0 && type === 'input') errors.push({ path: `${path}.type`, message: 'Only the first layer can be an input layer.' });
    if (index === last && type !== 'output') errors.push({ path: `${path}.type`, message: 'The last layer must have type "output".' });
    if (index !== last && type === 'output')
      errors.push({ path: `${path}.type`, message: 'Only the last layer can be an output layer (use "dense" for hidden layers).' });

    const size = layer.size;
    if (!isPositiveInt(size)) {
      errors.push({ path: `${path}.size`, message: '"size" must be an integer ≥ 1 (number of neurons).' });
      return;
    }
    if (size > LIMITS.maxLayerSize) {
      errors.push({ path: `${path}.size`, message: `At most ${LIMITS.maxLayerSize} neurons per layer are supported.` });
      return;
    }

    const role: LayerRole = index === 0 ? 'input' : index === last ? 'output' : 'hidden';
    let activation: ActivationId = role === 'input' ? 'linear' : role === 'output' ? 'linear' : 'relu';
    if (layer.activation !== undefined) {
      if (role === 'input') {
        warnings.push({ path: `${path}.activation`, message: 'Input layers do not apply an activation; the value is ignored.' });
      } else if (!isActivationId(layer.activation)) {
        errors.push({
          path: `${path}.activation`,
          message: `Unknown activation "${String(layer.activation)}". Available: ${ACTIVATION_IDS.join(', ')}.`,
        });
      } else {
        activation = layer.activation;
      }
    }
    if (activation === 'softmax' && role !== 'output')
      errors.push({ path: `${path}.activation`, message: 'Softmax is only supported on the output layer.' });
    if (activation === 'softmax' && size === 1)
      warnings.push({ path: `${path}.activation`, message: 'Softmax over a single neuron always outputs 1. Use "sigmoid" for one output.' });

    let customName: string | null = null;
    if (layer.name !== undefined) {
      if (typeof layer.name !== 'string' || !layer.name.trim()) errors.push({ path: `${path}.name`, message: '"name" must be a non-empty string.' });
      else customName = layer.name.trim();
    }

    let labels: string[] | null = null;
    if (layer.labels !== undefined) {
      if (!Array.isArray(layer.labels) || !layer.labels.every((l) => typeof l === 'string'))
        errors.push({ path: `${path}.labels`, message: '"labels" must be an array of strings.' });
      else if (layer.labels.length !== size)
        errors.push({ path: `${path}.labels`, message: `"labels" must have exactly ${size} entries (one per neuron).` });
      else labels = [...layer.labels];
    }

    let shape: [number, number] | null = null;
    if (layer.shape !== undefined) {
      const s = layer.shape;
      if (role !== 'input') warnings.push({ path: `${path}.shape`, message: '"shape" is only used for the input layer; ignored.' });
      else if (!Array.isArray(s) || s.length !== 2 || !isPositiveInt(s[0]) || !isPositiveInt(s[1]))
        errors.push({ path: `${path}.shape`, message: '"shape" must be [rows, cols] with positive integers.' });
      else if (s[0] * s[1] !== size)
        errors.push({ path: `${path}.shape`, message: `rows × cols (${s[0]} × ${s[1]} = ${s[0] * s[1]}) must equal size (${size}).` });
      else shape = [s[0], s[1]];
    }

    let range: [number, number] = [DEFAULT_INPUT_RANGE[0], DEFAULT_INPUT_RANGE[1]];
    if (layer.range !== undefined) {
      const r = layer.range;
      if (role !== 'input') warnings.push({ path: `${path}.range`, message: '"range" is only used for the input layer; ignored.' });
      else if (!Array.isArray(r) || r.length !== 2 || !r.every((v) => typeof v === 'number' && Number.isFinite(v)) || r[0] >= r[1])
        errors.push({ path: `${path}.range`, message: '"range" must be [min, max] with min < max.' });
      else range = [r[0], r[1]];
    }

    if (role === 'hidden') hiddenCounter++;
    const defaultName = role === 'input' ? 'Input' : role === 'output' ? 'Output' : `Hidden ${hiddenCounter}`;
    const prefix = role === 'input' ? 'I' : role === 'output' ? 'O' : `H${hiddenCounter}`;

    specs.push({
      index,
      role,
      size,
      activation,
      name: customName ?? defaultName,
      customName,
      prefix,
      labels,
      shape,
      range,
    });
  });

  if (errors.length) return fail();

  const neurons = specs.reduce((sum, l) => sum + l.size, 0);
  let connections = 0;
  for (let l = 1; l < specs.length; l++) connections += specs[l].size * specs[l - 1].size;
  if (neurons > LIMITS.maxNeurons)
    errors.push({ path: 'layers', message: `Too many neurons (${neurons}); the limit is ${LIMITS.maxNeurons}.` });
  if (connections > LIMITS.maxConnections)
    errors.push({
      path: 'layers',
      message: `Too many connections (${connections.toLocaleString('en-US')}); the limit is ${LIMITS.maxConnections.toLocaleString('en-US')}.`,
    });
  if (errors.length) return fail();

  const spec: NetworkSpec = {
    name: typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim() : 'Untitled network',
    description: typeof raw.description === 'string' ? raw.description : '',
    layers: specs,
  };
  return { ok: true, spec, warnings };
}

/** Parses JSON text and validates it. Syntax errors are reported as issues. */
export function parseConfigText(text: string): ConfigResult {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    return { ok: false, errors: [{ path: 'JSON', message: (error as Error).message }], warnings: [] };
  }
  return validateConfig(raw);
}

export class ConfigError extends Error {
  readonly issues: ConfigIssue[];
  constructor(issues: ConfigIssue[]) {
    super(issues.map((i) => `${i.path}: ${i.message}`).join('\n'));
    this.name = 'ConfigError';
    this.issues = issues;
  }
}

/** Validates and returns the spec, throwing a ConfigError when invalid. */
export function specFromConfig(config: unknown): NetworkSpec {
  const result = validateConfig(config);
  if (!result.ok) throw new ConfigError(result.errors);
  return result.spec;
}

/** Canonical configuration for a spec (what the JSON editor shows). */
export function specToConfig(spec: NetworkSpec): NetworkConfig {
  const config: NetworkConfig = { name: spec.name, layers: [] };
  if (spec.description) config.description = spec.description;
  for (const layer of spec.layers) {
    const out: LayerConfig = {
      type: layer.role === 'input' ? 'input' : layer.role === 'output' ? 'output' : 'dense',
      size: layer.size,
    };
    if (layer.role !== 'input') out.activation = layer.activation;
    if (layer.customName) out.name = layer.customName;
    if (layer.shape) out.shape = [layer.shape[0], layer.shape[1]];
    if (layer.role === 'input' && (layer.range[0] !== DEFAULT_INPUT_RANGE[0] || layer.range[1] !== DEFAULT_INPUT_RANGE[1]))
      out.range = [layer.range[0], layer.range[1]];
    if (layer.labels) out.labels = [...layer.labels];
    config.layers.push(out);
  }
  return config;
}

/** Compact JSON for nested values: `{ "type": "dense", "size": 4 }`. */
function compactJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(compactJson).join(', ')}]`;
  if (isObject(value))
    return `{ ${Object.entries(value)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${JSON.stringify(k)}: ${compactJson(v)}`)
      .join(', ')} }`;
  return JSON.stringify(value);
}

/** Pretty-prints a configuration with one layer per line (easier to edit by hand). */
export function formatConfig(config: NetworkConfig): string {
  const parts: string[] = [];
  if (config.name !== undefined) parts.push(`  "name": ${JSON.stringify(config.name)}`);
  if (config.description) parts.push(`  "description": ${JSON.stringify(config.description)}`);
  const layers = config.layers.map((layer) => `    ${compactJson(layer)}`).join(',\n');
  parts.push(`  "layers": [\n${layers}\n  ]`);
  return `{\n${parts.join(',\n')}\n}\n`;
}
