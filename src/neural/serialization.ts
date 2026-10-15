/**
 * Model file format (JSON, version 1): architecture + parameters + input.
 *
 * Designed to be trivially loadable from Python/PyTorch (python/mlp_bridge.py):
 * `weight` is a nested list with shape [out, in], exactly like
 * `nn.Linear.weight.tolist()`, and `output` stores the forward result computed
 * in the browser so both implementations can be cross-checked.
 */
import { toNested } from '../math/linalg';
import { specToConfig, validateConfig, type NetworkConfig, type NetworkSpec } from './config';
import { DEFAULT_INIT, isInitScheme, type InitSettings } from './init';
import type { NetworkModel } from './NetworkModel';

export const MODEL_FORMAT = 'neural-explorer-3d/mlp';
export const MODEL_FORMAT_VERSION = 1;
export const APP_ID = 'neural-explorer-3d@0.1.0';

export interface SerializedLayerParams {
  /** Index of the layer that owns the weights (≥ 1). */
  layer: number;
  /** Shape [n_layer, n_{layer-1}] — same as PyTorch nn.Linear.weight. */
  weight: number[][];
  bias: number[];
}

export interface SerializedModel {
  format: typeof MODEL_FORMAT;
  formatVersion: number;
  config: NetworkConfig;
  init: InitSettings;
  parameters: SerializedLayerParams[];
  input: number[];
  /** Network output for `input`, as computed by the app when exporting. */
  output: number[];
  meta: { app: string; exportedAt: string };
}

export function serializeModel(model: NetworkModel): SerializedModel {
  const spec = model.spec;
  const parameters: SerializedLayerParams[] = [];
  for (let l = 1; l < spec.layers.length; l++) {
    parameters.push({
      layer: l,
      weight: toNested(model.weights(l), spec.layers[l].size, spec.layers[l - 1].size),
      bias: Array.from(model.biases(l)),
    });
  }
  return {
    format: MODEL_FORMAT,
    formatVersion: MODEL_FORMAT_VERSION,
    config: specToConfig(spec),
    init: { ...model.init },
    parameters,
    input: Array.from(model.input),
    output: Array.from(model.output),
    meta: { app: APP_ID, exportedAt: new Date().toISOString() },
  };
}

/** Pretty JSON where arrays of numbers (weight rows, biases, inputs) stay on one line. */
export function stringifyModel(data: SerializedModel): string {
  return JSON.stringify(data, null, 2).replace(/\[\s*(-?\d[\d.eE+-]*(?:,\s*-?\d[\d.eE+-]*)*)\s*\]/g, (_, inner: string) => `[${inner.split(/,\s*/).join(', ')}]`);
}

export interface LoadedModel {
  spec: NetworkSpec;
  init: InitSettings;
  /** Null when the file only contained a configuration (fresh initialization). */
  weights: Float64Array[] | null;
  biases: Float64Array[] | null;
  input: Float64Array | null;
  warnings: string[];
}

export type LoadResult = { ok: true; model: LoadedModel } | { ok: false; errors: string[] };

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * Accepts either a full model file (`format: "neural-explorer-3d/mlp"`) or a
 * bare network configuration (`{ "layers": [...] }`).
 */
export function deserializeModel(raw: unknown): LoadResult {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return { ok: false, errors: ['The file must contain a JSON object.'] };
  const data = raw as Record<string, unknown>;

  // Bare configuration.
  if (data.format === undefined && Array.isArray(data.layers)) {
    const result = validateConfig(data);
    if (!result.ok) return { ok: false, errors: result.errors.map((e) => `${e.path}: ${e.message}`) };
    return {
      ok: true,
      model: { spec: result.spec, init: { ...DEFAULT_INIT }, weights: null, biases: null, input: null, warnings: result.warnings.map((w) => w.message) },
    };
  }

  if (data.format !== MODEL_FORMAT) return { ok: false, errors: [`Unknown file format "${String(data.format)}" (expected "${MODEL_FORMAT}").`] };
  if (typeof data.formatVersion !== 'number' || data.formatVersion > MODEL_FORMAT_VERSION)
    return { ok: false, errors: [`Unsupported format version ${String(data.formatVersion)} (this app reads up to ${MODEL_FORMAT_VERSION}).`] };

  const configResult = validateConfig(data.config);
  if (!configResult.ok) return { ok: false, errors: configResult.errors.map((e) => `config.${e.path}: ${e.message}`) };
  const spec = configResult.spec;
  const errors: string[] = [];
  const warnings = configResult.warnings.map((w) => w.message);

  const initRaw = (data.init ?? {}) as Record<string, unknown>;
  const init: InitSettings = {
    scheme: isInitScheme(initRaw.scheme) ? initRaw.scheme : DEFAULT_INIT.scheme,
    seed: isFiniteNumber(initRaw.seed) ? Math.floor(initRaw.seed) : DEFAULT_INIT.seed,
  };

  const weights: Float64Array[] = [new Float64Array(0)];
  const biases: Float64Array[] = [new Float64Array(0)];
  const parameters = Array.isArray(data.parameters) ? (data.parameters as unknown[]) : null;
  if (!parameters) errors.push('"parameters" must be an array.');
  for (let l = 1; l < spec.layers.length && parameters; l++) {
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    const entry = parameters.find((p) => typeof p === 'object' && p !== null && (p as SerializedLayerParams).layer === l) as
      | SerializedLayerParams
      | undefined;
    if (!entry) {
      errors.push(`Missing parameters for layer ${l}.`);
      continue;
    }
    const W = new Float64Array(rows * cols);
    if (!Array.isArray(entry.weight) || entry.weight.length !== rows) {
      errors.push(`Layer ${l}: "weight" must have ${rows} rows.`);
      continue;
    }
    entry.weight.forEach((row, j) => {
      if (!Array.isArray(row) || row.length !== cols || !row.every(isFiniteNumber)) {
        errors.push(`Layer ${l}: weight row ${j} must contain ${cols} finite numbers.`);
        return;
      }
      for (let i = 0; i < cols; i++) W[j * cols + i] = row[i];
    });
    if (!Array.isArray(entry.bias) || entry.bias.length !== rows || !entry.bias.every(isFiniteNumber)) {
      errors.push(`Layer ${l}: "bias" must contain ${rows} finite numbers.`);
      continue;
    }
    weights.push(W);
    biases.push(Float64Array.from(entry.bias));
  }

  let input: Float64Array | null = null;
  if (data.input !== undefined) {
    if (!Array.isArray(data.input) || data.input.length !== spec.layers[0].size || !data.input.every(isFiniteNumber))
      errors.push(`"input" must contain ${spec.layers[0].size} finite numbers.`);
    else input = Float64Array.from(data.input as number[]);
  }

  if (errors.length) return { ok: false, errors };
  return { ok: true, model: { spec, init, weights, biases, input, warnings } };
}

/** Applies a loaded file to an existing model (single batched update). */
export function applyLoadedModel(model: NetworkModel, loaded: LoadedModel): void {
  model.batch(() => {
    model.setArchitecture(loaded.spec, { preserveParams: false, reason: 'import', input: loaded.input ?? undefined });
    model.reinitialize(loaded.init, 'import');
    if (loaded.weights && loaded.biases) model.setParameters(loaded.weights, loaded.biases, 'import');
  });
}
