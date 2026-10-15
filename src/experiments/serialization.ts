/**
 * Experiments saved as JSON files, so they can be shared, repeated and
 * reloaded. Loading validates every field and reports readable errors; values
 * that do not fit the current model (e.g. a layer GPT-2 has but the toy model
 * has not) are checked by the workspace when the experiment is applied.
 */
import type { Experiment, Intervention, InterventionSpec, TensorId, TokenRef } from './types';

export const EXPERIMENT_FORMAT = 'neural-explorer-3d/experiment';
export const EXPERIMENT_FORMAT_VERSION = 1;

export interface ExperimentFile {
  format: typeof EXPERIMENT_FORMAT;
  formatVersion: number;
  savedAt: string;
  experiment: Experiment;
}

let counter = 0;

/** A short id for an intervention (unique within the page). */
export function newId(): string {
  counter = (counter + 1) % 1e6;
  return `${Date.now().toString(36)}${counter.toString(36)}`;
}

export function serializeExperiment(experiment: Experiment): string {
  const file: ExperimentFile = { format: EXPERIMENT_FORMAT, formatVersion: EXPERIMENT_FORMAT_VERSION, savedAt: new Date().toISOString(), experiment };
  return JSON.stringify(file, null, 2);
}

class FormatError extends Error {}

function fail(message: string): never {
  throw new FormatError(message);
}

function object(value: unknown, where: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) fail(`${where} must be an object`);
  return value as Record<string, unknown>;
}

function integer(value: unknown, where: string, min = 0, max = 1e6): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) fail(`${where} must be an integer between ${min} and ${max}`);
  return value;
}

function finite(value: unknown, where: string, limit = 1e4): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > limit) fail(`${where} must be a number between −${limit} and ${limit}`);
  return value;
}

function token(value: unknown, where: string): TokenRef {
  const o = object(value, where);
  if (typeof o.text !== 'string' || o.text.length > 100) fail(`${where}.text must be a short string`);
  return { id: integer(o.id, `${where}.id`, 0, 1e6), text: o.text };
}

const TENSORS: TensorId[] = ['embed', 'pos', 'q', 'k', 'v', 'o', 'mlpIn', 'mlpOut'];

function spec(value: unknown, where: string): InterventionSpec {
  const o = object(value, where);
  switch (o.kind) {
    case 'head':
      return { kind: 'head', layer: integer(o.layer, `${where}.layer`, 0, 99), head: integer(o.head, `${where}.head`, 0, 99), scale: finite(o.scale, `${where}.scale`) };
    case 'mlp':
      return { kind: 'mlp', layer: integer(o.layer, `${where}.layer`, 0, 99), scale: finite(o.scale, `${where}.scale`) };
    case 'neuron':
      if (o.mode !== 'scale' && o.mode !== 'set') fail(`${where}.mode must be "scale" or "set"`);
      return { kind: 'neuron', layer: integer(o.layer, `${where}.layer`, 0, 99), unit: integer(o.unit, `${where}.unit`, 0, 1e5), mode: o.mode, value: finite(o.value, `${where}.value`) };
    case 'embedding':
      return { kind: 'embedding', position: integer(o.position, `${where}.position`, 0, 4096), token: token(o.token, `${where}.token`), mix: finite(o.mix, `${where}.mix`, 10) };
    case 'remove':
      return { kind: 'remove', position: integer(o.position, `${where}.position`, 0, 4096) };
    case 'weight': {
      if (!TENSORS.includes(o.tensor as TensorId)) fail(`${where}.tensor must be one of ${TENSORS.join(', ')}`);
      let cell: { row: number; col: number; value: number } | null = null;
      if (o.cell !== null && o.cell !== undefined) {
        const c = object(o.cell, `${where}.cell`);
        cell = { row: integer(c.row, `${where}.cell.row`, 0, 1e5), col: integer(c.col, `${where}.cell.col`, 0, 1e5), value: finite(c.value, `${where}.cell.value`) };
      }
      return {
        kind: 'weight',
        tensor: o.tensor as TensorId,
        layer: integer(o.layer ?? 0, `${where}.layer`, 0, 99),
        head: integer(o.head ?? 0, `${where}.head`, 0, 99),
        scale: finite(o.scale ?? 1, `${where}.scale`),
        cell,
      };
    }
    default:
      fail(`${where}.kind "${String(o.kind)}" is not a known intervention`);
  }
}

/** Parses an experiment file (or a bare experiment object). Throws an Error with a readable message. */
export function parseExperiment(json: unknown): Experiment {
  try {
    const root = object(json, 'The file');
    const body = 'experiment' in root ? object(root.experiment, 'experiment') : root;
    if ('format' in root && root.format !== EXPERIMENT_FORMAT) fail(`This is not an experiment file (format "${String(root.format)}")`);
    if (body.model !== 'toy' && body.model !== 'gpt2') fail('experiment.model must be "toy" or "gpt2"');
    if (typeof body.text !== 'string' || body.text.length > 600) fail('experiment.text must be a string of at most 600 characters');
    const list = body.interventions ?? [];
    if (!Array.isArray(list) || list.length > 200) fail('experiment.interventions must be a list of at most 200 items');
    const interventions: Intervention[] = list.map((item, k) => {
      const where = `intervention ${k + 1}`;
      const o = object(item, where);
      return { ...spec(o, where), id: newId(), enabled: o.enabled !== false };
    });
    return {
      model: body.model,
      text: body.text,
      target: body.target === null || body.target === undefined ? null : token(body.target, 'experiment.target'),
      contrast: body.contrast === null || body.contrast === undefined ? null : token(body.contrast, 'experiment.contrast'),
      interventions,
    };
  } catch (error) {
    if (error instanceof FormatError) throw new Error(error.message);
    throw error;
  }
}
