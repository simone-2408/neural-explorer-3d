/**
 * Experiments on GPT-2 small, run by the Python backend
 * (python/server/experiments.py). The request carries the enabled
 * interventions with exactly the JSON shape of types.ts.
 */
import { post } from '../llm/client';
import type { LlmToken } from '../llm/types';
import type { AttentionPair, ComponentRef, Experiment, ExperimentResult, InterventionSpec, NeuronScan, ScaleCurve, SweepResult } from './types';

interface Request {
  text: string;
  interventions: InterventionSpec[];
  target: number | null;
  contrast: number | null;
}

/** The enabled interventions, without the UI fields (id, enabled). */
export function activeSpecs(experiment: Experiment, skip?: (i: InterventionSpec) => boolean): InterventionSpec[] {
  return experiment.interventions.filter((i) => i.enabled && !skip?.(i)).map(({ id: _id, enabled: _enabled, ...spec }) => spec as InterventionSpec);
}

function request(experiment: Experiment, interventions = activeSpecs(experiment)): Request {
  return { text: experiment.text, interventions, target: experiment.target?.id ?? null, contrast: experiment.contrast?.id ?? null };
}

export async function gpt2Experiment(experiment: Experiment, signal?: AbortSignal): Promise<ExperimentResult> {
  const result = await post<Omit<ExperimentResult, 'model' | 'source'>>('experiment', request(experiment), signal);
  return { ...result, model: 'gpt2', source: 'live' };
}

/** Effect of zeroing each head and each MLP layer alone (on the original model, or on top of the experiment). */
export async function gpt2Sweep(experiment: Experiment, result: ExperimentResult, baseline: 'original' | 'experiment', signal?: AbortSignal): Promise<SweepResult> {
  const body = { ...request(experiment, baseline === 'experiment' ? activeSpecs(experiment) : []), target: result.target.id, contrast: result.contrast?.id ?? null };
  const sweep = await post<Omit<SweepResult, 'source'>>('sweep', body, signal);
  return { ...sweep, baseline, source: 'live' };
}

/** Estimates for every neuron of a layer, measurements for the strongest 24 (and for `extra` units). */
export async function gpt2NeuronScan(experiment: Experiment, result: ExperimentResult, baseline: 'original' | 'experiment', layer: number, extra: number[] = [], signal?: AbortSignal): Promise<NeuronScan> {
  const body = { ...request(experiment, baseline === 'experiment' ? activeSpecs(experiment) : []), target: result.target.id, contrast: result.contrast?.id ?? null, layer, k: 24, extra };
  const scan = await post<Omit<NeuronScan, 'source'>>('neurons', body, signal);
  return { ...scan, source: 'live' };
}

export function gpt2Curve(experiment: Experiment, result: ExperimentResult, component: ComponentRef, scales: number[], signal?: AbortSignal): Promise<ScaleCurve> {
  return post<ScaleCurve>('curve', { ...request(experiment), target: result.target.id, contrast: result.contrast?.id ?? null, component, scales }, signal);
}

export function gpt2Attention(experiment: Experiment, layer: number, head: number, signal?: AbortSignal): Promise<AttentionPair> {
  return post<AttentionPair>('attention', { ...request(experiment), layer, head }, signal);
}

/** GPT-2 tokens of a short text (to pick a target, a contrast or a replacement token). */
export async function gpt2Tokenize(text: string): Promise<LlmToken[]> {
  return (await post<{ tokens: LlmToken[] }>('tokenize', { text })).tokens;
}
