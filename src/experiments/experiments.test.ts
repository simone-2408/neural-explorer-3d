import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { runTransformer } from '../transformer/forward';
import { loadTransformer, type TransformerFile } from '../transformer/model';
import { tokenize } from '../transformer/tokenizer';
import { effectBetween, logSoftmax, support } from './metrics';
import { toyPresets } from './presets';
import { parseExperiment, serializeExperiment } from './serialization';
import { applyWeightEdits, prepare, runToyExperiment, toyCurve, toyNeuronScan, toySweep } from './toyEngine';
import type { Experiment, Intervention, InterventionSpec } from './types';

const file = JSON.parse(readFileSync(join(process.cwd(), 'public/models/toy-transformer.json'), 'utf8')) as TransformerFile;
const model = loadTransformer(file);
const id = (word: string) => model.index.get(word)!;
const IT = "the cat didn't cross the river because it was";

let n = 0;
const item = (spec: InterventionSpec, enabled = true): Intervention => ({ ...spec, id: `t${n++}`, enabled });
const experiment = (text: string, interventions: Intervention[] = [], target: string | null = null, contrast: string | null = null): Experiment => ({
  model: 'toy',
  text,
  target: target ? { id: id(target), text: target } : null,
  contrast: contrast ? { id: id(contrast), text: contrast } : null,
  interventions,
});
const lastLogits = (text: string, interventions: InterventionSpec[]) => {
  const p = prepare(model, text, interventions);
  return runTransformer(p.model, p.tokens, p.interventions).logits.at(-1)!;
};
const expectSameLogits = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 10) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i], digits);
};

describe('metrics', () => {
  it('measures nothing when nothing changes, and KL ≥ 0 otherwise', () => {
    const base = logSoftmax([2, 1, 0, -1]);
    expect(effectBetween(base, base, 0, 1)).toEqual({ dlogp: 0, dp: 0, kl: 0, dlogit: 0 });
    const run = logSoftmax([0, 1, 0, -1]);
    const e = effectBetween(base, run, 0, 1);
    expect(e.dlogp).toBeLessThan(0);
    expect(e.dlogit).toBeCloseTo(-2, 12); // logit differences ignore the normalization
    expect(e.kl).toBeGreaterThan(0);
    expect(support(e, 'logp')).toBeCloseTo(-e.dlogp, 12);
    expect(support(e, 'kl')).toBe(e.kl);
    expect(effectBetween(base, run, 0, null).dlogit).toBeNull();
  });
});

describe('toy experiments', () => {
  it('an empty experiment changes nothing, and the default target is the top prediction', () => {
    const { result } = runToyExperiment(model, experiment('the capital of france is'));
    expect(result.target.text).toBe('paris');
    expect(result.effect.dlogp).toBe(0);
    expect(result.effect.kl).toBe(0);
    expect(result.delta.flat().every((v) => v === 0)).toBe(true);
    expect(result.delta).toHaveLength(1 + 2 * model.config.nLayers);
  });

  it('scale 1 is the identity; disabled interventions are ignored', () => {
    const base = lastLogits(IT, []);
    expectSameLogits(lastLogits(IT, [{ kind: 'head', layer: 1, head: 2, scale: 1 }, { kind: 'mlp', layer: 0, scale: 1 }, { kind: 'neuron', layer: 1, unit: 5, mode: 'scale', value: 1 }]), base, 12);
    const { result } = runToyExperiment(model, experiment(IT, [item({ kind: 'head', layer: 1, head: 2, scale: 0 }, false)]));
    expect(result.effect.kl).toBe(0);
  });

  it('W_O × s of a head is the same as scaling the head’s output by s', () => {
    for (const s of [0, 2]) {
      const viaHead = lastLogits(IT, [{ kind: 'head', layer: 1, head: 2, scale: s }]);
      const viaWeights = lastLogits(IT, [{ kind: 'weight', tensor: 'o', layer: 1, head: 2, scale: s, cell: null }]);
      expectSameLogits(viaWeights, viaHead);
    }
  });

  it('MLP × 0 is the same as zeroing W_out and its bias', () => {
    const viaMlp = lastLogits(IT, [{ kind: 'mlp', layer: 0, scale: 0 }]);
    const edited = applyWeightEdits(model, [{ kind: 'weight', tensor: 'mlpOut', layer: 0, head: 0, scale: 0, cell: null }]);
    const fc2 = edited.blocks[0].fc2;
    edited.blocks[0] = { ...edited.blocks[0], fc2: { ...fc2, bias: new Float64Array(fc2.bias.length) } };
    expectSameLogits(runTransformer(edited, tokenize(model, IT)).logits.at(-1)!, viaMlp);
  });

  it('weight edits copy the arrays and never touch the loaded model', () => {
    const before = model.positionEmbedding[5];
    const edited = applyWeightEdits(model, [{ kind: 'weight', tensor: 'pos', layer: 0, head: 0, scale: 0, cell: { row: 0, col: 5, value: 1.5 } }]);
    expect(model.positionEmbedding[5]).toBe(before);
    expect(edited.positionEmbedding[5]).toBe(1.5);
    expect(edited.positionEmbedding[6]).toBe(0);
    expect(edited.tokenEmbedding).toBe(model.tokenEmbedding);
  });

  it('an embedding swap with mix 1 equals writing the other word', () => {
    const swapped = lastLogits('the capital of france is', [{ kind: 'embedding', position: 4, token: { id: id('italy'), text: 'italy' }, mix: 1 }]);
    expectSameLogits(swapped, lastLogits('the capital of italy is', []));
  });

  it('removing a token renumbers the positions and leaves the earlier tokens untouched (causal attention)', () => {
    const { result } = runToyExperiment(model, experiment(IT, [item({ kind: 'remove', position: 6 }), item({ kind: 'remove', position: 0 })]));
    expect(result.alignment).toEqual([0, 1, 2, 3, 4, 5, 7, 8, 9]); // <s> (position 0) is never removed
    expect(result.tokensAfter.map((t) => t.text)).not.toContain('river');
    for (const stage of result.delta) {
      expect(stage[6]).toBeNull();
      for (let p = 0; p < 6; p++) expect(stage[p]).toBe(0);
      for (let p = 7; p < stage.length; p++) expect(stage[p]).toBeGreaterThan(0);
    }
  });

  it('sweeps match single runs', () => {
    const e = experiment(IT, [], 'hungry', 'deep');
    const { result } = runToyExperiment(model, e);
    const sweep = toySweep(model, e, result, 'original');
    const single = runToyExperiment(model, experiment(IT, [item({ kind: 'head', layer: 1, head: 2, scale: 0 })], 'hungry', 'deep')).result.effect;
    expect(sweep.heads[1][2].dlogp).toBeCloseTo(single.dlogp, 12);
    expect(sweep.heads[1][2].dlogit!).toBeCloseTo(single.dlogit!, 12);
    expect(sweep.heads).toHaveLength(model.config.nLayers);
    expect(sweep.mlps).toHaveLength(model.config.nLayers);
    // On top of the experiment, zeroing an already-zeroed head changes nothing.
    const off = experiment(IT, [item({ kind: 'head', layer: 1, head: 2, scale: 0 })], 'hungry', 'deep');
    const offSweep = toySweep(model, off, runToyExperiment(model, off).result, 'experiment');
    expect(offSweep.heads[1][2].kl).toBeCloseTo(0, 12);
  });

  it('scale curves pass through the original model at scale 1', () => {
    const e = experiment(IT, [item({ kind: 'head', layer: 1, head: 2, scale: 0 })], 'hungry', 'deep');
    const { result } = runToyExperiment(model, e);
    const curve = toyCurve(model, e, result, { kind: 'head', layer: 1, head: 2 }, [0, 1]);
    expect(curve.p[0]).toBeCloseTo(result.after.pTarget, 10);
    expect(curve.p[1]).toBeCloseTo(result.before.pTarget, 10);
    expect(curve.logitDiff).toHaveLength(2);
  });

  it('scans every neuron of a layer: measured and first-order estimate', () => {
    const e = experiment('the capital of france is', [], 'paris', 'rome');
    const { result } = runToyExperiment(model, e);
    const scan = toyNeuronScan(model, e, result, 'original', 1);
    expect(scan.metric).toBe('logitdiff');
    expect(scan.estimate).toHaveLength(model.config.dMlp);
    expect(scan.measured).toHaveLength(model.config.dMlp);
    expect(scan.activations[0]).toHaveLength(result.tokens.length);
    const unit = scan.measured.reduce((a, b) => (Math.abs(b.effect.dlogit!) > Math.abs(a.effect.dlogit!) ? b : a));
    expect(Math.sign(scan.estimate[unit.unit])).toBe(Math.sign(unit.effect.dlogit!));
  });
});

describe('presets', () => {
  const run = (presetId: string) => {
    const preset = toyPresets(model).find((p) => p.id === presetId)!;
    return runToyExperiment(model, preset.experiment).result;
  };

  it('do what their descriptions say', () => {
    const head = run('toy-it-head');
    expect(head.before.pTarget).toBeCloseTo(0.465, 2);
    expect(head.after.pTarget).toBeLessThan(0.005);
    expect(run('toy-amplify').after.pTarget).toBeGreaterThan(0.95);
    const removed = run('toy-remove');
    expect(removed.after.pTarget).toBeGreaterThan(0.6);
    expect(run('toy-fact').after.pTarget).toBeLessThan(0.001);
    expect(run('toy-swap').after.top[0].text).toBe('rome');
    const order = run('toy-order');
    expect(order.after.top[0].text).toBe('a');
    expect(order.after.pTarget).toBeLessThan(0.001);
    expect(['germany', 'italy', 'china', 'france']).toContain(run('toy-neuron').after.top[0].text);
    const weights = run('toy-weights');
    expect(weights.before.pTarget).toBeGreaterThan(0.99);
    expect(weights.after.pTarget).toBeLessThan(0.2);
  });

  it('the single neurons of MLP 1 barely matter for the fact (see “Where is the fact?”)', () => {
    const e = experiment('the capital of france is', [], 'paris');
    const scan = toyNeuronScan(model, e, runToyExperiment(model, e).result, 'original', 0);
    expect(Math.max(...scan.measured.map((m) => Math.abs(m.effect.dlogp)))).toBeLessThan(0.01);
  });
});

describe('serialization', () => {
  it('round-trips and validates', () => {
    const preset = toyPresets(model).find((p) => p.id === 'toy-swap')!;
    const back = parseExperiment(JSON.parse(serializeExperiment(preset.experiment)));
    expect(back.text).toBe(preset.experiment.text);
    expect(back.interventions.map(({ id: _id, ...rest }) => rest)).toEqual(preset.experiment.interventions.map(({ id: _id, ...rest }) => rest));
    expect(() => parseExperiment({ model: 'toy', text: 'x', interventions: [{ kind: 'head', layer: -1, head: 0, scale: 0 }] })).toThrow(/layer/);
    expect(() => parseExperiment({ model: 'bert', text: 'x' })).toThrow(/model/);
    expect(() => parseExperiment({ format: 'other', experiment: {} })).toThrow(/not an experiment/);
  });
});
