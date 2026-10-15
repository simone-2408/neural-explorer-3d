import { describe, expect, it } from 'vitest';
import { symmetricLayer, unitStats } from './diagnostics';
import { runHeadless, SCENARIOS } from './scenarios';

const scenario = (id: string) => SCENARIOS.find((s) => s.id === id)!;
const last = (run: ReturnType<typeof runHeadless>) => run.trainer.history[run.trainer.history.length - 1];

describe('Break the Network scenarios really break (and their fixes work)', () => {
  it('huge learning rate: the loss gets worse or diverges; the fix learns', () => {
    const broken = runHeadless(scenario('huge-lr').settings, 60);
    const first = broken.trainer.history[0].train.loss;
    const end = last(broken).train.loss;
    expect(!Number.isFinite(end) || end > first || last(broken).train.accuracy < 0.8).toBe(true);
    const fixed = runHeadless(scenario('huge-lr').fixed, 60);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.85);
  });

  it('tiny learning rate: almost no progress', () => {
    const broken = runHeadless(scenario('tiny-lr').settings, 30);
    const change = Math.abs(broken.trainer.history[0].train.loss - last(broken).train.loss) / broken.trainer.history[0].train.loss;
    expect(change).toBeLessThan(0.01);
    expect(broken.diagnosis?.id).toBe('slow');
    const fixed = runHeadless(scenario('tiny-lr').fixed, 150);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.9);
  });

  it('network too small: a line cannot separate the circles', () => {
    const broken = runHeadless(scenario('too-small').settings, 120);
    expect(last(broken).train.accuracy).toBeLessThan(0.75);
    expect(broken.diagnosis?.id).toBe('plateau');
    const fixed = runHeadless(scenario('too-small').fixed, 120);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.9);
  });

  it('deep sigmoid network: gradients vanish towards the input', () => {
    const broken = runHeadless(scenario('deep-sigmoid').settings, 20);
    const norms = last(broken).gradNorms;
    expect(norms[1] / norms[norms.length - 1]).toBeLessThan(1e-3);
    expect(broken.diagnosis?.id).toBe('vanishing');
    expect(last(broken).train.accuracy).toBeLessThan(0.7);
    const fixed = runHeadless(scenario('deep-sigmoid').fixed, 120);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.9);
  });

  it('zero initialization: neurons of a layer stay identical', () => {
    const broken = runHeadless(scenario('zero-init').settings, 60);
    expect(symmetricLayer(broken)).toBe(1);
    expect(broken.diagnosis?.id).toBe('symmetric');
    const fixed = runHeadless(scenario('zero-init').fixed, 150);
    expect(symmetricLayer(fixed)).toBe(-1);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.9);
  });

  it('too large initialization: tanh neurons saturate', () => {
    const broken = runHeadless(scenario('large-init').settings, 5);
    expect(unitStats(broken).saturatedFraction).toBeGreaterThanOrEqual(0.5);
    expect(broken.diagnosis?.id).toBe('saturated');
    const fixed = runHeadless(scenario('large-init').fixed, 5);
    expect(unitStats(fixed).saturatedFraction).toBeLessThan(0.2);
  });

  it('negative biases: ReLU neurons are dead and stay dead', () => {
    const broken = runHeadless(scenario('dead-relu').settings, 40);
    expect(unitStats(broken).deadFraction).toBeGreaterThanOrEqual(0.5);
    expect(broken.diagnosis?.id).toBe('dead-relu');
    expect(last(broken).train.accuracy).toBeLessThan(0.75);
    const fixed = runHeadless(scenario('dead-relu').fixed, 40);
    expect(unitStats(fixed).deadFraction).toBeLessThan(0.5);
    expect(last(fixed).train.accuracy).toBeGreaterThan(0.85);
  });

  it('overfitting: training accuracy far above test accuracy', () => {
    const broken = runHeadless(scenario('overfitting').settings, 600);
    const end = last(broken);
    expect(end.train.accuracy - (end.test?.accuracy ?? 1)).toBeGreaterThan(0.15);
    expect(broken.diagnosis?.id).toBe('overfitting');
    const fixed = runHeadless(scenario('overfitting').fixed, 600);
    const fixedEnd = last(fixed);
    expect(fixedEnd.train.accuracy - (fixedEnd.test?.accuracy ?? 1)).toBeLessThan(0.1);
  });
});
