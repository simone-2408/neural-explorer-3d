import { describe, expect, it } from 'vitest';
import { createRng } from '../math/random';
import { DEFAULT_OPTIMIZER } from '../training/optimizers';
import { backwardRnn, clipGradients, runRnn, zeroRnnGrads } from './forward';
import { RnnModel, type CellType } from './model';
import { daysTask, memoryTask } from './tasks';
import { evaluateRnn, generate, RnnTrainer } from './trainer';

describe('tasks', () => {
  it('days: the targets are the next letters of the week', () => {
    const task = daysTask(12);
    const seq = task.sample(createRng(3));
    expect(seq.inputs).toHaveLength(12);
    expect(seq.targets.slice(0, -1)).toEqual(seq.inputs.slice(1));
    const typed = task.encode('Friday sat');
    const targets = task.targetsFor(typed);
    // after "friday s" the next letter is known (saturday follows friday): "a"
    expect(task.outputSymbols[targets[7]]).toBe('a');
    // after a lone "s" nothing is known
    expect(task.targetsFor(task.encode('s'))[0]).toBe(-1);
  });

  it('memory: only the last step is scored, with the first letter', () => {
    const task = memoryTask(5);
    const seq = task.sample(createRng(1));
    expect(seq.inputs).toHaveLength(7);
    expect(seq.targets.slice(0, 6).every((t) => t === -1)).toBe(true);
    expect(seq.targets[6]).toBe(seq.inputs[0]);
    expect(task.encode('b x y ?')).toEqual([1, 4, 5, 7]); // A B C D x y z ? — spaces are not symbols
  });
});

describe('RNN', () => {
  for (const cell of ['rnn', 'gru'] as CellType[]) {
    it(`backpropagation through time matches finite differences (${cell})`, () => {
      const model = new RnnModel({ cell, inputSize: 5, hiddenSize: 6, outputSize: 4 }, 2);
      const inputs = [0, 3, 1, 4, 2, 2];
      const targets = [1, -1, 3, 0, -1, 2];
      const grads = zeroRnnGrads(model);
      backwardRnn(model, runRnn(model, inputs), targets, grads);
      const loss = () => {
        const { steps } = runRnn(model, inputs);
        return targets.reduce((sum, t, k) => (t < 0 ? sum : sum - Math.log(steps[k].probs[t])), 0);
      };
      const h = 1e-6;
      for (let k = 0; k < model.params.length; k++) {
        const p = model.params[k];
        for (const idx of [0, 7, Math.floor(p.length / 2), p.length - 1]) {
          if (idx >= p.length) continue;
          const keep = p[idx];
          p[idx] = keep + h;
          const plus = loss();
          p[idx] = keep - h;
          const minus = loss();
          p[idx] = keep;
          expect(grads[k][idx]).toBeCloseTo((plus - minus) / (2 * h), 6);
        }
      }
    });
  }

  it('the GRU with its update gate at 1 keeps its state unchanged', () => {
    const model = new RnnModel({ cell: 'gru', inputSize: 3, hiddenSize: 4, outputSize: 2 }, 5);
    // push the update gate's bias very high: z ≈ 1, so h_t ≈ h_{t−1}
    for (let i = 4; i < 8; i++) model.bih[i] = 50;
    const h0 = Float64Array.from([0.3, -0.2, 0.7, 0.1]);
    const { steps } = runRnn(model, [0, 1, 2, 1], h0);
    for (let i = 0; i < 4; i++) expect(steps[3].h[i]).toBeCloseTo(h0[i], 10);
  });

  it('clips gradients to a maximum global norm', () => {
    const grads = [Float64Array.from([3, 0]), Float64Array.from([0, 4])];
    expect(clipGradients(grads, 1)).toBeCloseTo(5, 12);
    expect(Math.hypot(grads[0][0], grads[1][1])).toBeCloseTo(1, 5);
  });

  it('learns the days of the week, and generates them', () => {
    const task = daysTask(24);
    const model = new RnnModel({ cell: 'rnn', inputSize: task.inputSymbols.length, hiddenSize: 24, outputSize: task.outputSymbols.length }, 1);
    const trainer = new RnnTrainer(model, task, { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.02 }, { batchSize: 16, seed: 2 });
    const before = evaluateRnn(model, trainer.testSet);
    for (let s = 0; s < 200; s++) trainer.trainStep();
    const after = evaluateRnn(model, trainer.testSet);
    expect(after.loss).toBeLessThan(before.loss * 0.1);
    expect(after.accuracy).toBeGreaterThan(0.9);
    // "friday s" can only continue as "saturday": the network remembers the previous word
    const text = generate(model, task.encode('friday s'), 12, 0, createRng(1)).map((i) => task.outputSymbols[i]).join('');
    expect(text.startsWith('aturday')).toBe(true);
  }, 20000);

  it('the gradient vanishes over a long gap in a vanilla RNN, much less in a GRU that starts remembering', () => {
    const task = memoryTask(30);
    const ratio = (cell: CellType, updateGateBias: number) => {
      const model = new RnnModel({ cell, inputSize: task.inputSymbols.length, hiddenSize: 16, outputSize: 4, updateGateBias }, 1);
      const trainer = new RnnTrainer(model, task, { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.01 }, { batchSize: 8, seed: 3 });
      trainer.trainStep();
      const n = trainer.lastHiddenGradNorms;
      return n[0] / n[n.length - 1];
    };
    expect(ratio('rnn', 0)).toBeLessThan(1e-4);
    expect(ratio('gru', 2)).toBeGreaterThan(1e-2);
  });
});
