import { describe, expect, it } from 'vitest';
import { isWorkspaceId } from '../app/state';
import { LESSONS, LESSON_GROUPS } from './catalog';

describe('lessons', () => {
  it('cover the eleven topics of the curriculum, in order, then CNNs and RNNs', () => {
    expect(LESSONS.map((l) => l.number)).toEqual(Array.from({ length: 13 }, (_, k) => k + 1));
    expect(LESSONS.map((l) => l.title)).toEqual([
      'What is a neuron?',
      'Weighted sum',
      'Activation functions',
      'Forward propagation',
      'Loss',
      'Backpropagation',
      'Gradient descent',
      'Embeddings',
      'Self-attention',
      'The Transformer',
      'Next-token prediction',
      'Convolution',
      'Recurrence',
    ]);
    expect(LESSON_GROUPS.flatMap((g) => g.lessons)).toEqual(LESSONS);
  });

  it('are made of steps that drive the scene, not pages of text', () => {
    const ids = new Set<string>();
    for (const lesson of LESSONS) {
      expect(ids.has(lesson.id)).toBe(false);
      ids.add(lesson.id);
      expect(isWorkspaceId(lesson.tab)).toBe(true);
      expect(lesson.steps.length).toBeGreaterThanOrEqual(4);
      // the first step sets up the scene, and every lesson asks the learner to do something
      expect(typeof lesson.steps[0].enter).toBe('function');
      expect(lesson.steps.some((s) => s.task)).toBe(true);
      for (const step of lesson.steps) {
        expect(step.title.length).toBeGreaterThan(0);
        if (Array.isArray(step.text)) for (const p of step.text) expect(p.split('*').length % 2).toBe(1); // balanced *emphasis*
      }
    }
  });
});
