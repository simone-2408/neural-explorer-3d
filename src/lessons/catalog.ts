/** All lessons, in order. */
import { ARCHITECTURE_LESSONS } from './architectureLessons';
import { NETWORK_LESSONS } from './networkLessons';
import { TRANSFORMER_LESSONS } from './transformerLessons';
import type { Lesson } from './types';

export const LESSONS: Lesson[] = [...NETWORK_LESSONS, ...TRANSFORMER_LESSONS, ...ARCHITECTURE_LESSONS];

export const LESSON_GROUPS: { title: string; lessons: Lesson[] }[] = [
  { title: 'Neural networks', lessons: NETWORK_LESSONS },
  { title: 'Transformers and language models', lessons: TRANSFORMER_LESSONS },
  { title: 'More architectures', lessons: ARCHITECTURE_LESSONS },
];
