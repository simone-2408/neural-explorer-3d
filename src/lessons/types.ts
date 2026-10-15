/**
 * Lessons as data. A lesson is a short sequence of steps; each step drives the
 * live scene (`enter`: switch tab, load a preset, select an element, start an
 * animation…) and may ask the learner to do something in it (`task`), checked
 * against the real model state. No step is a page of text: the scene is the
 * material, the text only points at it.
 */
import type { WorkspaceId } from '../app/state';
import type { LessonContext } from './context';

export interface LessonTask {
  /** What to do, in one sentence. */
  prompt: string;
  /** True once the learner has done it (polled while the step is shown). */
  check(ctx: LessonContext): boolean;
  /** Shown when the check passes. */
  done?: string;
}

export interface LessonStep {
  title: string;
  /** Short paragraphs; *text between asterisks* is emphasized. Functions read live values. */
  text: string[] | ((ctx: LessonContext) => string[]);
  /** Optional formula (TeX) under the text. */
  tex?: string;
  /** Puts the scene in the state the step talks about. */
  enter?(ctx: LessonContext): void | Promise<void>;
  task?: LessonTask;
}

export interface Lesson {
  id: string;
  number: number;
  title: string;
  /** One line for the catalog. */
  summary: string;
  /** Tab where the lesson starts (several lessons visit more than one). */
  tab: WorkspaceId;
  steps: LessonStep[];
}
