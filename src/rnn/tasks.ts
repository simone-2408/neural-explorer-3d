/**
 * Sequence tasks for the RNN tab, generated from a seed.
 *
 *   days    — predict the next character of the days of the week, written
 *             in order ("…friday saturday sunday monday…"): inside a word the
 *             next letter is easy, but after "s" it can be "a" (saturday) or
 *             "u" (sunday) — the model has to remember the previous word.
 *   memory  — read a letter (A–D), then `gap` meaningless symbols, then "?":
 *             answer the first letter. Only the last step is scored, so the
 *             gradient must travel back over the whole gap.
 */
import type { Rng } from '../math/random';

export type RnnTaskId = 'days' | 'memory';

export interface Sequence {
  inputs: number[];
  /** Target class at each step, −1 where nothing is predicted. */
  targets: number[];
}

export interface RnnTaskInfo {
  id: RnnTaskId;
  label: string;
  description: string;
}

export const RNN_TASKS: Record<RnnTaskId, RnnTaskInfo> = {
  days: {
    id: 'days',
    label: 'Next letter',
    description: 'The days of the week, in order, one letter at a time: predict the next letter.',
  },
  memory: {
    id: 'memory',
    label: 'Remember',
    description: 'A letter A–D, then a gap of meaningless symbols, then “?”: answer the first letter. Only the last step is scored.',
  },
};

export const DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
const WEEK = DAYS.join(' ') + ' ';

export interface RnnTask {
  info: RnnTaskInfo;
  inputSymbols: string[];
  outputSymbols: string[];
  /** Length of the training sequences. */
  length: number;
  sample(rng: Rng): Sequence;
  /** Symbols of a text that the task knows (others are dropped). */
  encode(text: string): number[];
  /** Targets for an input sequence typed by the user (null where unknown). */
  targetsFor(inputs: number[]): number[];
}

export function daysTask(length = 32): RnnTask {
  const symbols = [...new Set(WEEK)].sort((a, b) => (a === ' ' ? -1 : b === ' ' ? 1 : a.localeCompare(b)));
  const index = new Map(symbols.map((s, i) => [s, i]));
  const encode = (text: string) => [...text.toLowerCase()].flatMap((ch) => (index.has(ch) ? [index.get(ch)!] : []));
  return {
    info: RNN_TASKS.days,
    inputSymbols: symbols,
    outputSymbols: symbols,
    length,
    sample(rng) {
      const start = Math.floor(rng.next() * WEEK.length);
      const text = Array.from({ length: length + 1 }, (_, k) => WEEK[(start + k) % WEEK.length]).join('');
      const ids = encode(text);
      return { inputs: ids.slice(0, length), targets: ids.slice(1) };
    },
    encode,
    targetsFor(inputs) {
      // The next letter is known when the last letters typed occur at only one place of the week.
      const text = inputs.map((i) => symbols[i]).join('');
      const cycle = WEEK + WEEK;
      return inputs.map((_, t) => {
        const context = text.slice(Math.max(0, t - 11), t + 1);
        const places = new Set<number>();
        for (let at = cycle.indexOf(context); at >= 0; at = cycle.indexOf(context, at + 1)) places.add(at % WEEK.length);
        if (places.size !== 1) return -1;
        const [at] = places;
        return index.get(WEEK[(at + context.length) % WEEK.length]) ?? -1;
      });
    },
  };
}

export const SIGNALS = ['A', 'B', 'C', 'D'];
export const NOISE = ['x', 'y', 'z'];

export function memoryTask(gap = 6): RnnTask {
  const inputSymbols = [...SIGNALS, ...NOISE, '?'];
  const query = inputSymbols.length - 1;
  return {
    info: RNN_TASKS.memory,
    inputSymbols,
    outputSymbols: [...SIGNALS],
    length: gap + 2,
    sample(rng) {
      const signal = Math.floor(rng.next() * SIGNALS.length);
      const noise = Array.from({ length: gap }, () => SIGNALS.length + Math.floor(rng.next() * NOISE.length));
      const inputs = [signal, ...noise, query];
      return { inputs, targets: inputs.map((_, t) => (t === inputs.length - 1 ? signal : -1)) };
    },
    encode(text) {
      return [...text].flatMap((ch) => {
        const i = inputSymbols.indexOf(SIGNALS.includes(ch.toUpperCase()) ? ch.toUpperCase() : ch.toLowerCase());
        return i >= 0 ? [i] : [];
      });
    },
    targetsFor(inputs) {
      const signal = inputs[0] !== undefined && inputs[0] < SIGNALS.length ? inputs[0] : -1;
      return inputs.map((id, t) => (id === query && t > 0 ? signal : -1));
    },
  };
}

export function makeTask(id: RnnTaskId, gap: number): RnnTask {
  return id === 'days' ? daysTask() : memoryTask(gap);
}
