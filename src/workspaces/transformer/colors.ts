/**
 * Categorical colours of the Transformer views: one per attention head and
 * one per word group. Taken from the Okabe–Ito colour-blind-safe set, avoiding
 * the orange/blue pair that encodes signs; every colour is always paired with
 * a text label (head name, group name), never used alone.
 */
import type { ThemeId } from '../../visualization/palette';

const HEADS: Record<ThemeId, string[]> = {
  dark: ['#3ccf9d', '#e58fc4', '#f0e442', '#c9b8ff'],
  light: ['#00805e', '#b3478f', '#9a8700', '#6a4fd6'],
};

export function headColor(head: number, theme: ThemeId): string {
  const list = HEADS[theme];
  return list[head % list.length];
}

export function headColors(count: number, theme: ThemeId): string[] {
  return Array.from({ length: count }, (_, h) => headColor(h, theme));
}

export const GROUP_LABELS: Record<string, string> = {
  country: 'Countries',
  capital: 'Capitals',
  royal: 'Royals',
  person: 'People',
  animal: 'Animals',
  place: 'Places',
  adjective: 'Adjectives',
  number: 'Numbers',
  other: 'Other words',
};

const GROUPS: Record<ThemeId, Record<string, string>> = {
  dark: {
    country: '#e69f00',
    capital: '#56b4e9',
    royal: '#cc79a7',
    person: '#f0e442',
    animal: '#3ccf9d',
    place: '#ff7a45',
    adjective: '#b39dff',
    number: '#e2e8f0',
    other: '#6b7686',
  },
  light: {
    country: '#b77c00',
    capital: '#1f7fbf',
    royal: '#a8457f',
    person: '#8f8500',
    animal: '#00805e',
    place: '#c4410f',
    adjective: '#6a4fd6',
    number: '#3d4654',
    other: '#9aa4b1',
  },
};

export function groupColor(group: string | undefined, theme: ThemeId): string {
  const table = GROUPS[theme];
  return table[group ?? 'other'] ?? table.other;
}
