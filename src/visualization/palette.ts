/**
 * Colour palettes for the 3D scene and the 2D charts.
 *
 * Accessibility: the default diverging pair is blue/orange (distinguishable
 * with the common forms of colour blindness). Sign is never encoded by colour
 * alone: negative weights are also dashed and every value is shown as a
 * signed number in the inspector.
 */
export type ThemeId = 'dark' | 'light';
export type WeightPaletteId = 'blue-orange' | 'purple-green' | 'mono';

export interface ScenePalette {
  theme: ThemeId;
  weightPalette: WeightPaletteId;
  /** Words for the sign colours, used in explanations ("orange", "blue"…). */
  positiveName: string;
  negativeName: string;
  background: string;
  floorGrid: string;
  axis: string;
  /** Colour of positive values (weights and activations). */
  positive: string;
  /** Colour of negative values. */
  negative: string;
  /** Neuron colour at zero activation. */
  neuronOff: string;
  /** Structural colours, used when activation colouring is disabled. */
  neuronInput: string;
  neuronHidden: string;
  neuronOutput: string;
  plateFill: string;
  plateBorder: string;
  plateFocus: string;
  selection: string;
}

export const WEIGHT_PALETTES: Record<
  WeightPaletteId,
  {
    label: string;
    names: [positive: string, negative: string];
    dark: [positive: string, negative: string];
    light: [positive: string, negative: string];
  }
> = {
  'blue-orange': { label: 'Orange + / Blue −', names: ['orange', 'blue'], dark: ['#ffa94d', '#4dabf7'], light: ['#d9480f', '#1971c2'] },
  'purple-green': { label: 'Green + / Purple −', names: ['green', 'purple'], dark: ['#69db7c', '#d0a2ff'], light: ['#2b8a3e', '#862e9c'] },
  mono: { label: 'Monochrome (dashes only)', names: ['bright', 'dim'], dark: ['#f1f3f5', '#8d96a0'], light: ['#212529', '#a5adb6'] },
};

type BaseColors = Omit<ScenePalette, 'positive' | 'negative' | 'theme' | 'weightPalette' | 'positiveName' | 'negativeName'>;

const BASE: Record<ThemeId, BaseColors> = {
  dark: {
    background: '#0b0f15',
    floorGrid: '#1b2430',
    axis: '#5c6b7d',
    neuronOff: '#3b4553',
    neuronInput: '#6f8aa3',
    neuronHidden: '#8a95a5',
    neuronOutput: '#b2a5dc',
    plateFill: '#1d2734',
    plateBorder: '#3a4859',
    plateFocus: '#e6edf5',
    selection: '#ffffff',
  },
  light: {
    background: '#f3f5f8',
    floorGrid: '#d6dce4',
    axis: '#8a96a5',
    neuronOff: '#c6cdd6',
    neuronInput: '#5a7895',
    neuronHidden: '#7b8696',
    neuronOutput: '#7a69b6',
    plateFill: '#dbe2ea',
    plateBorder: '#9fabb9',
    plateFocus: '#1c2530',
    selection: '#0d1117',
  },
};

export function scenePalette(theme: ThemeId, weights: WeightPaletteId): ScenePalette {
  const entry = WEIGHT_PALETTES[weights];
  const [positive, negative] = entry[theme];
  return {
    theme,
    weightPalette: weights,
    positiveName: entry.names[0],
    negativeName: entry.names[1],
    ...BASE[theme],
    positive,
    negative,
  };
}
