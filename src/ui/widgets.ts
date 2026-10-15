/**
 * Reusable form/display widgets built with `h()`. Widgets that hold a value
 * expose `set*()` so panels can sync them with the model without being
 * re-created (re-creating an input while the user drags it breaks the drag).
 */
import { formatSigned, formatValue } from '../utils/format';
import { h, type Child } from './dom';
import { icon, type IconName } from './icons';

// ------------------------------------------------------------------ buttons

export interface ButtonOptions {
  label?: string;
  icon?: IconName;
  title?: string;
  onClick: (event: MouseEvent) => void;
  variant?: 'default' | 'primary' | 'ghost' | 'danger';
  small?: boolean;
  pressed?: boolean;
  disabled?: boolean;
  class?: string;
}

export function button(o: ButtonOptions): HTMLButtonElement {
  const classes = ['btn', `btn--${o.variant ?? 'default'}`];
  if (o.small) classes.push('btn--sm');
  if (!o.label) classes.push('btn--icon');
  if (o.class) classes.push(o.class);
  const el = h(
    'button',
    { type: 'button', class: classes.join(' '), title: o.title, 'aria-label': o.title ?? o.label, onClick: o.onClick, disabled: o.disabled },
    o.icon ? icon(o.icon) : null,
    o.label ? h('span', { class: 'btn__label' }, o.label) : null,
  );
  if (o.pressed !== undefined) el.setAttribute('aria-pressed', String(o.pressed));
  return el;
}

// ------------------------------------------------------------------ slider

export interface SliderOptions {
  label?: Child;
  min: number;
  max: number;
  step: number;
  value: number;
  digits?: number;
  /** Colour the track from 0 to the value (orange/blue like weights). */
  diverging?: boolean;
  ariaLabel: string;
  onInput(value: number): void;
}

export interface Slider {
  el: HTMLElement;
  setValue(value: number): void;
}

/** Range + numeric field. Typed values outside the range extend the range. */
export function slider(o: SliderOptions): Slider {
  const digits = o.digits ?? 3;
  let min = o.min;
  let max = o.max;
  const range = h('input', { type: 'range', class: `slider__range${o.diverging ? ' is-diverging' : ''}`, min, max, step: o.step, 'aria-label': o.ariaLabel });
  const number = h('input', { type: 'number', class: 'slider__number', step: 'any', 'aria-label': `${o.ariaLabel} (exact value)` });

  const paint = (value: number) => {
    const pct = (v: number) => ((Math.min(Math.max(v, min), max) - min) / (max - min)) * 100;
    const zero = o.diverging ? pct(0) : 0;
    const at = pct(value);
    range.style.setProperty('--from', `${Math.min(zero, at)}%`);
    range.style.setProperty('--to', `${Math.max(zero, at)}%`);
    range.dataset.sign = value < 0 ? 'neg' : 'pos';
  };
  const extend = (value: number) => {
    if (value < min) min = Math.floor(value);
    if (value > max) max = Math.ceil(value);
    range.min = String(min);
    range.max = String(max);
  };
  const setValue = (value: number) => {
    extend(value);
    if (document.activeElement !== range) range.value = String(value);
    if (document.activeElement !== number) number.value = value.toFixed(digits);
    paint(value);
  };

  range.addEventListener('input', () => {
    const value = Number(range.value);
    number.value = value.toFixed(digits);
    paint(value);
    o.onInput(value);
  });
  number.addEventListener('input', () => {
    const value = Number(number.value);
    if (number.value.trim() === '' || !Number.isFinite(value)) return;
    extend(value);
    range.value = String(value);
    paint(value);
    o.onInput(value);
  });
  number.addEventListener('change', () => {
    number.value = Number(range.value).toFixed(digits);
  });

  setValue(o.value);
  const el = h('div', { class: 'slider' }, o.label ? h('div', { class: 'slider__label' }, o.label) : null, h('div', { class: 'slider__controls' }, range, number));
  return { el, setValue };
}

// ------------------------------------------------------------------ toggle

export interface Toggle {
  el: HTMLLabelElement;
  set(checked: boolean): void;
}

export function toggle(o: { label: string; checked: boolean; title?: string; disabled?: boolean; onChange(checked: boolean): void }): Toggle {
  const input = h('input', { type: 'checkbox', role: 'switch', class: 'toggle__input', disabled: o.disabled });
  input.checked = o.checked;
  input.addEventListener('change', () => o.onChange(input.checked));
  const el = h(
    'label',
    { class: `toggle${o.disabled ? ' is-disabled' : ''}`, title: o.title },
    input,
    h('span', { class: 'toggle__track', 'aria-hidden': 'true' }, h('span', { class: 'toggle__thumb' })),
    h('span', { class: 'toggle__label' }, o.label),
  );
  return { el, set: (checked) => (input.checked = checked) };
}

// ------------------------------------------------------------------ select

export function select<T extends string>(o: {
  options: { value: T; label: string }[];
  value: T;
  ariaLabel: string;
  onChange(value: T): void;
}): HTMLSelectElement {
  const el = h('select', { class: 'select', 'aria-label': o.ariaLabel }, o.options.map((opt) => h('option', { value: opt.value }, opt.label)));
  el.value = o.value;
  el.addEventListener('change', () => o.onChange(el.value as T));
  return el;
}

// --------------------------------------------------------------- segmented

export interface Segmented<T extends string> {
  el: HTMLElement;
  set(value: T): void;
}

export function segmented<T extends string>(o: {
  options: { value: T; label: string; title?: string }[];
  value: T;
  ariaLabel: string;
  onChange(value: T): void;
}): Segmented<T> {
  const buttons = o.options.map((opt) =>
    h('button', { type: 'button', class: 'segmented__item', title: opt.title, dataset: { value: opt.value }, onClick: () => o.onChange(opt.value) }, opt.label),
  );
  const set = (value: T) => buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.value === value)));
  set(o.value);
  return { el: h('div', { class: 'segmented', role: 'group', 'aria-label': o.ariaLabel }, buttons), set };
}

// ----------------------------------------------------------------- section

const SECTION_STATE_KEY = 'ne3d.sections';

function readSectionState(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(SECTION_STATE_KEY) ?? '{}') as Record<string, boolean>;
  } catch {
    return {};
  }
}

/** Collapsible panel section; the open/closed state is remembered per id. */
export function section(o: { id: string; title: string; open?: boolean; aside?: Child; children: Child[] }): HTMLDetailsElement {
  const stored = readSectionState()[o.id];
  const el = h(
    'details',
    { class: 'section', dataset: { section: o.id } },
    h('summary', { class: 'section__header' }, h('span', { class: 'section__title' }, o.title), o.aside ? h('span', { class: 'section__aside' }, o.aside) : null),
    h('div', { class: 'section__body' }, o.children),
  );
  el.open = stored ?? o.open ?? true;
  el.addEventListener('toggle', () => {
    try {
      const state = readSectionState();
      state[o.id] = el.open;
      localStorage.setItem(SECTION_STATE_KEY, JSON.stringify(state));
    } catch {
      /* storage unavailable: ignore */
    }
  });
  return el;
}

// ----------------------------------------------------------------- display

/** Key/value row. */
export function kv(key: Child, value: Child, className = ''): HTMLElement {
  return h('div', { class: `kv ${className}` }, h('span', { class: 'kv__key' }, key), h('span', { class: 'kv__value' }, value));
}

/** Number with sign-aware styling: colour *and* explicit sign (+/−). */
export function num(value: number, options: { digits?: number; signed?: boolean; plain?: boolean } = {}): HTMLSpanElement {
  const text = options.signed ? formatSigned(value, options.digits) : formatValue(value, options.digits);
  const cls = options.plain ? 'num' : `num ${value > 0 ? 'num--pos' : value < 0 ? 'num--neg' : 'num--zero'}`;
  return h('span', { class: cls }, text);
}

export type Level = 1 | 2 | 3;

const LEVELS: Record<Level, { label: string; title: string }> = {
  1: { label: 'Visual', title: 'Level 1 — what you see in the 3D scene' },
  2: { label: 'Numbers', title: 'Level 2 — the actual values' },
  3: { label: 'Math', title: 'Level 3 — the formula behind it' },
};

/** Heading of one of the three explanation levels (visual → numbers → math). */
export function levelHeading(level: Level, extra?: Child): HTMLElement {
  return h(
    'h3',
    { class: `level-heading level-heading--${level}`, title: LEVELS[level].title },
    h('span', { class: 'level-heading__badge' }, String(level)),
    h('span', null, LEVELS[level].label),
    extra ? h('span', { class: 'level-heading__extra' }, extra) : null,
  );
}

export function note(children: Child, kind: 'info' | 'warning' = 'info'): HTMLElement {
  return h('div', { class: `note note--${kind}`, role: kind === 'warning' ? 'status' : undefined }, icon(kind === 'warning' ? 'warning' : 'info', 14), h('div', null, children));
}

/** Small horizontal bar centred on zero, for contributions (−1…1 of the row scale). */
export function signedBar(value: number, scale: number): HTMLElement {
  const t = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
  const bar = h('span', { class: `signed-bar__fill ${value < 0 ? 'is-neg' : 'is-pos'}` });
  bar.style.width = `${t * 50}%`;
  bar.style.left = value < 0 ? `${50 - t * 50}%` : '50%';
  return h('span', { class: 'signed-bar', 'aria-hidden': 'true' }, bar);
}
