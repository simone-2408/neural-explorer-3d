/** Displays shared by the CNN inspectors. */
import { maxAbs } from '../../../math/stats';
import { h } from '../../../ui/dom';
import { formatSigned, formatValue } from '../../../utils/format';

/**
 * A small k×k grid of numbers with a diverging background (colour = sign,
 * opacity = size), e.g. a kernel, an image window or their products.
 */
export function numberGrid(
  values: ArrayLike<number>,
  size: number,
  o: { colors: { positive: string; negative: string }; signed?: boolean; digits?: number; scale?: number; caption?: string; onCell?(index: number): void; selected?: number | null; missing?: boolean[] },
): HTMLElement {
  const scale = o.scale ?? (maxAbs(values) || 1);
  const grid = h('div', { class: 'cnn-grid', role: 'grid' });
  grid.style.setProperty('--n', String(size));
  for (let k = 0; k < size * size; k++) {
    const v = values[k];
    const missing = o.missing?.[k] ?? false;
    const t = Math.min(1, Math.abs(v) / scale);
    const cell = h(o.onCell ? 'button' : 'span', { class: `cnn-grid__cell${missing ? ' is-missing' : ''}${o.selected === k ? ' is-selected' : ''}`, title: missing ? 'outside the image (zero padding)' : formatSigned(v, 4) }, missing ? '·' : o.signed === false ? formatValue(v, o.digits ?? 2) : formatSigned(v, o.digits ?? 2));
    if (!missing) cell.style.background = `color-mix(in srgb, ${v < 0 ? o.colors.negative : o.colors.positive} ${Math.round(8 + 60 * t)}%, transparent)`;
    if (o.onCell && cell instanceof HTMLButtonElement) {
      cell.type = 'button';
      cell.addEventListener('click', () => o.onCell?.(k));
    }
    grid.append(cell);
  }
  return o.caption ? h('figure', { class: 'cnn-grid-figure' }, grid, h('figcaption', null, o.caption)) : grid;
}

export function stageName(kind: string, layer: number, pool: string): string {
  if (kind === 'input') return 'Input image';
  if (kind === 'conv') return `Conv ${layer + 1}`;
  if (kind === 'pool') return `${pool === 'max' ? 'Max' : 'Average'} pool ${layer + 1}`;
  return 'Output';
}
