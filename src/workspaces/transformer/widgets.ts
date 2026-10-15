/** Small displays used by the Transformer inspectors and panels. */
import { maxAbs } from '../../math/stats';
import { h, type Child } from '../../ui/dom';
import { formatPercent, formatProbability, formatSigned, formatValue } from '../../utils/format';

/**
 * A vector as a strip of diverging bars (the 2-D twin of a slab in the 3-D
 * view): bars grow up for positive values and down for negative ones.
 */
export function vectorStrip(
  values: ArrayLike<number>,
  o: { scale?: number; label?: Child; highlight?: number | null; onHover?(dim: number | null): void; compact?: boolean } = {},
): HTMLElement {
  const scale = o.scale ?? (maxAbs(values) || 1);
  const cells = Array.from({ length: values.length }, (_, i) => {
    const v = values[i];
    const t = Math.min(1, Math.abs(v) / scale);
    const bar = h('span', { class: `vstrip__bar ${v < 0 ? 'is-neg' : 'is-pos'}` });
    bar.style.height = `${Math.max(2, t * 50)}%`;
    const cell = h('span', { class: `vstrip__cell${o.highlight === i ? ' is-marked' : ''}`, title: `dimension ${i + 1}: ${formatSigned(v)}` }, bar);
    if (o.onHover) {
      cell.addEventListener('pointerenter', () => o.onHover?.(i));
      cell.addEventListener('pointerleave', () => o.onHover?.(null));
    }
    return cell;
  });
  const strip = h('div', { class: `vstrip${o.compact ? ' vstrip--compact' : ''}`, role: 'img', 'aria-label': `Vector of ${values.length} numbers` }, cells);
  strip.style.setProperty('--n', String(values.length));
  return o.label ? h('div', { class: 'vstrip-row' }, h('div', { class: 'vstrip-row__label' }, o.label), strip) : strip;
}

export interface BarRow {
  label: Child;
  value: number;
  /** Ghost bar behind (e.g. probability before filtering). */
  ghost?: number;
  text?: string;
  color?: string;
  highlight?: boolean;
  muted?: boolean;
  title?: string;
  onClick?(): void;
}

/** Horizontal bars, e.g. next-token probabilities or head contributions. */
export function barRows(rows: BarRow[], o: { max?: number; format?(v: number): string } = {}): HTMLElement {
  const max = o.max ?? Math.max(1e-12, ...rows.map((r) => Math.max(r.value, r.ghost ?? 0)));
  const format = o.format ?? formatProbability;
  return h(
    'div',
    { class: 'bars' },
    rows.map((r) => {
      const fill = h('span', { class: 'bars__fill' });
      fill.style.width = `${(Math.max(0, r.value) / max) * 100}%`;
      if (r.color) fill.style.background = r.color;
      const ghost = r.ghost !== undefined ? h('span', { class: 'bars__ghost' }) : null;
      if (ghost) ghost.style.width = `${(Math.max(0, r.ghost ?? 0) / max) * 100}%`;
      const content = [h('span', { class: 'bars__label' }, r.label), h('span', { class: 'bars__track' }, ghost, fill), h('span', { class: 'bars__value' }, r.text ?? format(r.value))];
      const cls = `bars__row${r.highlight ? ' is-highlight' : ''}${r.muted ? ' is-muted' : ''}`;
      return r.onClick ? h('button', { type: 'button', class: `${cls} bars__row--button`, title: r.title, onClick: r.onClick }, content) : h('div', { class: cls, title: r.title }, content);
    }),
  );
}

/** Attention matrix of one head with token labels (rows = query, columns = key). */
export function attentionGrid(o: {
  attention: ArrayLike<number>;
  labels: string[];
  color: string;
  query: number | null;
  key: number | null;
  onCell?(query: number, key: number): void;
}): HTMLElement {
  const T = o.labels.length;
  const readout = h('div', { class: 'attn-grid__readout' }, 'Rows: the token that reads (query) · columns: the tokens it reads from (keys). Hover a cell.');
  const grid = h('div', { class: 'attn-grid', role: 'grid', 'aria-label': 'Attention weights' });
  grid.style.setProperty('--n', String(T));
  grid.append(h('span', { class: 'attn-grid__corner' }));
  for (let j = 0; j < T; j++) grid.append(h('span', { class: `attn-grid__col${o.key === j ? ' is-focus' : ''}`, title: o.labels[j] }, o.labels[j]));
  for (let i = 0; i < T; i++) {
    grid.append(h('span', { class: `attn-grid__row${o.query === i ? ' is-focus' : ''}`, title: o.labels[i] }, o.labels[i]));
    for (let j = 0; j < T; j++) {
      const masked = j > i;
      const w = masked ? 0 : o.attention[i * T + j];
      const cell = h('button', {
        type: 'button',
        class: `attn-grid__cell${masked ? ' is-masked' : ''}${o.query === i ? ' is-row' : ''}${o.query === i && o.key === j ? ' is-selected' : ''}`,
        disabled: masked,
        'aria-label': masked ? `${o.labels[i]} cannot see ${o.labels[j]} (future token)` : `${o.labels[i]} → ${o.labels[j]}: ${formatPercent(w)}`,
        onClick: () => o.onCell?.(i, j),
      });
      if (!masked) cell.style.background = `color-mix(in srgb, ${o.color} ${Math.round(6 + 94 * Math.pow(w, 0.7))}%, transparent)`;
      cell.addEventListener('pointerenter', () => {
        readout.textContent = masked
          ? `“${o.labels[i]}” cannot see “${o.labels[j]}”: it comes later (causal mask).`
          : `“${o.labels[i]}” (position ${i}) reads “${o.labels[j]}” (position ${j}) with weight ${formatValue(w, 3)}`;
      });
      grid.append(cell);
    }
  }
  return h('div', { class: 'attn-grid-wrap' }, grid, readout);
}

/** Compact attention matrix (canvas), e.g. in the multi-head overview. */
export function attentionThumbnail(attention: ArrayLike<number>, T: number, color: string): HTMLCanvasElement {
  const canvas = h('canvas', { class: 'attn-thumb', width: T, height: T, 'aria-hidden': 'true' });
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const probe = document.createElement('span');
  probe.style.color = color;
  document.body.append(probe);
  const rgb = getComputedStyle(probe).color.match(/\d+(\.\d+)?/g)?.map(Number) ?? [200, 200, 200];
  probe.remove();
  const image = ctx.createImageData(T, T);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const k = 4 * (i * T + j);
      const masked = j > i;
      const w = masked ? 0 : attention[i * T + j];
      image.data[k] = rgb[0];
      image.data[k + 1] = rgb[1];
      image.data[k + 2] = rgb[2];
      image.data[k + 3] = masked ? 10 : Math.round(255 * (0.06 + 0.94 * Math.pow(w, 0.7)));
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

export function swatch(color: string): HTMLElement {
  const el = h('span', { class: 'swatch', 'aria-hidden': 'true' });
  el.style.background = color;
  return el;
}

/** A token as a small chip ("it" · 8). */
export function tokenChip(text: string, position?: number, extra = ''): HTMLElement {
  return h('span', { class: `tchip ${extra}`.trim() }, h('span', { class: 'tchip__text' }, text), position !== undefined ? h('span', { class: 'tchip__pos' }, String(position)) : null);
}

/** Attention matrix as a canvas with a hover readout: for long inputs, where a labelled grid would be unreadable. */
export function attentionCanvas(o: { attention: ArrayLike<number>; labels: string[]; color: string; query: number | null; onCell?(query: number, key: number): void }): HTMLElement {
  const T = o.labels.length;
  const canvas = attentionThumbnail(o.attention, T, o.color);
  canvas.classList.add('attn-canvas');
  const readout = h('div', { class: 'attn-grid__readout' }, `${T} × ${T} — rows: the token that reads, columns: the tokens it reads. Hover to read a cell.`);
  const marker = h('div', { class: 'attn-canvas__row', hidden: o.query === null });
  if (o.query !== null) {
    marker.style.top = `${(o.query / T) * 100}%`;
    marker.style.height = `${100 / T}%`;
  }
  const cellAt = (event: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    return {
      i: Math.min(T - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * T))),
      j: Math.min(T - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * T))),
    };
  };
  canvas.addEventListener('pointermove', (event) => {
    const { i, j } = cellAt(event);
    readout.textContent = j > i ? `“${o.labels[i]}” cannot see “${o.labels[j]}” (later token).` : `“${o.labels[i]}” (${i}) reads “${o.labels[j]}” (${j}): ${formatValue(o.attention[i * T + j], 3)}`;
  });
  if (o.onCell) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (event) => {
      const { i, j } = cellAt(event);
      o.onCell?.(i, j);
    });
  }
  return h('div', { class: 'attn-grid-wrap' }, h('div', { class: 'attn-canvas-wrap' }, canvas, marker), readout);
}
