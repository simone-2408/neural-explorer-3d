/**
 * Weight-matrix view: shows that the lines in the 3D scene ARE the entries of
 * a matrix. Row j = target neuron, column i = source neuron, so entry (j, i)
 * is the weight of connection i → j.
 *
 * Small matrices render as a numeric table, large ones as a heatmap canvas.
 * Clicking an entry selects the corresponding connection in the scene.
 */
import { maxAbs } from '../../math/stats';
import { formatSigned } from '../../utils/format';
import { h } from '../dom';

export interface Cell {
  row: number;
  col: number;
}

export interface MatrixViewOptions {
  rows: number;
  cols: number;
  /** Row-major values. */
  data: ArrayLike<number>;
  rowLabels: string[];
  colLabels: string[];
  bias?: ArrayLike<number>;
  selected?: Cell | null;
  /** Row / column to outline (e.g. the selected neuron's incoming weights). */
  focusRow?: number | null;
  focusCol?: number | null;
  colors: { positive: string; negative: string };
  onCellClick?(cell: Cell): void;
}

const TABLE_MAX_COLS = 8;
const TABLE_MAX_CELLS = 72;

function cellBackground(value: number, scale: number, colors: MatrixViewOptions['colors']): string {
  const t = scale > 0 ? Math.min(1, Math.abs(value) / scale) : 0;
  return `color-mix(in srgb, ${value < 0 ? colors.negative : colors.positive} ${Math.round(8 + 55 * t)}%, transparent)`;
}

export function matrixView(o: MatrixViewOptions): HTMLElement {
  return o.cols <= TABLE_MAX_COLS && o.rows * o.cols <= TABLE_MAX_CELLS ? matrixTable(o) : matrixHeatmap(o);
}

function matrixTable(o: MatrixViewOptions): HTMLElement {
  const scale = maxAbs(o.data);
  const head = h(
    'tr',
    null,
    h('th', { class: 'matrix__corner', scope: 'col' }, 'to \\ from'),
    o.colLabels.map((label, i) => h('th', { scope: 'col', class: o.focusCol === i ? 'is-focus' : '' }, label)),
    o.bias ? h('th', { scope: 'col', class: 'matrix__bias' }, 'b') : null,
  );
  const body = [];
  for (let j = 0; j < o.rows; j++) {
    const cells = [];
    for (let i = 0; i < o.cols; i++) {
      const value = o.data[j * o.cols + i];
      const isSelected = o.selected?.row === j && o.selected.col === i;
      const td = h(
        'td',
        {
          class: `matrix__cell${isSelected ? ' is-selected' : ''}${o.focusRow === j || o.focusCol === i ? ' is-focus' : ''}`,
          title: `W[${j + 1},${i + 1}]  ${o.colLabels[i]} → ${o.rowLabels[j]}: ${formatSigned(value)}`,
        },
        o.onCellClick
          ? h('button', { type: 'button', class: 'matrix__btn', onClick: () => o.onCellClick?.({ row: j, col: i }) }, formatSigned(value, 2))
          : formatSigned(value, 2),
      );
      td.style.background = cellBackground(value, scale, o.colors);
      cells.push(td);
    }
    body.push(
      h(
        'tr',
        null,
        h('th', { scope: 'row', class: o.focusRow === j ? 'is-focus' : '' }, o.rowLabels[j]),
        cells,
        o.bias ? h('td', { class: 'matrix__bias' }, formatSigned(o.bias[j], 2)) : null,
      ),
    );
  }
  return h('div', { class: 'matrix' }, h('table', { class: 'matrix__table' }, h('thead', null, head), h('tbody', null, body)));
}

function matrixHeatmap(o: MatrixViewOptions): HTMLElement {
  const scale = maxAbs(o.data);
  const canvas = h('canvas', { class: 'heatmap__canvas', width: o.cols, height: o.rows, 'aria-label': `Weight matrix heatmap, ${o.rows} × ${o.cols}` });
  paintMatrix(canvas, o.data, o.rows, o.cols, scale, o.colors);
  const marker = h('div', { class: 'heatmap__marker', hidden: true });
  const readout = h('div', { class: 'heatmap__readout' }, `${o.rows} × ${o.cols} — hover to read a weight${o.onCellClick ? ', click to select it' : ''}`);
  const wrap = h('div', { class: 'heatmap__wrap' }, canvas, marker);
  wrap.style.height = `${Math.min(260, Math.max(48, o.rows * 6))}px`;

  const placeMarker = (cell: Cell | null | undefined) => {
    if (!cell) {
      marker.hidden = true;
      return;
    }
    marker.hidden = false;
    marker.style.left = `${(cell.col / o.cols) * 100}%`;
    marker.style.top = `${(cell.row / o.rows) * 100}%`;
    marker.style.width = `max(3px, ${100 / o.cols}%)`;
    marker.style.height = `max(3px, ${100 / o.rows}%)`;
  };
  placeMarker(o.selected);

  const cellAt = (event: PointerEvent | MouseEvent): Cell => {
    const rect = canvas.getBoundingClientRect();
    return {
      col: Math.min(o.cols - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * o.cols))),
      row: Math.min(o.rows - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * o.rows))),
    };
  };
  canvas.addEventListener('pointermove', (event) => {
    const { row, col } = cellAt(event);
    readout.textContent = `W[${row + 1},${col + 1}]  ${o.colLabels[col]} → ${o.rowLabels[row]}: ${formatSigned(o.data[row * o.cols + col])}`;
  });
  if (o.onCellClick) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (event) => o.onCellClick?.(cellAt(event)));
  }
  return h('div', { class: 'heatmap' }, wrap, readout);
}

/** Paints values into a canvas at one pixel per entry (scaled up with CSS, pixelated). */
export function paintMatrix(
  canvas: HTMLCanvasElement,
  data: ArrayLike<number>,
  rows: number,
  cols: number,
  scale: number,
  colors: { positive: string; negative: string },
): void {
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const pos = parseHex(colors.positive);
  const neg = parseHex(colors.negative);
  // Colour = sign, opacity = |value| / scale (transparent cells show the panel background).
  const image = ctx.createImageData(cols, rows);
  for (let k = 0; k < rows * cols; k++) {
    const v = data[k];
    const t = scale > 0 ? Math.min(1, Math.abs(v) / scale) : 0;
    const c = v < 0 ? neg : pos;
    image.data[4 * k] = c[0];
    image.data[4 * k + 1] = c[1];
    image.data[4 * k + 2] = c[2];
    image.data[4 * k + 3] = Math.round(255 * (0.06 + 0.94 * Math.pow(t, 0.8)));
  }
  ctx.putImageData(image, 0, 0);
}

function parseHex(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
}
