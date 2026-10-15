/** A weight-matrix edit of the toy model: scale the whole matrix, or change one entry. */
import { tensorInfo } from '../../../experiments/toyEngine';
import type { Intervention } from '../../../experiments/types';
import { maxAbs } from '../../../math/stats';
import { matrixView } from '../../../ui/charts/matrixView';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, note, slider } from '../../../ui/widgets';
import { formatCount, formatSigned, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { tensorName } from '../describe';
import type { ExperimentsWorkspace } from '../ExperimentsWorkspace';

type WeightEdit = Extract<Intervention, { kind: 'weight' }>;

/** Entry picked in the heatmap, per edit (kept while the inspector is rebuilt). */
const picked = new Map<string, { row: number; col: number }>();

const FORMULAS: Record<WeightEdit['tensor'], string> = {
  embed: 'x_0 = E[\\text{token}] + P[\\text{position}], \\qquad \\text{logits} = \\mathrm{LN}_f(x_L)\\, E^{\\top}',
  pos: 'x_0 = E[\\text{token}] + P[\\text{position}]',
  q: 'q_i = W_Q\\, \\mathrm{LN}(x_i) + b_Q, \\qquad A_{ij} = \\mathrm{softmax}_j\\big(q_i \\cdot k_j / \\sqrt{d_k}\\big)',
  k: 'k_j = W_K\\, \\mathrm{LN}(x_j) + b_K, \\qquad A_{ij} = \\mathrm{softmax}_j\\big(q_i \\cdot k_j / \\sqrt{d_k}\\big)',
  v: 'v_j = W_V\\, \\mathrm{LN}(x_j) + b_V, \\qquad \\text{head}_i = \\sum_j A_{ij}\\, v_j',
  o: 'x_i \\leftarrow x_i + \\sum_h W_O^{(h)}\\, \\text{head}^{(h)}_i + b_O',
  mlpIn: 'a = \\mathrm{GELU}\\big(W_{in}\\, \\mathrm{LN}(x) + b_{in}\\big)',
  mlpOut: 'x \\leftarrow x + W_{out}\\, a + b_{out}',
};

export function weightInspector(ws: ExperimentsWorkspace, item: WeightEdit): InspectorView {
  const model = ws.toy!;
  const info = tensorInfo(model, item.tensor, item.layer, item.head);
  const source = info.array(model);
  const original = new Float64Array(info.rows * info.cols);
  for (let r = 0; r < info.rows; r++) for (let c = 0; c < info.cols; c++) original[r * info.cols + c] = source[info.index(r, c)];
  const edited = Float64Array.from(original, (v) => v * item.scale);
  if (item.cell) edited[item.cell.row * info.cols + item.cell.col] = item.cell.value;
  const range = maxAbs(original) || 1;
  const cell = item.cell ? { row: item.cell.row, col: item.cell.col } : (picked.get(item.id) ?? null);
  const frob = (a: Float64Array) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

  const scaleSlider = slider({
    label: 'Scale the whole matrix (1 = original, 0 = all zeros)',
    min: -2,
    max: 3,
    step: 0.05,
    value: item.scale,
    digits: 2,
    diverging: true,
    ariaLabel: 'Scale of the matrix',
    onInput: (v) => ws.updateIntervention(item.id, { scale: v }),
  });
  const cellBlock = cell
    ? (() => {
        const k = cell.row * info.cols + cell.col;
        const s = slider({
          label: `Entry [${info.rowLabel(cell.row)}, ${info.colLabel(cell.col)}] — original ${formatSigned(original[k], 4)}`,
          min: -3 * range,
          max: 3 * range,
          step: range / 100,
          value: edited[k],
          digits: 4,
          diverging: true,
          ariaLabel: 'Value of the selected weight',
          onInput: (v) => ws.updateIntervention(item.id, { cell: { row: cell.row, col: cell.col, value: v } }),
        });
        return h(
          'div',
          { class: 'stack' },
          s.el,
          h(
            'div',
            { class: 'button-row' },
            button({ label: 'Zero', small: true, onClick: () => ws.updateIntervention(item.id, { cell: { row: cell.row, col: cell.col, value: 0 } }) }),
            button({ label: 'Original', small: true, variant: 'ghost', disabled: !item.cell, onClick: () => ws.updateIntervention(item.id, { cell: null }) }),
          ),
        );
      })()
    : paragraph('Click an entry of the matrix to change that single weight.');

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Weight edit · toy model',
      title: tensorName(item.tensor, item.layer, item.head),
      subtitle: `${info.rows} × ${info.cols} = ${formatCount(info.rows * info.cols)} weights (${info.rowName} × ${info.colName})`,
      actions: [button({ icon: 'close', label: 'Remove', small: true, variant: 'ghost', onClick: () => ws.removeIntervention(item.id) })],
    }),
    block(1, 'What this matrix does', paragraph(info.role), item.tensor === 'embed' ? note('The token embeddings are tied: the same matrix turns the last vector into word scores. Editing a row changes both how the word enters the model and how strongly the model can predict it.', 'warning') : null),
    block(
      2,
      null,
      scaleSlider.el,
      h('div', { class: 'button-row' }, [0, 0.5, 2].map((v) => button({ label: v === 0 ? 'All zeros' : `×${v}`, small: true, onClick: () => ws.updateIntervention(item.id, { scale: v }) })), button({ icon: 'reset', label: 'Original', small: true, variant: 'ghost', disabled: item.scale === 1, onClick: () => ws.updateIntervention(item.id, { scale: 1 }) })),
      matrixView({
        rows: info.rows,
        cols: info.cols,
        data: edited,
        rowLabels: Array.from({ length: info.rows }, (_, r) => info.rowLabel(r)),
        colLabels: Array.from({ length: info.cols }, (_, c) => info.colLabel(c)),
        selected: cell,
        colors: { positive: ws.palette.positive, negative: ws.palette.negative },
        onCellClick: (picked_) => {
          picked.set(item.id, picked_);
          if (item.cell) ws.updateIntervention(item.id, { cell: { ...picked_, value: edited[picked_.row * info.cols + picked_.col] } });
          else ws.showInspector(true);
        },
      }),
      cellBlock,
      kv('‖W‖ original → edited', `${formatValue(frob(original), 3)} → ${formatValue(frob(edited), 3)}`),
    ),
    block(3, null, tex(FORMULAS[item.tensor], { display: true }), paragraph('Biases are not edited.')),
    block(null, 'Keep in mind', note('One weight among ' + formatCount(info.rows * info.cols) + ' rarely matters on its own: the computation is spread over many of them. Whole matrices tell more.')),
  );
  return { el, update: () => {} };
}
