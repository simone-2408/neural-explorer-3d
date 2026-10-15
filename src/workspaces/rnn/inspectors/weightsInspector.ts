/** A weight matrix of the recurrent network, shared by every step; for W_hh, why memory fades. */
import type { RnnWeights } from '../../../three/RnnView';
import { matrixView } from '../../../ui/charts/matrixView';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { RnnWorkspace } from '../RnnWorkspace';

/** Largest singular value of the rows [r0, r0 + rows) of a row-major matrix (power iteration on AᵀA). */
export function spectralNorm(W: Float64Array, cols: number, r0: number, rows: number): number {
  let v = new Float64Array(cols).fill(1 / Math.sqrt(cols));
  let sigma = 0;
  for (let it = 0; it < 60; it++) {
    const Av = new Float64Array(rows);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) Av[r] += W[(r0 + r) * cols + c] * v[c];
    const AtAv = new Float64Array(cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) AtAv[c] += W[(r0 + r) * cols + c] * Av[r];
    const n = Math.hypot(...AtAv);
    if (n === 0) return 0;
    sigma = Math.sqrt(n);
    v = AtAv.map((x) => x / n);
  }
  return sigma;
}

export function weightsInspector(ws: RnnWorkspace, which: RnnWeights): InspectorView {
  const m = ws.model;
  const { inputSize: I, hiddenSize: H, outputSize: O } = m.config;
  const gru = m.config.cell === 'gru';
  const G = m.gates * H;
  const sym = (s: string) => (s === ' ' ? '␣' : s);
  const gateOf = (r: number) => (gru ? `${'rzn'[Math.floor(r / H)]}${(r % H) + 1}` : `h${r + 1}`);
  const colors = { positive: ws.palette.positive, negative: ws.palette.negative };
  let title: string;
  let rows: number;
  let cols: number;
  let data: Float64Array;
  let rowLabels: string[];
  let colLabels: string[];
  if (which === 'ih') {
    title = 'W_ih — input → state';
    [rows, cols, data] = [G, I, m.Wih];
    rowLabels = Array.from({ length: G }, (_, r) => gateOf(r));
    colLabels = ws.task.inputSymbols.map(sym);
  } else if (which === 'hh') {
    title = 'W_hh — state → next state';
    [rows, cols, data] = [G, H, m.Whh];
    rowLabels = Array.from({ length: G }, (_, r) => gateOf(r));
    colLabels = Array.from({ length: H }, (_, c) => `h${c + 1}`);
  } else {
    title = 'W_out — state → prediction';
    [rows, cols, data] = [O, H, m.Wout];
    rowLabels = ws.task.outputSymbols.map(sym);
    colLabels = Array.from({ length: H }, (_, c) => `h${c + 1}`);
  }
  const norms = which === 'hh' ? (gru ? [0, 1, 2].map((g) => spectralNorm(m.Whh, H, g * H, H)) : [spectralNorm(m.Whh, H, 0, H)]) : [];
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Shared weights', title, subtitle: `${rows} × ${cols} = ${formatCount(rows * cols)} weights, used at every step` }),
    block(
      1,
      null,
      which === 'ih' ? paragraph('One column per input symbol: what that symbol adds to the state.') : null,
      which === 'hh' ? paragraph('How each number of the previous state feeds each number of the new one. This matrix is applied again at every step, so its effect compounds over time.') : null,
      which === 'out' ? paragraph('One row per output symbol: how strongly each number of the state votes for it.') : null,
      gru ? paragraph('GRU rows come in three blocks: reset gate (r), update gate (z) and candidate (n) — PyTorch’s order.') : null,
    ),
    block(
      2,
      null,
      matrixView({ rows, cols, data, rowLabels, colLabels, colors }),
      norms.length
        ? h(
            'div',
            { class: 'stack' },
            norms.map((s, g) => kv(gru ? `Largest singular value, ${['reset', 'update', 'candidate'][g]} block` : 'Largest singular value σ_max(W_hh)', formatValue(s, 3))),
            gru
              ? paragraph('In a GRU the state also travels directly: h_t contains z ⊙ h_{t−1}. When z is close to 1 that path multiplies the gradient by almost exactly 1 per step, whatever W_hh does.')
              : paragraph(norms[0] < 1 ? `σ_max < 1: every step multiplies the gradient flowing back by a matrix that shrinks vectors by at least ${formatValue(norms[0], 2)} — and tanh’s slope (≤ 1) shrinks it further. Over 30 steps that is a factor ${formatValue(Math.pow(norms[0], 30), 2)} or smaller.` : `σ_max ≥ 1: some directions can grow from step to step; tanh’s slope (≤ 1) usually keeps the gradient bounded, but large values can explode — gradient clipping guards against that.`),
          )
        : null,
    ),
    which === 'hh'
      ? block(3, null, tex('\\frac{\\partial h_{t+1}}{\\partial h_t} = \\mathrm{diag}\\big(1 - h_{t+1}^2\\big)\\, W_{hh}, \\qquad \\Big\\lVert \\frac{\\partial h_{t+k}}{\\partial h_t} \\Big\\rVert \\le \\sigma_{\\max}(W_{hh})^k', { display: true }), paragraph('Vanilla RNN; the bound uses |tanh′| ≤ 1.'))
      : null,
    which === 'hh' && !gru ? block(null, 'Keep in mind', note('The bound is an upper limit: the actual shrinking depends on the states visited. The gradient bars under the steps show the measured values for the current input.')) : null,
  );
  return { el, update: () => {} };
}
