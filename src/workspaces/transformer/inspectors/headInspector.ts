/**
 * One attention head: its attention matrix and a step-by-step walkthrough of
 * softmax(QKᵀ/√d_k)V for the focused token, with the real numbers.
 */
import { maxAbs } from '../../../math/stats';
import { headContributionNorms, headPatterns, norm } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, note, segmented } from '../../../ui/widgets';
import { formatPercent, formatSigned, formatValue, texNumber, texParen } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { attentionGrid, barRows, swatch, vectorStrip } from '../widgets';
import { quote } from './common';

const STEPS = ['Q · K · V', 'Scores', 'Softmax', 'Mix values', 'Output'];

export function headInspector(ws: TransformerWorkspace, layer: number, head: number, keyChoice: number | null): InspectorView {
  const model = ws.model!;
  const trace = ws.trace!;
  const block_ = trace.blocks[layer];
  const T = trace.tokens.length;
  const dk = model.config.dModel / model.config.nHeads;
  const i = Math.min(ws.focus, T - 1);
  const A = block_.attention[head];
  const S_ = block_.scores[head];
  let key = keyChoice ?? 0;
  if (keyChoice === null) for (let j = 1; j <= i; j++) if (A[i * T + j] > A[i * T + key]) key = j;
  key = Math.min(key, i);
  const tokens = trace.tokens.map((t) => t.text);
  const q = block_.q[head][i];
  const k = block_.k[head][key];
  const v = block_.v[head][key];
  const o = block_.headOut[head][i];
  const color = ws.headColor(head);
  const pattern = headPatterns(trace).find((p) => p.layer === layer && p.head === head);
  const step = Math.min(ws.qkvStep, STEPS.length - 1);
  const pairText = `${quote(tokens[i])} (position ${i}) reading ${quote(tokens[key])} (position ${key})`;

  const grid = attentionGrid({
    attention: A,
    labels: tokens,
    color,
    query: i,
    key,
    onCell: (qi, kj) => {
      ws.setFocus(qi);
      ws.select({ kind: 'head', layer, head, key: kj });
    },
  });

  const scale8 = Math.max(maxAbs(q), maxAbs(k), maxAbs(v), maxAbs(o)) || 1;
  const visibleKeys = Array.from({ length: i + 1 }, (_, j) => j);
  let stepContent: HTMLElement;
  switch (step) {
    case 0:
      stepContent = h(
        'div',
        null,
        paragraph(`Each token’s normalized vector LN₁(x) (${model.config.dModel} numbers) is multiplied by three matrices of this head (${dk}×${model.config.dModel} each). The query q says what the token is looking for, the key k what a token offers, the value v what it passes on if it is read.`),
        vectorStrip(block_.ln1[i], { label: `LN₁(x) of ${quote(tokens[i])}`, compact: true }),
        vectorStrip(q, { scale: scale8, label: `q of ${quote(tokens[i])}` }),
        vectorStrip(k, { scale: scale8, label: `k of ${quote(tokens[key])}` }),
        vectorStrip(v, { scale: scale8, label: `v of ${quote(tokens[key])}` }),
      );
      break;
    case 1:
      stepContent = h(
        'div',
        null,
        paragraph(`The query of ${quote(tokens[i])} is compared with the key of every token up to itself: dot product q·k, divided by √${dk} = ${formatValue(Math.sqrt(dk), 3)} to keep the scores in a reasonable range. Later tokens are hidden by the causal mask.`),
        h(
          'table',
          { class: 'data-table' },
          h('thead', null, h('tr', null, h('th', null, 'key'), h('th', null, 'q · k'), h('th', null, 'score'))),
          h(
            'tbody',
            null,
            Array.from({ length: T }, (_, j) =>
              h(
                'tr',
                { class: j === key ? 'is-highlight' : j > i ? 'is-muted' : '' },
                h('td', null, `${j} · ${tokens[j]}`),
                h('td', null, j > i ? 'masked' : formatSigned(S_[i * T + j] * Math.sqrt(dk), 3)),
                h('td', null, j > i ? '−∞' : formatSigned(S_[i * T + j], 3)),
              ),
            ),
          ),
        ),
      );
      break;
    case 2:
      stepContent = h(
        'div',
        null,
        paragraph('Softmax turns the scores into weights that are positive and add up to 1: the larger a score, the larger its share. These are the arcs in 3D and the row of the matrix above.'),
        barRows(
          visibleKeys.map((j) => ({ label: `${j} · ${tokens[j]}`, value: A[i * T + j], color, highlight: j === key, onClick: () => ws.select({ kind: 'head', layer, head, key: j }) })),
          { max: 1 },
        ),
      );
      break;
    case 3: {
      const contributions = visibleKeys
        .map((j) => ({ j, size: A[i * T + j] * norm(block_.v[head][j]) }))
        .sort((a, b) => b.size - a.size)
        .slice(0, 4);
      stepContent = h(
        'div',
        null,
        paragraph(`The head’s output for ${quote(tokens[i])} is the weighted average of the value vectors: o = Σ a_j v_j. Tokens with small weights contribute almost nothing.`),
        h('div', { class: 'field-label' }, 'Largest terms ‖a_j v_j‖'),
        barRows(contributions.map((c) => ({ label: `${c.j} · ${tokens[c.j]}`, value: c.size, color, highlight: c.j === key })), { format: (x) => formatValue(x, 3) }),
        vectorStrip(o, { scale: scale8, label: 'o = Σ a v' }),
      );
      break;
    }
    default: {
      const norms = headContributionNorms(trace, layer, i);
      stepContent = h(
        'div',
        null,
        paragraph(`The ${dk} numbers of o are mapped back to ${model.config.dModel} by this head’s slice of W_O and added to the residual stream of ${quote(tokens[i])}, together with the other heads.`),
        vectorStrip(block_.headContribution[head][i], { label: 'W_O · o', onHover: (d) => ws.markDimension(d) }),
        h('div', { class: 'field-label' }, `What each head of layer ${layer + 1} added to ${quote(tokens[i])} (length)`),
        barRows(
          norms.map((n, hd) => ({ label: `head ${hd + 1}`, value: n, color: ws.headColor(hd), highlight: hd === head, onClick: () => ws.select({ kind: 'head', layer, head: hd, key: null }) })),
          { format: (x) => formatValue(x, 2) },
        ),
      );
    }
  }

  const stepper = segmented<string>({
    options: STEPS.map((label, s) => ({ value: String(s), label: `${s + 1}. ${label}` })),
    value: String(step),
    ariaLabel: 'Walkthrough step',
    onChange: (v) => ws.setQkvStep(Number(v)),
  });

  const score = S_[i * T + key];
  const dot = score * Math.sqrt(dk);
  let denominator = 0;
  let max = -Infinity;
  for (let j = 0; j <= i; j++) max = Math.max(max, S_[i * T + j]);
  for (let j = 0; j <= i; j++) denominator += Math.exp(S_[i * T + j] - max);

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Attention head',
      title: `Layer ${layer + 1} · head ${head + 1}`,
      subtitle: h('span', null, swatch(color), ' ', pattern?.summary ?? ''),
    }),
    block(
      1,
      null,
      paragraph(`Rows: the token that reads (query). Columns: the tokens it can read (keys) — only itself and earlier tokens. Darker = more attention. Click a cell to follow that pair; the 3D arcs of this head are drawn in its colour.`),
      grid,
    ),
    block(
      2,
      `Step by step · ${pairText}`,
      h('div', { class: 'insp-picker' }, stepper.el),
      stepContent,
      h(
        'div',
        { class: 'button-row' },
        button({ icon: 'stepBack', label: 'Back', small: true, variant: 'ghost', disabled: step === 0, onClick: () => ws.setQkvStep(step - 1) }),
        button({ icon: 'stepForward', label: 'Next', small: true, variant: 'ghost', disabled: step === STEPS.length - 1, onClick: () => ws.setQkvStep(step + 1) }),
      ),
    ),
    block(
      3,
      null,
      tex(`\\operatorname{head}_{${head + 1}} = \\operatorname{softmax}\\!\\left(\\frac{Q K^{\\top}}{\\sqrt{d_k}} + M\\right) V, \\qquad d_k = ${dk}`, { display: true }),
      paragraph(`For the highlighted pair (${pairText}):`),
      tex(
        `\\begin{aligned} s &= \\frac{q \\cdot k}{\\sqrt{${dk}}} = \\frac{${texNumber(dot)}}{${texNumber(Math.sqrt(dk))}} = ${texNumber(score)} \\\\ a &= \\frac{e^{s}}{\\sum_{j \\le ${i}} e^{s_j}} = \\frac{e^{${texParen(score - max)}}}{${texNumber(denominator)}} = ${texNumber(A[i * T + key])} \\end{aligned}`,
        { display: true },
      ),
      paragraph('M is the causal mask: −∞ for later tokens, so their weight is exactly 0. (The numbers above subtract the largest score before exponentiating, which does not change the result.)'),
    ),
    block(
      null,
      'Keep in mind',
      note(`Attention weights show where a head reads from, not why the model predicts what it does: the values, W_O, the MLPs and the next layer matter too. A head with ${formatPercent(pattern?.bos ?? 0, 0)} of its attention on <s> is often “resting”: <s> works as a default place to look. The Experiments tab lets you switch a head off and measure its real effect.`),
    ),
  );
  return { el, update: () => {}, dispose: () => ws.markDimension(null) };
}
