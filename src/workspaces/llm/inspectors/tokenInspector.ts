/**
 * One token of the prompt: how BPE produced it, what the model predicts after
 * it, and two heuristic answers to "where does its information come from?"
 * (attention rollout, gradient × input) — presented as estimates.
 */
import { showToken } from '../../../llm/client';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatPercent, formatProbability, formatSigned, formatValue } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { LlmWorkspace } from '../LlmWorkspace';
import { tokenName, word } from './common';

export function tokenInspector(ws: LlmWorkspace, position: number): InspectorView {
  const a = ws.analysis!;
  const token = a.tokens[position];
  const [predicted, predictedP] = a.predictions[position];
  const next = a.tokens[position + 1];
  const isLast = position === a.T - 1;
  const label = (p: number) => `${p} · ${showToken(a.tokens[p].text)}`;

  // Attention rollout row of this position; the first token is set aside (attention sink).
  const row = a.rollout[position];
  const sink = row[0];
  const rest = row.slice(1, position + 1).reduce((s, v) => s + v, 0) || 1;
  const rollout = row.slice(1, position + 1).map((v, k) => ({ label: label(k + 1), value: v / rest }));

  let saliency: HTMLElement | null = null;
  if (isLast) {
    const values = a.saliency.values;
    const total = values.reduce((s, v) => s + Math.abs(v), 0) || 1;
    saliency = h(
      'div',
      null,
      h('div', { class: 'field-label' }, `Gradient × input for ${word(a, a.saliency.target)} (share of |total|, with sign)`),
      barRows(
        values.map((v, p) => ({ label: label(p), value: Math.abs(v) / total, text: formatSigned(v / total, 2), color: v >= 0 ? 'var(--pos)' : 'var(--neg)' })),
        { max: Math.max(...values.map((v) => Math.abs(v) / total)) },
      ),
    );
  }

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Token', title: `${tokenName(a, position)} · position ${position}`, subtitle: `token id ${token.id} of ${a.logits.length.toLocaleString('en-US')}` }),
    block(
      1,
      'How the text became this token',
      paragraph('GPT-2 splits text with byte-pair encoding (BPE): frequent words are one token, rarer words are cut into pieces, and any character can fall back to its raw bytes. A space is part of the token that follows it (shown as “·”, written “Ġ” inside BPE).'),
      kv('BPE piece', h('code', null, token.piece)),
      kv('UTF-8 bytes', h('code', null, token.bytes)),
      token.text.includes('�') ? note('This token is only part of a character: a multi-byte UTF-8 character (e.g. “è” = c3 a8) was split into separate byte tokens, which cannot be displayed alone.', 'warning') : null,
    ),
    block(
      2,
      'What comes next',
      kv('Model’s top prediction after it', `${word(a, predicted)} (${formatProbability(predictedP)})`),
      next ? kv('Actual next token', `${showToken(next.text)}${next.id === predicted ? ' ✓ predicted' : ''}`) : kv('Next token', h('span', null, entityLink('see the prediction →', () => ws.select({ kind: 'prediction' })))),
      h('div', { class: 'field-label' }, 'Length of its vector, layer by layer (E = embeddings)'),
      barRows(
        a.norms.map((n, l) => ({ label: l === 0 ? 'E' : `layer ${l}`, value: n[position], text: formatValue(n[position], 0), onClick: () => ws.select({ kind: 'cell', layer: l, position }) })),
        { format: (v) => formatValue(v, 0) },
      ),
    ),
    block(
      null,
      'Where does its information come from? (estimates)',
      position === 0
        ? paragraph('The first token can only see itself.')
        : [
            h('div', { class: 'field-label' }, `Attention rollout (first token set aside: it holds ${formatPercent(sink, 1)})`),
            barRows(rollout),
            saliency,
            note(
              `Two heuristics. Rollout multiplies the attention of all layers (averaged over heads, plus the residual connection): it ignores what the values carry and what the MLPs do. Gradient × input measures how much a tiny change of each input embedding would change the prediction: a local, linear estimate. When they disagree, neither is “the” explanation — the Experiments tab tests effects by intervening.`,
            ),
          ],
    ),
    block(
      3,
      null,
      tex('R = \\prod_{\\ell=L}^{1} \\tfrac{1}{2}\\big(\\bar{A}_\\ell + I\\big), \\qquad \\text{saliency}_i = \\nabla_{e_i} \\log p(y) \\cdot e_i', { display: true }),
      paragraph('Ā_ℓ is the attention of layer ℓ averaged over its heads (rows renormalized); e_i is the embedding of input token i.'),
    ),
  );
  return { el, update: () => {} };
}
