/**
 * A token's vector at one stage of the residual stream: its numbers, how it
 * was built (embedding + position, or previous vector + sub-layer output) and
 * what the logit lens reads from it.
 */
import { maxAbs } from '../../../math/stats';
import { headContributionNorms, logitLens, norm } from '../../../transformer/analysis';
import { residualAt } from '../../../transformer/forward';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatPercent, formatValue } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { barRows, vectorStrip } from '../widgets';
import { quote, stageKind, stageName, stagePicker, stageShort } from './common';

export function tokenInspector(ws: TransformerWorkspace, position: number, stage: number): InspectorView {
  const model = ws.model!;
  const trace = ws.trace!;
  const token = trace.tokens[position];
  const S = ws.stages;
  const vector = residualAt(trace, stage)[position];
  const { kind, layer } = stageKind(stage);
  const mark = (dim: number | null) => ws.markDimension(dim);

  // How this vector was built.
  let parts: { label: string; values: Float64Array }[];
  let explanation: string;
  let formula: string;
  if (kind === 'embed') {
    parts = [
      { label: `E[${quote(token.text)}]`, values: trace.tokenEmbeddings[position] },
      { label: `+ P[${position}]`, values: trace.positionEmbeddings[position] },
      { label: '= x₀', values: vector },
    ];
    explanation = `This is where ${quote(token.text)} enters the network: the row of the embedding matrix E for this word (token id ${token.id}) plus the vector P of position ${position}. Without P the model could not tell “dog chased cat” from “cat chased dog”.`;
    formula = `x^{(0)}_{${position}} = E_{${token.id}} + P_{${position}}`;
  } else if (kind === 'attention') {
    const block = trace.blocks[layer];
    parts = [
      { label: `x (${stageShort(stage - 1)})`, values: block.input[position] },
      { label: '+ attention', values: block.attnOut[position] },
      { label: '= x', values: vector },
    ];
    explanation = `Attention ${layer + 1} added to this vector what its ${model.config.nHeads} heads gathered from ${position === 0 ? 'this token itself (the first token can only see itself)' : 'this token and the tokens before it'}.`;
    formula = `x' = x + \\sum_{h=1}^{${model.config.nHeads}} W_O^{(h)}\\, \\operatorname{head}_h\\big(\\operatorname{LN}_1(x)\\big) + b_O`;
  } else {
    const block = trace.blocks[layer];
    parts = [
      { label: `x (${stageShort(stage - 1)})`, values: block.mid[position] },
      { label: '+ MLP', values: block.mlpOut[position] },
      { label: '= x', values: vector },
    ];
    explanation = `The MLP of layer ${layer + 1} added its output, computed from this vector alone: no other token is involved.`;
    formula = `x'' = x' + W_2\\, \\operatorname{GELU}\\big(W_1 \\operatorname{LN}_2(x') + b_1\\big) + b_2`;
  }
  const scale = Math.max(...parts.map((p) => maxAbs(p.values))) || 1;

  // Head contributions (attention stages).
  let heads: HTMLElement | null = null;
  if (kind === 'attention') {
    const norms = headContributionNorms(trace, layer, position);
    const T = trace.tokens.length;
    heads = h(
      'div',
      null,
      h('div', { class: 'field-label' }, 'Size of what each head added ‖W_O·head‖, and what it read most'),
      barRows(
        norms.map((n, hd) => {
          const a = trace.blocks[layer].attention[hd];
          let best = 0;
          for (let j = 1; j <= position; j++) if (a[position * T + j] > a[position * T + best]) best = j;
          return {
            label: `head ${hd + 1}`,
            value: n,
            color: ws.headColor(hd),
            text: `${formatValue(n, 2)} · ${quote(trace.tokens[best].text)} ${formatPercent(a[position * T + best], 0)}`,
            title: `Open head ${hd + 1} of layer ${layer + 1}`,
            onClick: () => ws.select({ kind: 'head', layer, head: hd, key: best }),
          };
        }),
        { format: (v) => formatValue(v, 2) },
      ),
    );
  }

  // Logit lens at every stage (layer-by-layer prediction).
  const lensRows = Array.from({ length: S }, (_, s) => {
    const top = logitLens(model, residualAt(trace, s)[position], 1)[0];
    return { label: stageShort(s), value: top.value, text: `${top.text} ${formatPercent(top.value, 0)}`, highlight: s === stage, onClick: () => ws.select({ kind: 'token', position, stage: s }) };
  });
  const lens = logitLens(model, vector, 5);
  const isFinal = stage === S - 1;
  const next = trace.tokens[position + 1];

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Residual stream',
      title: `${quote(token.text)} · position ${position}`,
      subtitle: `${stageName(stage)} · token id ${token.id}${token.unknown ? ' · unknown word' : ''}`,
    }),
    stagePicker(ws, position, stage),
    block(
      1,
      null,
      paragraph(explanation),
      paragraph('In 3D it is the outlined slab: one bar per number, along the depth. Hover the strips below to light up a dimension in every vector of the scene.'),
    ),
    block(
      2,
      null,
      parts.map((p) => vectorStrip(p.values, { scale, label: p.label, onHover: mark })),
      kv('Length ‖x‖', formatValue(norm(vector), 3)),
      kind !== 'embed' ? kv('Length of the added vector', formatValue(norm(parts[1].values), 3)) : null,
      heads,
      kind === 'mlp' ? h('div', null, entityLink(`Open the 128 units of MLP ${layer + 1} →`, () => ws.select({ kind: 'mlp', layer, position }))) : null,
    ),
    block(
      null,
      isFinal ? 'The prediction from this vector' : 'If the model had to answer now (logit lens)',
      barRows(lens.map((t) => ({ label: t.text, value: t.value }))),
      isFinal
        ? paragraph(next ? `This is the model’s actual prediction after ${quote(token.text)}. The next token in your text is ${quote(next.text)}.` : 'This is the model’s actual prediction for the next word.')
        : note('Logit lens: the final LayerNorm and unembedding applied to an intermediate vector. Earlier layers were not trained to be read this way: treat it as a hint of what the vector already points to, not as the model’s answer.'),
      h('div', { class: 'field-label' }, 'Top word at each stage'),
      barRows(lensRows),
    ),
    block(3, null, tex(formula, { display: true }), kind === 'embed' ? null : paragraph('The vector is never replaced, only added to: every layer reads the stream and writes its contribution back (residual connection).')),
  );
  return { el, update: () => {}, dispose: () => ws.markDimension(null) };
}
