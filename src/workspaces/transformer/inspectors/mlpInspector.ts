/** The MLP of one layer at one position: its 128 hidden units and what it writes back. */
import { norm, topUnits, unitVocabularyEffect } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatCount, formatSigned, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { vectorStrip } from '../widgets';
import { positionPicker, quote } from './common';

export function mlpInspector(ws: TransformerWorkspace, layer: number, position: number): InspectorView {
  const model = ws.model!;
  const trace = ws.trace!;
  const { dModel, dMlp } = model.config;
  const b = trace.blocks[layer];
  const token = trace.tokens[position];
  const act = b.mlpAct[position];
  const out = b.mlpOut[position];
  const active = Array.from(act).filter((v) => v > 0.05).length;
  const negative = Array.from(act).filter((v) => v < -0.05).length;
  const units = topUnits(trace, layer, position, 8);
  const params = dModel * dMlp + dMlp + dMlp * dModel + dModel;

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'MLP (feed-forward)', title: `MLP ${layer + 1} · ${quote(token.text)}`, subtitle: `${dModel} → ${dMlp} → ${dModel} · GELU · position ${position}` }),
    positionPicker(ws, position, (p) => {
      ws.setFocus(p);
      ws.select({ kind: 'mlp', layer, position: p });
    }),
    block(
      1,
      null,
      paragraph(`The MLP reads the normalized vector of ${quote(token.text)}, computes ${dMlp} hidden units, and writes ${dModel} numbers back into the stream. It sees this position only: the same weights are applied to every token separately, so information never moves between tokens here (that is attention’s job).`),
    ),
    block(
      2,
      null,
      vectorStrip(act, { label: `GELU(z) of the ${dMlp} units`, compact: true }),
      kv('Clearly positive · negative', `${active} · ${negative} of ${dMlp} units`),
      h('div', { class: 'field-label' }, 'Strongest units and the words their output pushes towards'),
      h(
        'table',
        { class: 'data-table' },
        h('thead', null, h('tr', null, h('th', null, 'unit'), h('th', null, 'z'), h('th', null, 'GELU(z)'), h('th', null, 'pushes towards'))),
        h(
          'tbody',
          null,
          units.map((u) => {
            const effect = unitVocabularyEffect(model, layer, u.unit, 3);
            const words = (u.act >= 0 ? effect.promoted : effect.suppressed).map((t) => t.text).join(', ');
            return h('tr', null, h('td', null, `#${u.unit + 1}`), h('td', null, formatSigned(u.pre, 2)), h('td', null, formatSigned(u.act, 2)), h('td', { class: 'muted' }, words));
          }),
        ),
      ),
      vectorStrip(out, { label: 'output (added to x)', onHover: (d) => ws.markDimension(d) }),
      kv('‖output‖ vs ‖x‖ before', `${formatValue(norm(out), 2)} vs ${formatValue(norm(b.mid[position]), 2)}`),
    ),
    block(
      3,
      null,
      tex('\\operatorname{MLP}(x) = W_2\\, \\operatorname{GELU}\\big(W_1 \\operatorname{LN}_2(x) + b_1\\big) + b_2, \\qquad \\operatorname{GELU}(z) = z\\,\\Phi(z)', { display: true }),
      paragraph(`W₁ is ${dMlp}×${dModel} and W₂ is ${dModel}×${dMlp}: ${formatCount(params)} parameters, the largest part of each layer. Φ is the standard normal CDF: GELU lets positive z through, nearly blocks negative z, and is smooth around 0.`),
    ),
    block(
      null,
      'Keep in mind',
      note('“Pushes towards” projects a unit’s output weights onto the vocabulary (through the final LayerNorm’s centring and gain, ignoring its normalization and the later layers). For negative activations the effect is reversed, so the table shows the words pushed when the unit is negative. A rough hint, not what the unit “means”.'),
    ),
  );
  return { el, update: () => {}, dispose: () => ws.markDimension(null) };
}
