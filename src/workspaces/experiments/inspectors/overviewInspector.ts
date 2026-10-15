/** Experiments inspector with nothing selected: the experiment, its measured effect and how to read it. */
import { METRICS } from '../../../experiments/metrics';
import { BACKEND_HINT } from '../../../llm/client';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { note } from '../../../ui/widgets';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { ExperimentsWorkspace } from '../ExperimentsWorkspace';
import { compareBars, effectRows } from '../widgets';

export function overviewInspector(ws: ExperimentsWorkspace): InspectorView {
  const r = ws.result;
  const e = ws.experiment;
  const preset = ws.preset[ws.modelKind];
  const gpt2 = ws.modelKind === 'gpt2';
  const active = e.interventions.filter((i) => i.enabled);
  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: `Experiment · ${gpt2 ? 'GPT-2 small' : 'toy Transformer'}`,
      title: preset?.label ?? 'Your experiment',
      subtitle: r ? ws.summary() : ws.busy ? 'Running…' : undefined,
    }),
    preset ? block(null, null, paragraph(preset.description), preset.tryNext ? note(h('span', null, h('strong', null, 'Try next: '), preset.tryNext)) : null) : null,
    block(
      1,
      'What you see',
      paragraph(
        'Left wall: every attention head (columns h1, h2…) and the MLP of every layer (last column, half a row higher because it runs after the attention). Each bar shows what switching that component off ',
        h('em', null, 'alone'),
        ' does to the measured token: toward you when the target relies on it, behind the wall when the component works against it. Switched-off components are red, scaled ones amber.',
      ),
      paragraph('Right wall: the residual stream of every token (columns) at every step (rows, aligned with the components that write there). Bar length shows how much each vector changed because of your interventions: the change starts where you intervene and can only spread upwards and to the right — a token never influences the tokens before it.'),
      paragraph('Click a tile or a token to inspect it and change it. Shift-click a tile to switch it off or on.'),
    ),
    r
      ? block(
          2,
          null,
          h('div', { class: 'insp-subheading' }, `Next token after “${ws.format(r.tokens[r.tokens.length - 1].text)}”`),
          compareBars(r.before.top, r.after.top, (t) => ws.format(t), { target: r.target.id }),
          h('div', { class: 'insp-subheading' }, `Measured on “${ws.format(r.target.text)}”${r.contrast ? ` against “${ws.format(r.contrast.text)}”` : ''}`),
          effectRows(r.effect, ws.metric, { contrast: !!r.contrast }),
          h('div', { class: 'insp-subheading' }, `Interventions (${active.length})`),
          active.length
            ? h(
                'ul',
                { class: 'ex-insp-list' },
                active.map((i) =>
                  h(
                    'li',
                    null,
                    entityLink(ws.describeIntervention(i), () =>
                      ws.select(
                        i.kind === 'head'
                          ? { kind: 'head', layer: i.layer, head: i.head }
                          : i.kind === 'mlp'
                            ? { kind: 'mlp', layer: i.layer }
                            : i.kind === 'neuron'
                              ? { kind: 'neuron', layer: i.layer, unit: i.unit }
                              : i.kind === 'weight'
                                ? { kind: 'weight', id: i.id }
                                : { kind: 'token', position: i.position, stage: null },
                      ),
                    ),
                  ),
                ),
              )
            : paragraph('None yet: the two runs are identical. Pick a tile with a long bar and switch it off.'),
        )
      : block(2, null, paragraph(ws.busy ? 'Running the experiment…' : gpt2 && !ws.live ? `This GPT-2 experiment has no recording. Pick a ready-made experiment on the left, or ${BACKEND_HINT}.` : 'No result yet.')),
    block(
      3,
      null,
      tex('\\Delta \\log p = \\log p_{\\text{after}}(t) - \\log p_{\\text{before}}(t)', { display: true }),
      r?.contrast ? tex('\\Delta\\,\\text{logit diff} = \\Delta\\big(\\ell_t - \\ell_c\\big)', { display: true }) : null,
      tex('\\mathrm{KL}(p_{\\text{before}}\\,\\|\\,p_{\\text{after}}) = \\sum_{v} p_{\\text{before}}(v)\\,\\log_2 \\frac{p_{\\text{before}}(v)}{p_{\\text{after}}(v)}', { display: true }),
      paragraph(`Bars: ${METRICS[ws.metric].description}`),
      tex('\\text{head } h \\text{ scaled by } s:\\quad x \\leftarrow x + \\sum_{h\'} W_O^{(h\')}\\,(s_{h\'}\\,\\text{head}_{h\'}) + b_O', { display: true }),
    ),
    block(
      null,
      'Keep in mind',
      note('Switching a component off (multiplying it by 0) is a strong, unnatural intervention: the model never saw that component silent during training. Other methods replace it with its average value, or with its value on a different input; they can give different answers.'),
      note('One input is one example. A component that matters here may not matter elsewhere — try other texts before drawing conclusions.'),
      note('Effects do not add up: components interact, and others can take over a job when one is removed. “Bars measured on: With interventions” shows what still matters once your changes are in place.'),
      note('A large effect shows that the computation passes through a component on this input. It does not tell what the component “means”.'),
      gpt2 ? note('Layers and heads count from 1 here. Papers on GPT-2 often count from 0: their head “9.6” is layer 10 · head 7.') : null,
    ),
  );
  return { el, update: () => {} };
}
