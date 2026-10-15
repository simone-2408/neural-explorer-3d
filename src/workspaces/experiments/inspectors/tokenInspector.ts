/** One token of the input: remove it or replace its embedding, and see how its vector changed. */
import { gpt2Tokenize } from '../../../experiments/gpt2Engine';
import { isSpecial } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { toast } from '../../../ui/overlays';
import { button, note, slider } from '../../../ui/widgets';
import { formatPercent, formatProbability } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { ExperimentsWorkspace } from '../ExperimentsWorkspace';

export function stageLabel(stage: number): string {
  if (stage === 0) return 'embedding';
  const layer = Math.floor((stage - 1) / 2) + 1;
  return (stage - 1) % 2 === 0 ? `L${layer} attention` : `L${layer} MLP`;
}

export function tokenInspector(ws: ExperimentsWorkspace, position: number, stage: number | null): InspectorView {
  const r = ws.result!;
  const token = r.tokens[position];
  const { removed, embedding } = ws.tokenEdits(position);
  const editable = ws.editable;
  const toy = ws.modelKind === 'toy' ? ws.toy : null;
  const special = !!toy && isSpecial(toy, token.id);
  const j = r.alignment.indexOf(position);
  const format = (t: string) => ws.format(t);

  // replacement controls
  let replaceControl: HTMLElement;
  if (toy) {
    const select = h(
      'select',
      { class: 'select', 'aria-label': 'Replace with', disabled: !editable || !!removed || special },
      h('option', { value: '' }, 'keep its own embedding'),
      toy.config.vocab
        .map((w, id) => ({ w, id }))
        .filter(({ id }) => !isSpecial(toy, id) && id !== token.id)
        .sort((a, b) => a.w.localeCompare(b.w))
        .map(({ w, id }) => h('option', { value: String(id) }, w)),
    );
    select.value = embedding ? String(embedding.token.id) : '';
    select.addEventListener('change', () => ws.setReplacement(position, select.value ? { id: Number(select.value), text: toy.config.vocab[Number(select.value)] } : null, embedding?.mix ?? 1));
    replaceControl = h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Give it the embedding of another word'), select);
  } else {
    const input = h('input', { type: 'text', class: 'text-input', placeholder: 'e.g. “ Paris” (with the space)', maxlength: 40, disabled: !editable || !!removed, 'aria-label': 'Replacement token' });
    const apply = async () => {
      if (!input.value.trim()) return;
      try {
        const tokens = await gpt2Tokenize(input.value);
        if (tokens.length !== 1) {
          toast(`“${input.value}” is ${tokens.length} tokens: a replacement must be a single token. GPT-2 words usually start with a space.`, 'warning', 6000);
          return;
        }
        ws.setReplacement(position, { id: tokens[0].id, text: tokens[0].text }, embedding?.mix ?? 1);
      } catch (error) {
        toast(error instanceof Error ? error.message : String(error), 'error');
      }
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') void apply();
    });
    replaceControl = h('div', { class: 'stack' }, h('span', { class: 'field__label' }, 'Give it the embedding of another token'), h('div', { class: 'ex-typed' }, input, button({ label: 'Replace', small: true, disabled: !editable || !!removed, onClick: () => void apply() })));
  }
  const mixSlider = embedding
    ? slider({
        label: `Mix: 0 = “${format(token.text)}”, 1 = “${format(embedding.token.text)}”`,
        min: 0,
        max: 1,
        step: 0.01,
        value: embedding.mix,
        digits: 2,
        ariaLabel: 'Mix between the two embeddings',
        onInput: (v) => ws.setReplacement(position, embedding.token, v, `mix:${position}`),
      })
    : null;
  if (mixSlider && !editable) mixSlider.el.querySelectorAll('input').forEach((i) => (i.disabled = true));

  const changes = r.delta.map((row) => row[position]);
  const unchanged = !removed && changes.every((v) => v === 0);
  const before = r.before.predictions[position];
  const after = j >= 0 ? r.after.predictions[j] : null;

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: `Token · position ${position}`,
      title: `“${format(token.text)}”`,
      subtitle: `id ${token.id}${removed ? ' · removed' : embedding ? ` · embedding → “${format(embedding.token.text)}”` : ''}`,
      actions: [button({ label: removed ? 'Restore' : 'Remove', small: true, variant: removed ? 'default' : 'danger', disabled: !editable || special, title: 'X', onClick: () => ws.toggleRemoved(position) })],
    }),
    block(
      1,
      'Intervene',
      paragraph('Removing a token deletes it from the input: the tokens after it move one position to the left, so their position embeddings change too. Replacing its embedding keeps the position and the text, and changes only the vector the token starts as.'),
      special ? note('<s> starts every sentence the toy model was trained on: it stays.') : null,
      replaceControl,
      mixSlider?.el ?? null,
      embedding ? button({ icon: 'reset', label: 'Keep its own embedding', small: true, variant: 'ghost', disabled: !editable, onClick: () => ws.setReplacement(position, null) }) : null,
      editable ? null : note('Recorded preset: start the GPT-2 backend to change it.'),
    ),
    block(
      2,
      null,
      h('div', { class: 'insp-subheading' }, 'How much its vector changed, step by step'),
      removed
        ? paragraph('Removed: it is not computed any more.')
        : barRows(
            changes.map((v, s) => ({ label: stageLabel(s), value: Math.min(1, Math.sqrt(v ?? 0)), text: formatPercent(v ?? 0, (v ?? 0) < 0.01 ? 2 : 1), highlight: s === stage })),
            { max: 1 },
          ),
      h('div', { class: 'muted' }, 'Relative change ‖x_after − x_before‖ / ‖x_before‖ (bars: square root, as in the 3D view).'),
      h('div', { class: 'insp-subheading' }, 'The model’s guess for the next token here'),
      paragraph(after ? (after.id === before.id ? `“${format(before.text)}” (${formatProbability(before.p)} → ${formatProbability(after.p)})` : `“${format(before.text)}” ${formatProbability(before.p)} → “${format(after.text)}” ${formatProbability(after.p)}`) : 'Removed token: no guess.'),
      unchanged && r.effect.kl > 0 ? note('Identical at every step: no edit acts on this token or on the tokens before it, and attention only looks backwards (causal mask).') : null,
    ),
    block(3, null, tex('x_0 = (1 - m)\\,E[\\text{token}] + m\\,E[\\text{replacement}] + P[\\text{position}]', { display: true })),
  );
  return { el, update: () => {} };
}
