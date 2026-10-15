/**
 * GPT-2's next-token distribution: the 50,257 logits with temperature, top-k
 * and top-p applied in the browser, and generation (live) or the recorded
 * continuations (demo).
 */
import { showToken } from '../../../llm/client';
import type { GenerateResult } from '../../../llm/types';
import { distribution, entropyBits } from '../../../transformer/sampling';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, note, slider } from '../../../ui/widgets';
import { formatCount, formatProbability, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { barRows } from '../../transformer/widgets';
import type { LlmWorkspace } from '../LlmWorkspace';
import { tokenName, word } from './common';

const SHOWN = 12;

function continuation(title: string, result: GenerateResult): HTMLElement {
  return h(
    'div',
    { class: 'continuation' },
    h('div', { class: 'field-label' }, title),
    h(
      'div',
      { class: 'continuation__tokens' },
      result.steps.map((s) =>
        h(
          'span',
          { class: 'continuation__token', title: `p = ${formatProbability(s.p)} · alternatives: ${s.alternatives.map(([, t, p]) => `${showToken(t)} ${formatProbability(p)}`).join(', ')}` },
          showToken(s.text),
          h('span', { class: 'continuation__p' }, formatProbability(s.p)),
        ),
      ),
    ),
  );
}

export function predictionInspector(ws: LlmWorkspace): InspectorView {
  const a = ws.analysis!;
  const logits = a.logits;

  const bars = live(() => {
    const options = ws.settings.sampling;
    const dist = distribution(logits, options);
    const rows = dist.order.slice(0, SHOWN).map((id) => ({
      label: word(a, id),
      value: dist.probs[id],
      ghost: dist.raw[id],
      muted: dist.probs[id] === 0,
      title: `token id ${id} · logit ${formatValue(logits[id], 2)}`,
      text: dist.probs[id] === 0 ? `cut (${formatProbability(dist.raw[id])})` : formatProbability(dist.probs[id]),
    }));
    const top = dist.order[0];
    return [
      barRows(rows, { max: 1 }),
      kv('Tokens that can be drawn', `${formatCount(dist.kept)} of ${formatCount(logits.length)}`),
      kv('Uncertainty (entropy)', `${formatValue(entropyBits(dist.probs), 2)} bits`),
      kv('Top token', `${word(a, top)} · logit ${formatValue(logits[top], 2)}`),
    ];
  });
  const set = (patch: Partial<typeof ws.settings.sampling>) => {
    ws.setSampling(patch);
    bars.update();
  };
  const s = ws.settings.sampling;
  const temperature = slider({ label: 'Temperature T', min: 0.05, max: 3, step: 0.05, value: s.temperature, digits: 2, ariaLabel: 'Temperature', onInput: (v) => set({ temperature: Math.max(0.05, v) }) });
  const topK = slider({ label: 'Top-k (0 = off)', min: 0, max: 100, step: 1, value: s.topK, digits: 0, ariaLabel: 'Top-k', onInput: (v) => set({ topK: Math.max(0, Math.round(v)) }) });
  const topP = slider({ label: 'Top-p (1 = off)', min: 0.05, max: 1, step: 0.01, value: s.topP, digits: 2, ariaLabel: 'Top-p', onInput: (v) => set({ topP: Math.min(1, Math.max(0.01, v)) }) });

  const generated: HTMLElement[] = [];
  if (a.source === 'live') {
    generated.push(
      h(
        'div',
        { class: 'button-row' },
        button({ icon: 'stepForward', label: 'Most likely', small: true, disabled: ws.busy, onClick: () => void ws.generate('greedy') }),
        button({ icon: 'dice', label: 'Sample', small: true, variant: 'primary', disabled: ws.busy, onClick: () => void ws.generate('sample') }),
        button({ icon: 'reset', label: 'Undo', small: true, variant: 'ghost', disabled: ws.busy || !ws.canUndo, onClick: () => void ws.undo() }),
      ),
      h('div', { class: 'muted' }, `Each click appends ${ws.settings.maxNewTokens} tokens chosen one at a time with the settings above (the sampling runs in Python with the same rule).`),
    );
    if (ws.generation) generated.push(continuation('Last continuation (hover a token for its alternatives)', ws.generation));
  } else if (a.demo) {
    generated.push(continuation('Recorded: always the most likely token', a.demo.greedy), continuation('Recorded: sampled at T = 1', a.demo.sampled), h('div', { class: 'muted' }, 'Start the backend to generate with your own settings.'));
  }

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Next-token prediction', title: `After ${tokenName(a, a.T - 1)}`, subtitle: `a probability for each of the ${formatCount(logits.length)} tokens` }),
    block(1, null, paragraph('GPT-2 scores every token of its vocabulary; softmax turns the scores (logits) into probabilities. The text is then continued one token at a time: pick a token, append it, run the model again.')),
    block(2, 'Probabilities', bars.el),
    block(null, 'Sampling', temperature.el, topK.el, topP.el, generated),
    block(
      3,
      null,
      tex('p_i = \\frac{e^{z_i/T}}{\\sum_{j=1}^{50257} e^{z_j/T}}', { display: true }),
      paragraph('Low temperature: almost always the top token (repetitive text). High temperature: rare tokens become likely (creative, then incoherent). Top-k and top-p cut the long tail of unlikely tokens before sampling.'),
    ),
    block(null, 'Keep in mind', note('A probability is not a fact check: GPT-2 small often prefers generic continuations (“ the”, “ a”) over the correct specific word.')),
  );
  return { el, update: () => bars.update() };
}
