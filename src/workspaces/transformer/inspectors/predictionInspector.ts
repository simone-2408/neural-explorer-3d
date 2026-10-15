/**
 * Next-token prediction at one position: logits → softmax with temperature →
 * top-k / top-p filtering → sampling. The sliders update the bars live.
 */
import { distribution, entropyBits } from '../../../transformer/sampling';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, note, slider } from '../../../ui/widgets';
import { formatPercent, formatProbability, formatValue, texNumber } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { barRows } from '../widgets';
import { quote } from './common';

const SHOWN = 12;

export function predictionInspector(ws: TransformerWorkspace, position: number): InspectorView {
  const model = ws.model!;
  const trace = ws.trace!;
  const T = trace.tokens.length;
  const p = Math.min(position, T - 1);
  const logits = trace.logits[p];
  const token = trace.tokens[p];
  const isLast = p === T - 1;
  const next = trace.tokens[p + 1];
  const vocab = model.config.vocab;

  const bars = live(() => {
    const options = ws.settings.sampling;
    const dist = distribution(logits, options);
    const rows = dist.order.slice(0, SHOWN).map((id) => ({
      label: vocab[id],
      value: dist.probs[id],
      ghost: dist.raw[id],
      muted: dist.probs[id] === 0,
      highlight: next?.id === id,
      title: `logit z = ${formatValue(logits[id], 3)} · z/T = ${formatValue(logits[id] / Math.max(options.temperature, 1e-3), 3)} · p = ${formatPercent(dist.probs[id], 2)}`,
      text: dist.probs[id] === 0 ? `cut (${formatProbability(dist.raw[id])})` : formatProbability(dist.probs[id]),
    }));
    const top = dist.order[0];
    return [
      barRows(rows, { max: 1 }),
      h('div', { class: 'muted' }, `Solid bar: probability used for sampling. Faint bar: before top-k / top-p. Showing ${SHOWN} of ${vocab.length} words.`),
      kv('Words that can be drawn', `${dist.kept} of ${vocab.length}`),
      kv('Uncertainty (entropy)', `${formatValue(entropyBits(dist.probs), 2)} bits`),
      tex(
        `p_{\\text{top}} = \\frac{e^{z/T}}{\\sum_j e^{z_j/T}} = \\frac{e^{${texNumber(logits[top], 2)}/${texNumber(options.temperature, 2)}}}{\\sum_j e^{z_j/${texNumber(options.temperature, 2)}}} = ${texNumber(dist.raw[top])}`,
        { display: true },
      ),
    ];
  });

  const set = (patch: Partial<typeof ws.settings.sampling>) => {
    ws.setSampling(patch);
    bars.update();
  };
  const sampling = ws.settings.sampling;
  const temperature = slider({ label: 'Temperature T', min: 0.05, max: 3, step: 0.05, value: sampling.temperature, digits: 2, ariaLabel: 'Temperature', onInput: (v) => set({ temperature: Math.max(0.05, v) }) });
  const topK = slider({ label: 'Top-k (0 = off)', min: 0, max: 20, step: 1, value: sampling.topK, digits: 0, ariaLabel: 'Top-k', onInput: (v) => set({ topK: Math.max(0, Math.round(v)) }) });
  const topP = slider({ label: 'Top-p (1 = off)', min: 0.05, max: 1, step: 0.01, value: sampling.topP, digits: 2, ariaLabel: 'Top-p', onInput: (v) => set({ topP: Math.min(1, Math.max(0.01, v)) }) });

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: 'Next-token prediction',
      title: `After ${quote(token.text)}`,
      subtitle: isLast ? `the next word of your text · position ${p + 1}` : `prediction at position ${p} · the actual next token is ${quote(next.text)}`,
    }),
    block(
      1,
      null,
      paragraph(`The model gives a score (logit) to each of its ${vocab.length} words; softmax turns the scores into probabilities. To write text, a word is picked, appended, and the model runs again.`),
      isLast ? null : paragraph('A Transformer predicts at every position at once: during training each position learned to predict the word that follows it. The highlighted bar is the word that actually comes next in your text.'),
    ),
    block(2, 'Probabilities', bars.el),
    block(
      null,
      'Sampling',
      temperature.el,
      topK.el,
      topP.el,
      h(
        'div',
        { class: 'button-row' },
        button({ label: 'Reset', small: true, variant: 'ghost', onClick: () => {
          set({ temperature: 1, topK: 0, topP: 1 });
          temperature.setValue(1);
          topK.setValue(0);
          topP.setValue(1);
        } }),
      ),
      isLast
        ? h(
            'div',
            { class: 'button-row' },
            button({ icon: 'stepForward', label: 'Most likely', small: true, onClick: () => ws.generate('greedy') }),
            button({ icon: 'dice', label: 'Sample', small: true, variant: 'primary', onClick: () => ws.generate('sample') }),
            button({ label: 'Sample ×5', small: true, onClick: () => ws.generate('sample', 5) }),
            button({ icon: 'reset', label: 'Undo', small: true, variant: 'ghost', disabled: !ws.canUndo, onClick: () => ws.undo() }),
          )
        : null,
      isLast ? h('div', { class: 'muted' }, `Random seed ${ws.seed} · `, h('button', { type: 'button', class: 'entity-link', onClick: () => ws.reseed() }, 'new seed')) : null,
    ),
    block(
      3,
      null,
      tex('p_i = \\frac{e^{z_i / T}}{\\sum_{j} e^{z_j / T}}', { display: true }),
      paragraph('T < 1 sharpens the distribution (T → 0 always picks the top word), T > 1 flattens it. Top-k keeps the k most likely words; top-p keeps the smallest set whose probabilities add up to at least p. The kept probabilities are then rescaled to sum to 1.'),
      tex('z = \\operatorname{LN}_f(x)\\, E^{\\top}', { display: true }),
    ),
    block(null, 'Keep in mind', note('Special tokens (<pad>, <s>, <unk>) are never sampled. After “.” the predictions mean nothing: the model was trained on single sentences.')),
  );
  return { el, update: () => bars.update() };
}
