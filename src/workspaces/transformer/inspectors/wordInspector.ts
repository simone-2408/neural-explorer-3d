/** A vocabulary word in the embedding space: its vector and its true nearest neighbours. */
import { cosine, nearestWords, vocabularyProjection, wordIds } from '../../../transformer/analysis';
import { h } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatPercent, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import { GROUP_LABELS, groupColor } from '../colors';
import type { TransformerWorkspace } from '../TransformerWorkspace';
import { barRows, vectorStrip } from '../widgets';

let projectionCache: { model: unknown; explained: number } | null = null;

export function wordInspector(ws: TransformerWorkspace, id: number): InspectorView {
  const model = ws.model!;
  const d = model.config.dModel;
  const word = model.config.vocab[id];
  const group = model.groups[word];
  const vector = model.tokenEmbedding.subarray(id * d, (id + 1) * d);
  const neighbours = nearestWords(model, vector, 8, [id]);
  const theme = ws.palette.theme;

  // Mean similarity to the words of the same group vs the other grouped words.
  let same = 0;
  let sameCount = 0;
  let other = 0;
  let otherCount = 0;
  for (const j of wordIds(model)) {
    if (j === id) continue;
    const g = model.groups[model.config.vocab[j]];
    if (!g || !group) continue;
    const c = cosine(vector, model.tokenEmbedding.subarray(j * d, (j + 1) * d));
    if (g === group) {
      same += c;
      sameCount++;
    } else {
      other += c;
      otherCount++;
    }
  }
  if (projectionCache?.model !== model) {
    const p = vocabularyProjection(model);
    projectionCache = { model, explained: p.explained.reduce((s, v) => s + v, 0) };
  }

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({ kind: 'Word embedding', title: word, subtitle: `${group ? GROUP_LABELS[group] : 'Other word'} · token id ${id}` }),
    block(
      1,
      null,
      paragraph('Each point is a word’s vector (its row of the embedding matrix E), projected to 3-D. Training moved words that are used in similar ways towards similar directions — nobody told the model which words are countries or animals.'),
    ),
    block(
      2,
      null,
      vectorStrip(vector, { label: `E[${word}]` }),
      h('div', { class: 'field-label' }, 'Nearest words in the full 32-D space (cosine similarity)'),
      barRows(
        neighbours.map((n) => ({
          label: n.text,
          value: Math.max(0, n.value),
          text: formatValue(n.value, 2),
          color: groupColor(model.groups[n.text], theme),
          onClick: () => ws.select({ kind: 'word', id: n.id }),
          title: `${GROUP_LABELS[model.groups[n.text] ?? 'other']} — click to select`,
        })),
        { max: 1, format: (v) => formatValue(v, 2) },
      ),
      group && sameCount && otherCount
        ? kv(`Mean similarity to ${GROUP_LABELS[group].toLowerCase()} · to other groups`, `${formatValue(same / sameCount, 2)} · ${formatValue(other / otherCount, 2)}`)
        : null,
    ),
    block(
      3,
      null,
      tex('\\cos(a, b) = \\frac{a \\cdot b}{\\lVert a \\rVert\\, \\lVert b \\rVert} \\in [-1, 1]', { display: true }),
      paragraph('1 = same direction, 0 = unrelated (perpendicular), −1 = opposite. The length of an embedding is ignored.'),
    ),
    block(
      null,
      'Keep in mind',
      note(`The 3-D view keeps only ${formatPercent(projectionCache.explained, 0)} of the variance of the 32-D embeddings: neighbours on screen can be misleading, the list above is exact. The same matrix E also scores the words at the output (tied embeddings), so a word’s vector serves both to read it and to predict it.`),
    ),
  );
  return { el, update: () => {} };
}
