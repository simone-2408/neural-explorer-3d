/** One filter of a conv layer: its kernel (editable), how it compares with classic kernels, what excites it. */
import { kernelOf, kernelSimilarities, topActivations } from '../../../cnn/analysis';
import { h } from '../../../ui/dom';
import { live, type InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { button, kv, note, signedBar, slider } from '../../../ui/widgets';
import { formatSigned, formatValue } from '../../../utils/format';
import { block, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { CnnWorkspace } from '../CnnWorkspace';
import { imageCanvas } from '../panels';
import { numberGrid } from './common';

/** Weight being edited, per filter (kept while the inspector is rebuilt). */
const editing = new Map<string, { c: number; q: number }>();

export function filterInspector(ws: CnnWorkspace, stageIndex: number, layer: number, f: number): InspectorView {
  const model = ws.model;
  const s = model.shapes[layer];
  const k = s.kernel;
  const key = `${layer}:${f}`;
  const colors = { positive: ws.palette.positive, negative: ws.palette.negative };
  const relu = model.config.activation === 'relu';

  const kernels = live(() => {
    const all = Array.from({ length: s.inChannels }, (_, c) => kernelOf(model, layer, f, c));
    const scale = Math.max(1e-6, ...all.map((kc) => Math.max(...Array.from(kc, Math.abs))));
    const sel = editing.get(key);
    return h(
      'div',
      { class: 'cnn-kernels' },
      all.map((kc, c) =>
        numberGrid(kc, k, {
          colors,
          scale,
          caption: s.inChannels > 1 ? `from map ${c + 1}` : 'kernel',
          selected: sel?.c === c ? sel.q : null,
          onCell: (q) => {
            editing.set(key, { c, q });
            ws.showInspector(true);
          },
        }),
      ),
    );
  });

  const sel = editing.get(key);
  const editor = sel
    ? (() => {
        const u = Math.floor(sel.q / k);
        const v = sel.q % k;
        const index = model.weightIndex(layer, f, sel.c, u, v);
        const control = slider({
          label: `Weight W[${f + 1}, ${sel.c + 1}, ${u}, ${v}]`,
          min: -2,
          max: 2,
          step: 0.01,
          value: model.convWeight(layer)[index],
          digits: 3,
          diverging: true,
          ariaLabel: 'Kernel weight',
          onInput: (value) => ws.setKernelWeight(layer, f, sel.c, u, v, value),
        });
        return h('div', { class: 'stack' }, control.el, button({ icon: 'close', label: 'Done', small: true, variant: 'ghost', onClick: () => (editing.delete(key), ws.showInspector(true)) }));
      })()
    : h('div', { class: 'muted' }, 'Click a weight to change it and watch the whole map (and the prediction) react.');

  const stats = live(() => {
    const S = s.size;
    const act = ws.trace.layers[layer].act.subarray(f * S * S, (f + 1) * S * S);
    const max = Math.max(...act);
    const active = act.filter((v) => v > 0).length / act.length;
    const sims = k === 3 && s.inChannels === 1 ? kernelSimilarities(kernelOf(model, layer, f, 0)) : [];
    return h(
      'div',
      { class: 'stack' },
      kv('Bias b', formatSigned(model.convBias(layer)[f], 4)),
      kv('Strongest response on this image', formatValue(max, 3)),
      relu ? kv('Cells active (z > 0) on this image', `${formatValue(active * 100, 0)}%`) : null,
      sims.length
        ? h(
            'div',
            null,
            h('div', { class: 'insp-subheading' }, 'Cosine similarity with classic kernels'),
            h(
              'div',
              { class: 'cnn-sims' },
              sims.slice(0, 4).map((r) => h('div', { class: 'cnn-sims__row' }, h('span', null, r.name), signedBar(r.similarity, 1), h('span', { class: 'cnn-sims__value' }, formatSigned(r.similarity, 2)))),
            ),
            h('div', { class: 'muted' }, '±1 = the same pattern (− = with the sign flipped), 0 = unrelated.'),
          )
        : null,
    );
  });

  // the test images that excite this filter most (a pass over the test set: skipped while training)
  const top = ws.running ? null : topActivations(model, ws.dataset.test, layer, f, 6);
  const dark = ws.palette.theme === 'dark';
  const topBlock = top
    ? h(
        'div',
        { class: 'cnn-top' },
        top.map((t) => {
          const canvas = imageCanvas(ws.dataset.test[t.sample].pixels, dark, 'cnn-top__image');
          const frame = h('span', { class: 'cnn-top__frame' });
          frame.style.left = `${(t.region.left / 16) * 100}%`;
          frame.style.top = `${(t.region.top / 16) * 100}%`;
          frame.style.width = `${((t.region.right - t.region.left + 1) / 16) * 100}%`;
          frame.style.height = `${((t.region.bottom - t.region.top + 1) / 16) * 100}%`;
          return h(
            'button',
            { type: 'button', class: 'cnn-top__item', title: `Test image ${t.sample} (${ws.classNames[ws.dataset.test[t.sample].label]}): response ${formatValue(t.value, 3)} — click to show it`, onClick: () => ws.setInputSample(t.sample) },
            h('span', { class: 'cnn-top__box' }, canvas, frame),
            h('span', { class: 'cnn-top__value' }, formatValue(t.value, 2)),
          );
        }),
      )
    : h('div', { class: 'muted' }, 'Shown when training pauses.');

  const el = h(
    'div',
    { class: 'inspector-view' },
    inspectorHeader({
      kind: `Filter · Conv ${layer + 1}`,
      title: `Filter ${f + 1}`,
      subtitle: `${s.inChannels} × ${k}×${k} weights + 1 bias = ${s.inChannels * k * k + 1} parameters, shared by all ${s.size * s.size} positions`,
      actions: [button({ icon: 'play', label: 'Slide', small: true, title: 'Animate this filter moving over its input', onClick: () => ws.startSlide(stageIndex, f) })],
    }),
    block(
      1,
      null,
      paragraph('A filter is a small grid of weights. Placed on a window of its input, it gives a large positive response where the window looks like the pattern of positive weights, and a negative one where it looks like the opposite.'),
      paragraph('The map of this filter (framed in the scene) is its response at every position. Training changes the weights; press Space and watch them.'),
    ),
    block(2, null, kernels.el, editor, stats.el, h('div', { class: 'insp-subheading' }, 'Test images where it responds most (frame: the region its strongest cell sees)'), topBlock),
    block(3, null, tex(`\\text{map}_{${f + 1}}[i, j] = ${relu ? '\\mathrm{ReLU}' : '\\tanh'}\\Big(b_{${f + 1}} + \\sum_{c=1}^{${s.inChannels}} \\sum_{u,v=0}^{${k - 1}} W_{${f + 1},c,u,v}\\; x_{c,\\,i+u-${(k - 1) / 2},\\,j+v-${(k - 1) / 2}}\\Big)`, { display: true })),
    block(null, 'Keep in mind', note('Resemblance to an edge or line detector is measured, not assumed; and what a filter contributes also depends on the layers after it. Filters of the second layer combine the maps below and are much harder to read.')),
  );
  return {
    el,
    update: () => {
      kernels.update();
      stats.update();
    },
  };
}
