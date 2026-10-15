/** One cell of a map: the input pixel, a convolution output (window × kernel) or a pooling output (2×2 window). */
import { receptiveField } from '../../../cnn/analysis';
import { h, type Child } from '../../../ui/dom';
import type { InspectorView } from '../../../ui/InspectorPanel';
import { tex } from '../../../ui/tex';
import { kv, note } from '../../../ui/widgets';
import { formatSigned, formatValue, texNumber, texParen } from '../../../utils/format';
import { block, entityLink, inspectorHeader, paragraph } from '../../network/inspectors/common';
import type { CnnSelection, CnnWorkspace } from '../CnnWorkspace';
import { numberGrid, stageName } from './common';

type CellSelection = Extract<CnnSelection, { kind: 'cell' }>;

export function cellInspector(ws: CnnWorkspace, sel: CellSelection): InspectorView {
  const stage = ws.stages[sel.stage];
  if (stage.kind === 'conv') return convCell(ws, sel);
  if (stage.kind === 'pool') return poolCell(ws, sel);
  return inputCell(ws, sel);
}

function header(ws: CnnWorkspace, sel: CellSelection, kind: string): HTMLElement {
  const stage = ws.stages[sel.stage];
  const pool = stage.layer >= 0 ? ws.model.shapes[stage.layer].pool : 'none';
  return inspectorHeader({
    kind,
    title: `${stageName(stage.kind, stage.layer, pool)}${stage.maps > 1 ? ` · map ${sel.map + 1}` : ''} · (${sel.i}, ${sel.j})`,
    subtitle: `row ${sel.i}, column ${sel.j} of a ${stage.size}×${stage.size} map`,
  });
}

function inputCell(ws: CnnWorkspace, sel: CellSelection): InspectorView {
  const v = ws.trace.input[sel.i * 16 + sel.j];
  const k = ws.model.shapes[0].kernel;
  const el = h(
    'div',
    { class: 'inspector-view' },
    header(ws, sel, 'Input pixel'),
    block(1, null, paragraph(`One pixel of the image: 0 is background, 1 is full ink. Every filter of Conv 1 reads it, from the ${k * k} positions whose ${k}×${k} window covers it.`)),
    block(2, null, kv('Value', formatValue(v, 3))),
  );
  return { el, update: () => {} };
}

function convCell(ws: CnnWorkspace, sel: CellSelection): InspectorView {
  const model = ws.model;
  const stage = ws.stages[sel.stage];
  const l = stage.layer;
  const s = model.shapes[l];
  const k = s.kernel;
  const p = (k - 1) / 2;
  const S = s.size;
  const t = ws.trace.layers[l];
  const colors = { positive: ws.palette.positive, negative: ws.palette.negative };
  const z = t.pre[(sel.map * S + sel.i) * S + sel.j];
  const a = t.act[(sel.map * S + sel.i) * S + sel.j];
  const bias = model.convBias(l)[sel.map];
  // per input channel: window, kernel, products and their sum
  const channels = Array.from({ length: s.inChannels }, (_, c) => {
    const window = new Float64Array(k * k);
    const kernel = new Float64Array(k * k);
    const products = new Float64Array(k * k);
    const missing: boolean[] = [];
    let sum = 0;
    for (let u = 0; u < k; u++) {
      for (let v = 0; v < k; v++) {
        const y = sel.i + u - p;
        const x = sel.j + v - p;
        const q = u * k + v;
        kernel[q] = model.convWeight(l)[model.weightIndex(l, sel.map, c, u, v)];
        const outside = y < 0 || x < 0 || y >= S || x >= S;
        missing.push(outside);
        window[q] = outside ? 0 : t.input[(c * S + y) * S + x];
        products[q] = window[q] * kernel[q];
        sum += products[q];
      }
    }
    return { window, kernel, products, missing, sum };
  });
  const relu = model.config.activation === 'relu';
  const region = receptiveField(model, l, sel.i, sel.j);
  const below = ws.stages[sel.stage - 1];
  const belowName = below.kind === 'input' ? 'the image' : below.kind === 'pool' ? `the pooled maps of layer ${below.layer + 1}` : `the maps of Conv ${below.layer + 1}`;
  const channelBlock = (c: number): Child => {
    const ch = channels[c];
    return h(
      'div',
      { class: 'cnn-conv-row' },
      s.inChannels > 1 ? h('div', { class: 'insp-subheading' }, `Input map ${c + 1}: Σ = ${formatSigned(ch.sum, 3)}`) : null,
      h(
        'div',
        { class: 'cnn-conv' },
        numberGrid(ch.window, k, { colors, caption: 'window x', missing: ch.missing }),
        h('span', { class: 'cnn-conv__op' }, '×'),
        numberGrid(ch.kernel, k, { colors, caption: 'kernel W' }),
        h('span', { class: 'cnn-conv__op' }, '='),
        numberGrid(ch.products, k, { colors, caption: 'products', missing: ch.missing }),
      ),
    );
  };
  const sumTex = channels.length === 1 ? texNumber(channels[0].sum) : channels.map((c) => texParen(c.sum)).join(' + ');
  const el = h(
    'div',
    { class: 'inspector-view' },
    header(ws, sel, 'Convolution output'),
    block(
      1,
      'What it computes',
      paragraph(`This cell looks at a ${k}×${k} window of ${belowName}${s.inChannels > 1 ? ` — the same position in all ${s.inChannels} maps` : ''} (framed in the scene). It multiplies each value by the matching weight of filter ${sel.map + 1}, adds everything up with the filter’s bias, and applies ${relu ? 'ReLU' : 'tanh'}.`),
      paragraph(`Every cell of this map uses the same ${s.inChannels * k * k} weights: only the window moves. That is why a convolution can find a pattern anywhere in the image.`),
      below.kind !== 'input' ? paragraph(`Through the layers below, this cell can be influenced by the image region rows ${region.top}–${region.bottom}, columns ${region.left}–${region.right} (also framed): deeper cells see larger regions.`) : null,
    ),
    block(
      2,
      null,
      s.inChannels <= 4 ? channels.map((_, c) => channelBlock(c)) : [channelBlock(0), paragraph(`… and ${s.inChannels - 1} more input maps.`)],
      kv('Sum of the products', formatSigned(channels.reduce((acc, c) => acc + c.sum, 0), 4)),
      kv('Bias b', formatSigned(bias, 4)),
      kv('z = sum + b', formatSigned(z, 4)),
      kv(`a = ${relu ? 'ReLU' : 'tanh'}(z)`, formatSigned(a, 4), 'kv--strong'),
      relu && z <= 0 ? note('z ≤ 0: ReLU outputs 0. This filter’s pattern is absent (or opposite) at this position.') : null,
      h('div', null, entityLink(`Filter ${sel.map + 1}: its kernel and what it responds to →`, () => ws.select({ kind: 'map', stage: sel.stage, map: sel.map }))),
    ),
    block(
      3,
      null,
      tex(`z_{${sel.map + 1},${sel.i},${sel.j}} = \\sum_{c}\\sum_{u,v} W_{${sel.map + 1},c,u,v}\\, x_{c,\\,${sel.i}+u-${p},\\,${sel.j}+v-${p}} + b_{${sel.map + 1}}`, { display: true }),
      tex(`= ${sumTex} + ${texParen(bias)} = ${texNumber(z)}`, { display: true }),
      tex(`a = ${relu ? '\\max(0,\\, z)' : '\\tanh(z)'} = ${texNumber(a)}`, { display: true }),
      paragraph('Outside the image the window reads zeros (“same” padding), so the map keeps the size of its input.'),
    ),
  );
  return { el, update: () => {} };
}

function poolCell(ws: CnnWorkspace, sel: CellSelection): InspectorView {
  const stage = ws.stages[sel.stage];
  const l = stage.layer;
  const s = ws.model.shapes[l];
  const t = ws.trace.layers[l];
  const S = s.size;
  const colors = { positive: ws.palette.positive, negative: ws.palette.negative };
  const window = new Float64Array(4);
  let winner = -1;
  const argmax = t.argmax[(sel.map * stage.size + sel.i) * stage.size + sel.j];
  for (let di = 0; di < 2; di++) {
    for (let dj = 0; dj < 2; dj++) {
      const idx = (sel.map * S + 2 * sel.i + di) * S + 2 * sel.j + dj;
      window[di * 2 + dj] = t.act[idx];
      if (idx === argmax) winner = di * 2 + dj;
    }
  }
  const value = t.pooled[(sel.map * stage.size + sel.i) * stage.size + sel.j];
  const max = s.pool === 'max';
  const el = h(
    'div',
    { class: 'inspector-view' },
    header(ws, sel, max ? 'Max pooling output' : 'Average pooling output'),
    block(
      1,
      null,
      paragraph(`${max ? 'Max' : 'Average'} pooling summarizes each 2×2 window of the map below into one value: the map gets half as wide and half as tall. It has no weights.`),
      max ? paragraph('Max pooling keeps the strongest response of the window: a pattern found anywhere inside it passes on, so small shifts of the image matter less. During training, only the winning cell receives a gradient.') : paragraph('Average pooling keeps the mean of the window; every cell receives a quarter of the gradient.'),
    ),
    block(
      2,
      null,
      numberGrid(window, 2, { colors, caption: `window of Conv ${l + 1}, map ${sel.map + 1}`, selected: max ? winner : null }),
      kv(max ? 'Maximum' : 'Average', formatSigned(value, 4), 'kv--strong'),
    ),
    block(3, null, tex(max ? `\\text{pool}_{${sel.i},${sel.j}} = \\max(${Array.from(window, (v) => texNumber(v)).join(',\\,')}) = ${texNumber(value)}` : `\\text{pool}_{${sel.i},${sel.j}} = \\tfrac14(${Array.from(window, (v) => texNumber(v)).join(' + ')}) = ${texNumber(value)}`, { display: true })),
  );
  return { el, update: () => {} };
}
