/** Left-panel sections of the CNN workspace: data (with a drawing pad), architecture, training. */
import { DisposeBag } from '../../core/DisposeBag';
import { IMAGE_DATASETS, IMAGE_SIZE, type ImageDatasetId } from '../../cnn/datasets';
import { CNN_LIMITS, type ConvLayerConfig } from '../../cnn/model';
import { OPTIMIZERS, type OptimizerId } from '../../training/optimizers';
import { LineChart } from '../../ui/charts/lineChart';
import { h, replaceChildren } from '../../ui/dom';
import { button, kv, note, section, segmented, slider } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import { frameThrottle } from '../../utils/schedule';
import type { Panel } from '../network/context';
import type { CnnWorkspace } from './CnnWorkspace';

/** Paints a 16×16 image into a canvas (one pixel per value; scaled up with CSS). */
export function paintImage(canvas: HTMLCanvasElement, pixels: ArrayLike<number>, dark: boolean): void {
  const size = Math.round(Math.sqrt(pixels.length));
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const image = ctx.createImageData(size, size);
  for (let k = 0; k < pixels.length; k++) {
    const v = Math.max(0, Math.min(1, pixels[k]));
    const c = dark ? Math.round(20 + 225 * v) : Math.round(245 - 225 * v);
    image.data[4 * k] = c;
    image.data[4 * k + 1] = c;
    image.data[4 * k + 2] = c;
    image.data[4 * k + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

export function imageCanvas(pixels: ArrayLike<number>, dark: boolean, className = 'cnn-thumb'): HTMLCanvasElement {
  const canvas = h('canvas', { class: className, 'aria-hidden': 'true' });
  paintImage(canvas, pixels, dark);
  return canvas;
}

// ------------------------------------------------------------------- data

export class DataPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: CnnWorkspace;
  private readonly datasets = h('div');
  private readonly gallery = h('div', { class: 'cnn-gallery', role: 'list', 'aria-label': 'Test images' });
  private readonly pad: HTMLCanvasElement;
  private readonly info = h('div', { class: 'panel-note' });

  constructor(ws: CnnWorkspace) {
    this.ws = ws;
    this.pad = h('canvas', { class: 'cnn-pad', width: IMAGE_SIZE, height: IMAGE_SIZE, 'aria-label': 'Drawing pad: draw with the mouse, right-click or Shift to erase' });
    this.bindPad();
    this.el = section({
      id: 'cnn-data',
      title: 'Images',
      children: [
        this.datasets,
        h('div', { class: 'field-label' }, 'Test images — click one to feed it to the network'),
        this.gallery,
        h('div', { class: 'field-label' }, 'Or draw your own (right-click or Shift erases)'),
        h(
          'div',
          { class: 'cnn-draw' },
          this.pad,
          h(
            'div',
            { class: 'stack' },
            button({ icon: 'close', label: 'Clear', small: true, onClick: () => ws.setDrawing(new Float64Array(IMAGE_SIZE * IMAGE_SIZE)) }),
            button({ icon: 'copy', label: 'Copy image', small: true, title: 'Start from the image shown now', onClick: () => ws.setDrawing(ws.inputPixels()) }),
          ),
        ),
        this.info,
      ],
    });
    for (const e of ['model', 'input', 'settings', 'progress'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private bindPad(): void {
    let painting: 'ink' | 'erase' | null = null;
    const apply = frameThrottle(() => this.ws.setDrawing(this.ws.drawing));
    this.bag.add(() => apply.cancel());
    const paint = (event: PointerEvent) => {
      const rect = this.pad.getBoundingClientRect();
      const x = Math.floor(((event.clientX - rect.left) / rect.width) * IMAGE_SIZE);
      const y = Math.floor(((event.clientY - rect.top) / rect.height) * IMAGE_SIZE);
      if (x < 0 || y < 0 || x >= IMAGE_SIZE || y >= IMAGE_SIZE) return;
      const d = this.ws.input.kind === 'drawing' ? this.ws.drawing : new Float64Array(IMAGE_SIZE * IMAGE_SIZE);
      const set = (px: number, py: number, v: number) => {
        if (px < 0 || py < 0 || px >= IMAGE_SIZE || py >= IMAGE_SIZE) return;
        const k = py * IMAGE_SIZE + px;
        d[k] = painting === 'erase' ? Math.min(d[k], 1 - v) : Math.max(d[k], v);
      };
      set(x, y, 1);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) set(x + dx, y + dy, painting === 'erase' ? 1 : 0.3);
      this.ws.drawing = d;
      if (this.ws.input.kind !== 'drawing') this.ws.setDrawing(d);
      else apply();
      paintImage(this.pad, d, this.ws.palette.theme === 'dark');
    };
    this.pad.addEventListener('contextmenu', (e) => e.preventDefault());
    this.pad.addEventListener('pointerdown', (e) => {
      painting = e.button === 2 || e.shiftKey ? 'erase' : 'ink';
      this.pad.setPointerCapture(e.pointerId);
      paint(e);
    });
    this.pad.addEventListener('pointermove', (e) => {
      if (painting) paint(e);
    });
    const stop = () => (painting = null);
    this.pad.addEventListener('pointerup', stop);
    this.pad.addEventListener('pointercancel', stop);
  }

  private render(): void {
    const { ws } = this;
    const dark = ws.palette.theme === 'dark';
    replaceChildren(
      this.datasets,
      segmented<ImageDatasetId>({
        options: (Object.keys(IMAGE_DATASETS) as ImageDatasetId[]).map((id) => ({ value: id, label: IMAGE_DATASETS[id].label, title: IMAGE_DATASETS[id].description })),
        value: ws.settings.dataset,
        ariaLabel: 'Dataset',
        onChange: (id) => ws.update({ dataset: id }),
      }).el,
      h('div', { class: 'muted' }, ws.dataset.info.description, ` ${formatCount(ws.dataset.train.length)} training and ${formatCount(ws.dataset.test.length)} test images, generated from a seed.`),
    );
    const current = ws.input.kind === 'sample' ? ws.input.index : -1;
    replaceChildren(
      this.gallery,
      ws.dataset.test.slice(0, 24).map((s, k) =>
        h(
          'button',
          { type: 'button', role: 'listitem', class: `cnn-gallery__item${k === current ? ' is-selected' : ''}`, title: `Test image ${k}: ${ws.classNames[s.label]}`, onClick: () => ws.setInputSample(k) },
          imageCanvas(s.pixels, dark),
        ),
      ),
    );
    paintImage(this.pad, ws.input.kind === 'drawing' ? ws.drawing : new Float64Array(IMAGE_SIZE * IMAGE_SIZE), dark);
    const label = ws.inputLabel();
    const probs = ws.trace.probs;
    let best = 0;
    for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
    replaceChildren(
      this.info,
      h('div', null, ws.input.kind === 'drawing' ? 'Your drawing' : `Test image ${current}`, label !== null ? ` · true class: ${ws.classNames[label]}` : '', ` · the network says: ${ws.classNames[best]} (${formatValue(probs[best] * 100, 1)}%)`),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ------------------------------------------------------------- architecture

export class ArchitecturePanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: CnnWorkspace;
  private readonly body = h('div', { class: 'stack' });

  constructor(ws: CnnWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'cnn-architecture', title: 'Architecture', children: [this.body] });
    for (const e of ['model', 'settings'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const arch = ws.settings.architecture;
    const setConvs = (convs: ConvLayerConfig[]) => ws.update({ architecture: { ...arch, convs } });
    const layer = (c: ConvLayerConfig, l: number) => {
      const filters = h('select', { class: 'select select--compact', 'aria-label': `Filters of conv layer ${l + 1}` }, Array.from({ length: CNN_LIMITS.maxFilters }, (_, k) => h('option', { value: String(k + 1) }, `${k + 1} filter${k ? 's' : ''}`)));
      filters.value = String(c.filters);
      filters.addEventListener('change', () => setConvs(arch.convs.map((x, i) => (i === l ? { ...x, filters: Number(filters.value) } : x))));
      return h(
        'div',
        { class: 'cnn-layer' },
        h('div', { class: 'cnn-layer__title' }, `Conv ${l + 1}`),
        filters,
        segmented<'3' | '5'>({ options: [{ value: '3', label: '3×3' }, { value: '5', label: '5×5' }], value: String(c.kernel) as '3' | '5', ariaLabel: 'Kernel size', onChange: (v) => setConvs(arch.convs.map((x, i) => (i === l ? { ...x, kernel: Number(v) as 3 | 5 } : x))) }).el,
        segmented<ConvLayerConfig['pool']>({
          options: [
            { value: 'max', label: 'Max pool', title: '2×2 max pooling: keeps the largest value of each window' },
            { value: 'avg', label: 'Avg pool', title: '2×2 average pooling' },
            { value: 'none', label: 'No pool', title: 'No pooling: the maps keep their size' },
          ],
          value: c.pool,
          ariaLabel: 'Pooling',
          onChange: (v) => setConvs(arch.convs.map((x, i) => (i === l ? { ...x, pool: v } : x))),
        }).el,
      );
    };
    const model = ws.model;
    replaceChildren(
      this.body,
      segmented<'1' | '2'>({
        options: [
          { value: '1', label: '1 conv layer' },
          { value: '2', label: '2 conv layers' },
        ],
        value: String(arch.convs.length) as '1' | '2',
        ariaLabel: 'Number of convolutional layers',
        onChange: (v) => {
          const second: ConvLayerConfig = { filters: 8, kernel: 3, pool: 'max' };
          setConvs(v === '1' ? arch.convs.slice(0, 1) : [...arch.convs, second].slice(0, 2));
        },
      }).el,
      arch.convs.map(layer),
      segmented<'relu' | 'tanh'>({
        options: [
          { value: 'relu', label: 'ReLU' },
          { value: 'tanh', label: 'tanh' },
        ],
        value: arch.activation,
        ariaLabel: 'Activation of the conv layers',
        onChange: (v) => ws.update({ architecture: { ...arch, activation: v } }),
      }).el,
      h('div', { class: 'field-label' }, 'Parameters'),
      h(
        'div',
        { class: 'kv-list' },
        model.shapes.map((s, l) => kv(`Conv ${l + 1} (${s.outChannels}×${s.inChannels}×${s.kernel}×${s.kernel} + ${s.outChannels})`, formatCount(model.convParameterCount(l)))),
        kv(`Dense (${model.config.classes}×${model.flatSize} + ${model.config.classes})`, formatCount(model.denseWeight.length + model.denseBias.length)),
        kv('Total', formatCount(model.parameterCount()), 'kv--strong'),
      ),
      note(`Weight sharing: Conv 1 computes ${formatCount(model.shapes[0].outChannels * model.shapes[0].size * model.shapes[0].size)} values with ${formatCount(model.convParameterCount(0))} parameters. A fully connected layer producing as many values from the same image would need ${formatCount(model.fullyConnectedEquivalent(0))}.`),
      h('div', { class: 'muted' }, 'Changing the architecture starts training again from new random weights.'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ----------------------------------------------------------------- training

export class TrainingPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: CnnWorkspace;
  private readonly controls = h('div', { class: 'stack' });
  private readonly lossChart = new LineChart({ title: 'Loss (cross-entropy)', log: true, xLabel: 'epoch', format: (v) => formatValue(v, 4) });
  private readonly accChart = new LineChart({ title: 'Accuracy', yMin: 0, yMax: 1, xLabel: 'epoch', format: (v) => `${formatValue(v * 100, 1)}%` });

  constructor(ws: CnnWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'cnn-training', title: 'Training', children: [this.controls, this.lossChart.el, this.accChart.el] });
    this.bag.add(ws.events.on('model', () => this.render()));
    this.bag.add(ws.events.on('settings', () => this.render()));
    this.bag.add(ws.events.on('progress', () => this.draw()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const o = ws.settings.optimizer;
    const lr = slider({
      label: 'Learning rate (log scale)',
      min: -4,
      max: -0.5,
      step: 0.05,
      value: Math.log10(o.learningRate),
      digits: 2,
      ariaLabel: 'Learning rate, log10',
      onInput: (v) => ws.update({ optimizer: { ...ws.settings.optimizer, learningRate: Math.pow(10, v) } }),
    });
    const batch = h('select', { class: 'select', 'aria-label': 'Batch size' }, [4, 8, 16, 32, 64].map((b) => h('option', { value: String(b) }, `batch ${b}`)));
    batch.value = String(ws.settings.batchSize);
    batch.addEventListener('change', () => ws.update({ batchSize: Number(batch.value) }));
    replaceChildren(
      this.controls,
      segmented<OptimizerId>({
        options: (Object.keys(OPTIMIZERS) as OptimizerId[]).map((id) => ({ value: id, label: OPTIMIZERS[id].label, title: OPTIMIZERS[id].description })),
        value: o.id,
        ariaLabel: 'Optimizer',
        onChange: (id) => ws.update({ optimizer: { ...ws.settings.optimizer, id } }),
      }).el,
      lr.el,
      h('div', { class: 'muted' }, `η = ${formatValue(o.learningRate, 4)}`),
      batch,
    );
    this.draw();
  }

  private draw(): void {
    const history = this.ws.trainer.history;
    const xs = history.map((p) => p.epoch);
    const dark = this.ws.palette.theme === 'dark';
    const train = dark ? '#a99bff' : '#5b46d6';
    const test = dark ? '#ffc861' : '#a8670d';
    requestAnimationFrame(() => {
      this.lossChart.draw([
        { label: 'train', color: train, values: history.map((p) => p.train.loss), xs },
        { label: 'test', color: test, values: history.map((p) => p.test.loss), xs, dash: [4, 3] },
      ]);
      this.accChart.draw([
        { label: 'train', color: train, values: history.map((p) => p.train.accuracy), xs },
        { label: 'test', color: test, values: history.map((p) => p.test.accuracy), xs, dash: [4, 3] },
      ]);
    });
  }

  dispose(): void {
    this.bag.dispose();
  }
}
