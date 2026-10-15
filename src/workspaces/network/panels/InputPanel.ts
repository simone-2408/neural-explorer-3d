/**
 * Input panel: sliders for small input layers, a paintable image for
 * image-shaped inputs (e.g. 28×28). Moving an input re-runs the forward pass
 * instantly, so the effect propagates through the 3D network.
 */
import { DisposeBag } from '../../../core/DisposeBag';
import { createRng, randomSeed } from '../../../math/random';
import { clamp } from '../../../math/stats';
import { INPUT_PATTERNS, makePattern } from '../../../neural/inputPatterns';
import { neuronTitle, outputSymbol } from '../../../neural/topology';
import { h, replaceChildren } from '../../../ui/dom';
import { tex } from '../../../ui/tex';
import { button, section, slider, type Slider } from '../../../ui/widgets';
import { formatValue } from '../../../utils/format';
import { frameThrottle } from '../../../utils/schedule';
import type { NetworkContext, Panel } from '../context';

const MAX_SLIDERS = 24;

export class InputPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkContext;
  private readonly bag = new DisposeBag();
  private readonly body = h('div', { class: 'input-panel' });
  private sliders: Slider[] = [];
  private painter: ImagePainter | null = null;
  private readonly summary = h('p', { class: 'panel-text muted' });

  constructor(ctx: NetworkContext) {
    this.ctx = ctx;
    this.el = section({ id: 'input', title: 'Input', children: [this.body] });
    this.bag.add(ctx.model.events.on('structure', () => this.render()));
    this.bag.add(ctx.model.events.on('activations', () => this.sync()));
    this.render();
  }

  dispose(): void {
    this.bag.dispose();
    this.painter?.dispose();
  }

  private render(): void {
    const { model } = this.ctx;
    const layer = model.spec.layers[0];
    this.sliders = [];
    this.painter?.dispose();
    this.painter = null;

    const randomize = () => {
      const rng = createRng(randomSeed());
      model.setInputs(Array.from({ length: layer.size }, () => Math.round(rng.uniform(layer.range[0], layer.range[1]) * 100) / 100));
    };
    const actions = h(
      'div',
      { class: 'button-row' },
      button({ label: 'Randomize', icon: 'dice', small: true, onClick: randomize }),
      button({ label: 'Zero', small: true, onClick: () => model.setInputs(new Float64Array(layer.size)) }),
    );

    if (layer.shape) {
      this.painter = new ImagePainter(this.ctx, layer.shape[0], layer.shape[1]);
      replaceChildren(
        this.body,
        h('p', { class: 'panel-text' }, `Draw on the ${layer.shape[0]}×${layer.shape[1]} input image (Shift or right button erases). Each pixel is one input neuron.`),
        this.painter.el,
        h(
          'div',
          { class: 'button-row' },
          INPUT_PATTERNS.map((p) =>
            button({ label: p.label, small: true, title: `Load the “${p.label}” pattern`, onClick: () => model.setInputs(makePattern(p.id, layer.shape![0], layer.shape![1], randomSeed())) }),
          ),
        ),
      );
      return;
    }

    if (layer.size > MAX_SLIDERS) {
      replaceChildren(this.body, h('p', { class: 'panel-text' }, `${layer.size} inputs — too many for individual sliders. Select an input neuron to edit its value.`), this.summary, actions);
      this.sync();
      return;
    }

    this.sliders = Array.from({ length: layer.size }, (_, i) =>
      slider({
        label: h('span', null, tex(outputSymbol(model.spec, 0, i)), layer.labels ? h('span', { class: 'muted' }, ` ${layer.labels[i]}`) : ` · ${neuronTitle(model.spec, 0, i)}`),
        min: layer.range[0],
        max: layer.range[1],
        step: 0.01,
        value: model.input[i],
        diverging: layer.range[0] < 0,
        digits: 2,
        ariaLabel: `Input ${neuronTitle(model.spec, 0, i)}`,
        onInput: (v) => model.setInput(i, v),
      }),
    );
    replaceChildren(this.body, h('p', { class: 'panel-text' }, 'Change the input and watch the activations update through the network.'), this.sliders.map((s) => s.el), actions);
  }

  private sync(): void {
    const { model } = this.ctx;
    this.sliders.forEach((s, i) => s.setValue(model.input[i]));
    this.painter?.paint();
    if (!this.sliders.length && !this.painter) {
      let min = Infinity;
      let max = -Infinity;
      for (const v of model.input) {
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
      this.summary.textContent = `Current values range from ${formatValue(min, 2)} to ${formatValue(max, 2)}.`;
    }
  }
}

/** A small canvas where each pixel is one input value; supports painting. */
class ImagePainter {
  readonly el: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: NetworkContext;
  private readonly rows: number;
  private readonly cols: number;
  private readonly bag = new DisposeBag();
  private readonly values: Float64Array;
  private readonly commit = frameThrottle(() => this.ctx.model.setInputs(this.values));

  constructor(ctx: NetworkContext, rows: number, cols: number) {
    this.ctx = ctx;
    this.rows = rows;
    this.cols = cols;
    this.values = Float64Array.from(ctx.model.input);
    this.canvas = h('canvas', { class: 'input-image', width: cols, height: rows, 'aria-label': 'Input image: draw with the mouse or finger' });
    this.el = h('div', { class: 'input-image__wrap' }, this.canvas);
    this.paint();

    let drawing = false;
    let erase = false;
    const strokeAt = (event: PointerEvent) => {
      const rect = this.canvas.getBoundingClientRect();
      const cx = ((event.clientX - rect.left) / rect.width) * cols - 0.5;
      const cy = ((event.clientY - rect.top) / rect.height) * rows - 0.5;
      const range = ctx.model.spec.layers[0].range;
      const radius = Math.max(1.2, cols / 22);
      for (let r = Math.floor(cy - radius - 1); r <= Math.ceil(cy + radius + 1); r++) {
        for (let c = Math.floor(cx - radius - 1); c <= Math.ceil(cx + radius + 1); c++) {
          if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
          const d = Math.hypot(r - cy, c - cx);
          const strength = clamp(1 - d / (radius + 0.5), 0, 1);
          if (strength <= 0) continue;
          const k = r * cols + c;
          this.values[k] = erase
            ? Math.max(range[0], this.values[k] - strength * (range[1] - range[0]))
            : Math.min(range[1], Math.max(this.values[k], range[0] + strength * (range[1] - range[0])));
        }
      }
      this.paintValues(this.values);
      this.commit();
    };
    this.bag.listen(this.canvas, 'pointerdown', (event) => {
      drawing = true;
      erase = event.shiftKey || event.button === 2;
      this.values.set(ctx.model.input);
      this.canvas.setPointerCapture(event.pointerId);
      strokeAt(event);
    });
    this.bag.listen(this.canvas, 'pointermove', (event) => {
      if (drawing) strokeAt(event);
    });
    this.bag.listen(this.canvas, 'pointerup', () => (drawing = false));
    this.bag.listen(this.canvas, 'pointercancel', () => (drawing = false));
    this.bag.listen(this.canvas, 'contextmenu', (event) => event.preventDefault());
  }

  /** Repaints from the model input. */
  paint(): void {
    this.paintValues(this.ctx.model.input);
  }

  private paintValues(values: ArrayLike<number>): void {
    const g = this.canvas.getContext('2d');
    if (!g) return;
    const range = this.ctx.model.spec.layers[0].range;
    const image = g.createImageData(this.cols, this.rows);
    for (let k = 0; k < this.rows * this.cols; k++) {
      const t = clamp((values[k] - range[0]) / (range[1] - range[0]), 0, 1);
      const v = Math.round(18 + 237 * t);
      image.data[4 * k] = v;
      image.data[4 * k + 1] = v;
      image.data[4 * k + 2] = v;
      image.data[4 * k + 3] = 255;
    }
    g.putImageData(image, 0, 0);
  }

  dispose(): void {
    this.commit.cancel();
    this.bag.dispose();
  }
}
