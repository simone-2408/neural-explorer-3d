/** Left-panel sections of the Training workspace. */
import { DisposeBag } from '../../core/DisposeBag';
import { ACTIVATIONS, type ActivationId } from '../../math/activations';
import { randomSeed } from '../../math/random';
import { INIT_SCHEMES, type InitScheme } from '../../neural/init';
import { DATASETS, datasetInfo, type DatasetId, type DatasetOptions } from '../../training/datasets';
import { LOSSES, lossFor } from '../../training/loss';
import { OPTIMIZERS, type OptimizerId } from '../../training/optimizers';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { tex } from '../../ui/tex';
import { button, kv, section, segmented, select, slider, toggle, type Segmented, type Slider, type Toggle } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { Panel } from '../network/context';
import { MAX_HIDDEN_LAYERS, MAX_HIDDEN_SIZE, NETWORK_TEMPLATES } from '../../training/session';
import type { TrainingWorkspace } from './TrainingWorkspace';

const HIDDEN_ACTIVATIONS: ActivationId[] = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'linear'];

// ------------------------------------------------------------------ dataset

export class DatasetPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly datasetSelect: HTMLSelectElement;
  private readonly description = h('p', { class: 'panel-text' });
  private readonly samples: Slider;
  private readonly noise: Slider;
  private readonly testSplit: Slider;
  private readonly counts = h('p', { class: 'panel-text muted' });

  constructor(private readonly ws: TrainingWorkspace) {
    const o = ws.settings.datasetOptions;
    this.datasetSelect = select<DatasetId>({
      options: DATASETS.map((d) => ({ value: d.id, label: `${d.label}${d.classes > 2 ? ` (${d.classes} classes)` : ''}` })),
      value: ws.settings.dataset,
      ariaLabel: 'Dataset',
      onChange: (dataset) => ws.update({ dataset, datasetOptions: { ...ws.settings.datasetOptions, noise: datasetInfo(dataset).noise } }),
    });
    this.samples = slider({ label: 'Samples', min: 20, max: 500, step: 10, value: o.samples, digits: 0, ariaLabel: 'Number of samples', onInput: (v) => this.debouncedUpdate({ samples: Math.round(v) }) });
    this.noise = slider({ label: 'Noise', min: 0, max: 0.5, step: 0.01, value: o.noise, digits: 2, ariaLabel: 'Noise', onInput: (v) => this.debouncedUpdate({ noise: v }) });
    this.testSplit = slider({ label: 'Test set (held-out fraction)', min: 0, max: 0.5, step: 0.05, value: o.testFraction, digits: 2, ariaLabel: 'Test fraction', onInput: (v) => this.debouncedUpdate({ testFraction: v }) });
    this.el = section({
      id: 'training-dataset',
      title: 'Dataset',
      children: [
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Data'), this.datasetSelect),
        this.description,
        this.samples.el,
        this.noise.el,
        this.testSplit.el,
        h('div', { class: 'button-row' }, button({ label: 'New samples', icon: 'dice', small: true, title: 'Draw a new random dataset of the same kind', onClick: () => ws.update({ datasetOptions: { ...ws.settings.datasetOptions, seed: randomSeed() } }) })),
        this.counts,
      ],
    });
    this.bag.add(ws.events.on('settings', () => this.sync()));
    this.sync();
  }

  private timer: ReturnType<typeof setTimeout> | undefined;
  private pending: Partial<DatasetOptions> = {};

  private debouncedUpdate(patch: Partial<DatasetOptions>): void {
    this.pending = { ...this.pending, ...patch };
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.ws.update({ datasetOptions: { ...this.ws.settings.datasetOptions, ...this.pending } });
      this.pending = {};
    }, 250);
  }

  private sync(): void {
    const ws = this.ws;
    const o = ws.settings.datasetOptions;
    this.datasetSelect.value = ws.settings.dataset;
    this.description.textContent = datasetInfo(ws.settings.dataset).description;
    this.samples.setValue(o.samples);
    this.noise.setValue(o.noise);
    this.testSplit.setValue(o.testFraction);
    this.counts.textContent = `${ws.dataset.train.length} training points · ${ws.dataset.test.length} test points · ${ws.dataset.classes} classes`;
  }

  dispose(): void {
    clearTimeout(this.timer);
    this.bag.dispose();
  }
}

// ------------------------------------------------------------------ network

export class TrainingNetworkPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly list = h('div', { class: 'layer-list' });
  private readonly info = h('div');
  private readonly initFormula = h('div', { class: 'formula-box' });
  private readonly schemeSelect: HTMLSelectElement;

  constructor(private readonly ws: TrainingWorkspace) {
    this.schemeSelect = select<InitScheme>({
      options: Object.values(INIT_SCHEMES).map((s) => ({ value: s.id, label: s.label })),
      value: ws.settings.init.scheme,
      ariaLabel: 'Initialization',
      onChange: (scheme) => ws.update({ init: { ...ws.settings.init, scheme } }),
    });
    const templates = select<string>({
      options: [{ value: '', label: 'Templates…' }, ...NETWORK_TEMPLATES.map((t) => ({ value: t.id, label: t.label }))],
      value: '',
      ariaLabel: 'Network template',
      onChange: (id) => {
        const template = NETWORK_TEMPLATES.find((t) => t.id === id);
        if (template) ws.update({ hidden: template.hidden.map((l) => ({ ...l })) });
        templates.value = '';
      },
    });
    const allActivation = select<string>({
      options: [{ value: '', label: 'Set all hidden to…' }, ...HIDDEN_ACTIVATIONS.map((id) => ({ value: id, label: ACTIVATIONS[id].label }))],
      value: '',
      ariaLabel: 'Activation of all hidden layers',
      onChange: (id) => {
        if (id) ws.update({ hidden: ws.settings.hidden.map((l) => ({ ...l, activation: id as ActivationId })) });
        allActivation.value = '';
      },
    });
    this.el = section({
      id: 'training-network',
      title: 'Network',
      children: [
        h('div', { class: 'field field--inline field--pair' }, templates, allActivation),
        h('div', { class: 'field__label field__label--row' }, h('span', null, 'Hidden layers'), h('span', { class: 'muted' }, 'neurons · activation')),
        this.list,
        button({
          label: 'Add hidden layer',
          icon: 'plus',
          small: true,
          class: 'btn--block',
          onClick: () => {
            const hidden = ws.settings.hidden;
            if (hidden.length >= MAX_HIDDEN_LAYERS) return;
            const last = hidden[hidden.length - 1] ?? { size: 4, activation: 'relu' as ActivationId };
            ws.update({ hidden: [...hidden.map((l) => ({ ...l })), { ...last }] });
          },
        }),
        this.info,
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Initialization'), this.schemeSelect),
        this.initFormula,
        h(
          'div',
          { class: 'button-row' },
          button({ label: 'Re-initialize', icon: 'refresh', small: true, title: 'Same scheme and seed: back to the initial weights', onClick: () => ws.reset() }),
          button({ label: 'New seed', icon: 'dice', small: true, onClick: () => ws.update({ init: { ...ws.settings.init, seed: randomSeed() } }) }),
        ),
      ],
    });
    this.bag.add(ws.events.on('settings', () => this.render()));
    this.render();
  }

  private render(): void {
    const ws = this.ws;
    const hidden = ws.settings.hidden;
    replaceChildren(
      this.list,
      h('div', { class: 'layer-row layer-row--input' }, h('span', { class: 'layer-row__name' }, 'Input'), h('span', { class: 'layer-row__static muted' }, '2 features (x₁, x₂)'), h('span', { class: 'layer-row__spacer' })),
      hidden.map((layer, k) => {
        const setSize = (size: number) => {
          const next = hidden.map((l) => ({ ...l }));
          next[k].size = Math.max(1, Math.min(MAX_HIDDEN_SIZE, size));
          ws.update({ hidden: next });
        };
        return h(
          'div',
          { class: 'layer-row layer-row--hidden' },
          h('span', { class: 'layer-row__name' }, `Hidden ${k + 1}`),
          h(
            'div',
            { class: 'stepper' },
            h('button', { type: 'button', class: 'stepper__btn', 'aria-label': 'Remove a neuron', disabled: layer.size <= 1, onClick: () => setSize(layer.size - 1) }, icon('minus', 12)),
            h('span', { class: 'stepper__value', style: 'display:inline-block;padding:3px 0' }, String(layer.size)),
            h('button', { type: 'button', class: 'stepper__btn', 'aria-label': 'Add a neuron', disabled: layer.size >= MAX_HIDDEN_SIZE, onClick: () => setSize(layer.size + 1) }, icon('plus', 12)),
          ),
          select<ActivationId>({
            options: HIDDEN_ACTIVATIONS.map((id) => ({ value: id, label: ACTIVATIONS[id].label })),
            value: layer.activation,
            ariaLabel: `Hidden ${k + 1} activation`,
            onChange: (activation) => ws.update({ hidden: hidden.map((l, i) => (i === k ? { ...l, activation } : { ...l })) }),
          }),
          h(
            'button',
            { type: 'button', class: 'btn btn--ghost btn--icon btn--sm layer-row__remove', 'aria-label': `Remove hidden layer ${k + 1}`, onClick: () => ws.update({ hidden: hidden.filter((_, i) => i !== k).map((l) => ({ ...l })) }) },
            icon('close', 12),
          ),
        );
      }),
    );
    const output = ws.model.spec.layers[ws.model.layerCount - 1];
    const loss = LOSSES[lossFor(ws.model.spec)];
    replaceChildren(
      this.info,
      kv('Output layer', `${output.size} · ${ACTIVATIONS[output.activation].label}`),
      kv('Loss', loss.label),
      tex(loss.tex, { display: true }),
      kv('Parameters', formatCount(ws.model.totalParameters()), 'kv--strong'),
    );
    this.schemeSelect.value = ws.settings.init.scheme;
    replaceChildren(this.initFormula, tex(INIT_SCHEMES[ws.settings.init.scheme].tex, { display: true }));
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------------- optimizer

/** Learning rate slider on a log scale: position = log10(η). */
export class OptimizerPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly optimizer: Segmented<OptimizerId>;
  private readonly rule = h('div', { class: 'formula-box' });
  private readonly description = h('p', { class: 'panel-text' });
  private readonly lrRange: HTMLInputElement;
  private readonly lrValue = h('span', { class: 'num' });
  private readonly batch: HTMLSelectElement;
  private readonly decay: HTMLSelectElement;
  private readonly stopAt: HTMLInputElement;

  constructor(private readonly ws: TrainingWorkspace) {
    this.optimizer = segmented<OptimizerId>({
      options: (Object.keys(OPTIMIZERS) as OptimizerId[]).map((id) => ({ value: id, label: OPTIMIZERS[id].label, title: OPTIMIZERS[id].description })),
      value: ws.settings.optimizer.id,
      ariaLabel: 'Optimizer',
      onChange: (id) => ws.update({ optimizer: { ...ws.settings.optimizer, id } }),
    });
    this.lrRange = h('input', { type: 'range', class: 'slider__range', min: -4, max: 1, step: 0.05, 'aria-label': 'Learning rate (log scale)' });
    this.lrRange.addEventListener('input', () => {
      const lr = Number(Math.pow(10, Number(this.lrRange.value)).toPrecision(2));
      this.lrValue.textContent = formatValue(lr, lr < 0.01 ? 4 : 3);
      ws.update({ optimizer: { ...ws.settings.optimizer, learningRate: lr } });
    });
    this.batch = select<string>({
      options: ['1', '4', '8', '16', '32', '64', 'full'].map((v) => ({ value: v, label: v === 'full' ? 'Full batch' : v })),
      value: String(ws.settings.batchSize),
      ariaLabel: 'Batch size',
      onChange: (v) => ws.update({ batchSize: v === 'full' ? 100000 : Number(v) }),
    });
    this.decay = select<string>({
      options: ['0', '0.0001', '0.001', '0.01', '0.1'].map((v) => ({ value: v, label: v === '0' ? 'None' : v })),
      value: String(ws.settings.optimizer.weightDecay),
      ariaLabel: 'Weight decay',
      onChange: (v) => ws.update({ optimizer: { ...ws.settings.optimizer, weightDecay: Number(v) } }),
    });
    this.stopAt = h('input', { type: 'number', class: 'input input--seed', min: 0, step: 50, value: ws.settings.stopAtEpoch, 'aria-label': 'Stop after epochs' });
    this.stopAt.addEventListener('change', () => ws.update({ stopAtEpoch: Math.max(0, Math.floor(Number(this.stopAt.value) || 0)) }));

    this.el = section({
      id: 'training-optimizer',
      title: 'Optimizer',
      children: [
        this.optimizer.el,
        this.rule,
        this.description,
        h('div', { class: 'slider' }, h('div', { class: 'slider__label' }, 'Learning rate η ', this.lrValue), h('div', { class: 'slider__controls' }, this.lrRange)),
        h('label', { class: 'field field--inline' }, h('span', { class: 'field__label' }, 'Batch size'), this.batch),
        h('label', { class: 'field field--inline' }, h('span', { class: 'field__label' }, 'Weight decay (L2)'), this.decay),
        h('label', { class: 'field field--inline' }, h('span', { class: 'field__label' }, 'Stop after epochs (0 = never)'), this.stopAt),
      ],
    });
    this.bag.add(ws.events.on('settings', () => this.sync()));
    this.sync();
  }

  private sync(): void {
    const o = this.ws.settings.optimizer;
    this.optimizer.set(o.id);
    replaceChildren(this.rule, tex(OPTIMIZERS[o.id].tex, { display: true }));
    this.description.textContent = OPTIMIZERS[o.id].description;
    if (document.activeElement !== this.lrRange) this.lrRange.value = String(Math.log10(o.learningRate));
    this.lrValue.textContent = formatValue(o.learningRate, o.learningRate < 0.01 ? 4 : 3);
    this.batch.value = this.ws.settings.batchSize >= 100000 ? 'full' : String(this.ws.settings.batchSize);
    this.decay.value = String(o.weightDecay);
    if (document.activeElement !== this.stopAt) this.stopAt.value = String(this.ws.settings.stopAtEpoch);
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------------- encoding

export class TrainingViewPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly encode: Segmented<'weights' | 'gradients'>;
  private readonly log: Toggle;
  private readonly delta: Toggle;

  constructor(private readonly ws: TrainingWorkspace) {
    this.encode = segmented({
      options: [
        { value: 'weights', label: 'Weights', title: 'Line thickness = |w|' },
        { value: 'gradients', label: 'Gradients', title: 'Line thickness = |∂L/∂w| of the last step' },
      ],
      value: ws.encode,
      ariaLabel: 'What the lines show',
      onChange: (encode) => ws.setEncoding({ encode }),
    });
    this.log = toggle({ label: 'Logarithmic gradient scale', checked: ws.gradientLog, title: 'Shows gradients that differ by orders of magnitude (vanishing gradients)', onChange: (gradientLog) => ws.setEncoding({ gradientLog }) });
    this.delta = toggle({ label: 'Neurons show δ = ∂L/∂z', checked: ws.showDelta, title: 'Colour the neurons by their error signal instead of their activation', onChange: (showDelta) => ws.setEncoding({ showDelta }) });
    this.el = section({
      id: 'training-view',
      title: '3D encoding',
      children: [
        h('div', { class: 'field' }, h('span', { class: 'field__label' }, 'Lines show'), this.encode.el),
        h('div', { class: 'toggle-list' }, this.log.el, this.delta.el),
        h('p', { class: 'panel-text muted' }, 'Gradients and δ are those of the last training step (averaged over its mini-batch).'),
      ],
    });
    this.bag.add(ws.events.on('settings', () => this.sync()));
  }

  private sync(): void {
    this.encode.set(this.ws.encode);
    this.log.set(this.ws.gradientLog);
    this.delta.set(this.ws.showDelta);
  }

  dispose(): void {
    this.bag.dispose();
  }
}
