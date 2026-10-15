/** Left-panel sections of the RNN workspace: presets, task and input, network, training. */
import { DisposeBag } from '../../core/DisposeBag';
import { CELLS, type CellType } from '../../rnn/model';
import { RNN_TASKS, type RnnTaskId } from '../../rnn/tasks';
import { OPTIMIZERS, type OptimizerId } from '../../training/optimizers';
import { LineChart } from '../../ui/charts/lineChart';
import { h, replaceChildren } from '../../ui/dom';
import { button, kv, note, section, segmented, slider, toggle } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { Panel } from '../network/context';
import { RNN_PRESETS, type RnnWorkspace } from './RnnWorkspace';

// ---------------------------------------------------------------- presets

export class PresetsPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: RnnWorkspace;
  private readonly list = h('div', { class: 'example-list' });
  private readonly info = h('div', { class: 'panel-note' });

  constructor(ws: RnnWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'rnn-presets', title: 'Ready-made setups', children: [this.list, this.info] });
    for (const e of ['model', 'settings'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    replaceChildren(
      this.list,
      RNN_PRESETS.map((p) =>
        h(
          'button',
          { type: 'button', class: `example${ws.preset?.id === p.id ? ' is-active' : ''}`, title: p.description, onClick: () => ws.applyPreset(p) },
          h('span', { class: 'example__label' }, p.label),
          h('span', { class: 'example__text' }, p.description),
        ),
      ),
    );
    replaceChildren(this.info, ws.preset ? note(h('span', null, h('strong', null, ws.preset.label), ' — ', ws.preset.description, ' Press Space to train.')) : null);
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ------------------------------------------------------------------- task

export class TaskPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: RnnWorkspace;
  private readonly body = h('div', { class: 'stack' });
  private readonly input: HTMLInputElement;
  private readonly chips = h('div', { class: 'token-chips', role: 'list', 'aria-label': 'Input symbols' });
  private readonly actions = h('div', { class: 'button-row' });

  constructor(ws: RnnWorkspace) {
    this.ws = ws;
    this.input = h('input', { type: 'text', class: 'text-input', spellcheck: false, autocomplete: 'off', maxlength: 80, 'aria-label': 'Input sequence' });
    this.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') ws.setText(this.input.value);
    });
    this.input.addEventListener('change', () => ws.setText(this.input.value));
    this.el = section({ id: 'rnn-task', title: 'Task and input', children: [this.body, h('div', { class: 'field-label' }, 'Input — one symbol per step (Enter to run)'), this.input, this.chips, this.actions] });
    for (const e of ['model', 'input', 'settings', 'selection'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const s = ws.settings;
    replaceChildren(
      this.body,
      segmented<RnnTaskId>({
        options: (Object.keys(RNN_TASKS) as RnnTaskId[]).map((id) => ({ value: id, label: RNN_TASKS[id].label, title: RNN_TASKS[id].description })),
        value: s.task,
        ariaLabel: 'Task',
        onChange: (task) => ws.update({ task }),
      }).el,
      h('div', { class: 'muted' }, RNN_TASKS[s.task].description),
      s.task === 'memory'
        ? slider({ label: 'Gap (meaningless symbols between the letter and “?”)', min: 1, max: 40, step: 1, value: s.gap, digits: 0, ariaLabel: 'Gap length', onInput: (v) => ws.update({ gap: Math.round(v) }) }).el
        : null,
    );
    if (document.activeElement !== this.input) this.input.value = ws.text;
    const sel = ws.selection && ws.selection.kind !== 'weights' ? ws.selection.t : -1;
    replaceChildren(
      this.chips,
      ws.inputs.map((id, t) =>
        h(
          'button',
          { type: 'button', role: 'listitem', class: `tchip tchip--button${t === sel ? ' is-selected' : ''}`, title: `step ${t}`, onClick: () => ws.select({ kind: 'step', t }) },
          h('span', { class: 'tchip__text' }, ws.task.inputSymbols[id] === ' ' ? '␣' : ws.task.inputSymbols[id]),
          h('span', { class: 'tchip__pos' }, String(t)),
        ),
      ),
    );
    replaceChildren(
      this.actions,
      s.task === 'days'
        ? [
            button({ icon: 'stepForward', label: 'Continue', small: true, title: 'Append 12 letters, always the most likely one', onClick: () => ws.continueText('greedy') }),
            button({ icon: 'dice', label: 'Sample', small: true, title: 'Append 12 letters drawn from the predictions (temperature 0.8)', onClick: () => ws.continueText('sample') }),
            button({ icon: 'refresh', label: 'Random text', small: true, variant: 'ghost', title: 'A random stretch of the week (N)', onClick: () => ws.newExample() }),
          ]
        : button({ icon: 'refresh', label: 'New example', small: true, title: 'A random letter and a random gap (N)', onClick: () => ws.newExample() }),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------------- network

export class NetworkPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: RnnWorkspace;
  private readonly body = h('div', { class: 'stack' });

  constructor(ws: RnnWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'rnn-network', title: 'Network', children: [this.body] });
    for (const e of ['model', 'settings'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const s = ws.settings;
    const m = ws.model;
    const { inputSize: I, hiddenSize: H, outputSize: O } = m.config;
    const G = m.gates;
    const hidden = h('select', { class: 'select', 'aria-label': 'Hidden size' }, [4, 8, 16, 24, 32, 48].map((n) => h('option', { value: String(n) }, `${n} hidden units`)));
    hidden.value = String(s.hidden);
    hidden.addEventListener('change', () => ws.update({ hidden: Number(hidden.value) }));
    replaceChildren(
      this.body,
      segmented<CellType>({
        options: (Object.keys(CELLS) as CellType[]).map((c) => ({ value: c, label: CELLS[c].label, title: CELLS[c].description })),
        value: s.cell,
        ariaLabel: 'Cell',
        onChange: (cell) => ws.update({ cell }),
      }).el,
      h('div', { class: 'muted' }, CELLS[s.cell].description),
      hidden,
      s.cell === 'gru'
        ? toggle({ label: 'Update gate starts open (bias +2)', title: 'Initial bias of the update gate: z ≈ 0.88 at the start, so the cell keeps its state unless the input says otherwise', checked: s.updateGateBias > 0, onChange: (v) => ws.update({ updateGateBias: v ? 2 : 0 }) }).el
        : null,
      h('div', { class: 'field-label' }, 'Parameters (shared by every step)'),
      h(
        'div',
        { class: 'kv-list' },
        kv(`W_ih (${G * H}×${I}) + b_ih`, formatCount(G * H * I + G * H)),
        kv(`W_hh (${G * H}×${H}) + b_hh`, formatCount(G * H * H + G * H)),
        kv(`W_out (${O}×${H}) + b_out`, formatCount(O * H + O)),
        kv('Total', formatCount(m.parameterCount()), 'kv--strong'),
      ),
      h('div', { class: 'muted' }, 'Changing the network starts training again from new random weights.'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// --------------------------------------------------------------- training

export class TrainingPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: RnnWorkspace;
  private readonly controls = h('div', { class: 'stack' });
  private readonly lossChart = new LineChart({ title: 'Loss (cross-entropy on the scored steps)', log: true, xLabel: 'step', format: (v) => formatValue(v, 4) });
  private readonly accChart = new LineChart({ title: 'Accuracy', yMin: 0, yMax: 1, xLabel: 'step', format: (v) => `${formatValue(v * 100, 1)}%` });

  constructor(ws: RnnWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'rnn-training', title: 'Training', children: [this.controls, this.lossChart.el, this.accChart.el] });
    for (const e of ['model', 'settings'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.bag.add(ws.events.on('progress', () => this.draw()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const o = ws.settings.optimizer;
    const batch = h('select', { class: 'select', 'aria-label': 'Batch size' }, [4, 8, 16, 32].map((b) => h('option', { value: String(b) }, `batch ${b}`)));
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
      slider({ label: 'Learning rate (log scale)', min: -4, max: -0.5, step: 0.05, value: Math.log10(o.learningRate), digits: 2, ariaLabel: 'Learning rate, log10', onInput: (v) => ws.update({ optimizer: { ...ws.settings.optimizer, learningRate: Math.pow(10, v) } }) }).el,
      h('div', { class: 'muted' }, `η = ${formatValue(o.learningRate, 4)}`),
      batch,
      slider({ label: 'Gradient clipping (max norm, 0 = off)', min: 0, max: 10, step: 0.5, value: ws.settings.clip, digits: 1, ariaLabel: 'Gradient clipping', onInput: (v) => ws.update({ clip: v }) }).el,
    );
    this.draw();
  }

  private draw(): void {
    const history = this.ws.trainer.history;
    const xs = history.map((p) => p.step);
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
