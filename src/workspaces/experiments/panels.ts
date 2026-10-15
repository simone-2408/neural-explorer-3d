/** Left-panel sections of the Experiments workspace. */
import { DisposeBag } from '../../core/DisposeBag';
import { gpt2Tokenize } from '../../experiments/gpt2Engine';
import { METRICS } from '../../experiments/metrics';
import { TENSORS } from '../../experiments/toyEngine';
import type { ExperimentModel, Intervention, MetricId, TensorId, TokenRef } from '../../experiments/types';
import { LOCAL_SETUP_URL, ONLINE_DEMO } from '../../llm/client';
import { isSpecial } from '../../transformer/analysis';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { toast } from '../../ui/overlays';
import { button, note, section, segmented, type Segmented } from '../../ui/widgets';
import { formatCount, formatProbability } from '../../utils/format';
import type { Panel } from '../network/context';
import type { ExperimentSelection, ExperimentsWorkspace } from './ExperimentsWorkspace';

// ------------------------------------------------------------------ model

export class ModelPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly choice: Segmented<ExperimentModel>;
  private readonly body = h('div', { class: 'stack' });

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.choice = segmented<ExperimentModel>({
      options: [
        { value: 'toy', label: 'Toy Transformer', title: 'The tiny model of the Transformer tab, computed in your browser' },
        { value: 'gpt2', label: 'GPT-2 small', title: 'A real language model, computed by the local Python backend' },
      ],
      value: ws.modelKind,
      ariaLabel: 'Model',
      onChange: (v) => ws.setModel(v),
    });
    this.el = section({ id: 'ex-model', title: 'Model', children: [this.choice.el, this.body] });
    for (const e of ['model', 'result', 'busy'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    this.choice.set(ws.modelKind);
    if (ws.modelKind === 'toy') {
      const c = ws.toy?.config;
      replaceChildren(
        this.body,
        h('div', { class: 'muted' }, c ? `${c.nLayers} layers × ${c.nHeads} heads · d = ${c.dModel} · MLP ${c.dMlp} neurons · ${c.vocab.length} words. Every change is recomputed exactly, in about a millisecond.` : 'Loading…'),
      );
      return;
    }
    const s = ws.status;
    const state = !ws.checked ? 'checking' : !ws.online ? 'offline' : s?.error ? 'error' : s?.ready ? 'ready' : 'loading';
    const labels = { ready: 'GPT-2 small running locally', loading: 'GPT-2 small is loading…', error: 'The backend could not load the model', offline: ONLINE_DEMO ? 'Online demo — recorded presets only' : 'Backend not running — recorded presets only', checking: 'Looking for the local backend…' };
    replaceChildren(
      this.body,
      h('div', { class: `backend__status is-${state}` }, h('span', { class: 'backend__dot', 'aria-hidden': 'true' }), labels[state]),
      state === 'ready' && s?.config ? h('div', { class: 'muted' }, `${formatCount(s.config.parameters)} parameters · ${s.config.layers} layers × ${s.config.heads} heads · MLP ${formatCount(s.config.dMlp)} neurons`) : null,
      state === 'offline' && ONLINE_DEMO
        ? h(
            'div',
            { class: 'muted' },
            'The presets below were recorded with GPT-2 small. To run your own experiments, run the app on your computer with its GPT-2 backend',
            LOCAL_SETUP_URL ? [' (', h('a', { href: LOCAL_SETUP_URL, target: '_blank', rel: 'noopener' }, 'how'), ').'] : '.',
          )
        : state === 'offline'
        ? h('div', { class: 'muted' }, 'The presets below were recorded with GPT-2 small. To run your own experiments, start the backend: ', h('code', null, 'npm run llm'), ' (first time: ', h('code', null, 'npm run llm:setup'), ').')
        : null,
      state === 'error' ? note(s?.error ?? '', 'warning') : null,
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ----------------------------------------------------------------- prompt

export class PromptPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly input: HTMLTextAreaElement;
  private readonly run: HTMLButtonElement;
  private readonly chips = h('div', { class: 'token-chips', role: 'list', 'aria-label': 'Tokens' });
  private readonly info = h('div', { class: 'panel-note' });

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.input = h('textarea', { class: 'text-input text-input--area', rows: 2, spellcheck: false, maxlength: 400, 'aria-label': 'Text given to the model' });
    this.input.value = ws.experiment.text;
    this.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        ws.setText(this.input.value);
      }
    });
    this.input.addEventListener('blur', () => {
      if (ws.modelKind === 'toy') ws.setText(this.input.value);
    });
    this.run = button({ icon: 'play', label: 'Run', small: true, variant: 'primary', title: 'Use this text (Enter)', onClick: () => ws.setText(this.input.value) });
    this.el = section({
      id: 'ex-prompt',
      title: 'Text',
      children: [this.input, h('div', { class: 'button-row' }, this.run), h('div', { class: 'field-label' }, 'Tokens — click one to remove or replace it'), this.chips, this.info],
    });
    for (const e of ['model', 'experiment', 'result', 'selection', 'busy'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const r = ws.result;
    if (document.activeElement !== this.input && this.input.value !== ws.experiment.text) this.input.value = ws.experiment.text;
    this.input.readOnly = !ws.editable;
    this.run.disabled = !ws.editable || ws.busy;
    const selected = ws.selection?.kind === 'token' ? ws.selection.position : null;
    replaceChildren(
      this.chips,
      r?.tokens.map((t, p) => {
        const edits = ws.tokenEdits(p);
        const cls = ['tchip', 'tchip--button'];
        if (edits.removed) cls.push('is-removed');
        if (edits.embedding) cls.push('is-edited');
        if (p === selected) cls.push('is-selected');
        if (ws.modelKind === 'toy' && ws.toy && isSpecial(ws.toy, t.id)) cls.push('is-special');
        return h(
          'button',
          { type: 'button', role: 'listitem', class: cls.join(' '), title: edits.removed ? 'Removed by the experiment' : edits.embedding ? `Embedding replaced by “${ws.format(edits.embedding.token.text)}”` : `token id ${t.id} · position ${p}`, onClick: () => ws.select({ kind: 'token', position: p, stage: null }) },
          h('span', { class: 'tchip__text' }, ws.format(t.text)),
          edits.embedding ? h('span', { class: 'tchip__pos' }, `→ ${ws.format(edits.embedding.token.text)}`) : h('span', { class: 'tchip__pos' }, String(p)),
        );
      }),
    );
    replaceChildren(
      this.info,
      !r ? h('div', { class: 'muted' }, ws.busy ? 'Running…' : ws.modelKind === 'gpt2' && !ws.live ? 'This text has no recording: pick a preset below or start the backend.' : 'No result yet.') : null,
      r?.truncated ? note('The text was cut to its first tokens.', 'warning') : null,
      ws.modelKind === 'toy' && r?.tokens.some((t) => t.text === ws.toy?.config.unk) ? note('Some words are not in the toy vocabulary and became <unk>.', 'warning') : null,
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------------- measure

export class MeasurePanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly target = h('select', { class: 'select', 'aria-label': 'Target token' });
  private readonly contrast = h('select', { class: 'select', 'aria-label': 'Contrast token' });
  private readonly typed = h('input', { type: 'text', class: 'text-input', placeholder: 'another token, e.g. “ Paris”', maxlength: 40, 'aria-label': 'Type a token' });
  private readonly typedRow: HTMLElement;
  private readonly metric: Segmented<MetricId>;
  private readonly metricNote = h('div', { class: 'muted' });

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.target.addEventListener('change', () => ws.setTarget(this.refOf(this.target.value)));
    this.contrast.addEventListener('change', () => ws.setContrast(this.contrast.value === '' ? null : this.refOf(this.contrast.value)));
    this.metric = segmented<MetricId>({
      options: (['logp', 'logit', 'p', 'kl'] as MetricId[]).map((m) => ({ value: m, label: METRICS[m].label, title: METRICS[m].description })),
      value: ws.settings.metric,
      ariaLabel: 'Metric of the bars',
      onChange: (v) => ws.update({ metric: v }),
    });
    const setTyped = async (as: 'target' | 'contrast') => {
      const text = this.typed.value;
      if (!text.trim()) return;
      try {
        const tokens = await gpt2Tokenize(text);
        if (tokens.length !== 1) {
          toast(`“${text}” is ${tokens.length} tokens (${tokens.map((t) => `“${t.text}”`).join(' ')}): the measure needs a single token. GPT-2 words usually start with a space.`, 'warning', 6000);
          return;
        }
        const ref = { id: tokens[0].id, text: tokens[0].text };
        if (as === 'target') ws.setTarget(ref);
        else ws.setContrast(ref);
        this.typed.value = '';
      } catch (error) {
        toast(error instanceof Error ? error.message : String(error), 'error');
      }
    };
    this.typedRow = h(
      'div',
      { class: 'ex-typed' },
      this.typed,
      button({ label: 'Target', small: true, title: 'Measure this token', onClick: () => void setTyped('target') }),
      button({ label: 'Contrast', small: true, variant: 'ghost', title: 'Compare the target with this token', onClick: () => void setTyped('contrast') }),
    );
    this.el = section({
      id: 'ex-measure',
      title: 'Measure',
      children: [
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Target: the next token whose probability is measured'), this.target),
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Contrast (optional): measure target − contrast'), this.contrast),
        this.typedRow,
        h('div', { class: 'field-label' }, 'Bars show'),
        this.metric.el,
        this.metricNote,
      ],
    });
    for (const e of ['model', 'experiment', 'result', 'settings'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private refOf(value: string): TokenRef {
    const [id, ...text] = value.split(':');
    return { id: Number(id), text: text.join(':') };
  }

  private render(): void {
    const { ws } = this;
    const r = ws.result;
    const e = ws.experiment;
    const value = (t: TokenRef) => `${t.id}:${t.text}`;
    const label = (t: TokenRef, p?: number) => `${ws.format(t.text)}${p !== undefined ? `  ${formatProbability(p)}` : ''}`;
    const top = r?.before.top ?? [];
    const options = (selected: TokenRef | null, withNone: boolean) => {
      const seen = new Set<number>();
      const list: HTMLElement[] = [];
      if (withNone) list.push(h('option', { value: '' }, 'none'));
      const group = h('optgroup', { label: 'Most likely next tokens (original model)' });
      for (const t of top) {
        if (ws.modelKind === 'toy' && ws.toy && isSpecial(ws.toy, t.id)) continue;
        seen.add(t.id);
        group.append(h('option', { value: value(t) }, label(t, t.p)));
      }
      list.push(group);
      if (selected && !seen.has(selected.id)) {
        list.push(h('option', { value: value(selected) }, label(selected)));
        seen.add(selected.id);
      }
      if (ws.modelKind === 'toy' && ws.toy) {
        const model = ws.toy;
        const words = model.config.vocab
          .map((w, id) => ({ id, text: w }))
          .filter((t) => !isSpecial(model, t.id) && !seen.has(t.id))
          .sort((a, b) => a.text.localeCompare(b.text));
        list.push(h('optgroup', { label: 'Other words' }, words.map((t) => h('option', { value: value(t) }, t.text))));
      }
      return list;
    };
    const target = e.target ?? r?.target ?? null;
    replaceChildren(this.target, options(target, false));
    if (target) this.target.value = value(target);
    replaceChildren(this.contrast, options(e.contrast, true));
    this.contrast.value = e.contrast ? value(e.contrast) : '';
    this.target.disabled = this.contrast.disabled = !ws.editable || !r;
    this.typedRow.hidden = ws.modelKind !== 'gpt2' || !ws.live;
    this.metric.set(ws.metric);
    const logitButton = this.metric.el.querySelector<HTMLButtonElement>('[data-value="logit"]');
    if (logitButton) {
      logitButton.disabled = !r?.contrast;
      logitButton.title = r?.contrast ? METRICS.logit.description : 'Choose a contrast token first';
    }
    replaceChildren(this.metricNote, METRICS[ws.metric].description);
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------- interventions

type AddKind = 'head' | 'mlp' | 'neuron' | 'weight';

export class InterventionsPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly list = h('div', { class: 'ex-list' });
  private readonly add = h('div', { class: 'ex-add' });
  private readonly actions = h('div', { class: 'button-row' });
  private addKind: AddKind = 'head';
  private form = { layer: 0, head: 0, unit: 0, tensor: 'q' as TensorId };

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.el = section({
      id: 'ex-interventions',
      title: 'Interventions',
      children: [
        this.list,
        h('div', { class: 'field-label' }, 'Add'),
        this.add,
        this.actions,
        h('div', { class: 'muted' }, 'Tip: click a tile in the 3D view to inspect it; shift-click switches it off or on. Token edits: click a token.'),
      ],
    });
    for (const e of ['model', 'experiment', 'result', 'selection'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private selectionOf(i: Intervention): ExperimentSelection {
    switch (i.kind) {
      case 'head':
        return { kind: 'head', layer: i.layer, head: i.head };
      case 'mlp':
        return { kind: 'mlp', layer: i.layer };
      case 'neuron':
        return { kind: 'neuron', layer: i.layer, unit: i.unit };
      case 'embedding':
      case 'remove':
        return { kind: 'token', position: i.position, stage: null };
      case 'weight':
        return { kind: 'weight', id: i.id };
    }
  }

  private render(): void {
    const { ws } = this;
    const items = ws.experiment.interventions;
    const editable = ws.editable;
    const s = ws.selection;
    const isSelected = (i: Intervention) => JSON.stringify(this.selectionOf(i)) === JSON.stringify(s);
    replaceChildren(
      this.list,
      items.length
        ? items.map((i) => {
            const check = h('input', { type: 'checkbox', 'aria-label': 'Apply this intervention', disabled: !editable });
            check.checked = i.enabled;
            check.addEventListener('change', () => ws.toggleIntervention(i.id));
            return h(
              'div',
              { class: `ex-item${i.enabled ? '' : ' is-disabled'}${isSelected(i) ? ' is-selected' : ''}`, dataset: { kind: i.kind } },
              check,
              h('button', { type: 'button', class: 'ex-item__label', title: 'Inspect', onClick: () => ws.select(this.selectionOf(i)) }, h('span', { class: 'ex-item__kind' }, KIND_LABEL[i.kind]), ws.describeIntervention(i)),
              button({ icon: 'close', title: 'Remove this intervention', small: true, variant: 'ghost', disabled: !editable, onClick: () => ws.removeIntervention(i.id) }),
            );
          })
        : h('div', { class: 'muted' }, 'None: the model runs untouched. The bars still show what switching off each component alone would do.'),
    );
    this.renderAdd();
    replaceChildren(
      this.actions,
      button({ icon: 'reset', label: 'Undo', small: true, variant: 'ghost', title: 'Undo the last change (⌘/Ctrl+Z)', disabled: !ws.canUndo, onClick: () => ws.undo() }),
      button({ icon: 'close', label: 'Clear', small: true, variant: 'ghost', disabled: !editable || !items.length, onClick: () => ws.clearInterventions() }),
      button({ icon: 'download', label: 'Save', small: true, variant: 'ghost', title: 'Download this experiment as JSON', onClick: () => ws.save() }),
      button({ icon: 'upload', label: 'Load', small: true, variant: 'ghost', title: 'Load an experiment file', onClick: () => void ws.load() }),
    );
  }

  private renderAdd(): void {
    const { ws } = this;
    const L = ws.layers;
    const H = ws.heads;
    const f = this.form;
    f.layer = Math.min(f.layer, L - 1);
    f.head = Math.min(f.head, H - 1);
    f.unit = Math.min(f.unit, ws.units - 1);
    const kinds: { value: AddKind; label: string; title: string }[] = [
      { value: 'head', label: 'Head', title: 'Switch off (or scale) one attention head' },
      { value: 'mlp', label: 'MLP', title: 'Switch off (or scale) the MLP of one layer' },
      { value: 'neuron', label: 'Neuron', title: 'Zero, scale or set one MLP neuron' },
    ];
    if (ws.modelKind === 'toy') kinds.push({ value: 'weight', label: 'Weights', title: 'Scale a weight matrix or change one weight' });
    if (!kinds.some((k) => k.value === this.addKind)) this.addKind = 'head';
    const kind = segmented<AddKind>({ options: kinds, value: this.addKind, ariaLabel: 'Kind of intervention', onChange: (v) => ((this.addKind = v), this.renderAdd()) });
    const numberSelect = (label: string, count: number, value: number, onChange: (v: number) => void, base = 1) => {
      const el = h('select', { class: 'select select--compact', 'aria-label': label }, Array.from({ length: count }, (_, k) => h('option', { value: String(k) }, `${label} ${k + base}`)));
      el.value = String(value);
      el.addEventListener('change', () => onChange(Number(el.value)));
      return el;
    };
    const fields: HTMLElement[] = [];
    let add: () => void;
    switch (this.addKind) {
      case 'head':
        fields.push(numberSelect('layer', L, f.layer, (v) => (f.layer = v)), numberSelect('head', H, f.head, (v) => (f.head = v)));
        add = () => {
          ws.setComponentScale({ kind: 'head', layer: f.layer, head: f.head }, 0);
          ws.select({ kind: 'head', layer: f.layer, head: f.head });
        };
        break;
      case 'mlp':
        fields.push(numberSelect('layer', L, f.layer, (v) => (f.layer = v)));
        add = () => {
          ws.setComponentScale({ kind: 'mlp', layer: f.layer }, 0);
          ws.select({ kind: 'mlp', layer: f.layer });
        };
        break;
      case 'neuron': {
        const unit = h('input', { type: 'number', class: 'text-input text-input--number', min: 0, max: ws.units - 1, step: 1, 'aria-label': 'Neuron number' });
        unit.value = String(f.unit);
        unit.addEventListener('change', () => (f.unit = Math.max(0, Math.min(ws.units - 1, Math.round(Number(unit.value) || 0)))));
        fields.push(numberSelect('MLP', L, f.layer, (v) => (f.layer = v)), h('label', { class: 'ex-inline' }, '#', unit));
        add = () => {
          ws.setNeuron(f.layer, f.unit, 'scale', 0);
          ws.select({ kind: 'neuron', layer: f.layer, unit: f.unit });
        };
        break;
      }
      case 'weight': {
        const tensor = h('select', { class: 'select', 'aria-label': 'Weight matrix' }, TENSORS.map((t) => h('option', { value: t.id }, t.label)));
        tensor.value = f.tensor;
        tensor.addEventListener('change', () => ((f.tensor = tensor.value as TensorId), this.renderAdd()));
        const info = TENSORS.find((t) => t.id === f.tensor)!;
        fields.push(tensor);
        if (info.perLayer) fields.push(numberSelect('layer', L, f.layer, (v) => (f.layer = v)));
        if (info.perHead) fields.push(numberSelect('head', H, f.head, (v) => (f.head = v)));
        add = () => ws.addWeightEdit(f.tensor, info.perLayer ? f.layer : 0, info.perHead ? f.head : 0);
        break;
      }
    }
    replaceChildren(
      this.add,
      kind.el,
      h('div', { class: 'ex-add__fields' }, fields, button({ icon: 'plus', label: this.addKind === 'weight' ? 'Edit' : 'Switch off', small: true, disabled: !ws.editable, onClick: () => add() })),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

const KIND_LABEL: Record<Intervention['kind'], string> = {
  head: 'head',
  mlp: 'MLP',
  neuron: 'neuron',
  embedding: 'token',
  remove: 'token',
  weight: 'weights',
};

// ---------------------------------------------------------------- presets

export class PresetsPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly ws: ExperimentsWorkspace;
  private readonly list = h('div', { class: 'example-list' });

  constructor(ws: ExperimentsWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'ex-presets', title: 'Ready-made experiments', children: [this.list] });
    for (const e of ['model', 'experiment'] as const) this.bag.add(ws.events.on(e, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const presets = ws.presets();
    const current = ws.preset[ws.modelKind];
    replaceChildren(
      this.list,
      presets.map((p) =>
        h(
          'button',
          { type: 'button', class: `example${current?.id === p.id ? ' is-active' : ''}`, title: p.description, onClick: () => ws.loadPreset(p) },
          h('span', { class: 'example__label' }, p.label, ws.modelKind === 'gpt2' && !ws.live ? h('span', { class: 'example__tag', title: 'Recorded result' }, icon('server', 12)) : null),
          h('span', { class: 'example__text' }, p.experiment.text.replace(/\n/g, '⏎')),
        ),
      ),
      presets.length ? null : h('div', { class: 'muted' }, ws.modelKind === 'gpt2' ? 'No recorded presets found (public/llm-demo/experiments.json).' : 'Loading…'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}
