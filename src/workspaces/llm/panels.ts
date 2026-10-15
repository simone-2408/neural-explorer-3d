/** Left-panel sections of the LLM workspace. */
import { DisposeBag } from '../../core/DisposeBag';
import { LOCAL_SETUP_URL, ONLINE_DEMO, showToken } from '../../llm/client';
import { h, replaceChildren } from '../../ui/dom';
import { toast } from '../../ui/overlays';
import { button, note, section, segmented, toggle, type Segmented, type Toggle } from '../../ui/widgets';
import { formatCount } from '../../utils/format';
import type { Panel } from '../network/context';
import type { LlmWorkspace } from './LlmWorkspace';

// ---------------------------------------------------------------- backend

export class BackendPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly body = h('div', { class: 'backend' });

  private readonly ws: LlmWorkspace;

  constructor(ws: LlmWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'llm-backend', title: 'Model', children: [this.body] });
    this.bag.add(ws.events.on('status', () => this.render()));
    this.bag.add(ws.events.on('analysis', () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const s = ws.status;
    let state: 'ready' | 'loading' | 'error' | 'offline' | 'checking';
    if (!ws.checked) state = 'checking';
    else if (!ws.online) state = 'offline';
    else if (s?.error) state = 'error';
    else if (s?.ready) state = 'ready';
    else state = 'loading';
    const labels = {
      ready: `GPT-2 small running locally`,
      loading: 'GPT-2 small is loading…',
      error: 'The backend could not load the model',
      offline: ONLINE_DEMO ? 'Online demo — recorded examples' : 'Backend not running — recorded examples',
      checking: 'Looking for the local backend…',
    };
    const copy = (text: string) => () => {
      void navigator.clipboard?.writeText(text).then(
        () => toast('Copied to the clipboard', 'success'),
        () => toast(text, 'info', 6000),
      );
    };
    replaceChildren(
      this.body,
      h('div', { class: `backend__status is-${state}` }, h('span', { class: 'backend__dot', 'aria-hidden': 'true' }), labels[state]),
      state === 'ready' && s?.config
        ? h('div', { class: 'muted' }, `${formatCount(s.config.parameters)} parameters · ${s.config.layers} layers × ${s.config.heads} heads · d = ${s.config.dModel} · vocabulary ${formatCount(s.config.vocab)} · up to ${s.config.maxTokens} tokens here`)
        : null,
      state === 'error' ? note(s?.error ?? '', 'warning') : null,
      state === 'offline' && ONLINE_DEMO
        ? h(
            'div',
            { class: 'muted' },
            'This online version has no Python backend: the examples below were computed with GPT-2 small in advance. To analyze your own text, run the app on your computer with its GPT-2 backend',
            LOCAL_SETUP_URL ? [' (', h('a', { href: LOCAL_SETUP_URL, target: '_blank', rel: 'noopener' }, 'how'), ').'] : '.',
          )
        : state === 'offline'
        ? h(
            'div',
            { class: 'stack' },
            h('div', { class: 'muted' }, 'The examples below were computed with GPT-2 small in advance. To analyze your own text, start the Python backend in a terminal (first time: install it).'),
            h('div', { class: 'command' }, h('code', null, 'npm run llm:setup'), button({ icon: 'copy', title: 'Copy', small: true, variant: 'ghost', onClick: copy('npm run llm:setup') })),
            h('div', { class: 'command' }, h('code', null, 'npm run llm'), button({ icon: 'copy', title: 'Copy', small: true, variant: 'ghost', onClick: copy('npm run llm') })),
            h('div', { class: 'muted' }, 'GPT-2 small (≈ 550 MB) is downloaded from Hugging Face the first time. This tab notices when the backend starts.'),
          )
        : null,
      ws.analysis ? h('div', { class: 'muted' }, ws.analysis.source === 'live' ? `Analysis computed live in ${ws.analysis.seconds.toFixed(2)} s.` : 'Showing a recorded analysis.') : null,
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
  private readonly input: HTMLTextAreaElement;
  private readonly runButton: HTMLButtonElement;
  private readonly chips = h('div', { class: 'token-chips', role: 'list', 'aria-label': 'Tokens' });
  private readonly info = h('div', { class: 'panel-note' });
  private readonly generate = h('div', { class: 'button-row' });

  private readonly ws: LlmWorkspace;

  constructor(ws: LlmWorkspace) {
    this.ws = ws;
    this.input = h('textarea', { class: 'text-input text-input--area', rows: 3, spellcheck: false, maxlength: 600, 'aria-label': 'Prompt for GPT-2' });
    this.input.value = ws.text;
    this.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        void ws.run(this.input.value);
      }
    });
    this.runButton = button({ icon: 'play', label: 'Analyze', small: true, variant: 'primary', title: 'Run GPT-2 on this text (⌘/Ctrl + Enter)', onClick: () => void ws.run(this.input.value) });
    this.el = section({
      id: 'llm-prompt',
      title: 'Prompt',
      children: [this.input, h('div', { class: 'button-row' }, this.runButton), h('div', { class: 'field-label' }, 'Tokens (byte-pair encoding)'), this.chips, this.info, h('div', { class: 'field-label' }, 'Continue the text'), this.generate],
    });
    for (const event of ['analysis', 'status', 'busy', 'selection'] as const) this.bag.add(ws.events.on(event, () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const a = ws.analysis;
    if (document.activeElement !== this.input && this.input.value !== ws.text) this.input.value = ws.text;
    this.runButton.disabled = ws.busy || !ws.live;
    this.input.readOnly = !ws.live;
    this.runButton.title = ws.live ? 'Run GPT-2 on this text (⌘/Ctrl + Enter)' : 'Start the backend to analyze your own text';
    const selected = ws.selection?.kind === 'token' ? ws.selection.position : ws.selection?.kind === 'cell' ? ws.selection.position : null;
    replaceChildren(
      this.chips,
      a?.tokens.map((t, p) =>
        h(
          'button',
          {
            type: 'button',
            role: 'listitem',
            class: `tchip tchip--button${p === selected ? ' is-selected' : ''}${t.text.includes('�') ? ' is-unknown' : ''}`,
            title: `id ${t.id} · BPE piece “${t.piece}” · bytes ${t.bytes}`,
            onClick: () => ws.select({ kind: 'token', position: p }),
          },
          h('span', { class: 'tchip__text' }, showToken(t.text)),
          h('span', { class: 'tchip__pos' }, String(t.id)),
        ),
      ),
    );
    replaceChildren(
      this.info,
      a ? h('div', null, `${a.T} tokens · “·” marks a space that belongs to the token. Click a token to inspect it.`) : h('div', { class: 'muted' }, ws.busy ? 'Working…' : 'No analysis yet.'),
      a?.truncated ? note('The text was cut to the first 48 tokens.', 'warning') : null,
    );
    replaceChildren(
      this.generate,
      ws.live
        ? [
            button({ icon: 'stepForward', label: 'Most likely', small: true, disabled: ws.busy, title: `Append ${ws.settings.maxNewTokens} tokens, always the most likely one (greedy)`, onClick: () => void ws.generate('greedy') }),
            button({ icon: 'dice', label: 'Sample', small: true, disabled: ws.busy, title: 'Append tokens drawn at random (temperature, top-k, top-p in the prediction inspector)', onClick: () => void ws.generate('sample') }),
            button({ icon: 'reset', label: 'Undo', small: true, variant: 'ghost', disabled: ws.busy || !ws.canUndo, onClick: () => void ws.undo() }),
          ]
        : h('div', { class: 'muted' }, 'Generation needs the backend. The prediction inspector shows the continuations recorded with the example.'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// --------------------------------------------------------------- examples

export class LlmExamplesPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly list = h('div', { class: 'example-list' });

  private readonly ws: LlmWorkspace;

  constructor(ws: LlmWorkspace) {
    this.ws = ws;
    this.el = section({ id: 'llm-examples', title: 'Examples', children: [this.list] });
    this.bag.add(ws.events.on('status', () => this.render()));
    this.bag.add(ws.events.on('analysis', () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    replaceChildren(
      this.list,
      ws.demos.map((d) => {
        const example = h(
          'button',
          { type: 'button', class: `example${ws.text === d.text ? ' is-active' : ''}`, title: d.description, onClick: () => void ws.openDemo(d) },
          h('span', { class: 'example__label' }, d.label),
          h('span', { class: 'example__text' }, d.text.replace(/\n/g, '⏎')),
        );
        const experiment = d.experiment;
        return experiment
          ? h('div', { class: 'example-group' }, example, button({ icon: 'flask', small: true, variant: 'ghost', title: 'Open the experiment on this text (Experiments tab)', onClick: () => ws.openExperiment(experiment) }))
          : example;
      }),
      ws.demos.length ? null : h('div', { class: 'muted' }, 'No recorded examples found (public/llm-demo).'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ---------------------------------------------------------------- display

export class LlmDisplayPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly view: Segmented<'tower' | 'journey'>;
  private readonly arcs: Toggle;
  private readonly queries: Segmented<'focus' | 'all'>;
  private readonly words: Toggle;
  private readonly head: HTMLSelectElement;
  private readonly towerOptions: HTMLElement;

  private readonly ws: LlmWorkspace;

  constructor(ws: LlmWorkspace) {
    this.ws = ws;
    this.view = segmented<'tower' | 'journey'>({
      options: [
        { value: 'tower', label: 'Layers', title: 'Logit-lens tower: what each layer would predict (J)' },
        { value: 'journey', label: 'Token journey', title: 'Each token’s vector moving through the layers, projected to 3-D (J)' },
      ],
      value: ws.settings.view,
      ariaLabel: 'View',
      onChange: (v) => ws.update({ view: v }),
    });
    this.words = toggle({ label: 'Words on every box', title: 'Otherwise only on the selected layer and the selected token', checked: ws.settings.showWords, onChange: (v) => ws.update({ showWords: v }) });
    this.arcs = toggle({ label: 'Attention arcs', checked: ws.settings.arcsOn, onChange: (v) => ws.update({ arcsOn: v }) });
    this.head = h('select', { class: 'select', 'aria-label': 'Head' });
    this.head.addEventListener('change', () => ws.update({ arcHead: Number(this.head.value) }));
    this.queries = segmented<'focus' | 'all'>({
      options: [
        { value: 'focus', label: 'One token' },
        { value: 'all', label: 'Every token' },
      ],
      value: ws.settings.arcQueries,
      ariaLabel: 'Arcs of',
      onChange: (v) => ws.update({ arcQueries: v }),
    });
    this.towerOptions = h(
      'div',
      { class: 'stack' },
      this.words.el,
      h(
        'div',
        { class: 'tt-legend' },
        h('div', { class: 'muted' }, 'Box height and colour: how probable the model’s final answer already is at that layer. Orange boxes: that layer’s top word is already the final answer.'),
      ),
      this.arcs.el,
      h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Head (of the block that produced the selected layer)'), this.head),
      this.queries.el,
    );
    this.el = section({ id: 'llm-view', title: 'View', children: [this.view.el, this.towerOptions] });
    this.bag.add(ws.events.on('settings', () => this.sync()));
    this.bag.add(ws.events.on('analysis', () => this.sync()));
    this.sync();
  }

  private sync(): void {
    const { ws } = this;
    const st = ws.settings;
    this.view.set(st.view);
    this.words.set(st.showWords);
    this.arcs.set(st.arcsOn);
    this.queries.set(st.arcQueries);
    const heads = ws.analysis?.heads ?? 12;
    if (this.head.options.length !== heads) replaceChildren(this.head, Array.from({ length: heads }, (_, k) => h('option', { value: String(k) }, `head ${k + 1}`)));
    this.head.value = String(st.arcHead);
    this.towerOptions.hidden = st.view !== 'tower';
  }

  dispose(): void {
    this.bag.dispose();
  }
}
