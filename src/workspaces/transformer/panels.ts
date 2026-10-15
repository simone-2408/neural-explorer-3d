/** Left-panel sections of the Transformer workspace. */
import { DisposeBag } from '../../core/DisposeBag';
import { unknownWords } from '../../transformer/tokenizer';
import { h, replaceChildren } from '../../ui/dom';
import { button, note, section, segmented, toggle, type Segmented, type Toggle } from '../../ui/widgets';
import { debounce } from '../../utils/schedule';
import type { Panel } from '../network/context';
import { GROUP_LABELS, groupColor } from './colors';
import { EXAMPLES, type TransformerWorkspace } from './TransformerWorkspace';
import { swatch } from './widgets';

// ------------------------------------------------------------------- text

export class TextPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ws: TransformerWorkspace;
  private readonly bag = new DisposeBag();
  private readonly input: HTMLInputElement;
  private readonly chips = h('div', { class: 'token-chips', role: 'list', 'aria-label': 'Tokens' });
  private readonly info = h('div', { class: 'panel-note' });
  private readonly undo: HTMLButtonElement;

  constructor(ws: TransformerWorkspace) {
    this.ws = ws;
    this.input = h('input', { type: 'text', class: 'text-input', spellcheck: false, autocomplete: 'off', maxlength: 200, 'aria-label': 'Text given to the model' });
    this.input.value = ws.text;
    const apply = debounce(() => ws.setText(this.input.value), 140);
    this.input.addEventListener('input', () => apply());
    this.input.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      apply.cancel();
      ws.setText(this.input.value);
    });
    this.bag.add(() => apply.cancel());
    this.undo = button({ icon: 'reset', label: 'Undo', small: true, variant: 'ghost', title: 'Remove the last generated word', onClick: () => ws.undo() });
    const generate = h(
      'div',
      { class: 'button-row' },
      button({ icon: 'stepForward', label: 'Most likely', small: true, title: 'Append the most likely next word (greedy decoding)', onClick: () => ws.generate('greedy') }),
      button({ icon: 'dice', label: 'Sample', small: true, title: 'Append a word drawn at random from the next-word probabilities (temperature, top-k, top-p in the prediction inspector)', onClick: () => ws.generate('sample') }),
      this.undo,
    );
    this.el = section({
      id: 'tt-text',
      title: 'Text',
      children: [
        this.input,
        h('div', { class: 'field-label' }, 'Next word'),
        generate,
        h('div', { class: 'field-label' }, 'Tokens'),
        this.chips,
        this.info,
      ],
    });
    this.bag.add(ws.events.on('trace', () => this.render()));
    this.bag.add(ws.events.on('focus', () => this.render()));
    this.bag.add(ws.events.on('selection', () => this.render()));
    this.render();
  }

  private render(): void {
    const { ws } = this;
    const trace = ws.trace;
    const model = ws.model;
    if (!trace || !model) return;
    if (document.activeElement !== this.input && this.input.value !== ws.text) this.input.value = ws.text;
    this.undo.disabled = !ws.canUndo;
    const selected = ws.selection?.kind === 'token' ? ws.selection.position : null;
    replaceChildren(
      this.chips,
      trace.tokens.map((token, p) =>
        h(
          'button',
          {
            type: 'button',
            role: 'listitem',
            class: `tchip tchip--button${token.unknown ? ' is-unknown' : ''}${p === 0 ? ' is-special' : ''}${p === ws.focus ? ' is-focus' : ''}${p === selected ? ' is-selected' : ''}`,
            title: token.unknown ? `“${ws.text.slice(token.start, token.end)}” is not in the vocabulary: the model sees <unk>` : `token id ${token.id} · position ${p}`,
            onClick: () => {
              ws.setFocus(p);
              ws.select({ kind: 'token', position: p, stage: ws.stages - 1 });
            },
          },
          h('span', { class: 'tchip__text' }, token.text),
          h('span', { class: 'tchip__pos' }, String(token.id)),
        ),
      ),
    );
    const unknown = unknownWords(ws.text, trace.tokens);
    const words = ws.text.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]/g)?.length ?? 0;
    const truncated = words + 1 > model.config.context;
    replaceChildren(
      this.info,
      h('div', null, `${trace.tokens.length} tokens (at most ${model.config.context}) · chips show the token id. Click one to inspect it.`),
      unknown.length ? note(`Not in the ${model.config.vocab.length}-word vocabulary: ${unknown.map((w) => `“${w}”`).join(', ')} → <unk>. The model knows nothing about these words.`, 'warning') : null,
      truncated ? note(`Only the first ${model.config.context - 1} words are used (context window).`, 'warning') : null,
      h('div', { class: 'muted' }, 'A word-level tokenizer: one token per word or punctuation mark. Real LLMs split text into sub-word pieces (see the LLM tab).'),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// --------------------------------------------------------------- examples

export class ExamplesPanel implements Panel {
  readonly el: HTMLElement;

  constructor(ws: TransformerWorkspace) {
    this.el = section({
      id: 'tt-examples',
      title: 'Examples',
      children: [
        h(
          'div',
          { class: 'example-list' },
          EXAMPLES.map((ex) =>
            h(
              'button',
              { type: 'button', class: 'example', title: ex.description, onClick: () => ws.loadExample(ex.text) },
              h('span', { class: 'example__label' }, ex.label),
              h('span', { class: 'example__text' }, ex.text),
            ),
          ),
        ),
        h('div', { class: 'muted' }, `The model only knows the sentences of its tiny training world (${ws.model?.training.templates.length ?? 10} templates, see the overview).`),
      ],
    });
  }

  dispose(): void {}
}

// -------------------------------------------------------------- attention

type LayerChoice = 'all' | `${number}`;

export class AttentionPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ws: TransformerWorkspace;
  private readonly bag = new DisposeBag();
  private readonly layer: Segmented<LayerChoice>;
  private readonly head: Segmented<LayerChoice>;
  private readonly queries: Segmented<'focus' | 'all'>;
  private readonly weak: Toggle;
  private readonly focusSelect: HTMLSelectElement;

  constructor(ws: TransformerWorkspace) {
    this.ws = ws;
    const { nLayers, nHeads } = ws.model!.config;
    const toChoice = (v: number | 'all'): LayerChoice => (v === 'all' ? 'all' : `${v}`);
    const fromChoice = (v: LayerChoice): number | 'all' => (v === 'all' ? 'all' : Number(v));
    this.layer = segmented<LayerChoice>({
      options: [...Array.from({ length: nLayers }, (_, l) => ({ value: `${l}` as LayerChoice, label: `Layer ${l + 1}` })), { value: 'all', label: 'Both' }],
      value: toChoice(ws.settings.arcLayer),
      ariaLabel: 'Layer of the arcs',
      onChange: (v) => ws.update({ arcLayer: fromChoice(v) }),
    });
    this.head = segmented<LayerChoice>({
      options: [...Array.from({ length: nHeads }, (_, k) => ({ value: `${k}` as LayerChoice, label: `H${k + 1}`, title: `Head ${k + 1}` })), { value: 'all', label: 'All' }],
      value: toChoice(ws.settings.arcHead),
      ariaLabel: 'Head of the arcs',
      onChange: (v) => ws.update({ arcHead: fromChoice(v) }),
    });
    this.queries = segmented<'focus' | 'all'>({
      options: [
        { value: 'focus', label: 'One token', title: 'Only the arcs of the focused token' },
        { value: 'all', label: 'Every token', title: 'The arcs of every token (busy!)' },
      ],
      value: ws.settings.arcQueries,
      ariaLabel: 'Whose arcs',
      onChange: (v) => ws.update({ arcQueries: v }),
    });
    this.focusSelect = h('select', { class: 'select', 'aria-label': 'Focused token' });
    this.focusSelect.addEventListener('change', () => ws.setFocus(Number(this.focusSelect.value)));
    this.weak = toggle({ label: 'Hide arcs below 5 %', checked: ws.settings.hideWeak, onChange: (v) => ws.update({ hideWeak: v }) });
    const legend = h('div', { class: 'head-legend' });
    const renderLegend = () =>
      replaceChildren(
        legend,
        Array.from({ length: nHeads }, (_, k) => h('span', { class: 'head-legend__item' }, swatch(ws.headColor(k)), `head ${k + 1}`)),
      );
    renderLegend();

    this.el = section({
      id: 'tt-attention',
      title: 'Attention arcs',
      children: [
        h('div', { class: 'field-label' }, 'Layer'),
        this.layer.el,
        h('div', { class: 'field-label' }, 'Head'),
        this.head.el,
        legend,
        h('div', { class: 'field-label' }, 'Arcs of'),
        this.queries.el,
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Focused token'), this.focusSelect),
        this.weak.el,
        h('div', { class: 'muted' }, 'An arc goes from a token that is read (lower row) to the token reading it (upper row). Thickness and opacity = attention weight.'),
      ],
    });
    this.bag.add(
      ws.events.on('settings', () => {
        this.sync();
        renderLegend();
      }),
    );
    this.bag.add(ws.events.on('trace', () => this.sync()));
    this.bag.add(ws.events.on('focus', () => this.sync()));
    this.sync();
  }

  private sync(): void {
    const { ws } = this;
    const st = ws.settings;
    this.layer.set(st.arcLayer === 'all' ? 'all' : `${st.arcLayer}`);
    this.head.set(st.arcHead === 'all' ? 'all' : `${st.arcHead}`);
    this.queries.set(st.arcQueries);
    this.weak.set(st.hideWeak);
    const tokens = ws.trace?.tokens ?? [];
    replaceChildren(
      this.focusSelect,
      tokens.map((t, p) => h('option', { value: String(p) }, `${p} · ${t.text}`)),
    );
    this.focusSelect.value = String(ws.focus);
    this.focusSelect.disabled = st.arcQueries === 'all';
  }

  dispose(): void {
    this.bag.dispose();
  }
}

// ----------------------------------------------------------------- view

export class DisplayPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ws: TransformerWorkspace;
  private readonly bag = new DisposeBag();
  private readonly mode: Segmented<'model' | 'embeddings'>;
  private readonly points: Segmented<'vocabulary' | 'sentence'>;
  private readonly space: Segmented<'normalized' | 'raw'>;
  private readonly everywhere: Toggle;
  private readonly embeddingOptions: HTMLElement;
  private readonly spaceRow: HTMLElement;
  private readonly legend = h('div', { class: 'tt-legend' });

  constructor(ws: TransformerWorkspace) {
    this.ws = ws;
    this.mode = segmented<'model' | 'embeddings'>({
      options: [
        { value: 'model', label: '3D model', title: 'The computation: residual stream, attention, MLP (E)' },
        { value: 'embeddings', label: 'Embedding space', title: 'Where the vectors lie, projected to 3-D (E)' },
      ],
      value: ws.settings.view,
      ariaLabel: 'View',
      onChange: (v) => ws.update({ view: v }),
    });
    this.points = segmented<'vocabulary' | 'sentence'>({
      options: [
        { value: 'vocabulary', label: 'Vocabulary', title: 'The embedding of every word of the vocabulary' },
        { value: 'sentence', label: 'This sentence', title: 'The vectors of this sentence, moving layer after layer (follows the pipeline)' },
      ],
      value: ws.settings.embedding,
      ariaLabel: 'Points',
      onChange: (v) => ws.update({ embedding: v }),
    });
    this.space = segmented<'normalized' | 'raw'>({
      options: [
        { value: 'normalized', label: 'Normalized', title: 'Each vector rescaled to zero mean and unit variance, as LayerNorm does before every sub-layer' },
        { value: 'raw', label: 'Raw', title: 'The residual stream as it is: its size grows layer after layer' },
      ],
      value: ws.settings.residualSpace,
      ariaLabel: 'Vectors',
      onChange: (v) => ws.update({ residualSpace: v }),
    });
    this.everywhere = toggle({ label: 'Predictions at every position', checked: ws.settings.predictionsEverywhere, onChange: (v) => ws.update({ predictionsEverywhere: v }) });
    this.spaceRow = h('div', null, h('div', { class: 'field-label' }, 'Vectors'), this.space.el);
    this.embeddingOptions = h('div', { class: 'stack' }, h('div', { class: 'field-label' }, 'Points'), this.points.el, this.spaceRow);

    this.el = section({ id: 'tt-view', title: 'View', children: [this.mode.el, this.embeddingOptions, this.everywhere.el, this.legend] });
    this.bag.add(ws.events.on('settings', () => this.sync()));
    this.bag.add(ws.events.on('trace', () => this.sync()));
    this.sync();
  }

  private sync(): void {
    const { ws } = this;
    const st = ws.settings;
    this.mode.set(st.view);
    this.points.set(st.embedding);
    this.space.set(st.residualSpace);
    this.everywhere.set(st.predictionsEverywhere);
    this.embeddingOptions.hidden = st.view !== 'embeddings';
    this.spaceRow.hidden = st.embedding !== 'sentence';
    this.everywhere.el.hidden = st.view !== 'model';
    const p = ws.palette;
    if (st.view === 'model') {
      replaceChildren(
        this.legend,
        h('div', { class: 'tt-legend__row' }, h('span', { class: 'tt-legend__bar is-pos' }), h('span', null, `bar up = positive (${p.positiveName})`)),
        h('div', { class: 'tt-legend__row' }, h('span', { class: 'tt-legend__bar is-neg' }), h('span', null, `bar down = negative (${p.negativeName})`)),
        h('div', { class: 'muted' }, 'Each slab is one vector of 32 numbers (one bar per number, along the depth). Bar height ∝ |value|; the tallest bars are capped.'),
      );
    } else {
      const model = ws.model;
      const groups = Object.keys(GROUP_LABELS);
      replaceChildren(
        this.legend,
        st.embedding === 'vocabulary'
          ? h(
              'div',
              { class: 'group-legend' },
              groups.map((g) => h('span', { class: 'group-legend__item' }, swatch(groupColor(g, p.theme)), GROUP_LABELS[g])),
            )
          : null,
        note(
          st.embedding === 'vocabulary'
            ? `A projection (PCA) of ${model?.config.dModel ?? 32}-dimensional vectors onto the 3 directions of largest variance: the axis labels say how much of the variance each keeps. Points that look close may be far apart in the full space — the word inspector lists the true nearest words.`
            : 'Each token’s vector at every stage, projected with one PCA fitted on all of them (<s> excluded). Use the pipeline (bottom) to move through the layers. A projection: distances are only approximate.',
        ),
      );
    }
  }

  dispose(): void {
    this.bag.dispose();
  }
}
