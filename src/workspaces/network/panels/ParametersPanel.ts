/** Parameters panel: initialization scheme, seed, re-initialization and model file. */
import { DisposeBag } from '../../../core/DisposeBag';
import { randomSeed } from '../../../math/random';
import { INIT_SCHEMES, type InitScheme } from '../../../neural/init';
import { h, replaceChildren } from '../../../ui/dom';
import { tex } from '../../../ui/tex';
import { button, kv, section, select } from '../../../ui/widgets';
import { formatCount } from '../../../utils/format';
import type { NetworkWorkspaceContext, Panel } from '../context';

export class ParametersPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkWorkspaceContext;
  private readonly bag = new DisposeBag();
  private readonly schemeSelect: HTMLSelectElement;
  private readonly seedInput: HTMLInputElement;
  private readonly formula = h('div', { class: 'formula-box' });
  private readonly description = h('p', { class: 'panel-text' });
  private readonly counts = h('div');

  constructor(ctx: NetworkWorkspaceContext) {
    this.ctx = ctx;
    const { model, actions } = ctx;

    this.schemeSelect = select<InitScheme>({
      options: Object.values(INIT_SCHEMES).map((s) => ({ value: s.id, label: s.label })),
      value: model.init.scheme,
      ariaLabel: 'Weight initialization scheme',
      onChange: (scheme) => model.reinitialize({ scheme }),
    });
    this.seedInput = h('input', { type: 'number', class: 'input input--seed', min: 0, step: 1, value: model.init.seed, 'aria-label': 'Random seed' });
    this.seedInput.addEventListener('change', () => {
      const seed = Math.max(0, Math.floor(Number(this.seedInput.value)));
      if (Number.isFinite(seed)) model.reinitialize({ seed });
    });

    this.el = section({
      id: 'parameters',
      title: 'Parameters',
      children: [
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Initialization'), this.schemeSelect),
        this.formula,
        this.description,
        h(
          'div',
          { class: 'field field--inline' },
          h('span', { class: 'field__label' }, 'Seed'),
          this.seedInput,
          button({ icon: 'dice', title: 'New random seed', small: true, onClick: () => model.reinitialize({ seed: randomSeed() }) }),
        ),
        h(
          'div',
          { class: 'button-row' },
          button({
            label: 'Reset weights',
            icon: 'refresh',
            small: true,
            title: 'Re-draw all weights and biases from the scheme and seed (undoes manual edits)',
            onClick: () => model.reinitialize(),
          }),
        ),
        this.counts,
        h('h4', { class: 'field__label' }, 'Model file'),
        h(
          'div',
          { class: 'button-row' },
          button({ label: 'Export', icon: 'download', small: true, title: 'Download architecture + weights + input as JSON', onClick: () => actions.exportModel() }),
          button({ label: 'Import', icon: 'upload', small: true, title: 'Load a model file or a bare configuration', onClick: () => void actions.importModel() }),
          button({ label: 'Copy config', icon: 'copy', small: true, title: 'Copy the JSON configuration', onClick: () => void actions.copyConfig() }),
        ),
        h('p', { class: 'panel-text muted' }, 'Model files load in PyTorch with python/mlp_bridge.py (see README).'),
      ],
    });

    this.bag.add(model.events.on('params', () => this.render()));
    this.bag.add(model.events.on('structure', () => this.render()));
    this.render();
  }

  dispose(): void {
    this.bag.dispose();
  }

  private render(): void {
    const { model } = this.ctx;
    const scheme = INIT_SCHEMES[model.init.scheme];
    this.schemeSelect.value = scheme.id;
    if (document.activeElement !== this.seedInput) this.seedInput.value = String(model.init.seed);
    replaceChildren(this.formula, tex(scheme.tex, { display: true }));
    this.description.textContent = scheme.description;
    const counts = model.parameterCounts();
    const weights = counts.reduce((s, c) => s + c.weights, 0);
    const biases = counts.reduce((s, c) => s + c.biases, 0);
    replaceChildren(this.counts, kv('Weights', formatCount(weights)), kv('Biases', formatCount(biases)), kv('Total parameters', formatCount(weights + biases), 'kv--strong'));
  }
}
