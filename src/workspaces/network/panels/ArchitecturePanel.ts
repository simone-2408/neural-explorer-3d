/**
 * Architecture panel: presets, an editable layer list and the JSON
 * configuration editor. Every change goes through a NetworkConfig, so the
 * scene is always generated from configuration.
 */
import { DisposeBag } from '../../../core/DisposeBag';
import { ACTIVATION_IDS, ACTIVATIONS, type ActivationId } from '../../../math/activations';
import { formatConfig, LIMITS, parseConfigText, specToConfig } from '../../../neural/config';
import { findPreset, PRESETS } from '../../../neural/presets';
import { h, replaceChildren } from '../../../ui/dom';
import { icon } from '../../../ui/icons';
import { button, section, select } from '../../../ui/widgets';
import { formatCount } from '../../../utils/format';
import { debounce } from '../../../utils/schedule';
import type { NetworkWorkspaceContext, Panel } from '../context';

export class ArchitecturePanel implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkWorkspaceContext;
  private readonly bag = new DisposeBag();
  private readonly presetSelect: HTMLSelectElement;
  private readonly presetDescription = h('p', { class: 'panel-text' });
  private readonly layerList = h('div', { class: 'layer-list', role: 'list' });
  private readonly summary = h('span', { class: 'section__count' });
  private readonly editor: ConfigEditor;

  constructor(ctx: NetworkWorkspaceContext) {
    this.ctx = ctx;
    const { store, model, actions } = ctx;

    this.presetSelect = select({
      options: [{ value: 'custom', label: 'Custom (edited)' }, ...PRESETS.map((p) => ({ value: p.id, label: p.label }))],
      value: store.state.presetId ?? 'custom',
      ariaLabel: 'Network preset',
      onChange: (id) => {
        if (id !== 'custom') actions.applyPreset(id);
      },
    });

    this.editor = new ConfigEditor(ctx);
    this.el = section({
      id: 'architecture',
      title: 'Architecture',
      aside: this.summary,
      children: [
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Preset'), this.presetSelect),
        this.presetDescription,
        h('div', { class: 'field__label field__label--row' }, h('span', null, 'Layers'), h('span', { class: 'muted' }, 'size · activation')),
        this.layerList,
        button({ label: 'Add hidden layer', icon: 'plus', small: true, class: 'btn--block', onClick: () => actions.addHiddenLayer() }),
        this.editor.el,
      ],
    });

    this.bag.add(model.events.on('structure', () => this.render()));
    this.bag.add(store.watch((s) => s.presetId, () => this.renderPreset()));
    this.bag.add(store.watch((s) => s.selection, () => this.highlightSelection()));
    this.render();
  }

  dispose(): void {
    this.bag.dispose();
    this.editor.dispose();
  }

  private renderPreset(): void {
    const id = this.ctx.store.state.presetId;
    this.presetSelect.value = id ?? 'custom';
    this.presetDescription.textContent = id ? (findPreset(id)?.description ?? '') : 'Edited architecture (not a preset).';
  }

  private render(): void {
    const { model, actions } = this.ctx;
    const { spec } = model;
    this.renderPreset();
    this.summary.textContent = `${formatCount(model.totalParameters())} params`;

    const rows = spec.layers.map((layer, l) => {
      const sizeInput = h('input', {
        type: 'number',
        class: 'stepper__value',
        min: 1,
        max: LIMITS.maxLayerSize,
        value: layer.size,
        'aria-label': `${layer.name} size`,
      });
      sizeInput.addEventListener('change', () => {
        const size = Math.round(Number(sizeInput.value));
        if (Number.isFinite(size) && size >= 1 && size !== layer.size) actions.setLayerSize(l, size);
        else sizeInput.value = String(layer.size);
      });
      const stepper = h(
        'div',
        { class: 'stepper' },
        h('button', { type: 'button', class: 'stepper__btn', 'aria-label': `Remove a neuron from ${layer.name}`, disabled: layer.size <= 1, onClick: () => actions.setLayerSize(l, layer.size - 1) }, icon('minus', 12)),
        sizeInput,
        h('button', { type: 'button', class: 'stepper__btn', 'aria-label': `Add a neuron to ${layer.name}`, disabled: layer.size >= LIMITS.maxLayerSize, onClick: () => actions.setLayerSize(l, layer.size + 1) }, icon('plus', 12)),
      );
      const activation =
        layer.role === 'input'
          ? h('span', { class: 'layer-row__static muted' }, layer.shape ? `${layer.shape[0]}×${layer.shape[1]}` : 'input')
          : select<ActivationId>({
              options: ACTIVATION_IDS.filter((id) => id !== 'softmax' || layer.role === 'output').map((id) => ({ value: id, label: ACTIVATIONS[id].label })),
              value: layer.activation,
              ariaLabel: `${layer.name} activation`,
              onChange: (id) => actions.setLayerActivation(l, id),
            });
      const remove =
        layer.role === 'hidden'
          ? h(
              'button',
              { type: 'button', class: 'btn btn--ghost btn--icon btn--sm layer-row__remove', title: `Remove ${layer.name}`, 'aria-label': `Remove ${layer.name}`, onClick: () => actions.removeLayer(l) },
              icon('close', 12),
            )
          : h('span', { class: 'layer-row__spacer' });
      return h(
        'div',
        { class: `layer-row layer-row--${layer.role}`, role: 'listitem', dataset: { layer: String(l) } },
        h('button', { type: 'button', class: 'layer-row__name', title: `Select ${layer.name}`, onClick: () => actions.select({ kind: 'layer', layer: l }) }, layer.name),
        stepper,
        activation,
        remove,
      );
    });
    replaceChildren(this.layerList, rows);
    this.highlightSelection();
    this.editor.syncFromModel();
  }

  private highlightSelection(): void {
    const selection = this.ctx.store.state.selection;
    for (const row of this.layerList.querySelectorAll<HTMLElement>('.layer-row')) {
      row.classList.toggle('is-selected', !!selection && Number(row.dataset.layer) === selection.layer);
    }
  }
}

/**
 * JSON editor with live validation. Valid edits are applied automatically
 * (after a short pause) when "Live apply" is on; Ctrl/⌘+Enter applies now.
 */
class ConfigEditor {
  readonly el: HTMLDetailsElement;
  private readonly ctx: NetworkWorkspaceContext;
  private readonly textarea: HTMLTextAreaElement;
  private readonly status = h('div', { class: 'config-status', role: 'status', 'aria-live': 'polite' });
  private readonly liveToggle: HTMLInputElement;
  private dirty = false;
  private readonly validateSoon = debounce(() => this.validate(this.liveToggle.checked), 450);

  constructor(ctx: NetworkWorkspaceContext) {
    this.ctx = ctx;
    this.textarea = h('textarea', { class: 'config-editor', spellcheck: false, rows: 10, 'aria-label': 'Network configuration (JSON)' });
    this.textarea.value = formatConfig(specToConfig(ctx.model.spec));
    this.textarea.addEventListener('input', () => {
      this.dirty = true;
      this.status.className = 'config-status';
      this.status.textContent = 'Editing…';
      this.validateSoon();
    });
    this.textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        this.validateSoon.cancel();
        this.validate(true);
      } else if (event.key === 'Tab' && !event.shiftKey) {
        event.preventDefault();
        this.textarea.setRangeText('  ', this.textarea.selectionStart, this.textarea.selectionEnd, 'end');
      }
    });
    this.textarea.addEventListener('blur', () => {
      if (!this.dirty) this.syncFromModel();
    });

    this.liveToggle = h('input', { type: 'checkbox' });
    this.liveToggle.checked = true;

    this.el = h(
      'details',
      { class: 'subsection' },
      h('summary', null, 'JSON configuration'),
      h('p', { class: 'panel-text' }, 'The scene is generated from this configuration. Edit it and the network rebuilds itself.'),
      this.textarea,
      this.status,
      h(
        'div',
        { class: 'button-row' },
        button({ label: 'Apply', small: true, variant: 'primary', title: 'Apply (Ctrl/⌘ + Enter)', onClick: () => this.validate(true) }),
        button({ label: 'Format', small: true, onClick: () => this.format() }),
        button({ label: 'Revert', small: true, onClick: () => this.revert() }),
        h('label', { class: 'inline-check' }, this.liveToggle, 'Live apply'),
      ),
    );
    this.setStatus('ok', 'Valid configuration · matches the scene');
  }

  dispose(): void {
    this.validateSoon.cancel();
  }

  /** Shows the current model configuration (unless the user has unapplied edits). */
  syncFromModel(): void {
    if (this.dirty) return;
    if (document.activeElement === this.textarea) return;
    this.textarea.value = formatConfig(specToConfig(this.ctx.model.spec));
    this.setStatus('ok', 'Valid configuration · matches the scene');
  }

  private validate(apply: boolean): void {
    const result = parseConfigText(this.textarea.value);
    if (!result.ok) {
      this.setStatus(
        'error',
        result.errors
          .slice(0, 4)
          .map((e) => `${e.path}: ${e.message}`)
          .join('\n') + (result.errors.length > 4 ? `\n… and ${result.errors.length - 4} more` : ''),
      );
      return;
    }
    const warnings = result.warnings.map((w) => `⚠ ${w.path}: ${w.message}`).join('\n');
    const current = formatConfig(specToConfig(this.ctx.model.spec));
    const next = formatConfig(specToConfig(result.spec));
    if (current === next) {
      this.dirty = false;
      this.setStatus('ok', `Valid configuration · matches the scene${warnings ? `\n${warnings}` : ''}`);
      return;
    }
    if (!apply) {
      this.setStatus('pending', `Valid configuration · press Apply (Ctrl/⌘ + Enter) to rebuild${warnings ? `\n${warnings}` : ''}`);
      return;
    }
    const errors = this.ctx.actions.applyConfig(specToConfig(result.spec), 'config');
    if (errors.length) {
      this.setStatus('error', errors.join('\n'));
      return;
    }
    this.dirty = false;
    this.setStatus('ok', `Applied · the scene was rebuilt from this configuration${warnings ? `\n${warnings}` : ''}`);
  }

  private format(): void {
    const result = parseConfigText(this.textarea.value);
    if (!result.ok) {
      this.validate(false);
      return;
    }
    this.textarea.value = formatConfig(specToConfig(result.spec));
    this.validate(false);
  }

  private revert(): void {
    this.dirty = false;
    this.textarea.value = formatConfig(specToConfig(this.ctx.model.spec));
    this.setStatus('ok', 'Reverted to the configuration of the current scene');
  }

  private setStatus(kind: 'ok' | 'error' | 'pending', message: string): void {
    this.status.className = `config-status config-status--${kind}`;
    replaceChildren(this.status, icon(kind === 'ok' ? 'check' : kind === 'error' ? 'warning' : 'info', 13), h('span', null, message));
  }
}
