/** View panel: layout, encodings, labels, colours and theme. */
import type { ViewOptions } from '../../../app/state';
import { DisposeBag } from '../../../core/DisposeBag';
import { MAX_NEURON_LABELS } from '../../../three/NetworkView';
import { h } from '../../../ui/dom';
import { section, segmented, select, slider, toggle, type Segmented, type Slider, type Toggle } from '../../../ui/widgets';
import { LAYOUT_MODES, type LayoutMode } from '../../../visualization/layout';
import { WEIGHT_PALETTES, type ThemeId, type WeightPaletteId } from '../../../visualization/palette';
import type { NetworkContext, Panel } from '../context';

type BooleanKey = { [K in keyof ViewOptions]: ViewOptions[K] extends boolean ? K : never }[keyof ViewOptions];

const TOGGLES: { key: BooleanKey; label: string; title: string }[] = [
  { key: 'colorByActivation', label: 'Colour neurons by activation', title: 'Colour and size of each neuron encode its output value' },
  { key: 'showValues', label: 'Show values', title: 'Write each neuron output next to it' },
  { key: 'showNeuronNames', label: 'Show neuron names', title: 'I1, H1-3, O1…' },
  { key: 'dimUnrelated', label: 'Dim unrelated elements', title: 'When something is selected, fade what is not connected to it' },
  { key: 'showPlates', label: 'Layer plates', title: 'Translucent plate behind each layer (click it to select the layer)' },
  { key: 'showFloor', label: 'Floor grid & signal arrow', title: 'Reference grid and the input → output direction' },
  { key: 'autoRotate', label: 'Auto-rotate', title: 'Slowly orbit around the network' },
];

export class ViewPanel implements Panel {
  readonly el: HTMLElement;
  private readonly ctx: NetworkContext;
  private readonly bag = new DisposeBag();
  private readonly toggles: Partial<Record<BooleanKey, Toggle>> = {};
  private readonly layout: Segmented<LayoutMode>;
  private readonly scale: Segmented<'global' | 'layer'>;
  private readonly theme: Segmented<ThemeId>;
  private readonly thickness: Slider;
  private readonly palette: HTMLSelectElement;
  private readonly labelNote = h('p', { class: 'panel-text muted' });

  constructor(ctx: NetworkContext) {
    this.ctx = ctx;
    const { store, actions } = ctx;
    const view = store.state.view;

    this.layout = segmented({
      options: LAYOUT_MODES.map((m) => ({ value: m.id, label: m.label, title: m.description })),
      value: view.layout,
      ariaLabel: 'Layer layout',
      onChange: (layout) => actions.setView({ layout }),
    });
    for (const t of TOGGLES) {
      this.toggles[t.key] = toggle({ label: t.label, title: t.title, checked: view[t.key], onChange: (checked) => actions.setView({ [t.key]: checked }) });
    }
    this.scale = segmented({
      options: [
        { value: 'global', label: 'Global', title: 'Brightness relative to the largest |a| in the whole network' },
        { value: 'layer', label: 'Per layer', title: 'Brightness relative to the largest |a| of each layer' },
      ],
      value: view.activationScale,
      ariaLabel: 'Activation brightness scale',
      onChange: (activationScale) => actions.setView({ activationScale }),
    });
    this.thickness = slider({
      label: 'Line thickness (visual only)',
      min: 0.3,
      max: 3,
      step: 0.05,
      value: view.thickness,
      digits: 2,
      ariaLabel: 'Connection thickness multiplier',
      onInput: (thickness) => actions.setView({ thickness }),
    });
    this.palette = select<WeightPaletteId>({
      options: (Object.keys(WEIGHT_PALETTES) as WeightPaletteId[]).map((id) => ({ value: id, label: WEIGHT_PALETTES[id].label })),
      value: view.weightPalette,
      ariaLabel: 'Sign colours',
      onChange: (weightPalette) => actions.setView({ weightPalette }),
    });
    this.theme = segmented({
      options: [
        { value: 'dark', label: 'Dark' },
        { value: 'light', label: 'Light' },
      ],
      value: view.theme,
      ariaLabel: 'Theme',
      onChange: (theme) => actions.setView({ theme }),
    });

    this.el = section({
      id: 'view',
      title: 'View',
      children: [
        h('div', { class: 'field' }, h('span', { class: 'field__label' }, 'Layer layout'), this.layout.el),
        h('div', { class: 'toggle-list' }, TOGGLES.map((t) => this.toggles[t.key]!.el)),
        this.labelNote,
        h('div', { class: 'field' }, h('span', { class: 'field__label' }, 'Activation brightness'), this.scale.el),
        this.thickness.el,
        h('label', { class: 'field' }, h('span', { class: 'field__label' }, 'Sign colours'), this.palette),
        h('div', { class: 'field' }, h('span', { class: 'field__label' }, 'Theme'), this.theme.el),
      ],
    });

    this.bag.add(store.watch((s) => s.view, () => this.sync()));
    this.bag.add(ctx.model.events.on('structure', () => this.sync()));
    this.sync();
  }

  dispose(): void {
    this.bag.dispose();
  }

  private sync(): void {
    const view = this.ctx.store.state.view;
    this.layout.set(view.layout);
    for (const t of TOGGLES) this.toggles[t.key]!.set(view[t.key]);
    this.scale.set(view.activationScale);
    this.thickness.setValue(view.thickness);
    this.palette.value = view.weightPalette;
    this.theme.set(view.theme);
    const tooMany = this.ctx.model.topology.neuronCount > MAX_NEURON_LABELS;
    this.labelNote.hidden = !tooMany;
    this.labelNote.textContent = tooMany ? `Neuron names and values are hidden in the scene for networks with more than ${MAX_NEURON_LABELS} neurons (use the inspector instead).` : '';
  }
}
