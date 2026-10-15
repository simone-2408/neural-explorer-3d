/** Help dialog: controls, the shortcuts of every tab (the current one first), visual encodings and educational notes. */
import { h } from '../ui/dom';
import { openModal } from '../ui/overlays';
import type { WorkspaceId } from './state';

type Shortcut = [keys: string[], action: string];

const CAMERA: Shortcut[] = [
  [['R'], 'Reset the camera'],
  [['F', 'double-click'], 'Focus the camera on the selection'],
  [['1', '2', '3', '4'], 'Preset views (see the toolbar above the 3D view)'],
  [['Esc'], 'Clear the selection'],
];

const TABS: { id: WorkspaceId; label: string; shortcuts: Shortcut[] }[] = [
  {
    id: 'network',
    label: 'Neural Network',
    shortcuts: [
      [['P'], 'Forward-pass mode on / off'],
      [['Space'], 'Play / pause the forward pass'],
      [['.', ','], 'Next / previous step of the forward pass'],
      [['Home'], 'Back to the first step'],
      [['S'], 'Follow the signal from the selection'],
      [['←', '→'], 'Previous / next layer (keeps the neuron position)'],
      [['↑', '↓'], 'Previous / next neuron in the layer'],
      [['L'], 'Look at the selected layer face-on'],
      [['G'], 'Cycle layer layout: grid → flat → ring'],
      [['V'], 'Show / hide values'],
      [['N'], 'Show / hide neuron names'],
      [['A'], 'Auto-rotate'],
    ],
  },
  {
    id: 'training',
    label: 'Training',
    shortcuts: [
      [['Space'], 'Train / pause'],
      [['B'], 'Animate one backpropagation step'],
      [['←', '→', '↑', '↓'], 'Move the selection through the network'],
      [['L'], 'Look at the selected layer face-on'],
    ],
  },
  {
    id: 'cnn',
    label: 'CNN',
    shortcuts: [
      [['Space'], 'Train / pause'],
      [['click'], 'Inspect a cell (its window and computation) or a filter (its kernel)'],
      [['N'], 'Next test image'],
      [['←', '→', '↑', '↓'], 'Move the selected cell'],
      [['Esc'], 'Stop the kernel animation, then clear the selection'],
    ],
  },
  {
    id: 'rnn',
    label: 'RNN',
    shortcuts: [
      [['Space'], 'Train / pause'],
      [['←', '→'], 'Previous / next step'],
      [['U'], 'Fold into one cell / unroll through time'],
      [['G'], 'Show / hide the gradients (backpropagation through time)'],
      [['N'], 'New example'],
    ],
  },
  {
    id: 'transformer',
    label: 'Transformer',
    shortcuts: [
      [['Space'], 'Play / pause the computation step by step'],
      [['.', ','], 'Next / previous step'],
      [['Home', 'End'], 'Back to the tokens / show the finished computation'],
      [['←', '→'], 'Previous / next token (or head, when a head is selected)'],
      [['↑', '↓'], 'Move through the layers'],
      [['E'], 'Model view ↔ embedding space'],
    ],
  },
  {
    id: 'llm',
    label: 'LLM',
    shortcuts: [
      [['Space'], 'Move through the layers automatically'],
      [['↑', '↓'], 'Previous / next layer'],
      [['←', '→'], 'Previous / next token (or head, when a head is selected)'],
      [['J'], 'Layers ↔ token journey'],
      [['⌘/Ctrl', 'Enter'], 'Analyze the prompt'],
    ],
  },
  {
    id: 'experiments',
    label: 'Experiments',
    shortcuts: [
      [['click'], 'Inspect a head, an MLP or a token'],
      [['shift-click'], 'Switch a head or an MLP off / on'],
      [['X'], 'Switch the selected component off / on (or remove / restore the selected token)'],
      [['⌘/Ctrl', 'Z'], 'Undo the last change'],
      [['←', '→', '↑', '↓'], 'Move the selection (heads, layers, tokens, steps)'],
    ],
  },
];

const GLOBAL: Shortcut[] = [
  [['T'], 'Light / dark theme'],
  [['[', ']'], 'Hide / show the side panels'],
  [['?'], 'This help'],
];

function table(rows: Shortcut[]): HTMLElement {
  return h(
    'table',
    { class: 'shortcut-table' },
    h(
      'tbody',
      null,
      rows.map(([keys, action]) => h('tr', null, h('td', null, keys.map((k, i) => [i ? ' ' : '', h('kbd', null, k)])), h('td', null, action))),
    ),
  );
}

export function openHelp(current: WorkspaceId = 'network'): void {
  const tabs = [...TABS].sort((a, b) => Number(b.id === current) - Number(a.id === current));
  openModal({
    title: 'Neural Explorer 3D — help',
    wide: true,
    content: h(
      'div',
      { class: 'help' },
      h(
        'section',
        null,
        h('h3', null, 'Explore'),
        h(
          'ul',
          null,
          h('li', null, h('strong', null, 'Rotate:'), ' drag with the left mouse button (one finger on touch screens).'),
          h('li', null, h('strong', null, 'Pan:'), ' drag with the right button, or Shift + drag (two fingers on touch screens).'),
          h('li', null, h('strong', null, 'Zoom:'), ' scroll wheel or pinch; zoom goes towards the cursor.'),
          h('li', null, h('strong', null, 'Inspect:'), ' click anything in the 3D view; the inspector on the right explains it with the real numbers. Hover for a quick tooltip.'),
          h('li', null, h('strong', null, 'Learn:'), ' the Learn tab opens guided lessons that drive the live scenes.'),
        ),
      ),
      h('section', null, h('h3', null, `Shortcuts — ${tabs[0].label}`), table([...tabs[0].shortcuts, ...CAMERA])),
      h('section', null, h('h3', null, 'Everywhere'), table(GLOBAL)),
      h(
        'section',
        null,
        h('h3', null, 'Other tabs'),
        tabs.slice(1).map((t) => h('details', { class: 'help__tab' }, h('summary', null, t.label), table(t.shortcuts))),
      ),
      h(
        'section',
        null,
        h('h3', null, 'What the visuals mean'),
        h(
          'ul',
          null,
          h('li', null, 'Sign is never encoded by colour alone: negative weights are also dashed, negative values point down (or behind a wall), and every value is available as a signed number in the inspector.'),
          h('li', null, 'Sizes, colours and positions are chosen for readability; the inspector always shows the exact values and the formula that produced them.'),
          h('li', null, 'The arrangement of neurons inside a layer (grid, flat, ring) has no mathematical meaning: in a dense layer every neuron is connected to every neuron of the previous layer anyway.'),
        ),
      ),
      h(
        'section',
        null,
        h('h3', null, 'Three levels for every concept'),
        h(
          'ol',
          null,
          h('li', null, h('strong', null, 'Visual'), ' — what you see in the scene.'),
          h('li', null, h('strong', null, 'Numbers'), ' — the actual inputs, weights, activations, probabilities.'),
          h('li', null, h('strong', null, 'Math'), ' — the formula, with the current numbers substituted.'),
        ),
      ),
      h(
        'section',
        null,
        h('h3', null, 'Educational notes'),
        h(
          'ul',
          null,
          h('li', null, 'Everything is computed, nothing is animated for show: the Neural Network, Training, CNN, RNN and Transformer tabs run real models in your browser (checked against PyTorch), the LLM and Experiments tabs run GPT-2 small (or show recordings of it).'),
          h('li', null, 'The network of the Neural Network tab starts with random weights: its outputs mean nothing until it is trained (Training tab).'),
          h('li', null, 'Projections (PCA), the logit lens, attention maps and attributions are probes with limits; each inspector states them next to the numbers.'),
        ),
      ),
    ),
  });
}
