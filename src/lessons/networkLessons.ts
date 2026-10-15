/** Lessons 1–7: neurons, layers and training (Neural Network and Training tabs). */
import { SCENARIOS } from '../training/scenarios';
import { DEFAULT_TRAINING } from '../training/session';
import { isModalOpen } from '../ui/overlays';
import { formatValue } from '../utils/format';
import type { LessonContext } from './context';
import type { Lesson } from './types';

const fmt = (v: number) => formatValue(v, 2);

/** The Network tab with a preset loaded and a selection (the camera is reset to frame it). */
function networkWith(ctx: LessonContext, preset: string) {
  const ws = ctx.network();
  if (ctx.store.state.presetId !== preset) ws.applyPreset(preset);
  ctx.store.set({ signalMode: 'off' });
  return ws;
}

function trainingTab(ctx: LessonContext, mode: 'train' | 'break' | 'compare') {
  return ctx.training(mode);
}

/** Back to the default training session (XOR, 2 → 4 → 4 → 1, Adam), untrained. */
function freshSession(ctx: LessonContext) {
  const ws = ctx.training('train');
  ws.pause();
  ws.update(structuredClone(DEFAULT_TRAINING));
  return ws;
}

export const NETWORK_LESSONS: Lesson[] = [
  {
    id: 'neuron',
    number: 1,
    title: 'What is a neuron?',
    summary: 'One neuron, two inputs: weights, bias and an activation function — with your hands on the sliders.',
    tab: 'network',
    steps: [
      {
        title: 'One neuron',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'perceptron');
          ws.networkScene.select({ kind: 'neuron', layer: 1, index: 0 });
          ws.networkScene.resetCamera();
        },
        text: [
          'This is the smallest possible network: *two inputs* (left) and *one neuron* (right, selected).',
          'The neuron receives the two input numbers along the two connections and computes a single number. The inspector on the right shows every step of that computation with the real values.',
        ],
      },
      {
        title: 'Inputs are numbers',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'perceptron');
          ws.networkScene.select({ kind: 'neuron', layer: 0, index: 0 });
          ctx.memo.set('x1', ws.model.input[0]);
        },
        text: (ctx) => [
          `An input is just a number: here x₁ = ${fmt(ctx.networkWs.model.input[0])}. It could be a pixel, a temperature, a word's feature…`,
          'The colour and size of a neuron show its value.',
        ],
        task: {
          prompt: 'Change x₁ with its slider (Input panel on the left, or in the inspector) and watch the output neuron react.',
          check: (ctx) => Math.abs(ctx.networkWs.model.input[0] - (ctx.memo.get('x1') as number)) > 0.05,
          done: 'Every change of the input is recomputed through the whole network immediately.',
        },
      },
      {
        title: 'Weights',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'perceptron');
          ws.networkScene.select({ kind: 'connection', layer: 1, target: 0, source: 0 });
        },
        text: [
          'Each connection multiplies the number it carries by its *weight*. Line thickness shows the size of the weight; colour and dashes show its sign.',
          'Training a network means finding good weights. For now you set them by hand.',
        ],
        task: {
          prompt: 'Drag the weight slider in the inspector until w₁ becomes negative.',
          check: (ctx) => ctx.networkWs.model.weight(1, 0, 0) < 0,
          done: 'A negative weight turns a large input into a push downwards.',
        },
      },
      {
        title: 'Weighted sum and bias',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'perceptron');
          ws.networkScene.select({ kind: 'neuron', layer: 1, index: 0 });
        },
        text: [
          'The neuron adds up its weighted inputs and a *bias* b: a number of its own, added whatever the inputs are. The result is z, the weighted sum.',
        ],
        tex: 'z = w_1 x_1 + w_2 x_2 + b',
        task: {
          prompt: 'Use the bias slider in the inspector to make the output a larger than 0.9.',
          check: (ctx) => ctx.networkWs.model.output[0] > 0.9,
          done: 'A large bias makes z large for every input, so the output stays near 1.',
        },
      },
      {
        title: 'The activation function',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'perceptron');
          ws.networkScene.select({ kind: 'neuron', layer: 1, index: 0 });
        },
        text: [
          'Last, z goes through an *activation function*. This neuron uses the sigmoid σ, which squashes any number into the range 0 to 1: very negative z gives almost 0, very positive z almost 1.',
          'In the inspector, the curve shows where this neuron sits on σ.',
        ],
        tex: 'a = \\sigma(z) = \\frac{1}{1 + e^{-z}}',
      },
      {
        title: 'That is a neuron',
        text: [
          'A neuron = a weighted sum + a bias + a simple non-linear function. Everything else in this app — layers, Transformers, GPT-2 — is built from this, repeated many times.',
          'Next lesson: many neurons working side by side.',
        ],
        tex: 'a = f\\Big(\\sum_i w_i x_i + b\\Big)',
      },
    ],
  },
  {
    id: 'weighted-sum',
    number: 2,
    title: 'Weighted sum',
    summary: 'Contributions, active and inactive neurons, and why a layer is a matrix times a vector.',
    tab: 'network',
    steps: [
      {
        title: 'A real network',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ws.networkScene.resetCamera();
          ws.networkScene.select({ kind: 'neuron', layer: 1, index: 0 });
        },
        text: [
          'Two inputs, two hidden layers of 4 neurons, one output. Every neuron of the first hidden layer computes its *own* weighted sum of the *same* two inputs, with its own weights.',
        ],
      },
      {
        title: 'Contributions',
        text: [
          'In the inspector, each term wᵢ·xᵢ is a *contribution* to z: positive terms push z up, negative terms pull it down, and the bias is added at the end.',
          'The formula is shown three times: with symbols, with the numbers substituted, and as the result.',
        ],
        tex: 'z_j = \\sum_i w_{ji}\\, x_i + b_j',
      },
      {
        title: 'Active and inactive',
        enter: (ctx) => {
          networkWith(ctx, 'default');
        },
        text: [
          'The hidden layers use *ReLU*: a = max(0, z). When z is negative the neuron outputs exactly 0 — it is *inactive* for this input and passes nothing on.',
        ],
        task: {
          prompt: 'Click the neurons of the first hidden layer until the inspector says one of them is inactive.',
          check: (ctx) => {
            const s = ctx.store.state.selection;
            const model = ctx.networkWs.model;
            return s?.kind === 'neuron' && s.layer === 1 && model.trace.z[1][s.index] <= 0;
          },
          done: 'An inactive ReLU outputs 0, and its slope is 0 too: during training no gradient flows through it for this input.',
        },
      },
      {
        title: 'A layer is a matrix',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ws.networkScene.select({ kind: 'layer', layer: 1 });
        },
        text: [
          'All the weighted sums of a layer at once are one *matrix–vector product*: row j of W holds the weights of neuron j. The layer inspector shows W with the current numbers — the lines in 3D are its entries.',
        ],
        tex: 'z^{(1)} = W^{(1)} x + b^{(1)}, \\qquad a^{(1)} = \\mathrm{ReLU}\\big(z^{(1)}\\big)',
      },
      {
        title: 'Layer after layer',
        text: [
          'The second hidden layer does exactly the same, with the outputs of the first layer as its inputs. Use ← and → to move the selection between layers.',
        ],
        tex: 'a^{(2)} = \\mathrm{ReLU}\\big(W^{(2)} a^{(1)} + b^{(2)}\\big)',
      },
    ],
  },
  {
    id: 'activations',
    number: 3,
    title: 'Activation functions',
    summary: 'ReLU, tanh, sigmoid and softmax — and why a network without them would be a single weighted sum.',
    tab: 'network',
    steps: [
      {
        title: 'Why a function at all?',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ws.networkScene.select({ kind: 'neuron', layer: 1, index: 1 });
        },
        text: [
          'Without an activation function, a weighted sum of weighted sums is still one weighted sum: two layers would collapse into one, and depth would add nothing.',
          'The non-linear function between layers is what lets a network bend its decision boundaries.',
        ],
        tex: 'W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\\, x + (W_2 b_1 + b_2)',
      },
      {
        title: 'Explore them',
        text: ['ReLU, sigmoid, tanh and GELU (used by Transformers) side by side, with their slopes.'],
        task: {
          prompt: 'Open the activation explorer with the *f(z)* button above the 3D view, try a few inputs, then close it.',
          check: () => isModalOpen(),
          done: 'The slope matters as much as the value: training follows slopes.',
        },
      },
      {
        title: 'ReLU or tanh?',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ws.networkScene.select({ kind: 'layer', layer: 1 });
        },
        text: [
          'tanh squashes z into (−1, 1): negative values survive as negative numbers instead of becoming 0.',
        ],
        task: {
          prompt: 'In the *Architecture* panel (left), change the activation of hidden layer 1 to tanh and watch its neurons change.',
          check: (ctx) => ctx.networkWs.model.spec.layers[1]?.activation === 'tanh',
          done: 'Same weights, different function: every value after this layer changed.',
        },
      },
      {
        title: 'Softmax: probabilities',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'classifier');
          const last = ws.model.layerCount - 1;
          ws.networkScene.select({ kind: 'layer', layer: last });
        },
        text: [
          'With several classes, the output layer uses *softmax*: it turns the output scores into positive numbers that add up to 1, which can be read as probabilities.',
          'Unlike the other functions, softmax looks at the whole layer at once.',
        ],
        tex: '\\mathrm{softmax}(z)_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
      },
    ],
  },
  {
    id: 'forward',
    number: 4,
    title: 'Forward propagation',
    summary: 'Watch the signal travel from the input to the output, then follow where an output comes from.',
    tab: 'network',
    steps: [
      {
        title: 'Layer by layer',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ws.networkScene.select(null);
          ws.networkScene.resetCamera();
          ctx.store.set({ signalMode: 'forward' });
        },
        text: [
          'The *forward pass* computes the network in order: each layer needs the outputs of the previous one. Only the input layer is computed now; the rest is waiting.',
        ],
        task: {
          prompt: 'Press *Space* (or ▶ in the bar below) and let the forward pass reach the output.',
          check: (ctx) => {
            const player = ctx.networkWs.forwardPlayer;
            const state = player.state;
            return state.active && state.step >= state.layers - 1;
          },
          done: 'Pulses travel along the connections; each neuron lights up once its weighted sum is complete.',
        },
      },
      {
        title: 'Follow the signal',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          const last = ws.model.layerCount - 1;
          ws.networkScene.select({ kind: 'neuron', layer: last, index: 0 });
          ctx.store.set({ signalMode: 'follow' });
        },
        text: [
          '*Follow the Signal* highlights where the output’s value comes from: brighter, thicker paths carry larger contributions |w·a|; the rest fades.',
          'It is a view of the arithmetic on this input — not an explanation of what the network “knows”.',
        ],
      },
      {
        title: 'Change the input',
        enter: (ctx) => {
          const ws = networkWith(ctx, 'default');
          ctx.memo.set('input', Array.from(ws.model.input));
          const last = ws.model.layerCount - 1;
          ws.networkScene.select({ kind: 'neuron', layer: last, index: 0 });
          ctx.store.set({ signalMode: 'follow' });
        },
        text: ['The paths depend on the input: change it and the network routes its signal differently (some ReLUs switch on or off).'],
        task: {
          prompt: 'Move an input slider (Input panel) and watch the highlighted paths change.',
          check: (ctx) => {
            const before = ctx.memo.get('input') as number[];
            const now = ctx.networkWs.model.input;
            return before.some((v, i) => Math.abs(v - now[i]) > 0.05);
          },
        },
      },
      {
        title: 'Untrained, but computed',
        enter: (ctx) => ctx.store.set({ signalMode: 'off' }),
        text: [
          'These weights are random: the output is computed exactly, but it means nothing yet. Making it mean something is the job of training — next lessons.',
        ],
      },
    ],
  },
  {
    id: 'loss',
    number: 5,
    title: 'Loss',
    summary: 'How wrong is the network? One number per point, averaged over the data — and watching it fall.',
    tab: 'training',
    steps: [
      {
        title: 'Data to learn',
        enter: (ctx) => freshSession(ctx).showSideTab('boundary'),
        text: [
          'Each point is an example with a label (its colour). The background shows what the network predicts everywhere in the plane.',
          'The network has random weights: its prediction is a guess, unrelated to the data.',
        ],
      },
      {
        title: 'The loss of one point',
        enter: (ctx) => {
          const ws = trainingTab(ctx, 'train');
          ctx.memo.set('probe', JSON.stringify(ws.probe));
        },
        text: [
          'For one point with label y (0 or 1) and predicted probability p, the *loss* measures how wrong the prediction is: 0 when it is perfectly right, large when it is confidently wrong.',
        ],
        tex: 'L = -\\big[\\,y \\log p + (1 - y) \\log (1 - p)\\,\\big]',
        task: {
          prompt: 'Click a data point in the plot (left): the inspector and the panel under the plot show its prediction and its loss.',
          check: (ctx) => {
            const ws = ctx.trainingWs;
            return ws.probe.kind === 'sample' && JSON.stringify(ws.probe) !== ctx.memo.get('probe');
          },
        },
      },
      {
        title: 'Train',
        enter: (ctx) => trainingTab(ctx, 'train').showSideTab('boundary'),
        text: [
          'Training lowers the *average* loss over all the points. A network that always answers 50% scores log 2 ≈ 0.69 per point.',
        ],
        task: {
          prompt: 'Press *Space* (or Train) and watch the loss curve fall and the background take the shape of the data.',
          check: (ctx) => {
            const h = ctx.trainingWs.trainer.history;
            return h.length > 2 && h[h.length - 1].train.loss < Math.min(0.45, h[0].train.loss - 0.15);
          },
          done: 'The loss fell: the weights were adjusted, step after step, to make the training points less surprising.',
        },
      },
      {
        title: 'Loss is not accuracy',
        text: (ctx) => {
          const h = ctx.trainingWs.trainer.history;
          const last = h[h.length - 1];
          return [
            `Accuracy counts right answers${last ? ` (now ${formatValue(last.train.accuracy * 100, 0)}%)` : ''}; the loss also rewards *confidence*. Two networks with the same accuracy can have very different losses.`,
            'The loss is what training actually minimizes, because its slope tells every weight which way to move.',
          ];
        },
      },
    ],
  },
  {
    id: 'backprop',
    number: 6,
    title: 'Backpropagation',
    summary: 'Gradients flow backwards: one training step, animated on one point, with the chain rule.',
    tab: 'training',
    steps: [
      {
        title: 'Which way to move?',
        enter: (ctx) => {
          const ws = trainingTab(ctx, 'train');
          ws.pause();
        },
        text: [
          'To lower the loss, each weight needs to know whether to go up or down, and how strongly: the *gradient* ∂L/∂w, the slope of the loss with respect to that weight.',
          '*Backpropagation* computes all of them at once, from the output back to the input.',
        ],
      },
      {
        title: 'Watch one step',
        enter: (ctx) => {
          const ws = trainingTab(ctx, 'train');
          ws.pause();
          ws.setProbe({ kind: 'sample', index: 3 });
          ws.animateBackprop();
        },
        text: [
          'One training step on one point: forward pass, loss, then the error signal δ travels *backwards* layer by layer, and finally every weight moves a little.',
          'Press *B* to play it again.',
        ],
      },
      {
        title: 'The chain rule',
        text: [
          'Each neuron’s error δ is the weighted sum of the errors of the neurons it feeds, times the slope of its own activation. A weight’s gradient is the error of its target times the activity of its source.',
        ],
        tex: '\\delta_j = f\'(z_j) \\sum_k w_{kj}\\, \\delta_k, \\qquad \\frac{\\partial L}{\\partial w_{ji}} = \\delta_j\\, a_i',
        task: {
          prompt: 'Click a connection: the inspector shows its gradient and how much the last step changed it.',
          check: (ctx) => ctx.store.state.selection?.kind === 'connection',
        },
      },
      {
        title: 'When gradients vanish',
        enter: (ctx) => {
          const ws = trainingTab(ctx, 'break');
          const scenario = SCENARIOS.find((s) => s.id === 'deep-sigmoid');
          if (scenario) ws.applyScenario(scenario);
          ws.showSideTab('gradients');
        },
        text: [
          'In a deep network of sigmoids, each layer multiplies δ by a slope of at most 0.25: the gradient shrinks layer after layer, and the first layers barely learn.',
          'Train a little and look at the gradient norms per layer (left). The scenario panel explains the fix.',
        ],
      },
    ],
  },
  {
    id: 'gradient-descent',
    number: 7,
    title: 'Gradient descent',
    summary: 'Step downhill on the loss: the learning rate, the landscape, and what happens when steps are too big.',
    tab: 'training',
    steps: [
      {
        title: 'The update rule',
        enter: (ctx) => freshSession(ctx).showSideTab('landscape'),
        text: [
          'Every step moves each weight a little *against* its gradient. The size of the step is the *learning rate* η.',
        ],
        tex: 'w \\leftarrow w - \\eta\\, \\frac{\\partial L}{\\partial w}',
      },
      {
        title: 'The loss landscape',
        enter: (ctx) => trainingTab(ctx, 'train').showSideTab('landscape'),
        text: [
          'The surface is the loss on a 2-D slice through weight space; the dot is the current network. The real space has dozens of dimensions — this is a projection, so the path can look curved where the real one is not.',
        ],
        task: {
          prompt: 'Train (Space) for about 30 epochs and watch the dot go downhill.',
          check: (ctx) => ctx.trainingWs.trainer.epochCount >= 30,
        },
      },
      {
        title: 'Steps too big',
        enter: (ctx) => {
          const ws = trainingTab(ctx, 'break');
          const scenario = SCENARIOS.find((s) => s.id === 'huge-lr');
          if (scenario) ws.applyScenario(scenario);
          ws.showSideTab('boundary');
        },
        text: [
          'Learning rate 30 with plain SGD: every step overshoots the valley, so the loss jumps around or explodes instead of falling.',
        ],
        task: {
          prompt: 'Train a few epochs and watch the loss, then press *Apply the fix* in the scenario panel.',
          check: (ctx) => ctx.trainingWs.scenarioFixed,
          done: 'With a learning rate of 0.3 the same network trains smoothly.',
        },
      },
      {
        title: 'Better steps',
        enter: (ctx) => {
          ctx.training('compare');
        },
        text: [
          'Optimizers differ in how they turn gradients into steps: *momentum* keeps part of the previous step, *Adam* also scales each weight’s step by the size of its recent gradients.',
          'Compare mode trains two networks side by side: change one setting and press Train.',
        ],
      },
    ],
  },
];
