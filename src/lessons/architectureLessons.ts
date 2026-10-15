/** Lessons 12–13: convolutional and recurrent networks (CNN and RNN tabs). */
import { formatCount, formatValue } from '../utils/format';
import { RNN_PRESETS } from '../workspaces/rnn/RnnWorkspace';
import type { LessonContext } from './context';
import type { Lesson } from './types';

const testAccuracy = (history: { test: { accuracy: number } }[]) => history[history.length - 1]?.test.accuracy ?? 0;

function convStage(ctx: LessonContext, layer: number): number {
  return ctx.cnnWs.stages.findIndex((s) => s.kind === 'conv' && s.layer === layer);
}

export const ARCHITECTURE_LESSONS: Lesson[] = [
  {
    id: 'convolution',
    number: 12,
    title: 'Convolution',
    summary: 'Filters that slide over an image, weight sharing, pooling — and filters that learn to detect lines.',
    tab: 'cnn',
    steps: [
      {
        title: 'An image is a grid of numbers',
        enter: (ctx) => {
          const ws = ctx.cnn();
          ws.pause();
          if (ws.settings.dataset !== 'lines') ws.update({ dataset: 'lines' });
          ws.setInputSample(0);
          ws.select(null);
          ws.setCameraView('iso');
        },
        text: [
          'At the bottom: a 16×16 image, 256 numbers between 0 (background) and 1 (ink). An MLP would connect every pixel to every neuron. A *convolutional* network looks at small windows instead.',
          'Each row above is one step of the computation; the top row holds the probabilities of the four classes (horizontal, vertical and two diagonals).',
        ],
      },
      {
        title: 'A filter slides',
        enter: (ctx) => {
          const ws = ctx.cnn();
          const stage = convStage(ctx, 0);
          ws.select({ kind: 'map', stage, map: 0 });
          ws.startSlide(stage, 0);
        },
        text: [
          'A *filter* is a 3×3 grid of weights. It is placed on every 3×3 window of the image; at each position it multiplies the 9 pixels by its 9 weights and adds them up (plus a bias). Each result is one cell of its *feature map*. The animation computes the map cell by cell.',
        ],
        tex: 'z_{i,j} = b + \\sum_{u,v} W_{u,v}\\, x_{i+u-1,\\,j+v-1}',
        task: {
          prompt: 'Click any cell of a Conv 1 map: the inspector shows its window, the kernel and the nine products.',
          check: (ctx) => {
            const s = ctx.cnnWs.selection;
            return s?.kind === 'cell' && ctx.cnnWs.stages[s.stage]?.kind === 'conv';
          },
        },
      },
      {
        title: 'The same weights everywhere',
        enter: (ctx) => ctx.cnn().stopSlide(),
        text: (ctx) => {
          const m = ctx.cnnWs.model;
          return [
            `Every cell of a map uses the same few weights: Conv 1 computes ${formatCount(m.shapes[0].outChannels * m.shapes[0].size * m.shapes[0].size)} values with only ${formatCount(m.convParameterCount(0))} parameters. A fully connected layer would need ${formatCount(m.fullyConnectedEquivalent(0))} to do the same.`,
            '*Weight sharing* is what lets a convolution find a pattern anywhere in the image, wherever the stroke happens to be.',
          ];
        },
      },
      {
        title: 'Pooling',
        enter: (ctx) => {
          const ws = ctx.cnn();
          const pool = ws.stages.findIndex((s) => s.kind === 'pool');
          if (pool >= 0) ws.select({ kind: 'cell', stage: pool, map: 0, i: 3, j: 3 });
        },
        text: [
          '*Max pooling* keeps the largest value of each 2×2 window: the maps get half as wide, and a pattern found anywhere inside a window passes on. It has no weights.',
        ],
      },
      {
        title: 'Filters are learned',
        enter: (ctx) => {
          const ws = ctx.cnn();
          ws.select({ kind: 'map', stage: convStage(ctx, 0), map: 0 });
        },
        text: [
          'Nobody designs these kernels: training adjusts them, like any other weights. Watch the kernel in the inspector while the network trains.',
        ],
        task: {
          prompt: 'Press *Space* to train until the test accuracy (bottom bar) is above 90%.',
          check: (ctx) => testAccuracy(ctx.cnnWs.trainer.history) > 0.9,
          done: 'Look at the filters now: several resemble line or edge detectors (the inspector measures how much).',
        },
      },
      {
        title: 'Deeper cells see more',
        enter: (ctx) => {
          const ws = ctx.cnn();
          ws.pause();
          const stage = convStage(ctx, 1);
          if (stage >= 0) ws.select({ kind: 'cell', stage, map: 0, i: 4, j: 4 });
        },
        text: (ctx) => {
          const last = ctx.cnnWs.trainer.history[ctx.cnnWs.trainer.history.length - 1];
          return [
            'A cell of Conv 2 reads a 3×3 window of the pooled maps, which already summarize 2×2 windows of Conv 1, which read 3×3 windows of the image: its *receptive field* covers a larger region of the image (the large frame on the input).',
            `Test accuracy now: ${last ? `${formatValue(last.test.accuracy * 100, 1)}%` : '—'} on images the network never trained on.`,
          ];
        },
      },
    ],
  },
  {
    id: 'recurrence',
    number: 13,
    title: 'Recurrence',
    summary: 'One cell reading a sequence step by step, a hidden state as memory, backpropagation through time and gates.',
    tab: 'rnn',
    steps: [
      {
        title: 'One step at a time',
        enter: (ctx) => {
          const ws = ctx.rnn();
          ws.pause();
          const preset = RNN_PRESETS.find((p) => p.id === 'days')!;
          if (ws.preset?.id !== 'days') ws.applyPreset(preset);
          ws.setText('friday saturday s');
          ws.setUnfold(true);
          ws.select({ kind: 'step', t: 0 });
        },
        text: [
          'A *recurrent* network reads a sequence one symbol per step. At each step it combines the new symbol with its *hidden state* h — a vector it carries from step to step, its only memory — and predicts the next symbol.',
          'Each column is one step: the letter read (bottom), the state (middle), the prediction (top).',
        ],
      },
      {
        title: 'The same cell, again and again',
        text: ['All the columns use exactly the same weights. Unrolling through time is just a way of drawing the loop.'],
        task: {
          prompt: 'Press *Fold* above the 3D view (or U) to see the single cell with its loop, then Unfold.',
          check: (ctx) => !ctx.rnnWs.unfolded,
          done: 'Folded: one cell, one loop. Unfold it again to see the steps.',
        },
      },
      {
        title: 'Memory',
        enter: (ctx) => {
          const ws = ctx.rnn();
          ws.setUnfold(true);
          ws.select({ kind: 'output', t: ws.inputs.length - 1 });
        },
        text: [
          'After “friday saturday s” the next letter is “u” (sunday follows saturday). To know it, the network must still remember “saturday” when it reads the final “s”.',
        ],
        task: {
          prompt: 'Train (Space) until the test accuracy is above 90%, then look at the last prediction.',
          check: (ctx) => testAccuracy(ctx.rnnWs.trainer.history) > 0.9,
          done: 'The last column now predicts “u”: the state carried the previous word.',
        },
      },
      {
        title: 'Backpropagation through time',
        enter: (ctx) => {
          const ws = ctx.rnn();
          ws.pause();
          ws.applyPreset(RNN_PRESETS.find((p) => p.id === 'memory-long-rnn')!);
          ws.update({ showGradients: true });
          ws.select(null);
        },
        text: [
          'Now a memory test: read a letter, then 30 meaningless symbols, then answer the first letter. Training sends the error back through every step; the bars under the inputs show how large the gradient still is when it reaches each step.',
          'In a vanilla RNN each step multiplies the gradient by tanh’s slope and by W_hh: it *vanishes* long before it reaches the first letter. Train a little and look at the bars.',
        ],
        tex: '\\Big\\lVert \\frac{\\partial h_{t+k}}{\\partial h_t} \\Big\\rVert \\le \\sigma_{\\max}(W_{hh})^k',
      },
      {
        title: 'Gates',
        enter: (ctx) => {
          const ws = ctx.rnn();
          ws.pause();
          ws.applyPreset(RNN_PRESETS.find((p) => p.id === 'memory-long-gru')!);
        },
        text: [
          'A *GRU* adds gates: the update gate z decides, number by number, how much of the old state to keep. When z is close to 1 the state — and the gradient — passes through almost unchanged.',
        ],
        tex: 'h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}',
        task: {
          prompt: 'Train the GRU until the test accuracy is above 90% (chance is 25%).',
          check: (ctx) => ctx.rnnWs.settings.cell === 'gru' && testAccuracy(ctx.rnnWs.trainer.history) > 0.9,
          done: 'Gated cells (GRU, LSTM) made long memories trainable; Transformers later replaced the loop with attention.',
        },
      },
    ],
  },
];
