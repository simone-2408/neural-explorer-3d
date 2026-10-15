# `src/lessons`

The Learn tab: 13 lessons that drive the live scenes instead of being pages of text.

| File | Content |
| --- | --- |
| `types.ts` | a lesson = steps; a step has a title, short text, an optional formula, `enter(ctx)` (sets up the scene) and an optional task with `check(ctx)` |
| `context.ts` | `LessonContext`: reaches the workspaces through their public actions (switching tabs is part of a lesson) |
| `networkLessons.ts` | 1 What is a neuron? · 2 Weighted sum · 3 Activation functions · 4 Forward propagation · 5 Loss · 6 Backpropagation · 7 Gradient descent |
| `transformerLessons.ts` | 8 Embeddings · 9 Self-attention · 10 The Transformer · 11 Next-token prediction |
| `architectureLessons.ts` | 12 Convolution · 13 Recurrence |
| `LessonPlayer.ts` | the card over the 3D view: step text, live task check, Back / Next, minimize |
| `LessonCatalog.ts` | the catalog dialog opened by the Learn tab |
| `progress.ts` | progress per lesson in `localStorage` |

Tasks are checked by polling the real model state (e.g. “make w₁ negative” checks the weight, “train until
the loss falls” reads the trainer's history), so a lesson never pretends.
