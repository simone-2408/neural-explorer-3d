# `src/cnn`

The small convolutional network of the CNN tab, trained in the browser. Pure logic (no three.js, no DOM),
tested in Node and against PyTorch.

| File | Content |
| --- | --- |
| `datasets.ts` | seeded synthetic 16×16 images: lines (4 orientations), shapes (square, plus, cross, ring), digits (5×7 font scaled ×2, shifted, noisy) |
| `model.ts` | `CnnModel`: 1–2 conv layers (3×3 or 5×5 kernels, "same" padding, ReLU or tanh, max / average / no 2×2 pooling) → flatten → dense → softmax; PyTorch layouts (`Conv2d.weight` = [out, in, k, k]) and initialization |
| `forward.ts` | forward pass with a full trace (pre-activation maps, activations, pooled maps, max-pool winners, flattened vector, logits, probabilities) and backpropagation (cross-entropy, un-pooling, convolution gradients) |
| `trainer.ts` | mini-batch training with the optimizers of the Training tab, evaluation, confusion matrix |
| `analysis.ts` | receptive fields, test images that excite a filter most, cosine similarity of a kernel with classic kernels (edge and line detectors) |

`cnn.test.ts` checks shapes and parameter counts, one convolution cell against its formula, gradients
against finite differences (max / average / no pooling, ReLU / tanh, 3×3 / 5×5), learning, and the
analyses. `tests/pytorch-parity.test.ts` rebuilds three configurations with `nn.Conv2d`, `nn.MaxPool2d`,
`nn.AvgPool2d` and `nn.Linear`: logits, loss and every gradient agree to about 10⁻¹⁶.

The 3D view is `src/three/CnnView.ts`; the tab is `src/workspaces/cnn`.
