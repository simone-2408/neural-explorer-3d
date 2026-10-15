# `src/training`

Training of the small MLPs of the Training tab. Pure logic (no three.js, no DOM), tested in Node and
against PyTorch (`npm run test:parity`).

| File | Content |
| --- | --- |
| `datasets.ts` | seeded 2-D datasets: XOR, AND, OR, linear, circles, moons, spiral, blobs; train / test split |
| `loss.ts` | MSE, binary cross-entropy, cross-entropy with softmax, with their derivatives |
| `backprop.ts` | backpropagation on the forward trace: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ f′(z⁽ˡ⁾), ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ; gradients use the layout of the weights |
| `optimizers.ts` | SGD, Momentum, Adam (per-parameter state, exportable) |
| `Trainer.ts` | mini-batch loop, epochs, history (loss, accuracy, gradient norms per layer) |
| `session.ts` | training settings and the network they describe; network templates |
| `diagnostics.ts` | live diagnostics of the classic failure modes, each a measured condition explained with its numbers |
| `scenarios.ts` | *Break it*: classic mistakes as (settings, fixed settings, explanation) |
| `landscape.ts` | loss on a 2-D slice of parameter space (random or PCA directions) |
| `comparisons.ts` | ready-made A/B comparisons for the *Compare* mode: same data and batches, one difference |

The 3D view and the panels live in `src/workspaces/training`.
