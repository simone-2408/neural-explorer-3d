# `src/transformer`

The tiny educational Transformer of the Transformer tab, computed in the browser. It was trained in PyTorch
(`python/toy_transformer`) and is loaded from `public/models/toy-transformer.json`.

| File | Content |
| --- | --- |
| `model.ts` | file format and loading into flat `Float64Array`s (PyTorch layout: `nn.Linear.weight` is `[out, in]`) |
| `loader.ts` | fetches the weights once |
| `tokenizer.ts` | word-level tokenizer: lower case, punctuation as tokens, `<s>` prepended, unknown words → `<unk>` |
| `forward.ts` | `runTransformer()`: the forward pass with a full `TransformerTrace` (Q, K, V per head, scores, attention, per-head contributions, residual stream before / after every sub-layer, MLP activations, logits) and optional interventions (scale heads or MLP layers, set or scale neurons, edit embeddings) |
| `sampling.ts` | temperature, top-k, top-p, sampling, entropy |
| `analysis.ts` | read-outs for the inspectors: top tokens, logit lens, nearest words, attention summaries, per-head contributions, the direct vocabulary effect of an MLP unit, PCA projections |

`transformer.test.ts` checks that the logits and attention equal the reference outputs computed by PyTorch
(to 8 and 9 decimals). Several read-outs are probes rather than ground truth (logit lens, direct effects,
PCA): the inspectors state their limits next to the numbers.
