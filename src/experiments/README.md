# `src/experiments`

Controlled experiments (the Experiments tab): change one part of a Transformer, run it again, and measure
what changed — always compared with the untouched model.

| File | Content |
| --- | --- |
| `types.ts` | an experiment as data (model, text, measured token, optional contrast token, interventions) and the result types shared by both engines |
| `toyEngine.ts` | the toy Transformer in the browser: weight edits (copy-on-write), token removal and embedding swaps, comparison, sweeps of all heads and MLP layers, neuron scans (measured + first-order estimate), scale curves |
| `gpt2Engine.ts` | the same operations on GPT-2 through the backend (`python/server/experiments.py`) |
| `metrics.ts` | Δ log p, Δ p, KL (bits), Δ logit difference; how much the target “relies on” a component |
| `presets.ts` | ready-made toy experiments (checked by the tests) and loading of the recorded GPT-2 ones |
| `serialization.ts` | experiment files (`neural-explorer-3d/experiment`, version 1) with validation |

Interventions: `head` (output × scale), `mlp` (output × scale), `neuron` (scale or set after GELU),
`embedding` (mix towards another token's embedding), `remove` (token removed from the input), `weight`
(toy model only: a weight matrix × scale and/or one entry set).

`experiments.test.ts` checks the engine (scale 1 is the identity, `W_O × 0` equals switching the head off,
removing a token leaves earlier tokens untouched, sweeps equal single runs…) and that every toy preset does
what its description says.
