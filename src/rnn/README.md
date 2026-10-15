# `src/rnn`

The recurrent networks of the RNN tab — a vanilla (Elman) RNN and a GRU — trained in the browser with
backpropagation through time. Pure logic, tested in Node and against PyTorch.

| File | Content |
| --- | --- |
| `model.ts` | `RnnModel`: exactly `torch.nn.RNN(tanh)` / `torch.nn.GRU` + `nn.Linear`, same parameter layouts (GRU rows ordered r, z, n) and initialization; optional update-gate bias for the GRU (starts by keeping its state) |
| `forward.ts` | the network unrolled through time with a full trace (input and recurrent parts, gates, states, predictions); BPTT with ‖∂L/∂h_t‖ at every step; global gradient clipping |
| `tasks.ts` | *Next letter* (the days of the week in order: "saturday" vs "sunday" needs the previous word) and *Remember* (a letter, a gap of meaningless symbols, then "?") |
| `trainer.ts` | mini-batches of sequences, mean cross-entropy over the scored steps, clipping, evaluation, text generation |

`rnn.test.ts` checks the tasks, BPTT against finite differences for both cells, that a GRU with its update
gate at 1 keeps its state, clipping, learning the days of the week (and continuing "friday s" with
"aturday"), and that over a 30-symbol gap the gradient reaching the first step vanishes in a vanilla RNN
(< 10⁻⁴ of the last step's) but not in a GRU that starts remembering (> 10⁻²).
`tests/pytorch-parity.test.ts` compares both cells with `nn.RNN` / `nn.GRU`: logits, loss and every BPTT
gradient agree to about 10⁻¹⁶.

The 3D view is `src/three/RnnView.ts`; the tab is `src/workspaces/rnn`.
