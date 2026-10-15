# Python / PyTorch

Everything here is optional: the app runs without Python. It is used for three things.

| Folder / file | What it does | Needs |
| --- | --- | --- |
| `mlp_bridge.py` | Checks the browser's MLPs against PyTorch, and imports PyTorch MLPs into the app | `pip install -r python/requirements.txt` |
| `cnn_rnn_bridge.py` | Checks the CNN and the RNN / GRU of the app against PyTorch | the same |
| `toy_transformer/` | Trains the tiny Transformer of the Transformer tab and exports it | the same |
| `server/` | The GPT-2 backend of the LLM and Experiments tabs (FastAPI) | `npm run llm:setup` |

## `mlp_bridge.py`

| Command | What it does |
| --- | --- |
| `python3 python/mlp_bridge.py verify file.model.json [...]` | rebuilds each model exported by the app as `nn.Sequential`, runs the forward pass in float64 and compares it with the output computed in the browser |
| `python3 python/mlp_bridge.py export-demo out.json [--seed N]` | creates an MLP in PyTorch (default `nn.Linear` initialization) and saves it in the app's format (then *Import* it) |
| `python3 python/mlp_bridge.py summary file.model.json` | prints the architecture and the parameter shapes |

| App (JSON) | PyTorch |
| --- | --- |
| `dense` / `output` layer of size n after a layer of size m | `nn.Linear(m, n)` |
| `parameters[l-1].weight` (n × m) | `linear.weight` |
| `parameters[l-1].bias` (n) | `linear.bias` |
| `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `softmax` | `nn.ReLU()`, `nn.LeakyReLU(0.01)`, `nn.Sigmoid()`, `nn.Tanh()`, `nn.GELU()`, `nn.Softmax(dim=-1)` |

`npm run test:parity` (from the project root) exports every preset from the TypeScript engine, verifies
them with this script, checks training steps (SGD, Momentum, Adam) against `torch.optim`, and the reverse
path (a model created in PyTorch loaded into the app).

## `cnn_rnn_bridge.py`

```bash
python3 python/cnn_rnn_bridge.py verify case.json [more.json ...]
```

Each case file (written by `tests/pytorch-parity.test.ts`) holds a model as the app stores it, a batch of
inputs and what the browser computed: logits, mean cross-entropy and the gradient of every parameter. The
script rebuilds the model in float64 — `nn.Conv2d` / `MaxPool2d` / `AvgPool2d` / `Linear` for the CNN,
`nn.RNN` or `nn.GRU` + `nn.Linear` for the recurrent network (one-hot inputs, `ignore_index=-1` for unscored
steps) — recomputes everything with autograd and prints the largest differences (≈ 10⁻¹⁶).

## `toy_transformer/`

```bash
python3 python/toy_transformer/train.py
```

Options: `--steps 4000`, `--seed 0`, `--init gpt2|pytorch` (GPT-2's initialization by default: N(0, 0.02),
residual projections scaled by 1/√(2·layers)), `--weight-decay 0.01`, `--out path`.

`corpus.py` generates the training sentences from templates (capitals and countries, royals, animals and
places, predators and prey, counting…). The model is GPT-style: pre-LayerNorm, causal self-attention, GELU
MLP, tied input/output embeddings; 2 layers, 4 heads, d = 32, MLP 128, context 16, 83 words, about 28,600
parameters.

The export (`public/models/toy-transformer.json`) contains the weights rounded to 6 significant digits,
the word groups used for colours, the training summary (sentences, steps, loss, probes) and **reference
outputs** (token ids, logits, attention) computed in float64 with the rounded weights. `npm test` checks
that the browser reproduces them (logits to 8 decimals).

## `server/` — the GPT-2 backend

```bash
npm run llm:setup
```

```bash
npm run llm
```

The first command creates `python/.venv` (with access to the system's site-packages, so an existing PyTorch
is reused) and installs `requirements-llm.txt`. The second starts uvicorn on http://127.0.0.1:8000; GPT-2
small (≈ 550 MB) is downloaded from Hugging Face into `~/.cache/huggingface` the first time and loads in a
background thread (`/api/llm/status` reports `loading` → `ready`). The Vite dev server proxies `/api` to it.

| File | Content |
| --- | --- |
| `app.py` | FastAPI app, request validation, errors (400 for a bad experiment, 503 while the model loads) |
| `llm.py` | `LLM`: loading, tokens (byte-level BPE pieces and bytes), `analyze` (logit lens, attention, token journey, rollout, gradient × input), `generate` |
| `experiments.py` | `Experiments`: interventions as forward hooks, `experiment`, `sweep`, `neurons`, `curve`, `attention`, `tokenize` |
| `record_demos.py` | records the examples of the LLM tab and the GPT-2 presets of the Experiments tab into `public/llm-demo/` (`npm run llm:demos`) |

Endpoints and payloads are described in `docs/ARCHITECTURE.md` § 8.2. Tensors are sent as little-endian
float32 / uint16 in base64 inside the JSON. All computations hold one lock: hooks are installed on the
shared model for the duration of a request.
