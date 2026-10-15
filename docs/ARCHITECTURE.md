# Neural Explorer 3D — architecture

How the project is built, **why** it is built this way, and where to extend it. Up to date with every
tab: Neural Network, Training, CNN, RNN, Transformer, LLM, Experiments and Learn.

---

## 1. Guiding principles

| Principle (from the specification) | How the code applies it |
| --- | --- |
| Every concept on **three levels**: visual → numbers → math | Every inspector is split into *1 · Visual*, *2 · Numbers*, *3 · Math* blocks; formulas show the real numbers substituted (KaTeX). |
| **The math does not depend on the rendering** | `math/`, `neural/`, `training/`, `cnn/`, `rnn/`, `transformer/`, `experiments/`, `visualization/` import neither three.js nor the DOM. Their tests run in Node. |
| **One consistent state** per model | `NetworkModel`, `TransformerTrace` and the experiment results are the single sources of truth; three.js and the UI only read them. |
| Networks are **generated from configuration** | Every change (preset, JSON editor, +/− neurons, import) goes through a validated `NetworkConfig`. |
| **Educational accuracy** | No unproven metaphors; the limits of every representation (random weights, PCA, logit lens, attention, attributions, ablations) are stated next to it. |
| **Performance** on older hardware | Render on demand, `InstancedMesh` everywhere (one draw call per kind of element), no post-processing, level of detail for large networks. |
| **Modularity** up to Transformers and LLMs | Workspaces, selections, inspectors and "traces" are explicit extension points (§ 9). |

---

## 2. Stack

| Library | Role | Why |
| --- | --- | --- |
| **TypeScript** (strict) | All the code | Types on tensor shapes, selections, configurations and backend payloads. |
| **Vite** | Dev server, build, `/api` proxy | Instant start, static build that works from any host. |
| **three.js** | 3D rendering | Required by the specification. Only official add-ons: `OrbitControls`, `CSS2DRenderer`. |
| **KaTeX** | Formulas (level 3) | Synchronous, fast typesetting without dependencies. |
| **Vitest** (dev) | Tests | The math core, models and engines are tested in Node. |
| **Python + PyTorch** (optional) | Parity checks, toy-Transformer training, GPT-2 backend | Only where a real model is needed; the app works without it. |
| **FastAPI + transformers** (optional) | GPT-2 backend | Hooks on a real model, JSON + base64 binary payloads. |

**No UI framework** (ADR-001): a tiny `h()` helper and class-based panels. Controls are created once and
their values updated in place, so a slider keeps working while the model recomputes 60 times per second.

---

## 3. Layers and dependencies

```
┌──────────────────────── app/   composition root, shell, UI state, help, workspace contract ────────────────────────┐
│ workspaces/  network · training · cnn · rnn · transformer · llm · experiments   lessons/  (drive the workspaces)   │
├────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────────────┤
│ ui/        (DOM)       │ three/       (WebGL)                 │ animation/  (AnimationController)                 │
├────────────────────────┴──────────────┬───────────────────────┴───────────────────────────────────────────────────┤
│ visualization/  layout · encodings · emphasis · palettes · selection            (no three.js, no DOM)            │
├───────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────┤
│ neural/ (MLP)  training/  cnn/  rnn/  transformer/ (toy model)  experiments/ (interventions)  llm/ (client)      │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ math/  activations · linalg · pca · random · stats                         core/  Emitter · Store · DisposeBag    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                  python/server  (FastAPI + PyTorch, optional, over HTTP)
```

Rule: **dependencies only point down.** `neural/` does not know three.js exists; `three/` does not know the
panels; panels never touch three.js — they call workspace actions.

### Folder map

```
src/
├── app/            App (composition root), shell (top bar, panels, stage), state, help, Workspace contract
├── core/           typed Emitter, observable Store, DisposeBag
├── math/           activations (+ derivatives + TeX), linear algebra, PCA (power iteration), seeded RNG, stats
├── neural/         MLP: configuration, topology, forward pass with trace, initialization, files, presets, signal flow
├── training/       datasets, losses, backpropagation, SGD / Momentum / Adam, Trainer, diagnostics, scenarios, landscape
├── cnn/            convolutional network: synthetic image datasets, model, forward / backward, trainer, analyses
├── rnn/            vanilla RNN and GRU: sequence tasks, model, BPTT, trainer
├── transformer/    toy Transformer: model loading, tokenizer, forward pass with full trace + interventions, sampling, analyses
├── llm/            backend client (fetch, decode base64 tensors), recorded demos, types of the payloads
├── experiments/    experiments as data, toy engine, GPT-2 engine, metrics, presets, JSON files
├── lessons/        lesson data (13 lessons), LessonContext, LessonPlayer, catalog, progress
├── visualization/  3D layout, visual encodings, emphasis, palettes, selection types
├── three/          Viewport, CameraController, NetworkView, CnnView, RnnView, TransformerView, EmbeddingView, LlmView, ExperimentView, ArcMesh, shaders, picking
├── animation/      AnimationController + easing
├── ui/             DOM helper, widgets, KaTeX, tooltip / toast / modal, InspectorPanel, charts
├── workspaces/     network/ · training/ · cnn/ · rnn/ · transformer/ · llm/ · experiments/ — panels, inspectors, bars, toolbars
├── utils/          number formatting, download, debounce / frame throttle
└── styles/         CSS with design tokens (dark and light themes)
```

---

## 4. The MLP model (Neural Network and Training tabs)

**Configuration → spec.** A tolerant `NetworkConfig` (JSON written by people) is validated by
`validateConfig()` — errors carry the path (`layers[2].size`) — and normalized into an immutable
`NetworkSpec` (roles, default activations, names).

**Parameters in PyTorch layout.** `W⁽ˡ⁾` is a row-major `Float64Array` of shape `[n_l, n_{l−1}]`, exactly
`torch.nn.Linear.weight`. Entry `W⁽ˡ⁾[j, i]` is the weight of connection *i (layer l−1) → j (layer l)*.

**Global indices mirror the buffers.** Neuron `(l, j)` → `neuronOffsets[l] + j`; connection `(l, j, i)` →
`connectionOffsets[l] + j·n_{l−1} + i`. The index of a connection inside its layer *is* the position of its
weight in the parameter buffer, so the instance buffers of three.js, the weight matrix in the inspector and
the lines in the scene are literally the same data.

**ForwardTrace.** `forwardInto()` fills `z[l]` and `a[l]` without allocating; it runs on every slider move
and every training step. The trace is the contract between "who computes" and "who shows".

**Training** (`src/training`). Manual backpropagation on the trace
(δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ f′(z⁽ˡ⁾), ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ), gradients in buffers with the same layout as
the weights, optimizers with per-parameter state. Every step is checked against PyTorch by
`npm run test:parity`. Scenarios (*Break it*) are pairs of settings: the mistake and its fix.

---

### 4.1 The CNN and the RNN (`src/cnn`, `src/rnn`)

Both follow the same pattern as the MLP: a model class holding `Float64Array` parameters in **PyTorch's
layouts** (`Conv2d.weight` [out, in, k, k]; `RNN` / `GRU` `weight_ih_l0`, `weight_hh_l0`, biases, GRU rows
ordered r, z, n), a forward pass that returns a full trace, a hand-written backward pass, and a trainer
using the shared optimizers. The data is generated from a seed (16×16 images of lines, shapes and digits;
the days of the week; a memory task with a configurable gap), so nothing is downloaded.

- The CNN uses "same" padding, stride 1 and 2×2 pooling; the trace keeps the max-pool winners, which both
  the backward pass (gradient routing) and the 3D view (highlighted winner) use.
- The RNN is unrolled through time; BPTT also returns ‖∂L/∂h_t‖ at every step — the measured version of the
  vanishing-gradient argument shown in the inspector (σ_max of W_hh, computed by power iteration).

Gradients are checked against finite differences in unit tests and against `torch.autograd` in
`npm run test:parity` (differences ≈ 10⁻¹⁶).

## 5. State

| | Models (`NetworkModel`, traces, results) | `Store<AppState>` |
| --- | --- | --- |
| Holds | architecture, parameters, inputs, computed values | selection, hover, view options, open panels, active tab |
| Shape | **mutable** objects with typed arrays | **immutable** object (a new one per change) |
| Notifications | events (`structure`, `params`, `activations`; per-workspace emitters) | `subscribe` / `watch(selector, callback)` |
| Why | large data updated at high frequency | small data, cheap equality by reference |

Workspaces of the Transformer, LLM and Experiments tabs keep their own state (text, selection, settings,
results) and emit typed events (`trace`, `selection`, `settings`, `result`, `busy`…) that panels and bars
subscribe to. Example flow — dragging a weight slider:

```
slider ─► model.setWeight() ─► forward recomputed ─► "params" + "activations" events
                                                      ├─► NetworkView.updateParams()      (one GPU upload)
                                                      ├─► NetworkView.updateActivations()
                                                      └─► InspectorPanel.refresh()        (at most once per frame)
```

---

## 6. Rendering

- **Viewport** — renderer, scene, camera, `OrbitControls` (zoom to cursor, pan), HTML labels
  (`CSS2DRenderer`) and the loop. **Render on demand**: a frame is drawn only when something changed.
- **One view class per tab**, each owning a group in the shared scene and disposed on unmount:
  `NetworkView` (instanced spheres and cylinders, dashed negative weights in the fragment shader),
  `TransformerView` (vectors as instanced bar tiles, attention arcs, MLP blocks, prediction cards),
  `EmbeddingView` (points and trails), `LlmView` (logit-lens tower), `ExperimentView` (component wall and
  residual-change wall), `CnnView` (feature maps as rows of cells, receptive-field windows and weight
  lines), `RnnView` (steps unrolled left to right, hidden-state tiles, recurrent ribbons, fold animation,
  BPTT gradient bars).
- **ArcMesh** — attention arcs as camera-facing quadratic Bézier ribbons evaluated in the vertex shader,
  all in one draw call, with screen-space picking.
- **Diverging bars** encode sign by direction (up/down, or in front of / behind a wall) as well as colour.
- **Picking** — screen-space for thin things (connections, arcs), ray casts against invisible boxes for
  tiles and cells.
- **Camera** — preset views framed on the real bounding box projected in the frustum; animated flights in
  spherical coordinates; the framing follows panel and window resizes until the user moves the camera; the
  camera position is remembered per tab.
- **Visual effects** (`Atmosphere`, `fx.ts`; on by default, ✨ button or **M**) — HDR rendering into a
  half-float target with 4× MSAA, bloom, a light vignette and colour fringe, Khronos PBR Neutral tone
  mapping (colours below the highlights are left as they are); a backdrop with faint nebulae, drifting dust
  and a floor grid fading into the distance, placed from the bounds the camera frames. Glow encodes values:
  materials share the `FX` uniforms and emit light in proportion to the value an element shows (activation,
  residual value, attention weight, probability, effect size), and in the Neural Network tab light flows
  along each connection as bright as its contribution w·a. The backdrop, dust and floor use neutral colours
  and carry no values. Light theme: gradient and floor only.

---

## 7. Animation

Everything time-based goes through the `AnimationController`, ticked once per frame by the Viewport:
tweens with easing, **channels** (a new animation on a channel cancels the previous one), *scaled*
animations obeying `timeScale` / `paused` (forward-pass player, backprop player, Transformer pipeline) and
UI animations that do not (camera, layout, bar morphs), `prefers-reduced-motion` respected.

---

## 8. The Transformer, the LLM backend and the experiments

### 8.1 Toy Transformer (`src/transformer`)

Trained in PyTorch (`python/toy_transformer/train.py`: GPT-2-style pre-LayerNorm, tied embeddings, GPT-2
initialization), exported to `public/models/toy-transformer.json` with weights rounded to 6 significant
digits **and reference outputs computed in float64 with the rounded weights** — so the TypeScript forward
pass can be tested exactly (logits to 8 decimals, attention to 9).

`runTransformer(model, tokens, interventions)` returns a `TransformerTrace` with every intermediate vector
(Q, K, V per head, scores, attention, per-head contributions, residual stream before / after each
sub-layer, MLP activations, logits). Interventions (scale heads or MLP layers, set or scale neurons, edit
embeddings) are applied inside the computation; the weights are never touched.

### 8.2 GPT-2 backend (`python/server`, `src/llm`)

FastAPI app with GPT-2 small loaded in a background thread at start-up (`/api/llm/status` reports it). The
Vite dev and preview servers proxy `/api` to `127.0.0.1:8000`; when the backend is down, the proxy answers
the status check quietly and the tabs fall back to recordings in `public/llm-demo`.

| Endpoint | Used by | Returns |
| --- | --- | --- |
| `GET /status` | LLM, Experiments | model, ready / loading / error, configuration |
| `POST /analyze` | LLM | tokens, next-token logits (float32, base64), logit lens per layer, attention (uint16, base64), norms, token journey (PCA), rollout, gradient × input |
| `POST /generate` | LLM | sampled or greedy continuation with alternatives per step |
| `POST /tokenize` | Experiments | GPT-2 tokens of a short text |
| `POST /experiment` | Experiments | before / after predictions, Δ log p, Δ p, KL, Δ logit diff, residual change per stage and position |
| `POST /sweep` | Experiments | the effect of zeroing every head and every MLP layer alone |
| `POST /neurons` | Experiments | gradient × activation estimates for all 3,072 neurons of a layer, measured effects of the strongest |
| `POST /curve` | Experiments | p(target) while one component is scaled |
| `POST /attention` | Experiments | one head's attention before / after the interventions |

Interventions are forward hooks (pre-hook on `attn.c_proj` for heads, hook on `mlp` for MLP layers, on
`mlp.act` for neurons; token edits through `inputs_embeds`). **One computation at a time**: every request
holds `llm.compute_lock`, because hooks live on the shared model and the server answers on several threads
(ADR-017). Batched sweeps compute the part of the network before each component's layer once and run only
the layers after it (`run_from`), about 2.4× faster than full batched passes.

### 8.3 Experiments (`src/experiments`)

An experiment is **data** — model, text, measured token, optional contrast token, list of interventions —
with the same JSON shape for the browser engine (`toyEngine.ts`, exact and instantaneous) and the Python
engine (`experiments.py`); files are versioned (`neural-explorer-3d/experiment`). Results share one type
(`ExperimentResult`, `SweepResult`, `NeuronScan`, `ScaleCurve`) whichever engine produced them, so the
workspace, the 3D view and the inspectors do not care where the numbers come from. Toy presets are verified
by unit tests; GPT-2 presets are recorded with their results by `npm run llm:demos`.

---

## 9. Extension points

- **Workspaces** — `app/workspace.ts`: `Workspace { mount(host); unmount() }`. Each tab receives the shared
  viewport, camera, store and panel slots, and keeps its own state between visits.
- **Selections and inspectors** — each workspace has a discriminated union of what can be selected and one
  `InspectorView` per kind, always with the three levels.
- **Layer types** — the configuration validator is where `conv2d` or `rnn` layers would be added.
- **Traces** — the visualization never computes; it reads traces / results. A new model (e.g. Pythia) needs
  a backend function returning the same payloads.
- **Lessons** — `src/lessons`: a lesson is data (steps with `enter`, text, formula, task); the
  `LessonContext` exposes the workspaces through their public actions, so lessons contain no rendering code.

---

## 10. Python and PyTorch

- `python/mlp_bridge.py` rebuilds exported MLPs as `nn.Sequential` and compares outputs, or exports an MLP
  created in PyTorch; `python/cnn_rnn_bridge.py` rebuilds the CNN (`nn.Conv2d`, pooling, `nn.Linear`) and the
  recurrent cells (`nn.RNN`, `nn.GRU`) and compares logits, loss and every gradient; `npm run test:parity`
  runs both (largest differences ≈ 2·10⁻¹⁶ in float64).
- `python/toy_transformer/` trains and exports the toy Transformer (§ 8.1).
- `python/server/` is the GPT-2 backend (§ 8.2); `record_demos.py` records the analyses and experiments used
  when the backend is off.

---

## 11. Architecture decision records

**ADR-001 · No UI framework.** TypeScript + `h()`. Minimal dependencies, full control over high-frequency
updates. Controls are created once and updated in place.

**ADR-002 · Imperative three.js.** Direct control over `InstancedMesh`, buffers and shaders; each view is a
class that *reads* the model.

**ADR-003 · Two kinds of state** (mutable models with events + immutable UI store), § 5.

**ADR-004 · Row-major `[out, in]` weights in `Float64Array`.** 1:1 with `nn.Linear`; float64 so that the
hand calculations shown in the inspector add up.

**ADR-005 · Global indices mirror the parameter buffers** (§ 4): no duplicated objects.

**ADR-006 · Screen-space picking** for thin geometry (lines, arcs).

**ADR-007 · Render on demand.**

**ADR-008 · Custom shaders with per-instance attributes** (colour, emphasis, dashes, values): one draw call
per kind of element; sign never encoded by colour alone.

**ADR-009 · KaTeX for the math level**, with the actual numbers substituted.

**ADR-010 · Versioned JSON files** for models, training sessions and experiments.

**ADR-011 · Python only where a real model is needed.** Training small networks and running the toy
Transformer happen in the browser: more responsive, nothing to install.

**ADR-012 · Deterministic, seeded initialization per layer**: resizing a layer does not change the others'
initial weights; every experiment is reproducible.

**ADR-013 · View preferences in `localStorage`** (theme, layout, open sections, lesson progress): comfort
only; models, sessions and experiments are saved explicitly as files.

**ADR-014 · The toy Transformer ships with reference outputs** computed from its *rounded* exported
weights, so the browser implementation is tested for exact agreement with PyTorch.

**ADR-015 · Recordings as a fallback.** The LLM and Experiments tabs never require the backend: recorded
analyses and experiments (with their sweeps) are static files, and the tabs switch to live computation as
soon as the backend answers.

**ADR-016 · Experiments are data, shared by two engines.** The same intervention list runs in TypeScript
on the toy model and in PyTorch on GPT-2, and can be saved, reloaded and re-run.

**ADR-017 · One model computation at a time in the backend.** Hooks are installed on the shared model for
the duration of a request; concurrent requests would see each other's hooks, so they are serialized with a
lock (requests are short and CPU-bound, so nothing is lost).

**ADR-018 · Lessons drive the real workspaces.** No separate "lesson scenes": a lesson calls the same
actions as the panels and checks its tasks against the real model state.

**ADR-019 · Synthetic data for the CNN and RNN tabs.** Seeded generators instead of downloaded datasets:
no licences or downloads, instant reproducible runs, and tasks chosen so that the learned behaviour can
be checked (strokes for edge-like filters, "saturday"/"sunday" for memory, a variable gap for vanishing
gradients).

**ADR-020 · Hand-written backward passes, verified twice.** The CNN and RNN gradients are written out
(no autodiff library), so the inspectors can show every intermediate value; finite differences and
PyTorch's autograd check them.

**ADR-021 · Visual effects that encode values, and can be switched off.** Glow and moving light make the
scenes easier to read at a glance (where the large values are, where the signal goes), so they are tied to
the same values as the colours, never decorative; the decorative parts (backdrop, dust, floor) stay
neutral. The ambient motion breaks render-on-demand (frames at 30 fps), so it stops with
`prefers-reduced-motion`, the post-processed image is capped at a pixel ratio of 1.5, and switching the
effects off restores the original renderer exactly (direct rendering, no tone mapping, no idle frames).

---

## 12. Educational accuracy

- Random weights are labelled as such; brightness and thickness are relative to a stated reference.
- The arrangement of neurons inside a layer has no mathematical meaning (stated in the help).
- PCA plots state the variance they keep; the loss landscape is labelled as a 2-D slice; the logit lens,
  attention maps, rollout and gradient × input are presented as probes with their limits.
- Ablations are described as strong, unnatural interventions measured on one input; effects do not add up;
  a large effect is not an explanation of what a component means. GPT-2 heads are named with 1-based layers
  and heads as in the rest of the app, with the 0-based notation of the papers alongside.
- Neuron states are described mathematically ("inactive: z ≤ 0, so ReLU outputs 0"), never
  anthropomorphically.

---

## 13. Measurements

| What | Time |
| --- | --- |
| MLP preset change, 784-32-16-10 (842 neurons, 25,760 weights; 17 draw calls) | 60–90 ms |
| Toy Transformer forward pass with full trace | ≈ 1 ms |
| Toy experiment + sweep of all heads and MLPs | < 50 ms |
| Toy neuron scan (128 neurons, measured + estimated) | ≈ 250 ms (deferred, off the slider path) |
| GPT-2 `/analyze` (≈ 15 tokens) | 0.2–0.35 s |
| GPT-2 `/experiment` | 0.15–0.35 s |
| GPT-2 `/sweep` (144 heads + 12 MLPs) | ≈ 1.2 s |
| GPT-2 `/neurons` (3,072 estimated, 24 measured) | ≈ 0.6 s |
| CNN, one epoch (600 images, conv 4 + conv 8, forward + backward) | ≈ 0.45 s |
| GRU, one training step (16 sequences × 32 steps, 24 units) | ≈ 15 ms |

Measured on this Mac (CPU) with the dev server.
