"""
Controlled experiments on GPT-2 small (Experiments tab).

An experiment is a list of interventions, sent by the browser with the same
JSON shape it uses for the toy Transformer (src/experiments/types.ts):

    {"kind": "head",      "layer": 9, "head": 6, "scale": 0}     head output × scale (0 = off)
    {"kind": "mlp",       "layer": 3, "scale": 0}                whole MLP output × scale
    {"kind": "neuron",    "layer": 5, "unit": 12, "mode": "scale" | "set", "value": 0}
    {"kind": "embedding", "position": 3, "token": {"id": 6342}, "mix": 1}   token embedding → another token's
    {"kind": "remove",    "position": 2}                         the token is removed from the input

Weights are never modified: everything is done with forward hooks on a copy of
the computation, so the same model serves every request.

What is measured (always at the last position, i.e. the next-token prediction):

    dlogp   change of log p(target)
    dp      change of p(target)
    kl      KL(before ‖ after) of the whole next-token distribution, in bits
    dlogit  change of logit(target) − logit(contrast), when a contrast token is given

and, for the 3-D view, the relative change ‖x_after − x_before‖ / ‖x_before‖ of
the residual stream at every stage (embeddings, then after attention / after
the MLP of each layer) and position.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch

from .llm import LLM

LN2 = math.log(2)


class BadRequest(ValueError):
    """An experiment that cannot be run as described (reported as HTTP 400)."""


# A per-row extra intervention for batched runs: one component scaled in one batch row.
Row = tuple  # ("head", layer, head, scale) | ("mlp", layer, scale) | ("neuron", layer, unit, scale)


@dataclass
class Edits:
    heads: dict[tuple[int, int], float] = field(default_factory=dict)
    mlps: dict[int, float] = field(default_factory=dict)
    neurons: list[tuple[int, int, str, float]] = field(default_factory=list)
    remove: set[int] = field(default_factory=set)
    replace: dict[int, tuple[int, float]] = field(default_factory=dict)

    @staticmethod
    def parse(items: list[dict], llm: LLM, T: int, skip: dict | None = None) -> "Edits":
        """Validates the interventions; `skip` leaves out one component (for scale curves)."""
        c = llm.require().config
        layers, heads, units, vocab = c.n_layer, c.n_head, 4 * c.n_embd, c.vocab_size
        e = Edits()

        def check(value, upper: int, name: str) -> int:
            if not isinstance(value, int) or not 0 <= value < upper:
                raise BadRequest(f"{name} must be an integer in [0, {upper - 1}], got {value!r}")
            return value

        def number(value, name: str) -> float:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e4:
                raise BadRequest(f"{name} must be a finite number, got {value!r}")
            return float(value)

        for item in items:
            kind = item.get("kind")
            if kind == "head":
                key = (check(item.get("layer"), layers, "layer"), check(item.get("head"), heads, "head"))
                if skip == {"kind": "head", "layer": key[0], "head": key[1]}:
                    continue
                e.heads[key] = e.heads.get(key, 1.0) * number(item.get("scale"), "scale")
            elif kind == "mlp":
                layer = check(item.get("layer"), layers, "layer")
                if skip == {"kind": "mlp", "layer": layer}:
                    continue
                e.mlps[layer] = e.mlps.get(layer, 1.0) * number(item.get("scale"), "scale")
            elif kind == "neuron":
                layer, unit = check(item.get("layer"), layers, "layer"), check(item.get("unit"), units, "unit")
                if skip == {"kind": "neuron", "layer": layer, "unit": unit}:
                    continue
                mode = item.get("mode")
                if mode not in ("scale", "set"):
                    raise BadRequest(f"neuron mode must be 'scale' or 'set', got {mode!r}")
                e.neurons.append((layer, unit, mode, number(item.get("value"), "value")))
            elif kind == "remove":
                e.remove.add(check(item.get("position"), T, "position"))
            elif kind == "embedding":
                position = check(item.get("position"), T, "position")
                token = item.get("token") or {}
                e.replace[position] = (check(token.get("id"), vocab, "token id"), number(item.get("mix", 1), "mix"))
            elif kind == "weight":
                raise BadRequest("Weight edits are only available for the toy Transformer (GPT-2's weights stay read-only).")
            else:
                raise BadRequest(f"Unknown intervention kind {kind!r}")
        if len(e.remove) >= T:
            raise BadRequest("At least one token must remain.")
        return e


def effect(base: torch.Tensor, run: torch.Tensor, target: int, contrast: int | None) -> dict:
    """Change of the last-position prediction between two log-probability vectors (float64)."""
    dlogit = None
    if contrast is not None:
        dlogit = float((run[target] - run[contrast]) - (base[target] - base[contrast]))
    return {
        "dlogp": round(float(run[target] - base[target]), 6),
        "dp": round(float(run[target].exp() - base[target].exp()), 6),
        "kl": round(max(0.0, float((base.exp() * (base - run)).sum()) / LN2), 6),
        "dlogit": None if dlogit is None else round(dlogit, 6),
    }


class Experiments:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    # --------------------------------------------------------------- plumbing

    def tokens_of(self, text: str | None, ids: list[int] | None) -> list[int]:
        ids, _ = self.llm.encode(text, ids)
        return ids

    def prepare(self, ids: list[int], e: Edits) -> tuple[list[int], list[int], torch.Tensor | None]:
        """Edited input: (ids after removals, original position of each, input embeddings if replaced)."""
        model = self.llm.require()
        alignment = [p for p in range(len(ids)) if p not in e.remove]
        ids2 = [ids[p] for p in alignment]
        if not e.replace:
            return ids2, alignment, None
        wte = model.transformer.wte.weight
        embeds = wte[torch.tensor(ids2)].clone()
        for position, (token, mix) in e.replace.items():
            if position in alignment:
                j = alignment.index(position)
                embeds[j] = (1 - mix) * wte[ids[position]] + mix * wte[token]
        return ids2, alignment, embeds

    def forward(self, ids: list[int], embeds: torch.Tensor | None, batch: int = 1, attentions: bool = False):
        model = self.llm.require()
        if embeds is not None:
            return model(inputs_embeds=embeds.unsqueeze(0).expand(batch, -1, -1), output_attentions=attentions)
        return model(torch.tensor([ids] * batch), output_attentions=attentions)

    def run_from(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Last-position log-probabilities after running blocks layer… from the residual stream `hidden` (batch × T × d).

        Batched sweeps start each component's rows at its own layer: everything
        before it is identical to the baseline run and is computed once.
        """
        model = self.llm.require()
        for block in model.transformer.h[layer:]:
            hidden = block(hidden)[0]  # GPT-2's eager attention applies the causal mask itself
        return model.lm_head(model.transformer.ln_f(hidden[:, -1])).double().log_softmax(-1)

    def baseline(self, ids2: list[int], embeds: torch.Tensor | None, e: Edits) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run with the edits: last-position log-probabilities and the residual stream at every stage."""
        capture: dict[int, torch.Tensor] = {}
        with self.install(e, capture=capture):
            logits = self.forward(ids2, embeds).logits[0, -1]
        return logits.double().log_softmax(-1), capture

    @contextmanager
    def install(
        self,
        e: Edits,
        rows: list[Row] | None = None,
        capture: dict[int, torch.Tensor] | None = None,
        act_capture: tuple[int, list[torch.Tensor]] | None = None,
    ) -> Iterator[None]:
        """
        Hooks applying `e` to every batch row, plus `rows[r]` to batch row r only.
        `capture` receives the residual stream at each stage (0 = embeddings,
        2l + 1 = after attention l, 2l + 2 = after MLP l); `act_capture` the MLP
        activations of one layer, kept in the autograd graph.
        """
        model = self.llm.require()
        c = model.config
        size = c.n_embd // c.n_head
        rows = rows or []
        handles = []
        for layer, block in enumerate(model.transformer.h):
            head_base = {h: s for (l, h), s in e.heads.items() if l == layer}
            head_rows = [(r, row[2], row[3]) for r, row in enumerate(rows) if row[0] == "head" and row[1] == layer]
            if head_base or head_rows:
                def pre(_module, args, head_base=head_base, head_rows=head_rows):
                    x = args[0].clone()
                    for h, s in head_base.items():
                        x[..., h * size : (h + 1) * size] *= s
                    for r, h, s in head_rows:
                        x[r, :, h * size : (h + 1) * size] *= s
                    return (x,) + args[1:]

                handles.append(block.attn.c_proj.register_forward_pre_hook(pre))

            mlp_base = e.mlps.get(layer)
            mlp_rows = [(r, row[2]) for r, row in enumerate(rows) if row[0] == "mlp" and row[1] == layer]
            if mlp_base is not None or mlp_rows:
                def mlp_hook(_module, _inputs, output, mlp_base=mlp_base, mlp_rows=mlp_rows):
                    out = output * mlp_base if mlp_base is not None else output.clone()
                    for r, s in mlp_rows:
                        out[r] = out[r] * s
                    return out

                handles.append(block.mlp.register_forward_hook(mlp_hook))

            unit_base = [(u, mode, v) for (l, u, mode, v) in e.neurons if l == layer]
            unit_rows = [(r, row[2], row[3]) for r, row in enumerate(rows) if row[0] == "neuron" and row[1] == layer]
            if unit_base or unit_rows:
                def act_hook(_module, _inputs, output, unit_base=unit_base, unit_rows=unit_rows):
                    out = output.clone()
                    for u, mode, v in unit_base:
                        out[..., u] = v if mode == "set" else out[..., u] * v
                    for r, u, s in unit_rows:
                        out[r, :, u] = out[r, :, u] * s
                    return out

                handles.append(block.mlp.act.register_forward_hook(act_hook))

            if act_capture is not None and act_capture[0] == layer:
                handles.append(block.mlp.act.register_forward_hook(lambda _m, _i, output: act_capture[1].append(output)))

            if capture is not None:
                handles.append(block.register_forward_pre_hook(lambda _m, args, layer=layer: capture.__setitem__(2 * layer, args[0].detach())))
                handles.append(block.ln_2.register_forward_pre_hook(lambda _m, args, layer=layer: capture.__setitem__(2 * layer + 1, args[0].detach())))
                if layer == c.n_layer - 1:
                    handles.append(block.register_forward_hook(lambda _m, _i, output, layer=layer: capture.__setitem__(2 * layer + 2, output[0].detach())))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def target_ids(self, base_last: torch.Tensor, target: int | None, contrast: int | None) -> tuple[int, int | None]:
        vocab = self.llm.require().config.vocab_size
        t = int(base_last.argmax()) if target is None else int(target)
        for value, name in ((t, "target"), (contrast, "contrast")):
            if value is not None and not 0 <= value < vocab:
                raise BadRequest(f"{name} token id out of range")
        return t, contrast

    def ref(self, token: int | None) -> dict | None:
        return None if token is None else {"id": int(token), "text": self.llm.word(token)}

    def outcome(self, lp: torch.Tensor, logits: torch.Tensor, target: int, contrast: int | None) -> dict:
        """Prediction summary of one run: lp = log-probabilities (T × V, float64), logits = raw logits."""
        last = lp[-1]
        top = last.exp().topk(10)
        best = lp.max(-1)
        return {
            "top": [{"id": int(i), "text": self.llm.word(int(i)), "p": round(float(p), 6)} for p, i in zip(top.values, top.indices)],
            "pTarget": float(last[target].exp()),
            "logitTarget": round(float(logits[-1, target]), 4),
            "logitContrast": None if contrast is None else round(float(logits[-1, contrast]), 4),
            "predictions": [{"id": int(i), "text": self.llm.word(int(i)), "p": round(float(p), 5)} for p, i in zip(best.values.exp(), best.indices)],
        }

    # ------------------------------------------------------------- endpoints

    @torch.no_grad()
    def tokenize(self, text: str) -> dict:
        self.llm.require()
        ids = self.llm.tokenizer.encode(text or "")[:64]
        return {"tokens": self.llm.token_info(ids)}

    @torch.no_grad()
    def experiment(self, text: str | None, ids: list[int] | None, items: list[dict], target: int | None, contrast: int | None) -> dict:
        model = self.llm.require()
        start = time.time()
        ids, truncated = self.llm.encode(text, ids)
        e = Edits.parse(items, self.llm, len(ids))
        ids2, alignment, embeds = self.prepare(ids, e)
        before_cap: dict[int, torch.Tensor] = {}
        with self.install(Edits(), capture=before_cap):
            before = self.forward(ids, None).logits[0]
        after_cap: dict[int, torch.Tensor] = {}
        with self.install(e, capture=after_cap):
            after = self.forward(ids2, embeds).logits[0]
        lp0, lp1 = before.double().log_softmax(-1), after.double().log_softmax(-1)
        t, c = self.target_ids(lp0[-1], target, contrast)
        stages = 2 * model.config.n_layer + 1
        delta = []
        for s in range(stages):
            r0, r1 = before_cap[s][0].double(), after_cap[s][0].double()
            row = []
            for p in range(len(ids)):
                if p not in alignment:
                    row.append(None)
                    continue
                j = alignment.index(p)
                row.append(round(float((r1[j] - r0[p]).norm() / r0[p].norm().clamp_min(1e-9)), 5))
            delta.append(row)
        return {
            "tokens": self.llm.token_info(ids),
            "tokensAfter": self.llm.token_info(ids2),
            "alignment": alignment,
            "truncated": truncated,
            "target": self.ref(t),
            "contrast": self.ref(c),
            "before": self.outcome(lp0, before, t, c),
            "after": self.outcome(lp1, after, t, c),
            "effect": effect(lp0[-1], lp1[-1], t, c),
            "delta": delta,
            "layers": model.config.n_layer,
            "heads": model.config.n_head,
            "seconds": round(time.time() - start, 3),
        }

    @torch.no_grad()
    def sweep(self, text: str | None, ids: list[int] | None, items: list[dict], target: int | None, contrast: int | None) -> dict:
        """Zero every head alone and every MLP layer alone (one batched pass) on top of `items`."""
        model = self.llm.require()
        start = time.time()
        ids = self.tokens_of(text, ids)
        e = Edits.parse(items, self.llm, len(ids))
        ids2, _, embeds = self.prepare(ids, e)
        base, capture = self.baseline(ids2, embeds, e)
        t, c = self.target_ids(base, target, contrast)
        L, H = model.config.n_layer, model.config.n_head
        heads: list[list[dict]] = []
        mlps: list[dict] = []
        for layer in range(L):
            rows: list[Row] = [("head", layer, h, 0.0) for h in range(H)] + [("mlp", layer, 0.0)]
            with self.install(e, rows=rows):
                runs = self.run_from(layer, capture[2 * layer].expand(len(rows), -1, -1))
            effects = [effect(base, runs[r], t, c) for r in range(len(rows))]
            heads.append(effects[:H])
            mlps.append(effects[H])
        return {
            "heads": heads,
            "mlps": mlps,
            "baseline": "experiment" if items else "original",
            "target": self.ref(t),
            "contrast": self.ref(c),
            "p": float(base[t].exp()),
            "seconds": round(time.time() - start, 3),
        }

    def neurons(self, text: str | None, ids: list[int] | None, items: list[dict], layer: int, target: int | None, contrast: int | None, k: int = 24, extra: list[int] | None = None) -> dict:
        """
        MLP neurons of one layer: a first-order (gradient × activation) estimate
        of the effect of zeroing each of them, then the measured effect of
        zeroing the k with the largest estimates.
        """
        model = self.llm.require()
        start = time.time()
        if not 0 <= layer < model.config.n_layer:
            raise BadRequest("layer out of range")
        ids = self.tokens_of(text, ids)
        e = Edits.parse(items, self.llm, len(ids))
        ids2, _, embeds = self.prepare(ids, e)
        acts: list[torch.Tensor] = []
        with torch.enable_grad():
            with self.install(e, act_capture=(layer, acts)):
                logits = self.forward(ids2, embeds).logits[0, -1]
            lp = logits.log_softmax(-1)
            t, c = self.target_ids(lp.detach(), target, contrast)
            metric = lp[t] - lp[c] if c is not None else lp[t]
            (grad,) = torch.autograd.grad(metric, acts[0])
        a, g = acts[0][0].detach(), grad[0]
        estimate = -(a * g).sum(0).double()  # Δ metric ≈ Σ_p (0 − a_p) ∂metric/∂a_p
        top = [int(u) for u in estimate.abs().argsort(descending=True)[:k]]
        top += [u for u in dict.fromkeys(extra or []) if 0 <= u < estimate.numel() and u not in top]  # units the user asked about
        with torch.no_grad():
            base, capture = self.baseline(ids2, embeds, e)
            rows: list[Row] = [("neuron", layer, u, 0.0) for u in top]
            with self.install(e, rows=rows):
                runs = self.run_from(layer, capture[2 * layer].expand(len(rows), -1, -1))
            # Direct effect on the vocabulary (as in the toy model): each unit's output weights, centred and
            # scaled like the final LayerNorm does, projected on the unembedding; later layers are ignored.
            w = model.transformer.h[layer].mlp.c_proj.weight[top]  # Conv1D weight: (3072, 768) → (k, 768)
            direction = (w - w.mean(-1, keepdim=True)) * model.transformer.ln_f.weight
            vocab_logits = direction @ model.transformer.wte.weight.T  # (k, V)

        def words(row: torch.Tensor, sign: int) -> list[dict]:
            top = (sign * row).topk(5)
            return [{"id": int(i), "text": self.llm.word(int(i)), "value": round(float(sign * v), 4)} for v, i in zip(top.values, top.indices)]

        return {
            "layer": layer,
            "metric": "logitdiff" if c is not None else "logp",
            "estimate": [round(float(v), 5) for v in estimate],
            "measured": [{"unit": u, "effect": effect(base, runs[r], t, c)} for r, u in enumerate(top)],
            "activations": {str(u): [round(float(v), 4) for v in a[:, u]] for u in top},
            "vocab": {str(u): {"promoted": words(vocab_logits[r], 1), "suppressed": words(vocab_logits[r], -1)} for r, u in enumerate(top)},
            "target": self.ref(t),
            "contrast": self.ref(c),
            "seconds": round(time.time() - start, 3),
        }

    @torch.no_grad()
    def curve(self, text: str | None, ids: list[int] | None, items: list[dict], component: dict, scales: list[float], target: int | None, contrast: int | None) -> dict:
        """p(target) while one component is scaled, with the other interventions in place."""
        start = time.time()
        ids = self.tokens_of(text, ids)
        kind = component.get("kind")
        if kind not in ("head", "mlp", "neuron"):
            raise BadRequest("component must be a head, an MLP layer or a neuron")
        skip = {key: component[key] for key in ("kind", "layer", "head", "unit") if key in component}
        e = Edits.parse(items, self.llm, len(ids), skip=skip)
        Edits.parse([{**component, "scale": 1, "mode": "scale", "value": 1}], self.llm, len(ids))  # validates the component
        ids2, _, embeds = self.prepare(ids, e)
        if kind == "head":
            rows: list[Row] = [("head", component["layer"], component["head"], float(s)) for s in scales]
        elif kind == "mlp":
            rows = [("mlp", component["layer"], float(s)) for s in scales]
        else:
            rows = [("neuron", component["layer"], component["unit"], float(s)) for s in scales]
        base, capture = self.baseline(ids2, embeds, e)
        t, c = self.target_ids(base, target, contrast)
        layer = component["layer"]
        with self.install(e, rows=rows):
            lp = self.run_from(layer, capture[2 * layer].expand(len(rows), -1, -1))
        return {
            "component": component,
            "scales": scales,
            "p": [round(float(v), 6) for v in lp[:, t].exp()],
            "logitDiff": None if c is None else [round(float(v), 5) for v in lp[:, t] - lp[:, c]],
            "seconds": round(time.time() - start, 3),
        }

    @torch.no_grad()
    def attention(self, text: str | None, ids: list[int] | None, items: list[dict], layer: int, head: int) -> dict:
        """Attention pattern of one head, without and with the interventions."""
        model = self.llm.require()
        if not (0 <= layer < model.config.n_layer and 0 <= head < model.config.n_head):
            raise BadRequest("layer or head out of range")
        ids = self.tokens_of(text, ids)
        e = Edits.parse(items, self.llm, len(ids))
        ids2, alignment, embeds = self.prepare(ids, e)
        before = self.forward(ids, None, attentions=True).attentions[layer][0, head]
        with self.install(e):
            after = self.forward(ids2, embeds, attentions=True).attentions[layer][0, head]
        rounded = lambda m: [[round(float(v), 4) for v in row] for row in m]
        return {"before": rounded(before), "after": rounded(after), "alignment": alignment}
