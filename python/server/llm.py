"""
GPT-2 small, instrumented for the LLM tab and the Experiments tab.

Everything returned is computed from one forward pass with the attention
probabilities and the residual stream of every layer:

    tokens            byte-level BPE pieces and their text
    logits            the next-token logits (all 50,257, float32, base64)
    lens              logit lens: final LayerNorm + unembedding applied to the
                      residual stream after every layer (0 = embeddings)
    attention         12 layers × 12 heads × T × T, quantized to uint16 (base64)
    journey           3-D PCA of every token's (normalized) residual vector,
                      layer after layer
    rollout           attention rollout (Abnar & Zuidema, 2020): a heuristic
                      estimate of how much each input token reaches each position
    saliency          gradient × input: how much a small change of each input
                      token's embedding would change log p(predicted token)

Interventions for the Experiments tab live in experiments.py.
"""
from __future__ import annotations

import base64
import threading
import time

import numpy as np
import torch

MODEL_NAME = "gpt2"
MAX_TOKENS = 48
LENS_K = 5


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from bytes to printable characters (a space becomes 'Ġ')."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def filtered_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Same rule as src/transformer/sampling.ts: temperature, then top-k, then top-p, renormalized."""
    t = max(temperature, 1e-3)
    probs = torch.softmax(logits.double() / t, dim=-1)
    order = torch.argsort(probs, descending=True)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    cumulative = 0.0
    kept = 0
    for i in order.tolist():
        if top_k > 0 and kept >= top_k:
            break
        if top_p < 1 and cumulative >= top_p:
            break
        keep[i] = True
        kept += 1
        cumulative += float(probs[i])
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    return probs / probs.sum()


class ModelNotReady(RuntimeError):
    """The model is still loading or failed to load."""


class LLM:
    # Grad mode is per thread in PyTorch and the web server answers on worker threads:
    # every entry point therefore runs under torch.no_grad() explicitly.

    def __init__(self, name: str = MODEL_NAME) -> None:
        self.name = name
        self.model = None
        self.tokenizer = None
        self.error: str | None = None
        self.loading = False
        self.load_seconds = 0.0
        self._lock = threading.Lock()
        # One computation at a time: interventions install hooks on the shared model,
        # and the web server answers requests on several threads.
        self.compute_lock = threading.RLock()

    # ------------------------------------------------------------- loading

    def load(self) -> None:
        with self._lock:
            if self.model is not None or self.loading:
                return
            self.loading = True
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            start = time.time()
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.name)
            model = GPT2LMHeadModel.from_pretrained(self.name, attn_implementation="eager").eval()
            with torch.no_grad():
                model(torch.tensor([[464, 3139]]))  # warm-up
            self.model = model
            self.load_seconds = time.time() - start
        except Exception as error:  # reported by /status
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.loading = False

    def status(self) -> dict:
        info: dict = {"model": self.name, "ready": self.model is not None, "loading": self.loading, "error": self.error}
        if self.model is not None:
            c = self.model.config
            info["config"] = {
                "layers": c.n_layer,
                "heads": c.n_head,
                "dModel": c.n_embd,
                "dMlp": 4 * c.n_embd,
                "vocab": c.vocab_size,
                "context": c.n_positions,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "maxTokens": MAX_TOKENS,
            }
            info["loadSeconds"] = round(self.load_seconds, 1)
        return info

    def require(self):
        if self.model is None:
            raise ModelNotReady(self.error or ("The model is still loading" if self.loading else "The model is not loaded"))
        return self.model

    # ------------------------------------------------------------- helpers

    def encode(self, text: str | None, ids: list[int] | None = None) -> tuple[list[int], bool]:
        if ids is None:
            ids = self.tokenizer.encode(text or "")
        if not ids:
            ids = [self.tokenizer.eos_token_id]  # <|endoftext|> as a start-of-text token
        truncated = len(ids) > MAX_TOKENS
        return ids[:MAX_TOKENS], truncated

    def token_info(self, ids: list[int]) -> list[dict]:
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        return [
            {"id": int(i), "text": self.tokenizer.decode([int(i)]), "piece": p, "bytes": bytes(BYTE_DECODER.get(c, 63) for c in p).hex(" ")}
            for i, p in zip(ids, pieces)
        ]

    def word(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)])

    @torch.no_grad()
    def residuals(self, ids: list[int]) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Logits, the raw residual stream before block 0 … after block 11 (13 tensors T×d) and attentions."""
        model = self.require()
        captured: list[torch.Tensor] = []
        handle = model.transformer.h[-1].register_forward_hook(lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out)[0]))
        try:
            out = model(torch.tensor([ids]), output_attentions=True, output_hidden_states=True)
        finally:
            handle.remove()
        # hidden_states[12] already went through ln_f: use the raw output of the last block instead.
        residuals = [h[0] for h in out.hidden_states[:-1]] + [captured[0]]
        attention = torch.stack([a[0] for a in out.attentions])  # (L, H, T, T)
        return out.logits[0], residuals, attention

    @torch.no_grad()
    def lens(self, residual: torch.Tensor) -> torch.Tensor:
        model = self.require()
        return model.lm_head(model.transformer.ln_f(residual))

    # ------------------------------------------------------------- analysis

    @torch.no_grad()
    def analyze(self, text: str | None = None, ids: list[int] | None = None) -> dict:
        model = self.require()
        ids, truncated = self.encode(text, ids)
        start = time.time()
        logits, residuals, attention = self.residuals(ids)
        T = len(ids)
        final = logits.argmax(-1)  # the model's own top prediction at each position

        lens = []
        for r in residuals:
            lg = self.lens(r)
            probs = lg.softmax(-1)
            top = probs.topk(LENS_K, dim=-1)
            final_logit = lg.gather(-1, final[:, None])
            lens.append({
                "top": [[[int(i), round(float(p), 5)] for p, i in zip(top.values[t], top.indices[t])] for t in range(T)],
                "finalProb": [round(float(v), 5) for v in probs.gather(-1, final[:, None])[:, 0]],
                "finalRank": [int(v) for v in (lg > final_logit).sum(-1)],
            })
        next_top = logits[-1].softmax(-1).topk(20)
        word_ids = {i for layer in lens for row in layer["top"] for i, _ in row} | set(final.tolist()) | {int(i) for i in next_top.indices}
        words = {int(i): self.word(int(i)) for i in word_ids}

        # token journey: PCA of normalized residual vectors (position 0 excluded from the fit)
        normed = [torch.nn.functional.layer_norm(r, (r.shape[-1],)) for r in residuals]
        stack = torch.stack(normed).double().numpy()  # (13, T, d)
        fit = stack[:, 1:, :].reshape(-1, stack.shape[-1]) if T > 1 else stack.reshape(-1, stack.shape[-1])
        mean = fit.mean(0)
        _, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
        variance = s**2
        explained = (variance[:3] / variance.sum()).tolist() if variance.sum() > 0 else [0, 0, 0]
        coords = (stack - mean) @ vt[:3].T  # (13, T, 3)

        # attention rollout: average heads, add the residual (identity), renormalize, multiply layers
        rollout = torch.eye(T, dtype=torch.float64)
        for layer in attention.double():
            a = 0.5 * layer.mean(0) + 0.5 * torch.eye(T, dtype=torch.float64)
            a = a / a.sum(-1, keepdim=True)
            rollout = a @ rollout

        saliency = self.saliency(ids, int(final[-1]))
        attention_u16 = np.round(attention.numpy().astype(np.float64) * 65535).astype("<u2")
        return {
            "model": self.name,
            "tokens": self.token_info(ids),
            "truncated": truncated,
            "logits": b64(logits[-1].numpy().astype("<f4")),
            "nextTop": [[int(i), round(float(p), 6)] for p, i in zip(next_top.values, next_top.indices)],
            "predictions": [[int(i), round(float(p), 5)] for i, p in zip(final.tolist(), logits.softmax(-1).max(-1).values.tolist())],
            "lens": lens,
            "words": words,
            "attention": b64(attention_u16),
            "norms": [[round(float(v), 3) for v in r.norm(dim=-1)] for r in residuals],
            "journey": {"coords": np.round(coords, 4).tolist(), "explained": [round(float(v), 4) for v in explained]},
            "rollout": np.round(rollout.numpy(), 5).tolist(),
            "saliency": {"target": int(final[-1]), "values": [round(float(v), 6) for v in saliency]},
            "seconds": round(time.time() - start, 3),
        }

    def saliency(self, ids: list[int], target: int) -> list[float]:
        """Gradient × input of log p(target | text) w.r.t. each token embedding (summed over dimensions)."""
        model = self.require()
        with torch.enable_grad():
            embeds = model.transformer.wte(torch.tensor([ids])).detach().requires_grad_(True)
            logp = model(inputs_embeds=embeds).logits[0, -1].log_softmax(-1)[target]
            (grad,) = torch.autograd.grad(logp, embeds)
        return (grad * embeds).sum(-1)[0].tolist()

    # ----------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, text: str, temperature: float, top_k: int, top_p: float, max_new_tokens: int, seed: int) -> dict:
        model = self.require()
        ids, _ = self.encode(text)
        generator = torch.Generator().manual_seed(int(seed))
        steps = []
        past = None
        current = torch.tensor([ids])
        for _ in range(max(1, min(max_new_tokens, 40))):
            out = model(current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            if temperature <= 0:
                probs = torch.softmax(logits.double(), -1)
                choice = int(probs.argmax())
            else:
                probs = filtered_probs(logits, temperature, top_k, top_p)
                choice = int(torch.multinomial(probs.float(), 1, generator=generator))
            top = probs.topk(5)
            steps.append({
                "id": choice,
                "text": self.word(choice),
                "p": round(float(probs[choice]), 5),
                "alternatives": [[int(i), self.word(int(i)), round(float(p), 5)] for p, i in zip(top.values, top.indices)],
            })
            current = torch.tensor([[choice]])
        return {"steps": steps, "text": text + "".join(s["text"] for s in steps)}
