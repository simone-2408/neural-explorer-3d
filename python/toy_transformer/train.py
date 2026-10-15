#!/usr/bin/env python3
"""
Train the tiny educational Transformer of the Transformer tab and export
it to public/models/toy-transformer.json.

    python3 python/toy_transformer/train.py [--steps 4000] [--seed 0]

Architecture (GPT-style, pre-LayerNorm, like GPT-2 but tiny):
    x0 = E[token] + P[position]
    for each block:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
    logits = LN_f(x) · Eᵀ          (tied input/output embeddings)

The exported weights are rounded to 6 significant digits; the reference
outputs stored in the file are computed with the *rounded* weights in float64,
so the TypeScript implementation can be checked exactly (src/transformer).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from corpus import COUNTRIES, GROUPS, TEMPLATES, corpus

SPECIALS = ["<pad>", "<s>", "<unk>"]
CONTEXT = 16
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_MLP = 128


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_MLP)
        self.fc2 = nn.Linear(D_MLP, D_MODEL)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, dh = N_HEADS, D // N_HEADS
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = scores.softmax(dim=-1)
        y = (attention @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x, attention


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        attentions = []
        for block in self.blocks:
            x, attention = block(x)
            attentions.append(attention)
        logits = self.ln_f(x) @ self.tok.weight.T
        return logits, attentions


def init_gpt2(model: TinyGPT) -> None:
    """GPT-2 initialization: N(0, 0.02) weights, zero biases, residual projections
    scaled by 1/sqrt(2 * layers). PyTorch's default N(0, 1) embeddings would leave a
    large random component (norm ≈ sqrt(d)) on every word, hiding the structure the
    training gives to the embedding space."""
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            nn.init.zeros_(p)
        elif "ln" in name:
            nn.init.ones_(p)
        else:
            std = 0.02 / math.sqrt(2 * N_LAYERS) if name.endswith(("proj.weight", "fc2.weight")) else 0.02
            nn.init.normal_(p, std=std)


def embedding_structure(model: TinyGPT, vocab: list[str]) -> dict:
    """How much the token embeddings are organised by the semantic groups of the corpus."""
    groups = {word: group for group, members in GROUPS.items() for word in members}
    words = [i for i, w in enumerate(vocab) if w not in SPECIALS]
    with torch.no_grad():
        e = model.tok.weight[words].double()
        unit = e / e.norm(dim=1, keepdim=True)
        cos = unit @ unit.T
        labels = [groups.get(vocab[i]) for i in words]
        same, other = [], []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                if labels[a] is None or labels[b] is None:
                    continue
                (same if labels[a] == labels[b] else other).append(float(cos[a, b]))
        centred = e - e.mean(dim=0)
        eig = torch.linalg.eigvalsh(centred.T @ centred).flip(0)
        explained = (eig[:3] / eig.sum()).tolist()
    return {
        "withinGroupCosine": round(sum(same) / len(same), 4),
        "acrossGroupCosine": round(sum(other) / len(other), 4),
        "pcaExplained": [round(v, 4) for v in explained],
    }


def encode(words: list[str], index: dict[str, int]) -> list[int]:
    return [index["<s>"]] + [index.get(w, index["<unk>"]) for w in words][: CONTEXT - 1]


def round6(t: torch.Tensor) -> list:
    return [float(f"{v:.6g}") for v in t.reshape(-1).tolist()]


def nested(t: torch.Tensor) -> list:
    flat = round6(t)
    if t.dim() == 1:
        return flat
    rows, cols = t.shape
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=["gpt2", "pytorch"], default="gpt2")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "public" / "models" / "toy-transformer.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sentences = corpus(30000, args.seed)
    words = sorted({w for s in sentences for w in s} | {w for group in GROUPS.values() for w in group})
    vocab = SPECIALS + words
    index = {w: i for i, w in enumerate(vocab)}
    pad = index["<pad>"]

    sequences = [encode(s, index) + [pad] for s in sentences]  # last target of each sentence is padding (ignored)
    length = max(len(s) for s in sequences)
    data = torch.full((len(sequences), length), pad, dtype=torch.long)
    for i, s in enumerate(sequences):
        data[i, : len(s)] = torch.tensor(s)

    model = TinyGPT(len(vocab))
    if args.init == "gpt2":
        init_gpt2(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    start = time.time()
    for step in range(args.steps):
        batch = data[torch.randint(0, len(data), (128,))]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits, _ = model(inputs)
        loss = loss_fn(logits.reshape(-1, len(vocab)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    print(f"trained in {time.time() - start:.1f}s")
    structure = embedding_structure(model, vocab)
    print("embedding structure:", structure)

    # Round the weights and continue in float64 with the rounded values. The
    # conversion must come first: copying the rounded decimals into float32
    # tensors would change them again (float32 keeps only ~7 digits).
    model = model.double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.tensor(round6(p), dtype=torch.float64).reshape(p.shape))

    def predict(text: str) -> tuple[list[int], torch.Tensor, list[torch.Tensor]]:
        ids = encode(text.split(), index)
        with torch.no_grad():
            logits, attentions = model(torch.tensor([ids]))
        return ids, logits[0], [a[0] for a in attentions]

    # Facts the model should have learned.
    correct = 0
    for country, capital in COUNTRIES.items():
        ids, logits, _ = predict(f"the capital of {country} is")
        correct += int(vocab[int(logits[-1].argmax())] == capital)
    print(f"capitals: {correct}/{len(COUNTRIES)} correct")

    probes = [
        "the capital of france is",
        "the animal didn't cross the street because it was",
        "the queen is a",
        "rome is the capital of",
        "one two three",
    ]
    probe_results = []
    for text in probes:
        ids, logits, _ = predict(text)
        p = logits[-1].softmax(-1)
        top = p.topk(5)
        probe_results.append({"text": text, "top": [[vocab[int(i)], round(float(v), 4)] for v, i in zip(top.values, top.indices)]})
        print(text, "→", probe_results[-1]["top"][:3])

    references = []
    for text in ["the capital of france is", "the animal didn't cross the street because it was tired", "two three four"]:
        ids, logits, attentions = predict(text)
        references.append({
            "text": text,
            "ids": ids,
            "logits": [round(float(v), 10) for v in logits.reshape(-1).tolist()],
            "attention": [[round(float(v), 10) for v in a.reshape(-1).tolist()] for a in attentions],
        })

    blocks = []
    for b in model.blocks:
        blocks.append({
            "ln1": {"weight": round6(b.ln1.weight), "bias": round6(b.ln1.bias)},
            "qkv": {"weight": nested(b.qkv.weight), "bias": round6(b.qkv.bias)},
            "proj": {"weight": nested(b.proj.weight), "bias": round6(b.proj.bias)},
            "ln2": {"weight": round6(b.ln2.weight), "bias": round6(b.ln2.bias)},
            "fc1": {"weight": nested(b.fc1.weight), "bias": round6(b.fc1.bias)},
            "fc2": {"weight": nested(b.fc2.weight), "bias": round6(b.fc2.bias)},
        })
    groups = {word: group for group, members in GROUPS.items() for word in members}
    out = {
        "format": "neural-explorer-3d/toy-transformer",
        "formatVersion": 1,
        "config": {
            "vocab": vocab,
            "dModel": D_MODEL,
            "nHeads": N_HEADS,
            "nLayers": N_LAYERS,
            "dMlp": D_MLP,
            "context": CONTEXT,
            "layerNormEps": 1e-5,
            "activation": "gelu",
            "bos": "<s>",
            "pad": "<pad>",
            "unk": "<unk>",
            "tiedEmbeddings": True,
        },
        "weights": {
            "tokenEmbedding": nested(model.tok.weight),
            "positionEmbedding": nested(model.pos.weight),
            "blocks": blocks,
            "lnFinal": {"weight": round6(model.ln_f.weight), "bias": round6(model.ln_f.bias)},
        },
        "training": {
            "sentences": len(sentences),
            "steps": args.steps,
            "finalLoss": round(float(loss.item()), 4),
            "parameters": sum(p.numel() for p in model.parameters()),
            "templates": TEMPLATES,
            "capitalsCorrect": correct,
            "init": args.init,
            "embeddingStructure": structure,
            "probes": probe_results,
            "pytorch": torch.__version__,
        },
        "groups": groups,
        "reference": references,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
