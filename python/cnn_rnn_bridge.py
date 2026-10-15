#!/usr/bin/env python3
"""
Checks the CNN and the recurrent networks of the app against PyTorch.

    python3 python/cnn_rnn_bridge.py verify case.json [more.json ...]

Each case file (written by tests/pytorch-parity.test.ts) holds a model as the
app stores it, a batch of inputs and the results computed in the browser
(logits, mean cross-entropy loss and the gradient of every parameter). The
model is rebuilt with torch.nn modules in float64 — nn.Conv2d / MaxPool2d /
AvgPool2d / Linear for the CNN, nn.RNN or nn.GRU + nn.Linear for the
recurrent network — and the same quantities are recomputed by autograd.

Parameter layouts are PyTorch's:
    CNN   [conv1.weight [F,C,k,k], conv1.bias, conv2.weight, conv2.bias, …, fc.weight [classes, flat], fc.bias]
    RNN   [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, out.weight, out.bias]   (GRU rows: r, z, n)
"""
from __future__ import annotations

import json
import sys

import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def tensor(values, shape) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).reshape(shape)


def build_cnn(config: dict, params: list[list[float]]) -> nn.Module:
    layers: list[nn.Module] = []
    channels = 1
    for conv in config["convs"]:
        layers.append(nn.Conv2d(channels, conv["filters"], conv["kernel"], padding=(conv["kernel"] - 1) // 2))
        layers.append(nn.ReLU() if config["activation"] == "relu" else nn.Tanh())
        if conv["pool"] == "max":
            layers.append(nn.MaxPool2d(2))
        elif conv["pool"] == "avg":
            layers.append(nn.AvgPool2d(2))
        channels = conv["filters"]
    features = nn.Sequential(*layers)
    with torch.no_grad():
        flat = features(torch.zeros(1, 1, config["size"], config["size"])).numel()
    model = nn.Sequential(features, nn.Flatten(), nn.Linear(flat, config["classes"]))
    with torch.no_grad():
        for p, values in zip(model.parameters(), params):
            p.copy_(tensor(values, p.shape))
    return model


def check_cnn(case: dict) -> dict:
    model = build_cnn(case["config"], case["params"])
    size = case["config"]["size"]
    x = tensor(case["images"], (len(case["images"]), 1, size, size))
    labels = torch.tensor(case["labels"])
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return {
        "logits": (logits.detach().flatten() - torch.tensor(case["expected"]["logits"])).abs().max().item(),
        "loss": abs(loss.item() - case["expected"]["loss"]),
        "grads": max((p.grad.flatten() - torch.tensor(g)).abs().max().item() for p, g in zip(model.parameters(), case["expected"]["grads"])),
    }


def check_rnn(case: dict) -> dict:
    cfg = case["config"]
    I, H, O = cfg["inputSize"], cfg["hiddenSize"], cfg["outputSize"]
    cell = nn.RNN(I, H, nonlinearity="tanh", batch_first=True) if cfg["cell"] == "rnn" else nn.GRU(I, H, batch_first=True)
    out = nn.Linear(H, O)
    params = [cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0, cell.bias_hh_l0, out.weight, out.bias]
    with torch.no_grad():
        for p, values in zip(params, case["params"]):
            p.copy_(tensor(values, p.shape))
    inputs = torch.tensor(case["inputs"])  # (batch, T)
    targets = torch.tensor(case["targets"])  # (batch, T), −1 = not scored
    x = nn.functional.one_hot(inputs, I).double()
    states, _ = cell(x)
    logits = out(states)
    loss = nn.functional.cross_entropy(logits.reshape(-1, O), targets.reshape(-1), ignore_index=-1)
    loss.backward()
    return {
        "logits": (logits.detach().flatten() - torch.tensor(case["expected"]["logits"])).abs().max().item(),
        "loss": abs(loss.item() - case["expected"]["loss"]),
        "grads": max((p.grad.flatten() - torch.tensor(g)).abs().max().item() for p, g in zip(params, case["expected"]["grads"])),
    }


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "verify":
        raise SystemExit(__doc__)
    results = {}
    for path in sys.argv[2:]:
        with open(path) as f:
            case = json.load(f)
        results[path] = check_cnn(case) if case["kind"] == "cnn" else check_rnn(case)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
