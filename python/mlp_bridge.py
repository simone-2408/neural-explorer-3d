#!/usr/bin/env python3
"""
Bridge between Neural Explorer 3D model files and PyTorch.

    python python/mlp_bridge.py verify model.json [more.json ...]
        Rebuild each exported model as a torch.nn.Sequential, run the forward
        pass in float64 and compare it with the output computed in the browser.

    python python/mlp_bridge.py export-demo out.json [--seed 0]
        Create an MLP in PyTorch (default nn.Linear initialization) and save it
        in the app format: load it in the app with "Parameters → Import".

    python python/mlp_bridge.py summary model.json
        Print the architecture and parameter shapes.

Model file (format "neural-explorer-3d/mlp", version 1):
    config      architecture, same JSON as the app's configuration editor
    parameters  per layer l >= 1: weight [out, in] (== nn.Linear.weight) and bias [out]
    input       input vector
    output      forward result for `input`, computed by the exporter
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

FORMAT = "neural-explorer-3d/mlp"
FORMAT_VERSION = 1

# Same definitions as src/math/activations.ts (PyTorch defaults).
ACTIVATIONS = {
    "linear": lambda: None,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,  # exact erf formulation
    "softmax": lambda: nn.Softmax(dim=-1),
}


def layer_activation(layer: dict) -> str:
    """Default activations mirror src/neural/config.ts."""
    if "activation" in layer:
        return layer["activation"]
    return "relu" if layer["type"] == "dense" else "linear"


def build_sequential(config: dict) -> nn.Sequential:
    """nn.Linear(n_{l-1}, n_l) followed by the activation, for every layer after the input."""
    layers = config["layers"]
    modules: list[nn.Module] = []
    for previous, layer in zip(layers, layers[1:]):
        modules.append(nn.Linear(previous["size"], layer["size"]))
        activation = ACTIVATIONS[layer_activation(layer)]()
        if activation is not None:
            modules.append(activation)
    return nn.Sequential(*modules)


def linear_layers(model: nn.Sequential) -> list[nn.Linear]:
    return [m for m in model if isinstance(m, nn.Linear)]


def load_model_file(path: str | Path) -> tuple[nn.Sequential, torch.Tensor, dict]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    if data.get("formatVersion", 0) > FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format version {data.get('formatVersion')}")
    model = build_sequential(data["config"]).double()
    parameters = {p["layer"]: p for p in data["parameters"]}
    with torch.no_grad():
        for l, linear in enumerate(linear_layers(model), start=1):
            linear.weight.copy_(torch.tensor(parameters[l]["weight"], dtype=torch.float64))
            linear.bias.copy_(torch.tensor(parameters[l]["bias"], dtype=torch.float64))
    x = torch.tensor(data["input"], dtype=torch.float64)
    return model, x, data


def model_to_dict(model: nn.Sequential, config: dict, x: torch.Tensor, seed: int) -> dict:
    model = model.double()
    with torch.no_grad():
        y = model(x.double())
    return {
        "format": FORMAT,
        "formatVersion": FORMAT_VERSION,
        "config": config,
        "init": {"scheme": "pytorch", "seed": seed},
        "parameters": [
            {"layer": l, "weight": linear.weight.tolist(), "bias": linear.bias.tolist()}
            for l, linear in enumerate(linear_layers(model), start=1)
        ],
        "input": x.tolist(),
        "output": y.tolist(),
        "meta": {
            "app": f"mlp_bridge.py (PyTorch {torch.__version__})",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
        },
    }


def cmd_verify(args: argparse.Namespace) -> int:
    failures = 0
    for file in args.files:
        model, x, data = load_model_file(file)
        with torch.no_grad():
            y = model(x)
        expected = torch.tensor(data["output"], dtype=torch.float64)
        diff = (y - expected).abs().max().item()
        ok = diff <= args.tolerance
        failures += not ok
        name = data["config"].get("name", Path(file).stem)
        print(f"{'OK  ' if ok else 'FAIL'} {name}: max |PyTorch - browser| = {diff:.3e}  (output {[round(v, 6) for v in y.tolist()]})")
    return 1 if failures else 0


def target_tensor(config: dict, labels: torch.Tensor) -> torch.Tensor:
    """Same targets as src/training/loss.ts (targetValue)."""
    output = config["layers"][-1]
    size = output["size"]
    if size == 1:
        negative = -1.0 if layer_activation(output) == "tanh" else 0.0
        return torch.where(labels > 0, 1.0, negative).double().reshape(-1, 1)
    return torch.nn.functional.one_hot(labels.long(), size).double()


def batch_loss(config: dict, out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = config["layers"][-1]
    activation = layer_activation(output)
    if activation == "softmax":
        return nn.functional.nll_loss(torch.log(out.clamp_min(1e-12)), labels.long())
    if activation == "sigmoid" and output["size"] == 1:
        return nn.functional.binary_cross_entropy(out.reshape(-1), labels.double())
    return 0.5 * ((out - target_tensor(config, labels)) ** 2).sum(dim=1).mean()


def cmd_train(args: argparse.Namespace) -> int:
    """Replays the batches recorded by the TypeScript Trainer and prints the final parameters."""
    session = json.loads(Path(args.file).read_text())
    model_data = session["model"]
    model = build_sequential(model_data["config"]).double()
    parameters = {p["layer"]: p for p in model_data["parameters"]}
    with torch.no_grad():
        for l, linear in enumerate(linear_layers(model), start=1):
            linear.weight.copy_(torch.tensor(parameters[l]["weight"], dtype=torch.float64))
            linear.bias.copy_(torch.tensor(parameters[l]["bias"], dtype=torch.float64))
    x = torch.tensor(session["x"], dtype=torch.float64).reshape(-1, 2)
    y = torch.tensor(session["y"])
    o = session["optimizer"]
    if o["id"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=o["learningRate"], weight_decay=o["weightDecay"])
    elif o["id"] == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=o["learningRate"], momentum=o["momentum"], weight_decay=o["weightDecay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=o["learningRate"], betas=(o["beta1"], o["beta2"]), eps=o["epsilon"], weight_decay=o["weightDecay"])
    losses = []
    for batch in session["batches"]:
        index = torch.tensor(batch, dtype=torch.long)
        optimizer.zero_grad()
        loss = batch_loss(model_data["config"], model(x[index]), y[index])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    result = {
        "losses": losses,
        "parameters": [
            {"layer": l, "weight": linear.weight.tolist(), "bias": linear.bias.tolist()}
            for l, linear in enumerate(linear_layers(model), start=1)
        ],
    }
    print(json.dumps(result))
    return 0


def cmd_export_demo(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    config = {
        "name": "PyTorch demo MLP",
        "layers": [
            {"type": "input", "size": 2},
            {"type": "dense", "size": 4, "activation": "relu"},
            {"type": "dense", "size": 4, "activation": "tanh"},
            {"type": "output", "size": 1, "activation": "sigmoid"},
        ],
    }
    model = build_sequential(config)  # PyTorch default initialization
    data = model_to_dict(model, config, torch.tensor([0.7, 0.2]), args.seed)
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.out}: output {data['output']}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    model, x, data = load_model_file(args.file)
    print(model)
    for name, tensor in model.state_dict().items():
        print(f"  {name:12s} {tuple(tensor.shape)}")
    print(f"input  {x.tolist()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="compare the browser forward pass with PyTorch")
    verify.add_argument("files", nargs="+")
    verify.add_argument("--tolerance", type=float, default=1e-9)
    verify.set_defaults(func=cmd_verify)

    demo = sub.add_parser("export-demo", help="export a PyTorch MLP in the app format")
    demo.add_argument("out")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_export_demo)

    train = sub.add_parser("train", help="replay a training session recorded by the app (prints JSON)")
    train.add_argument("file")
    train.set_defaults(func=cmd_train)

    summary = sub.add_parser("summary", help="print architecture and parameter shapes")
    summary.add_argument("file")
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
