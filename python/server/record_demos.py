"""
Records GPT-2 analyses and experiments into public/llm-demo/, so the LLM and
Experiments tabs work without the backend (e.g. on a static host). From the
project root:

    npm run llm:demos
"""
from __future__ import annotations

import json
from pathlib import Path

from .experiments import Experiments
from .llm import LLM

OUT = Path(__file__).resolve().parents[2] / "public" / "llm-demo"

DEMOS = [
    ("capital", "A fact?", "The capital of France is", "GPT-2 small is not sure: “ Paris” is only fifth. Compare with the toy Transformer."),
    ("ioi", "Who gets the drink?", "When Mary and John went to the store, John gave a drink to", "Indirect object: the model must pick the name that was not repeated (“ Mary”)."),
    ("induction", "Repeated text", "The cat sat on the mat. The cat sat on the", "Copying from earlier in the context: look for heads that attend to the token after the previous “the”."),
    ("days", "Days of the week", "Monday, Tuesday, Wednesday,", "A sequence: the logit lens shows the answer appearing only in the last layers."),
    ("winograd", "What is “it”?", "The trophy doesn't fit into the suitcase because it is too", "A classic ambiguity test: GPT-2 small splits its bet between “small” and “big”."),
    ("italian", "Not English", "La capitale d'Italia è", "Byte-level BPE: unfamiliar words break into many pieces, “è” into two raw bytes."),
    ("code", "Code", "def add(a, b):\n    return", "Indentation and newlines are tokens too."),
]

IOI = "When Mary and John went to the store, John gave a drink to"
INDUCTION = "The cat sat on the mat. The cat sat on the"


def heads(*pairs: tuple[int, int]) -> list[dict]:
    return [{"kind": "head", "layer": l, "head": h, "scale": 0} for l, h in pairs]


# GPT-2 presets of the Experiments tab. Numbers in the descriptions were measured with this script.
EXPERIMENTS = [
    {
        "id": "gpt2-ioi-movers",
        "demo": "ioi",
        "label": "Name mover heads",
        "description": "GPT-2 small expects “ Mary” (45%). Wang et al. (2022) call three heads “name movers”: they copy the right name to the output. Here: layer 10 · heads 7 and 10, layer 11 · head 1 (9.6, 9.9, 10.0 in the paper, which counts from 0). Switched off together, p(Mary) only drops to 25%: other heads take over part of the job.",
        "tryNext": "Bars: on this one sentence, which heads matter most when removed one at a time? Not the same ones — single-head ablation misses redundant heads.",
        "text": IOI, "target": " Mary", "contrast": " John",
        "interventions": heads((9, 6), (9, 9), (10, 0)),
    },
    {
        "id": "gpt2-ioi-negative",
        "label": "Heads against the answer",
        "description": "Some heads push the other way. Switching off layer 11 · head 8 and layer 12 · head 11 (the paper’s “negative name movers” 10.7 and 11.10) raises p(Mary) from 45% to 87%.",
        "text": IOI, "target": " Mary", "contrast": " John",
        "interventions": heads((10, 7), (11, 10)),
    },
    {
        "id": "gpt2-ioi-swap",
        "label": "Change who gives",
        "description": "Inside the model, the second “ John” is given the embedding of “ Mary” (“…store, Mary gave a drink to”). Now “ John” becomes the most likely answer.",
        "text": IOI, "target": " Mary", "contrast": " John",
        "interventions": [{"kind": "embedding", "position": 9, "token": " Mary", "mix": 1}],
    },
    {
        "id": "gpt2-ioi-remove",
        "label": "Remove the giver",
        "description": "Without the second “ John” (“…the store, gave a drink to”) the model no longer knows which name to avoid: p(Mary) falls from 45% to 7%.",
        "text": IOI, "target": " Mary", "contrast": " John",
        "interventions": [{"kind": "remove", "position": 9}],
    },
    {
        "id": "gpt2-induction",
        "demo": "induction",
        "label": "Copying from the context",
        "description": "From the last “ the”, five heads attend to “ mat”, the token that followed “ the” the first time — the “induction” pattern (Olsson et al., 2022): layer 6 · heads 2 and 6, layer 7 · head 10, layer 8 · heads 3 and 11 (5.1, 5.5, 6.9, 7.2, 7.10 counting from 0). Switched off together, p(“ mat”) drops from 59% to 15%.",
        "tryNext": "Switch layer 8 · head 3 back on: p(mat) rises to 70%, above the original. Effects of heads do not simply add up.",
        "text": INDUCTION, "target": " mat", "contrast": None,
        "interventions": heads((5, 1), (5, 5), (6, 9), (7, 2), (7, 10)),
    },
    {
        "id": "gpt2-mlp0",
        "label": "The first MLP is special",
        "description": "Zeroing GPT-2’s first MLP wrecks this prompt (p(“ Paris”) 6.4% → 0.02%) — and almost every other prompt. Interpretability work suggests GPT-2 small uses it almost as part of the token embedding: a big effect does not show where a fact is stored.",
        "text": "The Eiffel Tower is located in the city of", "target": " Paris", "contrast": None,
        "interventions": [{"kind": "mlp", "layer": 0, "scale": 0}],
    },
]


def record_experiments(llm: LLM) -> list[dict]:
    ex = Experiments(llm)
    token = lambda text: {"id": llm.tokenizer.encode(text)[0], "text": text} if text else None
    out = []
    for spec in EXPERIMENTS:
        items = [{**i, "token": token(i["token"])} if i["kind"] == "embedding" else dict(i) for i in spec["interventions"]]
        target, contrast = token(spec["target"]), token(spec["contrast"])
        result = ex.experiment(spec["text"], None, items, target["id"], contrast["id"] if contrast else None)
        sweep = ex.sweep(spec["text"], None, [], target["id"], contrast["id"] if contrast else None)
        out.append({
            "id": spec["id"],
            "label": spec["label"],
            "description": spec["description"],
            **({"tryNext": spec["tryNext"]} if "tryNext" in spec else {}),
            "experiment": {"model": "gpt2", "text": spec["text"], "target": target, "contrast": contrast, "interventions": [{**i, "enabled": True} for i in items]},
            "result": result,
            "sweep": sweep,
        })
        print(f"{spec['id']:20s} p({spec['target']!r}) {result['before']['pTarget']:.3f} → {result['after']['pTarget']:.3f}  ({result['seconds'] + sweep['seconds']:.1f} s)")
    return out


def main() -> None:
    llm = LLM()
    llm.load()
    if llm.model is None:
        raise SystemExit(llm.error)
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for demo_id, label, text, description in DEMOS:
        analysis = llm.analyze(text)
        analysis["demo"] = {
            "greedy": llm.generate(text, 0, 0, 1, 10, 0),
            "sampled": llm.generate(text, 1.0, 0, 1, 10, 1),
        }
        file = f"{demo_id}.json"
        (OUT / file).write_text(json.dumps(analysis, separators=(",", ":"), ensure_ascii=False))
        entry = {"id": demo_id, "label": label, "text": text, "description": description, "file": file}
        linked = next((e["id"] for e in EXPERIMENTS if e.get("demo") == demo_id), None)
        if linked:
            entry["experiment"] = linked
        index.append(entry)
        print(f"{demo_id:10s} {len(analysis['tokens']):3d} tokens → {analysis['words'][analysis['nextTop'][0][0]]!r} ({(OUT / file).stat().st_size / 1024:.0f} KB)")
    (OUT / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    for old in OUT.glob("*.sweep.json"):  # format of an earlier version
        old.unlink()
    experiments = record_experiments(llm)
    (OUT / "experiments.json").write_text(json.dumps(experiments, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
