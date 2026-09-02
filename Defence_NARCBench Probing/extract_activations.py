
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import transformers.activations as _ta
if not hasattr(_ta, "PytorchGELUTanh"):
    class _PytorchGELUTanh(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")
    _ta.PytorchGELUTanh = _PytorchGELUTanh

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layer_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _get_layers(model):
    if hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    raise AttributeError(f"Cannot find transformer layers in {type(model).__name__}")


class ActivationHooks:

    def __init__(self, model, target_layers: list[int]):
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        layers = _get_layers(model)
        for idx in target_layers:
            self.hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output

            self.captured[idx] = hidden[0, -1, :].float().cpu()
        return hook_fn

    def get_and_clear(self):
        result = dict(self.captured)
        self.captured.clear()
        return result

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def _chat_kwargs(tokenizer):
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen3" in name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def build_gen_text(tokenizer, turn: dict) -> str:
    """Prompt + response, matching NARCBench's _build_gen_text exactly --
    activations_gen.npz (what the probe was trained on) was built from this
    exact construction, not the input-only variant."""
    messages = [
        {"role": "system", "content": turn["system_prompt"]},
        {"role": "user", "content": turn["user_prompt"]},
        {"role": "assistant", "content": turn["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, **_chat_kwargs(tokenizer))


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def extract(model, tokenizer, turns: list[dict], target_layers: list[int]):
    hooks = ActivationHooks(model, target_layers)
    activations = {layer: [] for layer in target_layers}
    try:
        for turn in turns:
            text = build_gen_text(tokenizer, turn)
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model(input_ids)
            captured = hooks.get_and_clear()
            for layer in target_layers:
                activations[layer].append(captured[layer].numpy())
    finally:
        hooks.remove()
    return {layer: np.stack(vecs) for layer, vecs in activations.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--layers", required=True, help="e.g. 26-30 (must match probe's layer)")
    ap.add_argument("--turns", type=Path, required=True,
                     help="JSON list of {agent_name, system_prompt, user_prompt, response}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    turns = json.loads(args.turns.read_text())
    target_layers = parse_layer_range(args.layers)

    model, tokenizer = load_model(args.model)
    acts = extract(model, tokenizer, turns, target_layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{f"layer_{l}": arr for l, arr in acts.items()})
    meta_out = args.out.with_suffix("").with_suffix(".meta.json")
    meta_out.write_text(json.dumps([{"agent_name": t["agent_name"]} for t in turns], indent=2))
    print(f"Saved {args.out} ({len(turns)} turns x {len(target_layers)} layers)")


if __name__ == "__main__":
    main()
