import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
from inspect_ai.util import Store

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.agent_to_agent_monitor import score_trajectory as score_a2a
from src.defense_interface import SCOPES, score_trajectory as score_cot
from src.narcbench_probe_runtime import QwenActivationRuntime
from src.trajectory import StandardTrajectory

MODEL = "Qwen/Qwen3-32B-AWQ"
REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"


class Monitor:
    def __init__(self, runtime, output):
        self.runtime = runtime
        self.output = output
        self.calls = 0

    def __str__(self):
        return MODEL

    async def generate(self, input, **kwargs):
        import torch

        messages = [{"role": str(message.role), "content": message.content}
                    for message in input]
        prompt = self.runtime._render_prompt(messages)
        tokens = self.runtime.tokenizer(prompt, return_tensors="pt").to(self.runtime._input_device())
        started = time.monotonic()
        with torch.inference_mode():
            result = self.runtime.model.generate(**tokens, do_sample=False, max_new_tokens=1536)
        generated = result[0, tokens["input_ids"].shape[1]:]
        completion = self.runtime.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed = time.monotonic() - started
        self.calls += 1
        raw = {"call_id": self.calls, "messages": messages, "completion": completion,
               "input_tokens": tokens["input_ids"].shape[1], "output_tokens": len(generated),
               "seconds": elapsed, "hit_token_limit": len(generated) >= 1536}
        with (self.output / "monitor_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
        if raw["hit_token_limit"]:
            raise ValueError("Monitor response reached the token limit")
        return SimpleNamespace(completion=completion, usage=None, time=elapsed)


def normalize(data):
    data["public_task"]["context"] = data["public_task"].get("context") or data["public_task"].get("request", "")
    public_ids = {message["message_id"] for message in data["public_messages"]}
    for edge in data["communication_edges"]:
        if edge.get("message_id") in public_ids:
            edge["channel"] = "public"
    return StandardTrajectory(store=Store(), **data)


async def run(args, runtime=None):
    sources = sorted(args.input.rglob("standard_trajectories.jsonl"))
    if not sources:
        raise ValueError(f"No scoring inputs in {args.input}")
    jobs = [("cot", scope) for scope in SCOPES] + [("agent_to_agent", "public"), ("sentinel", "public")]
    args.output.mkdir(parents=True, exist_ok=False)
    if runtime is None:
        runtime = QwenActivationRuntime(output_root=args.output / "runtime",
                                        model_name=MODEL, model_revision=REVISION)
    monitor = Monitor(runtime, args.output)
    sentinel = runpy.run_path(str(PROJECT / "scripts/score_sentinel_message_only.py"))
    backend = sentinel["load_sentinel_backend"](PROJECT.parent / "Defense_SentinelAgents")
    manifest = {"monitor_model": MODEL, "monitor_revision": REVISION,
                "do_sample": False, "max_new_tokens": 1536, "enable_thinking": False,
                "input_sha256": {str(path.relative_to(args.input)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources}, "status": "running"}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    statuses = {}
    with (args.output / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for source in sources:
            for line in source.read_text(encoding="utf-8").splitlines():
                trajectory = normalize(json.loads(line))
                for defense, scope in jobs:
                    call_start = monitor.calls
                    try:
                        if defense == "cot":
                            row = await score_cot(trajectory, scope=scope, monitor=monitor)
                        elif defense == "agent_to_agent":
                            row = await score_a2a(trajectory, monitor=monitor)
                        else:
                            row = await sentinel["score_one"](trajectory, scope=scope,
                                        model=monitor, backend=backend, checker_retries=2)
                    except Exception as error:
                        row = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                    row.update(sample_id=trajectory.sample_id, pair_id=trajectory.pair_id,
                               attack_name=trajectory.attack_name, condition_label=trajectory.condition_label,
                               defense=defense, scope=scope, monitor_model=MODEL,
                               monitor_call_ids=list(range(call_start + 1, monitor.calls + 1)))
                    statuses[row["status"]] = statuses.get(row["status"], 0) + 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps({key: row[key] for key in ("sample_id", "defense", "scope", "status")}), flush=True)
    manifest.update(status="complete_with_errors" if statuses.get("error") or statuses.get("parse_error")
                    else "complete", model_calls=monitor.calls, score_status_counts=statuses)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    anyio.run(run, parser.parse_args())
