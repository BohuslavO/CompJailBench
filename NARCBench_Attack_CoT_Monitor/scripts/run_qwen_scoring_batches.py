import argparse
import hashlib
import json
import platform
import runpy
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile


def completed_archive(path, payload, scorer):
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt archive: {path}")
        manifest = json.loads(archive.read("manifest.json"))
        expected_hash = hashlib.sha256(payload).hexdigest()
        matching = manifest["input_sha256"] == {
            "agentharm/standard_trajectories.jsonl": expected_hash
        }
        matching = matching and archive.read(scorer.name) == scorer.read_bytes()
        if not matching or manifest["status"] != "complete":
            return False
        scores = [json.loads(line) for line in archive.read("scores.jsonl").splitlines()]
        calls = [json.loads(line) for line in archive.read("monitor_calls.jsonl").splitlines()]
        samples = {json.loads(line)["sample_id"] for line in payload.splitlines()}
        scopes = {
            ("cot", "public-action-only"),
            ("cot", "observable-reasoning-only"),
            ("cot", "public-action-plus-observable-reasoning"),
            ("agent_to_agent", "public"),
            ("sentinel", "public"),
        }
        keys = {(r["sample_id"], r["defense"], r["scope"]) for r in scores}
        expected = {(sample, defense, scope) for sample in samples for defense, scope in scopes}
        references = sorted(call for row in scores for call in row["monitor_call_ids"])
        call_ids = sorted(call["call_id"] for call in calls)
        return (
            len(scores) == len(expected)
            and keys == expected
            and all(row["status"] == "scored" for row in scores)
            and references == call_ids == list(range(1, len(calls) + 1))
            and len(calls) == manifest["model_calls"]
            and all(not call["hit_token_limit"] for call in calls)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", default=[f"{n:02d}" for n in range(14, 23)])
    parser.add_argument("--gpu-memory-gib", type=float, default=12.5)
    parser.add_argument("--cpu-memory-gib", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    inputs_root = project / "results/qwen_r0/scoring_inputs"
    scorer = Path(__file__).with_name("score_qwen_pilot_monitors.py")
    args.output.mkdir(parents=True, exist_ok=True)
    pending = []
    seen = set()
    for batch in args.batches:
        folder = "smoke" if batch == "smoke" else f"batch-{batch}"
        inputs = inputs_root / folder
        payload = (inputs / "agentharm/standard_trajectories.jsonl").read_bytes()
        samples = [json.loads(line) for line in payload.splitlines()]
        keys = {(row["pair_id"], row["condition_label"]) for row in samples}
        pairs = {row["pair_id"] for row in samples}
        expected = {(pair, condition) for pair in pairs for condition in ("attack", "benign_control")}
        if not samples or len(keys) != len(samples) or keys != expected or seen & pairs:
            raise ValueError(f"Invalid or overlapping paired input: {batch}")
        seen.update(pairs)
        archives = sorted(args.output.glob(f"qwen-r0-agentharm-batch-{batch}-scores-*.zip"))
        if any(completed_archive(path, payload, scorer) for path in archives):
            print(f"SKIP completed batch {batch}", flush=True)
        else:
            pending.append((batch, inputs))
    print(f"Validated {len(seen)} pairs; {len(pending)} batches pending", flush=True)
    if args.validate_only or not pending:
        return
    import anyio
    import torch

    module = runpy.run_path(str(scorer))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in ("transformers", "gptqmodel", "inspect-ai")},
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "gpu_memory_gib": args.gpu_memory_gib,
        "cpu_memory_gib": args.cpu_memory_gib,
    }
    if not environment["devices"]:
        raise ValueError("CUDA GPU unavailable")
    (args.output / f"environment-{stamp}.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment), flush=True)
    runtime = module["QwenActivationRuntime"](
        output_root=args.output / f"runtime-{stamp}",
        model_name=module["MODEL"],
        model_revision=module["REVISION"],
        gpu_memory_gib=args.gpu_memory_gib,
        cpu_memory_gib=args.cpu_memory_gib,
    )
    for batch, inputs in pending:
        output = args.output / f"qwen-r0-agentharm-batch-{batch}-scores-{stamp}"
        print(f"Starting batch {batch}", flush=True)
        try:
            anyio.run(module["run"], SimpleNamespace(input=inputs, output=output), runtime)
        finally:
            if output.is_dir():
                archive = output.with_suffix(".zip")
                staged = archive.with_suffix(".zip.tmp")
                with ZipFile(staged, "w", compression=ZIP_DEFLATED) as bundle:
                    for path in sorted(output.glob("*.json*")):
                        bundle.write(path, path.name)
                    bundle.write(scorer, scorer.name)
                    bundle.write(args.output / f"environment-{stamp}.json", "environment.json")
                staged.replace(archive)
                print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
