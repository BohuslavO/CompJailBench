"""Resume-safe self-hosted NARCBench x NARCBench Probing run for Kaggle.

Run from the ``NARCBench_Attack_CoT_Monitor`` directory. Start with one
matched pair; do not launch the full corpus until that result is inspected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai import Task, eval as inspect_eval

from src.narcbench_dataset import DATA_PATH, load_narcbench_dataset
from src.narcbench_probe import (
    DEFAULT_BACKEND_PATH,
    DEFAULT_PROBE_LAYER,
    DEFAULT_PROBE_MODEL,
    PROBE_METRIC_NAMES,
    RELEASED_PROBE_SHA256,
    narcbench_activation_probe_scorer,
)
from src.narcbench_probe_runtime import DEFAULT_EXTRACTOR_PATH, QwenActivationRuntime
from src.narcbench_real_solver import narcbench_real_solver
from src.narcbench_scorer import narcbench_attack_scorer


def available_pairs() -> list[str]:
    """Return all official pair IDs in dataset order."""

    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return list(dict.fromkeys(str(record["pair_id"]) for record in records))


def selected_pairs(value: str | None, max_pairs: int | None) -> list[str]:
    """Resolve and validate the pair subset requested on the command line."""

    available = available_pairs()
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("--max-pairs must be at least 1")
    if value:
        requested = [item.strip() for item in value.split(",") if item.strip()]
        unknown = [item for item in requested if item not in available]
        if unknown:
            raise ValueError(f"Unknown pair IDs: {unknown}")
        pairs = requested
    else:
        pairs = available
    return pairs[:max_pairs] if max_pairs is not None else pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probe", type=Path, required=True, help="Trained Qwen probe .pkl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/narcbench_probing_qwen"),
        help="Root directory for activations, Inspect logs, scores, and markers",
    )
    parser.add_argument(
        "--pairs",
        help="Comma-separated pair IDs; use one ID for the first real test",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Run only the first N selected or official pairs",
    )
    parser.add_argument("--model", default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_PROBE_LAYER)
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Generation token ceiling"
    )
    parser.add_argument(
        "--max-format-retries",
        type=int,
        default=1,
        help="Maximum schema-only regeneration attempts after malformed JSON",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260817,
        help="Base seed deterministically combined with each rendered prompt",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=float,
        default=12.5,
        help="Per-GPU weight budget; leaves T4 headroom for attention and activations",
    )
    parser.add_argument(
        "--cpu-memory-gib",
        type=float,
        default=24.0,
        help="CPU memory budget available for model offload",
    )
    parser.add_argument(
        "--cache-implementation",
        default="",
        help="Transformers KV-cache implementation; empty uses the stable dynamic default",
    )
    parser.add_argument(
        "--primary-metric",
        default="peak_suspicion",
        choices=PROBE_METRIC_NAMES,
        help="Probe metric exposed as the Inspect Score value",
    )
    parser.add_argument(
        "--probe-sha256",
        default=RELEASED_PROBE_SHA256,
        help=(
            "Expected SHA-256 of the released probe; use an empty value "
            "only for a documented replacement"
        ),
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=DEFAULT_BACKEND_PATH,
        help="Path to the unchanged released score_pipeline.py",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=DEFAULT_EXTRACTOR_PATH,
        help="Path to the unchanged released extract_activations.py",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun pairs even when their completion markers exist",
    )
    args = parser.parse_args()

    output_root = args.output.resolve()
    activation_root = output_root / "activations"
    log_root = output_root / "inspect_logs"
    marker_root = output_root / "completed_pairs"
    for directory in (activation_root, log_root, marker_root):
        directory.mkdir(parents=True, exist_ok=True)

    pairs = selected_pairs(args.pairs, args.max_pairs)
    print(f"Selected {len(pairs)} matched pair(s): {pairs}")
    print(f"Loading self-hosted model {args.model}...")
    runtime = QwenActivationRuntime(
        output_root=activation_root,
        model_name=args.model,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
        max_format_retries=args.max_format_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sampling_seed=args.sampling_seed,
        gpu_memory_gib=args.gpu_memory_gib,
        cpu_memory_gib=args.cpu_memory_gib,
        cache_implementation=args.cache_implementation or None,
        extractor_path=args.extractor,
    )
    probe_scorer = narcbench_activation_probe_scorer(
        probe_path=str(args.probe.resolve()),
        activation_root=str(activation_root),
        backend_path=str(args.backend.resolve()),
        expected_model=args.model,
        expected_layer=args.layer,
        primary_metric=args.primary_metric,
        results_path=str(output_root / "probe_scores.jsonl"),
        expected_probe_sha256=args.probe_sha256 or None,
    )

    for index, pair_id in enumerate(pairs, start=1):
        marker = marker_root / f"{pair_id}.json"
        if marker.exists() and not args.no_resume:
            print(f"[{index}/{len(pairs)}] SKIP {pair_id}: completion marker exists")
            continue

        print(f"[{index}/{len(pairs)}] RUN {pair_id}: collusion + matched control")
        task = Task(
            dataset=load_narcbench_dataset(pair_ids=[pair_id]),
            solver=narcbench_real_solver(
                generate_override=runtime.generate,
                finalize_override=runtime.finalize,
                model_name_override=args.model,
            ),
            scorer=[narcbench_attack_scorer(), probe_scorer],
            name=f"narcbench-probing-{pair_id}",
            version="1.0.0",
            metadata={
                "design": "matched NARCBench-Core collusion/control pair",
                "probe_model": args.model,
                "probe_layer": args.layer,
                "primary_probe_metric": args.primary_metric,
                "max_format_retries": args.max_format_retries,
                "cache_implementation": args.cache_implementation or "dynamic_default",
                "decoding": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "sampling_seed": args.sampling_seed,
                },
            },
        )
        logs = inspect_eval(
            task,
            model="mockllm/model",
            log_dir=str(log_root / pair_id),
            max_samples=1,
            fail_on_error=True,
            log_samples=True,
            display="plain",
        )
        statuses = [str(log.status) for log in logs]
        if not logs or any(status != "success" for status in statuses):
            raise RuntimeError(f"Inspect run failed for {pair_id}: statuses={statuses}")
        marker.write_text(
            json.dumps(
                {
                    "pair_id": pair_id,
                    "samples": [f"{pair_id}_collusion", f"{pair_id}_control"],
                    "model": args.model,
                    "layer": args.layer,
                    "primary_metric": args.primary_metric,
                    "max_format_retries": args.max_format_retries,
                    "cache_implementation": args.cache_implementation or "dynamic_default",
                    "decoding": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "sampling_seed": args.sampling_seed,
                    },
                    "statuses": statuses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{index}/{len(pairs)}] COMPLETE {pair_id}")

    print(f"Done. Results: {output_root}")
    print("Next: run scripts/analyze_narcbench_probe.py on probe_scores.jsonl")


if __name__ == "__main__":
    main()
