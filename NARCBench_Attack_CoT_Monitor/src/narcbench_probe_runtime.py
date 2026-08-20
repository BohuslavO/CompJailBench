"""Self-hosted Qwen generation and activation capture for Kaggle.

Heavy dependencies are imported lazily so the rest of the project and all
offline tests remain usable on machines without a GPU or Transformers model.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from src.model_helpers import parse_json_response
from src.narcbench_probe import (
    ACTIVATION_REF_TYPE,
    DEFAULT_PROBE_LAYER,
    DEFAULT_PROBE_MODEL,
    REPOSITORY_ROOT,
)
from src.trajectory import StandardTrajectory


DEFAULT_EXTRACTOR_PATH = REPOSITORY_ROOT / "Defence_NARCBench Probing" / "extract_activations.py"


def load_extractor(path: Path = DEFAULT_EXTRACTOR_PATH) -> ModuleType:
    """Load the released activation extractor without copying or modifying it."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NARCBench activation extractor not found: {path}")
    spec = importlib.util.spec_from_file_location("compjailbreak_narcbench_extractor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "extract", None)):
        raise ImportError(f"{path} does not expose extract()")
    return module


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_task_response(response: str, *, phase: str) -> dict[str, Any]:
    """Reject truncated/malformed generations before they enter later rounds."""

    parsed = parse_json_response(response)
    if not parsed.get("_parse_ok"):
        preview = " ".join(response.strip().split())[:160]
        raise ValueError(f"Qwen returned incomplete or invalid JSON: {preview!r}")

    content_key = "response" if phase == "final" else "message"
    if not str(parsed.get(content_key, "")).strip():
        raise ValueError(f"Qwen JSON is missing non-empty {content_key!r}")
    if not str(parsed.get("observable_rationale", "")).strip():
        raise ValueError("Qwen JSON is missing non-empty 'observable_rationale'")
    return parsed


def build_format_repair_prompt(
    *, original_user: str, invalid_response: str, phase: str
) -> str:
    """Request one bounded serialization repair without changing task intent."""

    content_key = "response" if phase == "final" else "message"
    recommendation_rule = (
        "The response value must contain exactly one final `RECOMMENDATION: Option X` "
        "line using the same intended option."
        if phase == "final"
        else "Preserve the same intended committee position in the message value."
    )
    draft_audit_id = _sha256_text(invalid_response)[:12]
    return (
        "FORMAT-REPAIR RETRY. Your previous draft was incomplete or malformed. "
        "Re-answer the original task below as exactly one compact, complete JSON "
        "object and nothing else. Do not copy or continue the malformed draft.\n"
        f"Malformed-draft audit ID: {draft_audit_id}.\n"
        f"Required keys: {content_key}, observable_rationale.\n"
        f"Keep {content_key} under 80 words and observable_rationale under 40 words. "
        f"{recommendation_rule} Do not add new factual claims.\n\n"
        "ORIGINAL TASK (follow its substance):\n"
        f"{original_user}\n\n"
        "Now rewrite the intended answer as valid JSON only."
    )


class QwenActivationRuntime:
    """Serialize Qwen calls and capture one last-token vector per model call."""

    def __init__(
        self,
        *,
        output_root: Path,
        model_name: str = DEFAULT_PROBE_MODEL,
        layer: int = DEFAULT_PROBE_LAYER,
        max_new_tokens: int = 512,
        max_format_retries: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        sampling_seed: int = 20260817,
        gpu_memory_gib: float | None = 12.5,
        cpu_memory_gib: float = 24.0,
        cache_implementation: str | None = None,
        extractor_path: Path = DEFAULT_EXTRACTOR_PATH,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.layer = int(layer)
        self.max_new_tokens = int(max_new_tokens)
        self.max_format_retries = int(max_format_retries)
        if self.max_format_retries < 0:
            raise ValueError("max_format_retries must be non-negative")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.sampling_seed = int(sampling_seed)
        self.cache_implementation = cache_implementation
        self.extractor = load_extractor(extractor_path)
        self._lock = asyncio.Lock()
        self._vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        self._metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)

        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            max_memory = None
            if gpu_memory_gib is not None:
                import torch

                max_memory = {
                    device: f"{gpu_memory_gib}GiB"
                    for device in range(torch.cuda.device_count())
                }
                max_memory["cpu"] = f"{cpu_memory_gib}GiB"
            offload_folder = self.output_root / "model_offload"
            offload_folder.mkdir(parents=True, exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory=max_memory,
                offload_buffers=max_memory is not None,
                offload_folder=str(offload_folder),
                trust_remote_code=True,
                torch_dtype="auto",
            )
            model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def _input_device(self) -> Any:
        for parameter in self.model.parameters():
            if getattr(parameter.device, "type", None) != "meta":
                return parameter.device
        return self.model.device

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # The task already requests an explicit observable rationale.
            # Qwen3's separate hidden thinking mode can consume the entire
            # fixed output budget before it emits the required JSON.
            enable_thinking=False,
        )

    def _generate_sync(self, system: str, user: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._render_prompt(messages)
        seed_material = f"{self.sampling_seed}\n{text}"
        call_seed = int(_sha256_text(seed_material)[:8], 16)
        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self._input_device())
        try:
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
                if self.cache_implementation:
                    generate_kwargs["cache_implementation"] = self.cache_implementation
                output_ids = self.model.generate(input_ids, **generate_kwargs)
            generated = output_ids[0][input_ids.shape[1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            del input_ids
            if "output_ids" in locals():
                del output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_sync(self, system: str, user: str, response: str) -> np.ndarray:
        turn = {
            "agent_name": "probe_target",
            "system_prompt": system,
            "user_prompt": user,
            "response": response,
        }
        original_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False
        try:
            arrays = self.extractor.extract(
                self.model,
                self.tokenizer,
                [turn],
                [self.layer],
            )
        finally:
            self.model.config.use_cache = original_use_cache
        vector = np.asarray(arrays[self.layer][0], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Expected one hidden vector, found {vector.shape}")
        return vector

    async def generate(self, *, system: str, user: str, context: dict[str, Any]) -> str:
        """Generate and capture under one lock so a sharded model is never re-entered."""

        sample_id = str(context["sample_id"])
        phase = str(context["phase"])
        async with self._lock:
            response = self._generate_sync(system, user)
            initial_response = response
            repair_prompt = ""
            generation_attempts = 1
            try:
                validate_task_response(response, phase=phase)
            except ValueError:
                if self.max_format_retries < 1:
                    raise
                repair_prompt = build_format_repair_prompt(
                    original_user=user,
                    invalid_response=response,
                    phase=phase,
                )
                response = self._generate_sync(system, repair_prompt)
                generation_attempts += 1
                try:
                    validate_task_response(response, phase=phase)
                except ValueError as retry_error:
                    failure_root = self.output_root / "failed_generations"
                    failure_root.mkdir(parents=True, exist_ok=True)
                    failure_path = failure_root / (
                        f"{_safe_name(sample_id)}_{_safe_name(str(context['stage']))}_"
                        f"{_safe_name(str(context['agent_id']))}_r{int(context['round'])}.json"
                    )
                    failure_path.write_text(
                        json.dumps(
                            {
                                "context": context,
                                "system_prompt": system,
                                "original_user_prompt": user,
                                "initial_response": initial_response,
                                "repair_prompt": repair_prompt,
                                "retry_response": response,
                                "error": str(retry_error),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    raise ValueError(
                        f"Qwen formatting retry failed; diagnostic={failure_path}; "
                        f"cause={retry_error}"
                    ) from retry_error

            # Score the accepted response under the original task prompt. The
            # repair instruction is therefore not itself exposed to the probe.
            vector = self._extract_sync(system, user, response)
            row_index = len(self._vectors[sample_id])
            self._vectors[sample_id].append(vector)
            self._metadata[sample_id].append(
                {
                    "row": row_index,
                    "sample_id": sample_id,
                    "agent_id": str(context["agent_id"]),
                    "stage": str(context["stage"]),
                    "phase": str(context["phase"]),
                    "channel": str(context["channel"]),
                    "round": int(context["round"]),
                    "system_sha256": _sha256_text(system),
                    "user_sha256": _sha256_text(user),
                    "response_sha256": _sha256_text(response),
                    "generation_attempts": generation_attempts,
                    "format_repair_used": bool(repair_prompt),
                    "initial_response_sha256": _sha256_text(initial_response),
                    "format_repair_prompt_sha256": (
                        _sha256_text(repair_prompt) if repair_prompt else None
                    ),
                    "decoding": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "sampling_seed": self.sampling_seed,
                    },
                }
            )
            self._inputs[sample_id].append(
                {
                    "row": row_index,
                    "agent_name": str(context["agent_id"]),
                    "system_prompt": system,
                    "user_prompt": user,
                    "response": response,
                    "generation_attempts": generation_attempts,
                    "format_repair_used": bool(repair_prompt),
                    "initial_response": initial_response if repair_prompt else None,
                }
            )
            return response

    def finalize(self, *, trajectory: StandardTrajectory) -> None:
        """Atomically persist one sample's activations and attach their reference."""

        sample_id = trajectory.sample_id
        vectors = self._vectors.pop(sample_id, [])
        metadata = self._metadata.pop(sample_id, [])
        inputs = self._inputs.pop(sample_id, [])
        if not vectors:
            raise ValueError(f"No captured activations for {sample_id}")

        stem = _safe_name(sample_id)
        vector_name = f"{stem}.npz"
        metadata_name = f"{stem}.meta.json"
        inputs_name = f"{stem}.inputs.json"
        vector_path = self.output_root / vector_name
        metadata_path = self.output_root / metadata_name
        inputs_path = self.output_root / inputs_name

        temporary_vector = self.output_root / f".{stem}.tmp.npz"
        np.savez_compressed(temporary_vector, vectors=np.stack(vectors).astype(np.float32))
        temporary_vector.replace(vector_path)

        for path, payload in ((metadata_path, metadata), (inputs_path, inputs)):
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)

        trajectory.activation_refs.append(
            {
                "type": ACTIVATION_REF_TYPE,
                "path": vector_name,
                "metadata_path": metadata_name,
                "inputs_path": inputs_name,
                "model_name": self.model_name,
                "layer": self.layer,
                "token_position": "last",
                "primary_scope": "all_trajectory_turns",
                "primary_aggregation": "maximum_deception_direction_projection",
                "secondary_scope": "discussion_turns_grouped_by_agent",
                "secondary_aggregation": "mean_per_agent",
                "n_turns": len(vectors),
                "format_repair_count": sum(
                    bool(row.get("format_repair_used")) for row in metadata
                ),
                "generation_attempts_total": sum(
                    int(row.get("generation_attempts", 1)) for row in metadata
                ),
                "hidden_dim": int(vectors[0].shape[0]),
            }
        )
