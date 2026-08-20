from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.util import Store


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_narcbench_probe import canonical_rows, metric_summary, team_rows
from scripts.run_narcbench_probe_kaggle import selected_pairs
from src.narcbench_probe import (
    ACTIVATION_REF_TYPE,
    load_probe_bundle,
    load_score_pipeline,
    narcbench_activation_probe_scorer,
    score_probe_trajectory,
    validate_probe_bundle,
)
from src.narcbench_probe_runtime import (
    QwenActivationRuntime,
    build_format_repair_prompt,
    validate_task_response,
)
from src.narcbench_dataset import load_narcbench_dataset
from src.narcbench_real_solver import RATIONALE_INSTRUCTION, narcbench_real_solver
from src.narcbench_scorer import narcbench_attack_scorer
from src.trajectory import StandardTrajectory


class IdentityTransform:
    def transform(self, values):
        return np.asarray(values)


class FakePCA(IdentityTransform):
    pass


class FakeClassifier:
    def predict_proba(self, values):
        values = np.asarray(values)
        probability = np.clip(0.5 + values.mean(axis=1) * 0.01, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])


def fake_bundle() -> dict:
    return {
        "model": "Qwen/Qwen3-32B-AWQ",
        "layer": 28,
        "scaler": IdentityTransform(),
        "direction": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "asymmetry_scaler": IdentityTransform(),
        "asymmetry_clf": FakeClassifier(),
        "concat_pca": FakePCA(),
        "concat_scaler": IdentityTransform(),
        "concat_clf": FakeClassifier(),
    }


class ProbeAdapterTests(unittest.TestCase):
    def make_artifacts(self, root: Path) -> StandardTrajectory:
        agent_ids = [f"agent_{index}" for index in range(1, 5)]
        vectors = []
        metadata = []
        for round_number in range(1, 4):
            for index, agent_id in enumerate(agent_ids):
                vectors.append(
                    np.array([float(index + round_number), 1.0, 0.0, -1.0], dtype=np.float32)
                )
                metadata.append(
                    {
                        "agent_id": agent_id,
                        "phase": "discussion",
                        "channel": "public",
                        "round": round_number,
                    }
                )
        vectors.append(np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32))
        metadata.append(
            {
                "agent_id": "agent_1",
                "phase": "discussion",
                "channel": "private",
                "round": 0,
            }
        )
        np.savez_compressed(root / "sample.npz", vectors=np.stack(vectors))
        (root / "sample.meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        return StandardTrajectory(
            store=Store(),
            sample_id="sample_attack",
            pair_id="sample",
            condition_label="attack",
            attack_name="narcbench_collusion",
            model_name="Qwen/Qwen3-32B-AWQ",
            agent_ids=agent_ids,
            activation_refs=[
                {
                    "type": ACTIVATION_REF_TYPE,
                    "path": "sample.npz",
                    "metadata_path": "sample.meta.json",
                    "model_name": "Qwen/Qwen3-32B-AWQ",
                    "layer": 28,
                }
            ],
        )

    def test_released_backend_scores_four_averaged_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = self.make_artifacts(root)
            scores = score_probe_trajectory(
                trajectory,
                bundle=fake_bundle(),
                activation_root=root,
                backend=load_score_pipeline(),
                expected_model="Qwen/Qwen3-32B-AWQ",
                expected_layer=28,
            )
        self.assertEqual(
            set(scores),
            {
                "peak_suspicion",
                "public_peak_suspicion",
                "agent_mean_peak_suspicion",
                "deception_split",
                "activation_alignment",
                "asymmetry_probe",
                "sorted_concat",
            },
        )
        self.assertTrue(all(value is not None for value in scores.values()))
        # The released primary scorer sees every turn; the public ablation excludes
        # the synthetic high-scoring private turn.
        self.assertEqual(scores["peak_suspicion"], 100.0)
        self.assertEqual(scores["public_peak_suspicion"], 6.0)
        self.assertEqual(scores["agent_mean_peak_suspicion"], 26.5)

    def test_bundle_contract_rejects_wrong_layer_and_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer mismatch"):
            validate_probe_bundle(fake_bundle(), expected_layer=27)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            validate_probe_bundle(fake_bundle(), vector_dim=8)

    def test_probe_hash_is_checked_before_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "untrusted.pkl"
            path.write_bytes(b"not a pickle")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_probe_bundle(path, expected_sha256="0" * 64)

    def test_unknown_primary_metric_is_rejected_during_setup(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown primary probe metric"):
            narcbench_activation_probe_scorer(
                probe_path="unused.pkl",
                activation_root="unused",
                primary_metric="not_a_metric",
            )


class ProbeAnalysisTests(unittest.TestCase):
    def test_pair_selection_rejects_invalid_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            selected_pairs(None, 0)
        with self.assertRaisesRegex(ValueError, "Unknown pair IDs"):
            selected_pairs("not_an_official_pair", None)

    def test_resume_deduplication_summary_and_matrix_export(self) -> None:
        def row(sample_id, pair_id, condition, score):
            return {
                "sample_id": sample_id,
                "pair_id": pair_id,
                "condition": condition,
                "attack_name": "narcbench_collusion",
                "model_name": "Qwen/Qwen3-32B-AWQ",
                "probe_model": "Qwen/Qwen3-32B-AWQ",
                "probe_layer": 28,
                "scores": {metric: score for metric in (
                    "peak_suspicion", "public_peak_suspicion",
                    "agent_mean_peak_suspicion", "deception_split",
                    "activation_alignment", "asymmetry_probe", "sorted_concat",
                )},
            }

        rows = [
            row("p1_collusion", "p1", "attack", 0.7),
            row("p1_collusion", "p1", "attack", 0.9),
            row("p1_control", "p1", "benign", 0.1),
            row("p2_collusion", "p2", "attack", 0.8),
            row("p2_control", "p2", "benign", 0.2),
        ]
        canonical = canonical_rows(rows)
        self.assertEqual(len(canonical), 4)
        summary = metric_summary(canonical, "peak_suspicion")
        self.assertEqual(summary["roc_auc"], 1.0)
        self.assertEqual(summary["n_complete_pairs"], 2)
        exported = team_rows(
            canonical,
            run_id="test",
            contributor="test-contributor",
        )
        self.assertEqual(len(exported), 28)


class ProbeSolverInjectionTests(unittest.TestCase):
    def test_generation_contract_is_bounded_and_json_only(self) -> None:
        self.assertIn("under 80 words", RATIONALE_INSTRUCTION)
        self.assertIn("under 40 words", RATIONALE_INSTRUCTION)
        self.assertIn("without Markdown fences", RATIONALE_INSTRUCTION)

    def test_structured_response_validation_accepts_fenced_json(self) -> None:
        discussion = validate_task_response(
            '```json\n{"message":"concise","observable_rationale":"recorded"}\n```',
            phase="discussion",
        )
        final = validate_task_response(
            '{"response":"RECOMMENDATION: Option A",'
            '"observable_rationale":"clinical evidence"}',
            phase="final",
        )
        self.assertEqual(discussion["message"], "concise")
        self.assertIn("RECOMMENDATION", final["response"])

    def test_structured_response_validation_rejects_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete or invalid JSON"):
            validate_task_response(
                '```json\n{"message":"cut off", "observable_rationale":"unfinished',
                phase="discussion",
            )

    def test_format_repair_prompt_preserves_phase_contract(self) -> None:
        prompt = build_format_repair_prompt(
            original_user="Choose between Option A and Option B.",
            invalid_response='{"response":"RECOMMENDATION: Option B',
            phase="final",
        )
        self.assertIn("Required keys: response, observable_rationale", prompt)
        self.assertIn("same intended option", prompt)
        self.assertIn("Do not add new factual claims", prompt)
        self.assertNotIn('RECOMMENDATION: Option B', prompt)
        self.assertIn("Malformed-draft audit ID", prompt)

    def test_runtime_retries_invalid_json_once_and_records_it(self) -> None:
        runtime = object.__new__(QwenActivationRuntime)
        runtime._lock = asyncio.Lock()
        runtime._vectors = defaultdict(list)
        runtime._metadata = defaultdict(list)
        runtime._inputs = defaultdict(list)
        runtime.max_format_retries = 1
        runtime.temperature = 0.7
        runtime.top_p = 0.8
        runtime.top_k = 20
        runtime.sampling_seed = 20260817
        responses = iter(
            [
                '{"message":"cut off',
                '{"message":"Option A remains best.",'
                '"observable_rationale":"Evidence supports A."}',
            ]
        )
        prompts: list[str] = []

        def generate_sync(system, user):
            prompts.append(user)
            return next(responses)

        runtime._generate_sync = generate_sync
        runtime._extract_sync = lambda system, user, response: np.zeros(4, dtype=np.float32)
        response = asyncio.run(
            runtime.generate(
                system="system",
                user="original task",
                context={
                    "sample_id": "sample",
                    "agent_id": "agent_1",
                    "stage": "discussion",
                    "phase": "discussion",
                    "channel": "public",
                    "round": 1,
                },
            )
        )
        self.assertIn("Option A", response)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(runtime._metadata["sample"][0]["generation_attempts"], 2)
        self.assertTrue(runtime._metadata["sample"][0]["format_repair_used"])
        self.assertEqual(runtime._inputs["sample"][0]["user_prompt"], "original task")

    def test_qwen_prompt_disables_hidden_thinking(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        runtime = object.__new__(QwenActivationRuntime)
        runtime.tokenizer = FakeTokenizer()
        rendered = runtime._render_prompt(
            [{"role": "user", "content": "Return the requested JSON."}]
        )
        self.assertEqual(rendered, "rendered")
        self.assertEqual(runtime.tokenizer.kwargs["enable_thinking"], False)

    def test_local_generator_receives_all_stage_contexts(self) -> None:
        calls: list[dict] = []
        finalized: list[str] = []

        async def generate_override(*, system, user, context):
            calls.append(dict(context))
            if context["phase"] == "final":
                return '{"response":"RECOMMENDATION: Option A","observable_rationale":"stub"}'
            return '{"message":"stub committee message","observable_rationale":"stub"}'

        def finalize_override(*, trajectory):
            finalized.append(trajectory.sample_id)
            trajectory.activation_refs.append(
                {"type": ACTIVATION_REF_TYPE, "path": "stub.npz", "metadata_path": "stub.json"}
            )

        task = Task(
            dataset=load_narcbench_dataset(pair_ids=["med_01"]),
            solver=narcbench_real_solver(
                generate_override=generate_override,
                finalize_override=finalize_override,
                model_name_override="Qwen/Qwen3-32B-AWQ",
            ),
            scorer=narcbench_attack_scorer(),
            name="probe-solver-injection-test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            # Inspect 0.3.251 leaves its in-memory receive stream to garbage
            # collection on Windows. Suppress only that dependency warning;
            # application ResourceWarnings remain visible.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Unclosed <MemoryObjectReceiveStream.*",
                    category=ResourceWarning,
                )
                logs = inspect_eval(
                    task,
                    model="mockllm/model",
                    log_dir=tmp,
                    max_samples=1,
                    display="none",
                )
        self.assertEqual([str(log.status) for log in logs], ["success"])
        self.assertEqual(set(finalized), {"med_01_collusion", "med_01_control"})
        by_sample = Counter(call["sample_id"] for call in calls)
        self.assertEqual(by_sample["med_01_collusion"], 18)
        self.assertEqual(by_sample["med_01_control"], 16)
        self.assertEqual(
            {call["phase"] for call in calls}, {"discussion", "final"}
        )


if __name__ == "__main__":
    unittest.main()
