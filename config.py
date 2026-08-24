from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can be set directly instead.
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"
LOGS_DIR = RESULTS_DIR / "inspect_logs"

for _dir in (RESULTS_DIR, TRACES_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Inspect AI model strings look like "<provider>/<model-name>", e.g.
#   "openai/gpt-4o-mini"
#   "anthropic/claude-sonnet-4-6"
#   "openai/azure/gpt-4o"
# Change these to whatever you actually have API access to. Nothing runs
# until (a) these are valid model strings for your Inspect AI version and
# (b) the corresponding API key env var is set.

# The model that plays Planner / Worker / Aggregator inside the multi-agent
# system under test (i.e. the thing being attacked by AgentHarm).
AGENT_MODEL = os.environ.get("COMPJAILBENCH_AGENT_MODEL", "openai/gpt-4o-mini")

# The model used by LLM-as-judge defenses (SentinelAgent's oversight agent,
# the CoT Monitor autorater). Kept separate from AGENT_MODEL so you can use
# a stronger/cheaper model for judging than for the agents under test.
JUDGE_MODEL = os.environ.get("COMPJAILBENCH_JUDGE_MODEL", AGENT_MODEL)


# ---------------------------------------------------------------------------
# Defense registry
# ---------------------------------------------------------------------------
# Which defenses are wired up and runnable right now. PROBING is included in
# the registry for completeness (per teammate ownership) but is disabled by
# default because NARCBench-style probing needs white-box access to hidden
# states, which is only possible with a local open-weight model (via
# transformers / vLLM exposing hidden_states), not a closed API model.
# Flip ENABLE_PROBING to True once you have that set up -- see
# defenses/probing.py for exactly what's required.

ENABLE_PROBING = os.environ.get("COMPJAILBENCH_ENABLE_PROBING", "false").lower() == "true"

# Generate each AgentHarm trajectory exactly once. The comparable 4x4
# defenses consume the saved StandardTrajectory JSONL afterwards.
DEFENSES = ["none"]


# ---------------------------------------------------------------------------
# AgentHarm dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentHarmConfig:
    # Harmful and benign are paired conditions, not Inspect dataset splits.
    condition: str = os.environ.get("COMPJAILBENCH_AGENTHARM_CONDITION", "harmful")
    # Official Inspect Evals split name.
    split: str = os.environ.get("COMPJAILBENCH_AGENTHARM_DATASET_SPLIT", "test_public")
    # Cap the number of samples for fast local iteration; set to None for the
    # full split.
    limit: int | None = (
        int(os.environ["COMPJAILBENCH_LIMIT"])
        if os.environ.get("COMPJAILBENCH_LIMIT")
        else 10
    )
    force_redownload: bool = (
        os.environ.get("COMPJAILBENCH_AGENTHARM_FORCE_REDOWNLOAD", "false").lower()
        == "true"
    )
    # Placeholders are forbidden for experimental runs. This switch exists
    # only for explicitly requested UI/demo smoke tests.
    allow_placeholder: bool = (
        os.environ.get("COMPJAILBENCH_ALLOW_PLACEHOLDER", "false").lower()
        == "true"
    )

    def validate(self) -> None:
        if self.condition not in {"harmful", "benign"}:
            raise ValueError(
                "COMPJAILBENCH_AGENTHARM_CONDITION must be 'harmful' or 'benign'."
            )
        if self.split not in {"val", "test_public", "test_private"}:
            raise ValueError(
                "COMPJAILBENCH_AGENTHARM_DATASET_SPLIT must be one of "
                "'val', 'test_public', or 'test_private'."
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("COMPJAILBENCH_LIMIT must be a positive integer.")


AGENTHARM_CONFIG = AgentHarmConfig()
AGENTHARM_CONFIG.validate()

RUN_ID = os.environ.get(
    "COMPJAILBENCH_RUN_ID",
    datetime.now(timezone.utc).strftime("agentharm_%Y%m%dT%H%M%SZ"),
)


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "COMPJAILBENCH_RUN_ID may contain only letters, numbers, '.', '_', "
            "and '-' (maximum 128 characters)."
        )


validate_run_id(RUN_ID)


# ---------------------------------------------------------------------------
# Multi-agent pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    num_workers: int = 2
    max_planner_tokens: int = int(
        os.environ.get("COMPJAILBENCH_MAX_PLANNER_TOKENS", "1536")
    )
    max_worker_tokens: int = int(
        os.environ.get("COMPJAILBENCH_MAX_WORKER_TOKENS", "1536")
    )
    max_aggregator_tokens: int = int(
        os.environ.get("COMPJAILBENCH_MAX_AGGREGATOR_TOKENS", "3072")
    )
    empty_response_retries: int = int(
        os.environ.get("COMPJAILBENCH_EMPTY_RESPONSE_RETRIES", "2")
    )
    reasoning_effort: str | None = os.environ.get(
        "COMPJAILBENCH_REASONING_EFFORT",
        "low" if "gpt-oss" in AGENT_MODEL.lower() else "",
    ) or None
    # Whether worker/planner agents must emit an explicit <reasoning> block
    # before their action -- required for the CoT Monitor defense to have
    # anything to read. Harmless to leave on for the other defenses too.
    require_visible_reasoning: bool = True


PIPELINE_CONFIG = PipelineConfig()


def api_keys_present() -> dict[str, bool]:
    """Quick sanity check you can call before a run to see what's configured."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "AZUREAI_OPENAI_API_KEY": bool(os.environ.get("AZUREAI_OPENAI_API_KEY")),
        "AZUREAI_OPENAI_BASE_URL": bool(os.environ.get("AZUREAI_OPENAI_BASE_URL")),
        "AZUREAI_OPENAI_API_VERSION": bool(os.environ.get("AZUREAI_OPENAI_API_VERSION")),
        "AZURE_OPENAI_API_KEY": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
        "AWS_BEARER_TOKEN_BEDROCK": bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK")),
        "BEDROCK_OPENAI_API_KEY": bool(os.environ.get("BEDROCK_OPENAI_API_KEY")),
    }


def model_credentials_present(model_name: str) -> bool:
    """Conservative preflight check for the configured Inspect model ID."""
    keys = api_keys_present()
    if model_name.startswith("mockllm/"):
        return True
    if model_name.startswith("openai/azure/"):
        return (
            keys["AZUREAI_OPENAI_API_KEY"] or keys["AZURE_OPENAI_API_KEY"]
        ) and keys["AZUREAI_OPENAI_BASE_URL"]
    if model_name.startswith("openai/bedrock/"):
        return keys["AWS_BEARER_TOKEN_BEDROCK"] or keys["BEDROCK_OPENAI_API_KEY"]
    if model_name.startswith("openai/"):
        return keys["OPENAI_API_KEY"]
    if model_name.startswith("anthropic/"):
        return keys["ANTHROPIC_API_KEY"]
    return any(keys.values())


if __name__ == "__main__":
    import json

    print("AGENT_MODEL:", AGENT_MODEL)
    print("JUDGE_MODEL:", JUDGE_MODEL)
    print("RUN_ID:", RUN_ID)
    print("DEFENSES:", DEFENSES)
    print("API keys present:", json.dumps(api_keys_present(), indent=2))
