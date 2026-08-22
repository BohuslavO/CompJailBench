from __future__ import annotations

import os
from dataclasses import dataclass, field
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
#   "azureai/gpt-4o"
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

DEFENSES = ["none", "g_safeguard", "sentinel_agent", "cot_monitor"] + (
    ["probing"] if ENABLE_PROBING else []
)


# ---------------------------------------------------------------------------
# AgentHarm dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentHarmConfig:
    # "harmful" or "benign" split of AgentHarm.
    split: str = os.environ.get("COMPJAILBENCH_AGENTHARM_SPLIT", "harmful")
    # Inspect Evals task variant: "agentharm" (standard) or "agentharm_avoidance"
    # (the "reasoning about a request without acting" variant). See
    # datasets/agentharm_loader.py.
    variant: str = os.environ.get("COMPJAILBENCH_AGENTHARM_VARIANT", "agentharm")
    # Cap the number of samples for fast local iteration; set to None for the
    # full split.
    limit: int | None = (
        int(os.environ["COMPJAILBENCH_LIMIT"])
        if os.environ.get("COMPJAILBENCH_LIMIT")
        else 10
    )


AGENTHARM_CONFIG = AgentHarmConfig()


# ---------------------------------------------------------------------------
# Multi-agent pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    num_workers: int = 2
    max_planner_tokens: int = 800
    max_worker_tokens: int = 800
    max_aggregator_tokens: int = 800
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
        "AZURE_OPENAI_API_KEY": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
        "AZURE_OPENAI_ENDPOINT": bool(os.environ.get("AZURE_OPENAI_ENDPOINT")),
    }


if __name__ == "__main__":
    import json

    print("AGENT_MODEL:", AGENT_MODEL)
    print("JUDGE_MODEL:", JUDGE_MODEL)
    print("DEFENSES:", DEFENSES)
    print("API keys present:", json.dumps(api_keys_present(), indent=2))