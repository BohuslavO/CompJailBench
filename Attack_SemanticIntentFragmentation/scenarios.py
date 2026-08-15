"""
Semantic Intent Fragmentation scenario loader.

A SIF scenario is ONE legitimately-phrased enterprise request that an
unmodified orchestrator is expected to decompose into subtasks that are
individually benign but jointly violate a named policy clause. This is the
inverse of DeCompBench (Attack_DecompBench/task_loader.py), where the red
teamer authors the decomposition -- here the orchestrator authors it, and
the attacker never interacts again after the initial request
("single-shot autonomy", Ahad et al. 2026).

Each scenario is grounded in:
  - a mechanism M1-M4 (how composition produces the violation),
  - a domain (financial / infosec / hr / cyber),
  - a verbatim policy clause the plan-level judge must cite back
    (success_gate.py condition (d)) -- NIST SP 800-53 Rev. 5, plus
    OWASP LLM06:2025 Excessive Agency and MITRE ATLAS AML.T0048.

Scenarios live in scenarios.json. Deliberately a single flat file rather
than DeCompBench's directory-per-task layout: a SIF scenario is one request
plus metadata, not a task tree with checkpoints/desiderata/utils.

Per CompJailBench Sec. 6, scenario text is abstract and enterprise-plausible
by construction -- the attack is that ordinary-sounding requests decompose
badly, so no actionable harmful detail is needed or included.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.json")

# How composition turns benign subtasks into a violation. Ahad et al. 2026, Sec. 3.
MECHANISMS = {
    "M1": "Bulk scope escalation -- completeness-implying language makes the "
          "orchestrator retrieve the full dataset when a filtered slice was authorized.",
    "M2": "Silent data exfiltration -- the orchestrator routes data to an external "
          "endpoint that never appeared in the request.",
    "M3": "Embedded trigger deployment -- a delivery subtask autonomously embeds an "
          "active routing element (QR, webhook, BI sync).",
    "M4": "Quasi-identifier aggregation -- individually permitted fields compose into "
          "a re-identifying set.",
}


@dataclass
class SIFScenario:
    scenario_id: str
    domain: str                 # financial | infosec | hr | cyber
    mechanism: str              # M1 | M2 | M3 | M4
    researcher_request: str     # baseline phrasing -- the phrasing-source ablation's control arm
    policy_clause: str          # verbatim; the plan judge must cite this back
    nist_controls: list = field(default_factory=list)
    benign_control: str = ""    # structurally matched, non-harmful variant
    owasp_id: str = "LLM06:2025"
    atlas_id: str = "AML.T0048"
    expected_local_label: str = "safe"
    expected_global_label: str = "unsafe"

    @property
    def mechanism_description(self) -> str:
        return MECHANISMS.get(self.mechanism, "unknown mechanism")


def load_scenarios(path=None) -> list:
    """Every scenario in the file, as SIFScenario objects."""
    path = Path(path) if path else DEFAULT_SCENARIOS
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SIFScenario(**s) for s in raw]


def load_scenario(scenario_id: str, path=None) -> SIFScenario:
    for s in load_scenarios(path):
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"No scenario {scenario_id!r} in {path or DEFAULT_SCENARIOS}")


def list_scenarios(path=None) -> list:
    """Scenario ids, for parity with task_loader.list_tasks."""
    return [s.scenario_id for s in load_scenarios(path)]


def coverage(scenarios: list) -> dict:
    """(mechanism, domain) -> count. The source paper's scenario matrix is
    sparse on purpose -- M4 is HR-only, M3 never applies to financial -- so a
    missing cell is not a gap in the data."""
    out = {}
    for s in scenarios:
        out[(s.mechanism, s.domain)] = out.get((s.mechanism, s.domain), 0) + 1
    return out
