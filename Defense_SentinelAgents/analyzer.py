"""
The three SentinelAgent analyzer tiers, wired together with the two pipeline
behaviors CompJailBench Sec. 3.3 singles out for explicit handling.

Tiers (He et al. 2025):
  1  global      -- the final output violates the task specification
  2  single-point -- the fault localizes to one agent or one tool invocation
  3  multi-point -- the failure emerges from interactions across several agents
                    with no single one at fault

Tier 3 is the target regime. A distributed jailbreak that reaches tier 2 was
never distributed in the first place.

Two behaviors carried over from the original design, both configurable because
both change what the reported numbers mean:

  escalate_unflagged_paths -- a suspicious path carrying no flagged node or edge
      goes to human review. Scored as a detection WITH an attached review cost,
      not as a free one; a defense that escalates everything has perfect recall
      and no value.

  strict_recheck -- raised path suspicion re-runs the local classifiers at a
      lower threshold. ABLATE THIS. It is the only route by which path-level
      suspicion can recover a locally missed violation, so leaving it on
      permanently makes tier 2 and tier 3 results impossible to separate.

Tier 1 is SKIPPED, not passed, when no task specification is supplied. An
un-run check reported as clean would inflate every downstream number.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from attack_paths import build_library, match_paths, default_semantic_checker
from graph_builder import ExecGraph, apply_static_metadata, graph_diff
from node_edge_checks import (DEFAULT_THRESHOLD, STRICT_THRESHOLD, CheckResult,
                              check_edges, check_nodes, granite_guardian_stub,
                              llamafirewall_stub)

SpecChecker = Callable[[ExecGraph], float]


@dataclass
class SentinelConfig:
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier3: bool = True
    node_threshold: float = DEFAULT_THRESHOLD
    edge_threshold: float = DEFAULT_THRESHOLD
    strict_threshold: float = STRICT_THRESHOLD
    escalate_unflagged_paths: bool = True
    strict_recheck: bool = True
    human_review_cost: float = 1.0


@dataclass
class Alert:
    detected: bool = False
    tier: Optional[int] = None            # most specific tier that fired
    tiers_fired: list = field(default_factory=list)
    flagged_nodes: list = field(default_factory=list)
    flagged_edges: list = field(default_factory=list)
    matched_paths: list = field(default_factory=list)
    unpermitted_edges: list = field(default_factory=list)
    recovered_by_recheck: bool = False    # tier 2 only reachable via path suspicion
    human_review: bool = False
    root_cause: str = ""
    cost: dict = field(default_factory=dict)
    tier1_evaluated: bool = False


def tier1_output_check(graph: ExecGraph, spec_checker: Optional[SpecChecker],
                       threshold: float = DEFAULT_THRESHOLD) -> Optional[bool]:
    """Does the final output violate the task specification? Returns None when
    no spec checker was supplied -- 'not evaluated', which is not 'clean'."""
    if spec_checker is None:
        return None
    return spec_checker(graph) >= threshold


def tier2_localize(node_result: CheckResult, edge_result: CheckResult) -> tuple:
    """Fault attributable to a specific agent or tool invocation."""
    entities = ([f"node:{n}" for n in node_result.flagged] +
                [f"edge:{i}" for i in edge_result.flagged])
    return bool(entities), entities


def tier3_path_check(graph: ExecGraph, library: list,
                     semantic_checker=default_semantic_checker) -> list:
    """Subgraphs matching known risk patterns -- the only layer that can see a
    composition whose every element is locally benign."""
    return match_paths(graph, library, semantic_checker)


def analyze(dynamic: ExecGraph, library: list = None, config: SentinelConfig = None,
            static: ExecGraph = None, spec_checker: Optional[SpecChecker] = None,
            node_checker=granite_guardian_stub, edge_checker=llamafirewall_stub,
            semantic_checker=default_semantic_checker) -> Alert:
    """Run the enabled tiers over one execution graph and aggregate one alert."""
    config = config or SentinelConfig()
    library = build_library("seen_signatures") if library is None else library
    started = time.perf_counter()

    alert = Alert()
    classifier_calls = 0

    # The permission half of the node check compares called tools against
    # declared ones, and only the static graph knows what was declared.
    if static is not None:
        apply_static_metadata(dynamic, static)

    node_result = CheckResult()
    edge_result = CheckResult()
    if config.enable_tier2:
        node_result = check_nodes(dynamic, node_checker, config.node_threshold)
        edge_result = check_edges(dynamic, edge_checker, config.edge_threshold)
        classifier_calls += node_result.calls + edge_result.calls

    if static is not None:
        alert.unpermitted_edges = [e.signature() for e in graph_diff(static, dynamic)]

    # Tier 1 -- global.
    tier1 = tier1_output_check(dynamic, spec_checker) if config.enable_tier1 else None
    alert.tier1_evaluated = tier1 is not None
    if tier1:
        alert.tiers_fired.append(1)

    # Tier 3 -- compositional. Run before tier 2 is finalized because a match
    # can trigger the strict re-check that tier 2 then reads.
    matches = []
    if config.enable_tier3:
        matches = tier3_path_check(dynamic, library, semantic_checker)
        alert.matched_paths = matches
        if matches:
            alert.tiers_fired.append(3)

    if matches and config.strict_recheck and config.enable_tier2:
        strict_nodes = check_nodes(dynamic, node_checker, config.strict_threshold)
        strict_edges = check_edges(dynamic, edge_checker, config.strict_threshold)
        classifier_calls += strict_nodes.calls + strict_edges.calls
        recovered = (set(strict_nodes.flagged) - set(node_result.flagged)) or \
                    (set(strict_edges.flagged) - set(edge_result.flagged))
        alert.recovered_by_recheck = bool(recovered)
        node_result, edge_result = strict_nodes, strict_edges

    # Tier 2 -- single-point, using whichever threshold the re-check left in force.
    localized, entities = tier2_localize(node_result, edge_result) if config.enable_tier2 \
        else (False, [])
    alert.flagged_nodes = list(node_result.flagged)
    alert.flagged_edges = list(edge_result.flagged)
    if localized:
        alert.tiers_fired.append(2)

    # A path with nothing flagged under it is exactly the case the original
    # design routes to a human.
    if matches and not localized and config.escalate_unflagged_paths:
        alert.human_review = True

    alert.detected = bool(alert.tiers_fired)
    # Most specific attribution available: a named entity beats a path, and a
    # path beats "the output looked wrong".
    for candidate in (2, 3, 1):
        if candidate in alert.tiers_fired:
            alert.tier = candidate
            break

    alert.root_cause = _root_cause(alert, entities, matches, tier1)
    alert.cost = {
        "nodes": len(dynamic.nodes),
        "edges": len(dynamic.edges),
        "classifier_calls": classifier_calls,
        "path_patterns": len(library),
        "human_review_cost": config.human_review_cost if alert.human_review else 0.0,
        "seconds": time.perf_counter() - started,
    }
    return alert


def _root_cause(alert: Alert, entities: list, matches: list, tier1: Optional[bool]) -> str:
    if not alert.detected:
        reason = "no tier fired"
        if not alert.tier1_evaluated:
            reason += " (tier 1 not evaluated: no task spec supplied)"
        return reason
    parts = []
    if 2 in alert.tiers_fired:
        recovered = " (recovered only by the strict re-check)" if alert.recovered_by_recheck else ""
        parts.append(f"localized to {', '.join(entities)}{recovered}")
    if 3 in alert.tiers_fired:
        ids = ", ".join(m.pattern_id for m in matches)
        review = ", escalated to human review" if alert.human_review else ""
        parts.append(f"attack path(s) {ids}{review}")
    if tier1:
        parts.append("final output violates the task specification")
    return "; ".join(parts)
