"""
Routing strategies for CompJailBench's distributed-jailbreak attack.

Each strategy takes a DeCompTask (ordered list of Turns) and a list of
agent roles, and returns an assignment: which agent handles which
turn. This is the "Role-based information routing" mechanism itself.

Three strategies:
  - random_baseline: turns assigned uniformly at random, ignoring role
    fit. Control condition -- isolates whether routing *by role* is
    what drives the effect, vs. just splitting the task at all.
  - least_informed_routing: each turn goes to whichever agent has seen
    the fewest turns so far, so no single agent accumulates enough of
    the task to recognize the overall harmful pattern.
  - role_override_routing: each turn goes to the agent whose declared
    specialty matches the turn's capability/service (e.g. a "database"
    role gets postgres.* turns), even when that concentrates more
    context on one agent than least_informed_routing would.
"""
import random
from dataclasses import dataclass, field

from task_loader import DeCompTask


@dataclass
class AgentRole:
    name: str
    system_prompt: str
    specialties: list = field(default_factory=list)  # capability prefixes, e.g. ["postgres"]


def random_baseline(task: DeCompTask, roles: list, seed: int = 0) -> dict:
    rng = random.Random(seed)
    return {turn.turn_id: rng.choice(roles).name for turn in task.turns}


def least_informed_routing(task: DeCompTask, roles: list) -> dict:
    seen_count = {r.name: 0 for r in roles}
    assignment = {}
    for turn in task.turns:
        least_agent = min(seen_count, key=seen_count.get)
        assignment[turn.turn_id] = least_agent
        seen_count[least_agent] += 1
    return assignment


def role_override_routing(task: DeCompTask, roles: list, default: str = None) -> dict:
    default = default or roles[0].name
    assignment = {}
    for turn in task.turns:
        matched = default
        if turn.capability_id:
            prefix = turn.capability_id.split(".")[0]
            for role in roles:
                if prefix in role.specialties:
                    matched = role.name
                    break
        assignment[turn.turn_id] = matched
    return assignment


STRATEGIES = {
    "random_baseline": random_baseline,
    "least_informed_routing": least_informed_routing,
    "role_override_routing": role_override_routing,
}
