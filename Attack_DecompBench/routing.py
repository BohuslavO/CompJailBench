import random
from dataclasses import dataclass, field

from task_loader import DeCompTask


@dataclass
class AgentRole:
    name: str
    system_prompt: str
    specialties: list = field(default_factory=list)


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
        if turn.service_hint:
            for role in roles:
                if turn.service_hint in role.specialties:
                    matched = role.name
                    break
        assignment[turn.turn_id] = matched
    return assignment


STRATEGIES = {
    "random_baseline": random_baseline,
    "least_informed_routing": least_informed_routing,
    "role_override_routing": role_override_routing,
}
