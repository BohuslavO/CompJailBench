"""
Smoke test for the attack pipeline skeleton. Runs against the REAL
downloaded abuse-report-suppression task (7 turns), all 3 routing
strategies, with the stub LLM (no API key needed). Verifies: the
loader parses real files, every strategy produces a full assignment,
the graph actually terminates and returns one trajectory record per
turn, and each record has the generic shape extract_activations.py
expects on the defense side.
"""
import sys

from task_loader import load_task, list_tasks
from routing import AgentRole, STRATEGIES
from pipeline import run_task





TASKS_ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\decompbench_data\tasks"
TASK_DIR = f"{TASKS_ROOT}/abuse-report-suppression"

ROLES = [
    AgentRole(name="db_specialist", system_prompt="You are a database operations specialist.", specialties=["postgres", "redis"]),
    AgentRole(name="docs_specialist", system_prompt="You are a document management specialist.", specialties=["owncloud", "gitlab", "plane"]),
    AgentRole(name="generalist", system_prompt="You are a general-purpose operations assistant.", specialties=["filesystem", "terminal", "rocketchat", "voip-gateway"]),
]


def main():
    print("=== list_tasks ===")
    slugs = list_tasks(TASKS_ROOT)
    print(f"{len(slugs)} tasks found, e.g. {slugs[:3]}")
    assert len(slugs) > 0

    print("\n=== load_task ===")
    task = load_task(TASK_DIR)
    print(f"slug={task.slug} turns={len(task.turns)} services={task.services} difficulty={task.difficulty}")
    assert task.slug == "abuse-report-suppression"
    assert len(task.turns) == 7

    print("\n=== routing strategies ===")
    for name in STRATEGIES:
        assignment = STRATEGIES[name](task, ROLES)
        assert len(assignment) == len(task.turns), f"{name} didn't assign every turn"
        counts = {}
        for agent in assignment.values():
            counts[agent] = counts.get(agent, 0) + 1
        print(f"{name}: {assignment}")
        print(f"  -> per-agent turn counts: {counts}")

    print("\n=== full pipeline run (stub LLM, all 3 strategies) ===")
    for name in STRATEGIES:
        trajectory = run_task(TASK_DIR, ROLES, name)
        assert len(trajectory) == len(task.turns), f"{name}: expected {len(task.turns)} records, got {len(trajectory)}"
        for rec in trajectory:
            assert set(rec.keys()) >= {"agent_name", "turn_id", "label", "system_prompt", "user_prompt", "response"}
        agents_used = sorted(set(r["agent_name"] for r in trajectory))
        print(f"{name}: {len(trajectory)} records, agents used: {agents_used}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
