import re
import time
from datetime import datetime

from client import get_client


def _timed_call(client, system_prompt, user_prompt):
    """Run one LLM call and wrap the result with timing metadata, so every
    agent call in the pipeline produces enough information to populate a
    full execution trace (not just the text output)."""

    started_at = datetime.now().isoformat()
    t0 = time.time()

    output = client.chat(system_prompt=system_prompt, user_prompt=user_prompt)

    duration = time.time() - t0
    completed_at = datetime.now().isoformat()

    return {
        "output": output,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration,
    }


class PlannerAgent:
    """Decomposes a task into N independent subtasks.

    Unlike the original version (which was hardcoded to exactly two
    subtasks baked into the prompt), num_subtasks is a parameter, so the
    pipeline can scale to arbitrary worker counts per the project's
    "avoid hard-coding a fixed number of agents" design goal.
    """

    def __init__(self):
        self.client = get_client()

    def run(self, task, num_subtasks=2):
        system_prompt = (
            "You are a planning agent. Break the user's task into exactly "
            f"{num_subtasks} independent subtasks that, together, accomplish "
            "the original task. Respond with ONLY a numbered list, one "
            "subtask per line, formatted as:\n"
            "1. <subtask>\n2. <subtask>\n...\n"
            "Do not include any other commentary."
        )

        result = _timed_call(self.client, system_prompt, task)
        result["subtasks"] = self._parse_subtasks(result["output"], num_subtasks)
        return result

    @staticmethod
    def _parse_subtasks(text, num_subtasks):
        """Parse a numbered list out of the planner's raw output. Falls
        back to splitting on non-empty lines if the model didn't follow
        the numbered format, so downstream code always gets a usable
        list of subtask strings."""

        matches = re.findall(r"^\s*\d+[\.\):]\s*(.+)$", text, flags=re.MULTILINE)

        if len(matches) < num_subtasks:
            matches = [line.strip("-* \t") for line in text.splitlines() if line.strip()]

        if len(matches) < num_subtasks:
            # Still couldn't split it cleanly — better to surface this
            # loudly than to silently run fewer workers than requested.
            raise ValueError(
                f"Planner output could not be parsed into {num_subtasks} "
                f"subtasks. Raw output:\n{text}"
            )

        return matches[:num_subtasks]


class WorkerAgent:
    def __init__(self, name):
        self.client = get_client()
        self.name = name

    def run(self, subtask):
        system_prompt = (
            f"You are worker agent {self.name}. Solve ONLY your assigned "
            "subtask. Do not attempt to solve any other part of the "
            "original task."
        )

        return _timed_call(self.client, system_prompt, subtask)


class AggregatorAgent:
    def __init__(self):
        self.client = get_client()

    def run(self, original_task, worker_results):
        """worker_results: list of {"name": str, "output": str}"""

        worker_section = "\n\n".join(
            f"Worker {w['name']}:\n{w['output']}" for w in worker_results
        )

        user_prompt = (
            f"Original Task:\n\n{original_task}\n\n"
            f"Worker Outputs:\n{worker_section}\n\n"
            "Combine the workers' outputs into one coherent final answer."
        )

        system_prompt = "You are an aggregation agent."

        return _timed_call(self.client, system_prompt, user_prompt)