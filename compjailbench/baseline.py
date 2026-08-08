from datetime import datetime
import time

from client import get_client


class SingleAgentBaseline:
    """Direct single-agent baseline: one model, one prompt, no decomposition.
    This is the comparison point that lets us claim decomposition changes
    completion/detection rates at all — without it, a high multi-agent
    completion rate proves nothing on its own.
    """

    def __init__(self):
        self.client = get_client()

    def run(self, sample):
        task = sample["task"]

        started_at = datetime.now().isoformat()
        t0 = time.time()

        output = self.client.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt=task,
        )

        duration = time.time() - t0
        completed_at = datetime.now().isoformat()

        execution_trace = {
            "sample_id": sample["id"],
            "sample_name": sample.get("name"),
            "task": task,
            "started_at": started_at,
            "completed_at": completed_at,
            "nodes": [{
                "id": "single_agent",
                "type": "single_agent",
                "name": "single_agent",
                "input": task,
                "output": output,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration,
            }],
            "edges": [],
            "final_output": output,
        }

        return {"execution_trace": execution_trace}