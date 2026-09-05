from datetime import datetime

from agents import PlannerAgent, WorkerAgent, AggregatorAgent
from monitors import DummyMonitor


class CompJailBench:
    """Runs the planner -> workers -> aggregator pipeline for one sample
    and produces a full execution trace (not just the final output).

    num_workers is configurable so the topology isn't hardcoded to two
    workers named "A" and "B" — this generalizes to any worker count,
    with worker names generated as A, B, C, ... (or W1, W2, ... beyond 26).
    """

    def __init__(self, num_workers=2, monitor=None):
        self.num_workers = num_workers
        self.planner = PlannerAgent()
        self.workers = [
            WorkerAgent(name=self._worker_name(i)) for i in range(num_workers)
        ]
        self.aggregator = AggregatorAgent()

        # Safety monitor is injected rather than hardcoded, so it can be
        # swapped for AgentHarmMonitor / AgentToAgentMonitor without touching
        # the pipeline itself.
        self.monitor = monitor if monitor is not None else DummyMonitor()

    @staticmethod
    def _worker_name(i):
        if i < 26:
            return chr(ord("A") + i)
        return f"W{i + 1}"

    def run(self, sample):
        task = sample["task"]
        started_at = datetime.now().isoformat()

        nodes = []
        edges = []

        ##################################################
        # Planner
        ##################################################

        plan = self.planner.run(task, num_subtasks=self.num_workers)

        nodes.append({
            "id": "planner",
            "type": "planner",
            "name": "planner",
            "input": task,
            "output": plan["output"],
            "started_at": plan["started_at"],
            "completed_at": plan["completed_at"],
            "duration_seconds": plan["duration_seconds"],
        })
        edges.append({"source": "user", "target": "planner", "type": "task_assignment"})

        ##################################################
        # Workers
        ##################################################

        worker_results = []

        for worker, subtask in zip(self.workers, plan["subtasks"]):
            result = worker.run(subtask)

            node_id = f"worker_{worker.name}"
            nodes.append({
                "id": node_id,
                "type": "worker",
                "name": worker.name,
                "input": subtask,
                "output": result["output"],
                "started_at": result["started_at"],
                "completed_at": result["completed_at"],
                "duration_seconds": result["duration_seconds"],
            })
            edges.append({"source": "planner", "target": node_id, "type": "subtask_delegation"})

            worker_results.append({"name": worker.name, "output": result["output"]})

        ##################################################
        # Aggregator
        ##################################################

        agg = self.aggregator.run(task, worker_results)

        nodes.append({
            "id": "aggregator",
            "type": "aggregator",
            "name": "aggregator",
            "input": worker_results,
            "output": agg["output"],
            "started_at": agg["started_at"],
            "completed_at": agg["completed_at"],
            "duration_seconds": agg["duration_seconds"],
        })

        for worker in self.workers:
            edges.append({
                "source": f"worker_{worker.name}",
                "target": "aggregator",
                "type": "output_submission",
            })

        edges.append({"source": "aggregator", "target": "final_output", "type": "final_response"})

        completed_at = datetime.now().isoformat()

        ##################################################
        # Build execution trace
        ##################################################

        execution_trace = {
            "sample_id": sample["id"],
            "sample_name": sample.get("name"),
            "task": task,
            "started_at": started_at,
            "completed_at": completed_at,
            "nodes": nodes,
            "edges": edges,
            "final_output": agg["output"],
        }

        ##################################################
        # Run safety monitor
        ##################################################

        monitor_result = self.monitor.analyze(execution_trace)

        return {
            "execution_trace": execution_trace,
            "monitor": monitor_result,
        }