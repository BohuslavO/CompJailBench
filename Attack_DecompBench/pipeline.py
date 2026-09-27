from typing import Callable, TypedDict

from langgraph.graph import StateGraph, END

from task_loader import DeCompTask, load_task
from routing import AgentRole, STRATEGIES
from benign_rewriter import generate_benign_task


class PipelineState(TypedDict):
    task: DeCompTask
    roles: list
    assignment: dict
    turn_index: int
    trajectory: list


def stub_llm(system_prompt: str, user_prompt: str) -> str:
    return f"[stub response for: {user_prompt[:60]}...]"


def worker_node(state: PipelineState, call_llm: Callable[[str, str], str]) -> dict:
    task = state["task"]
    turn = task.turns[state["turn_index"]]
    agent_name = state["assignment"][turn.turn_id]
    role = next(r for r in state["roles"] if r.name == agent_name)

    response = call_llm(role.system_prompt, turn.prompt)

    record = {
        "agent_name": agent_name,
        "turn_id": turn.turn_id,
        "label": turn.label,
        "system_prompt": role.system_prompt,
        "user_prompt": turn.prompt,
        "response": response,
    }
    return {
        "trajectory": state["trajectory"] + [record],
        "turn_index": state["turn_index"] + 1,
    }


def should_continue(state: PipelineState) -> str:
    return "worker" if state["turn_index"] < len(state["task"].turns) else "end"


def build_pipeline(call_llm: Callable[[str, str], str] = stub_llm):
    graph = StateGraph(PipelineState)
    graph.add_node("worker", lambda s: worker_node(s, call_llm))
    graph.set_entry_point("worker")
    graph.add_conditional_edges("worker", should_continue, {"worker": "worker", "end": END})
    return graph.compile()


def run_task(task_dir: str, roles: list, strategy_name: str, call_llm: Callable = stub_llm,
             condition: str = "attack") -> list:
    task = load_task(task_dir)
    if condition == "benign_control":
        task = generate_benign_task(task, call_llm)
    assignment = STRATEGIES[strategy_name](task, roles)

    initial_state: PipelineState = {
        "task": task,
        "roles": roles,
        "assignment": assignment,
        "turn_index": 0,
        "trajectory": [],
    }
    app = build_pipeline(call_llm)
    final_state = app.invoke(initial_state, config={"recursion_limit": 200})
    return final_state["trajectory"]
