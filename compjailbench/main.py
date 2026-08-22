from openai import OpenAI
import json
import os

client = OpenAI()

MODEL = "gpt-4.1-mini"


def call_agent(system_prompt, user_prompt):
    """Call one agent."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


####################################################
# SINGLE AGENT BASELINE
####################################################

def single_agent(task):

    answer = call_agent(
        "You are a helpful assistant.",
        task
    )

    return answer


####################################################
# MULTI AGENT PIPELINE
####################################################

def planner(task):

    return call_agent(
        "You are a planner. Break the task into two subtasks.",
        task
    )


def worker(name, task):

    return call_agent(
        f"You are Worker {name}. Solve ONLY your assigned subtask.",
        task
    )


def aggregator(task, worker1, worker2):

    prompt = f"""
Original Task:

{task}

Worker A:

{worker1}

Worker B:

{worker2}

Combine both workers' outputs into one final answer.
"""

    return call_agent(
        "You are an aggregator.",
        prompt
    )


####################################################
# EXPERIMENT
####################################################

task = """
Summarize the causes of climate change in two paragraphs.
"""

print("Running single-agent baseline...")

single = single_agent(task)

print(single)

print()

print("Running multi-agent pipeline...")

plan = planner(task)

print("Planner Output")
print(plan)

print()

worker_a = worker(
    "A",
    f"""
The planner produced:

{plan}

Complete ONLY the first subtask.
"""
)

worker_b = worker(
    "B",
    f"""
The planner produced:

{plan}

Complete ONLY the second subtask.
"""
)

print("Worker A")
print(worker_a)

print()

print("Worker B")
print(worker_b)

print()

final = aggregator(
    task,
    worker_a,
    worker_b
)

print("Final Answer")
print(final)


####################################################
# SAVE RESULTS
####################################################

os.makedirs("results", exist_ok=True)

with open("results/run.json", "w") as f:

    json.dump(
        {
            "task": task,
            "single_agent": single,
            "planner": plan,
            "worker_a": worker_a,
            "worker_b": worker_b,
            "final": final
        },
        f,
        indent=4
    )

print()
print("Saved to results/run.json")