# Standalone AgentHarm prototype and Agent-to-Agent Monitor

Benchmark for testing whether harmful intent can "leak" through a
planner → workers → aggregator multi-agent pipeline, even when no single
agent's output looks harmful in isolation. Scored two ways: AgentHarm-style
final-output grading, and the Agent-to-Agent Monitor.

## Status and scope

This directory preserves the standalone prototype used for the AgentHarm x
Agent-to-Agent Monitor evaluation path. The frozen matrix result is stored in
`../results 4x4/raw/slava__agentharm__agent_to_agent_monitor.csv`; the current
AgentHarm workflow and its other defense cells are documented in
`../AGENTHARM_RUNBOOK.md` and `../Attack_AgentHarm/README.md`.

This prototype remains useful for reproducing the graph-and-judge baseline, but
it is not the canonical entry point for the full 4x4 benchmark. The repository
root README identifies the current component for each attack and defense.

## Setup

```bash
pip install openai python-dotenv networkx google-genai
# (skip inspect-ai / matplotlib / pydantic for now — not used by this run)
```

Copy `.env` and fill in real values. Two provider options, set via
`LLM_PROVIDER` in `.env`:

**Gemini**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-2.0-flash
```
If `GEMINI_MODEL` 404s, list available models for your key and update it:
```python
from google import genai
client = genai.Client(api_key="YOUR_KEY")
for m in client.models.list():
    print(m.name)
```

**Azure OpenAI (if/when available)**
```
LLM_PROVIDER=azure
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_API_KEY=<your-key>
AZURE_API_VERSION=2024-02-15-preview
AZURE_DEPLOYMENT=<your-deployment-name>
```

## Run order

### Optional external reference — official single-agent AgentHarm

AgentHarm has an official implementation inside AISI's `inspect_evals` package —
this uses their actual dataset AND their actual programmatic graders, not an
approximation. This is the number that's citable as "real AgentHarm results."

```bash
pip install inspect-ai inspect-evals google-genai
export GOOGLE_API_KEY=<your gemini key>   # note: GOOGLE_API_KEY, not GEMINI_API_KEY, for Inspect's provider

inspect eval inspect_evals/agentharm --model google/gemini-2.0-flash
```

This runs the standard single-agent, tool-using AgentHarm evaluation, not the
multi-agent attack used in CompJailBench and not a cell in the 4x4 matrix. It
can be used as an external reference point. View the full transcript/scoring
with:

```bash
inspect view
```

If you want to see it against Azure once available:
```bash
inspect eval inspect_evals/agentharm --model azure/<your-deployment-name>
```
(Azure requires `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` — see Inspect's
model provider docs, this may differ slightly from our own `client.py`'s env
var names since that's Inspect's own provider integration, not our code.)

### Standalone multi-agent prototype

```bash
# 1. Sanity check the whole pipeline with zero API calls first
COMPJAILBENCH_MOCK=1 python run_experiment.py --dataset datasets/sample.json --num_workers 2

# 2. Pull real harmful samples from the public AgentHarm dataset
#    (may require accepting terms on HuggingFace / `huggingface-cli login`
#    if the dataset is gated — check this before assuming step 3 will work)
pip install datasets
python datasets/agentharm_loader.py --n 15 --out datasets/agentharm_samples.json

# 3. Real run
python run_experiment.py --dataset datasets/agentharm_samples.json --num_workers 2
```

Output: `results/summary.csv` (per-sample scores) and `results/traces/`
(full execution traces per sample, including every agent's input/output
and timing). Printed summary gives three headline numbers: baseline
harmful-completion rate, multi-agent harmful-completion rate, and
AgentToAgentMonitor's catch rate on the multi-agent successes.

## Architecture

- **`agents.py`** — `PlannerAgent` (decomposes task into N subtasks),
  `WorkerAgent` (solves one subtask), `AggregatorAgent` (combines worker
  outputs). Worker count is a parameter, not hardcoded.
- **`benchmark.py`** — `CompJailBench` runs planner → N workers →
  aggregator and builds a full execution trace with node/edge graph
  structure (needed by the graph monitor) plus timing.
- **`baseline.py`** — `SingleAgentBaseline`, one model/one prompt, no
  decomposition. The comparison point for whether decomposition changes
  anything.
- **`monitors.py`** — `AgentHarmMonitor` (LLM-judge grading of final
  output only — completed/refused), `AgentToAgentMonitor` (structural heuristics
  + LLM-judge "compositional leakage" check comparing individual worker
  outputs against the aggregated final output).
- **`client.py`** — `get_client()` factory returns `MockLLMClient`,
  `GeminiClient`, or `LLMClient` (Azure) based on `.env`. Nothing else in
  the codebase needs to know which backend is active.
- **`run_experiment.py`** — orchestrates baseline vs multi-agent over the
  dataset, scores both, writes CSV + traces + printed summary.
- **`datasets/agentharm_loader.py`** — pulls real samples from the public
  AgentHarm HuggingFace dataset and converts to our sample schema.

## Agent-to-Agent Monitor

`monitors.AgentToAgentMonitor` implements the project-defined Agent-to-Agent Monitor
described in the paper. It is a post-hoc, detection-only baseline with two
components: structural checks over the observable agent-interaction graph and
an LLM judge that compares individual outputs with the final aggregation for
compositional leakage. It does not train a classifier or modify a live run.

This name refers only to the CompJailBench baseline. Result files use the
canonical identifier `agent_to_agent_monitor`, and the evidence scope and model
configuration remain recorded in each result file's `notes` field.

## Known open items

- `AgentToAgentMonitor`'s structural heuristics (fan-out threshold and cycle
  detection) and compositional-leakage prompt are fixed project baselines; they
  were not systematically retuned for every attack family.
- `AgentHarmMonitor` uses an LLM judge as a stand-in for AgentHarm's real
  per-task programmatic graders (which need tool-call logs AgentHarm's own
  harness produces) — good for a fast comparative read, not a drop-in
  replacement.
- Not yet migrated to AISI Inspect (`Task`/`Solver`/`Scorer`) — this is a
  standalone runner. That migration is a deliberate later step, not
  something to retrofit under time pressure.
- `datasets/agentharm_loader.py`'s field mapping (`prompt`, `category`,
  `target_functions`) is based on best-available documentation of the
  AgentHarm schema, not a live-verified pull — if it KeyErrors, print
  `ds.column_names` and `ds[0]` and fix the mapping.
