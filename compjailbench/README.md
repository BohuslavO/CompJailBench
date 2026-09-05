# CompJailBench — Setup & Run Guide

Benchmark for testing whether harmful intent can "leak" through a
planner → workers → aggregator multi-agent pipeline, even when no single
agent's output looks harmful in isolation. Scored two ways: AgentHarm-style
final-output grading, and the Agent-to-Agent Monitor.

## Status

The full pipeline is built and has been validated end-to-end in **mock
mode** (no API calls, deterministic fake responses) — the code path works,
including arbitrary worker counts (tested with 2 and 3 workers). What's
NOT yet validated: behavior against a real model, since the team writing
this doesn't have API access yet. That's the one remaining step.

## Setup

```bash
pip install openai python-dotenv networkx google-genai
# (skip inspect-ai / matplotlib / pydantic for now — not used by this run)
```

Copy `.env` and fill in real values. Two provider options, set via
`LLM_PROVIDER` in `.env`:

**Gemini (recommended — free, no credit card required)**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=<get free at https://aistudio.google.com/app/apikey>
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

### Step 0 — Real AgentHarm via the official implementation (do this first)

AgentHarm has an official implementation inside AISI's `inspect_evals` package —
this uses their actual dataset AND their actual programmatic graders, not an
approximation. This is the number that's citable as "real AgentHarm results."

```bash
pip install inspect-ai inspect-evals google-genai
export GOOGLE_API_KEY=<your gemini key>   # note: GOOGLE_API_KEY, not GEMINI_API_KEY, for Inspect's provider

inspect eval inspect_evals/agentharm --model google/gemini-2.0-flash
```

This runs the standard (single-agent, tool-using) AgentHarm eval — i.e. NOT yet
through our decomposition pipeline. It's the baseline reference point: "how
does this model do on AgentHarm normally." View the full transcript/scoring
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

### Step 1 — Our own multi-agent decomposition pipeline (approximate scorers)

```bash
# 1. Sanity check the whole pipeline with zero API calls first
COMPJAILBENCH_MOCK=1 python run_experiment.py --dataset datasets/mock_samples.json --num_workers 2

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
