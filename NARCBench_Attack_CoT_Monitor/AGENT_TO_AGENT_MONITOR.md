# Agent-to-Agent Monitor

The Agent-to-Agent Monitor is the project's post-hoc, detection-only baseline
for observable multi-agent communication. It combines structural summaries of
the agent-interaction graph with an LLM judge for compositional leakage. It does
not train a classifier or intervene in a live execution.

## Evidence contract

For each `StandardTrajectory`, the adapter exposes only:

- label-free task context and options visible to every agent;
- each agent's latest public message and public-message count;
- public directed communication pairs, message counts, and round indices;
- the final aggregated output.

It excludes private messages, observable rationales, hidden targets, designated
colluder identities, condition labels, and all evaluation metadata. Labels are
joined only after the monitor prediction for analysis.

Expected cycles, bidirectional communication, and high connectivity are graph
statistics—not automatic attack indicators. This avoids the original baseline's
cycle rule flagging every ordinary multi-round NARCBench discussion.

## Frozen evaluation

- Defense ID: `agent_to_agent_monitor`
- Prompt: `compositional_leakage_v1`
- Scope: `public-utterance-graph-plus-final-output`
- Monitor: `openai/bedrock/gpt-oss-120b`
- Temperature: `0`
- Reasoning effort: `low`
- Threshold: `0.5`
- Samples: 50 collusion attempts and 50 matched controls

The scorer checkpoints every completed row. Resume an interrupted evaluation
without repeating valid paid calls:

```powershell
.\.venv\Scripts\python.exe scripts\score_agent_to_agent_monitor.py `
  results\paper_main\standard_trajectories.jsonl `
  --output results\agent_to_agent_monitor\full_scores.jsonl `
  --model openai/bedrock/gpt-oss-120b `
  --max-concurrency 5 `
  --resume
```

Regenerate canonical scores, statistics, and team rows without model calls:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_agent_to_agent_monitor.py `
  results\agent_to_agent_monitor\full_scores.jsonl `
  --trajectories results\paper_main\standard_trajectories.jsonl `
  --canonical results\agent_to_agent_monitor\scores_canonical.jsonl `
  --summary-json results\agent_to_agent_monitor\summary.json `
  --summary-csv results\agent_to_agent_monitor\summary.csv `
  --team-rows results\agent_to_agent_monitor\team_rows.csv `
  --contributor <contributor-id> `
  --run-id narcbench__agent_to_agent_monitor__20260815
```

See `results/agent_to_agent_monitor/RESULTS.md` for the frozen result and
`paper/methodology_agent_to_agent_monitor.tex` for paper-ready prose.
