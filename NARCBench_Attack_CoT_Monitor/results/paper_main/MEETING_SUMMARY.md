# NARCBench attack x observable-reasoning monitor: meeting handoff

## Status

The NARCBench x CoT-monitor and G-Safeguard-inspired cells are published on
Beibarys's branch. The Sentinel Agents message-only cell is complete locally
and awaiting review before publication.

- 50 matched NARCBench-Core scenarios
- 100 trajectories: 50 collusion and 50 honest control
- 1,700 attacker-model calls
- 300 post-hoc monitor-model calls
- Three monitor scopes: public action, observable rationale, and both combined
- 100 valid scores per scope; zero invalid outputs in the final log
- Attacker and monitor: `openai/bedrock/gpt-oss-120b` in separate Inspect roles
- Inspect AI 0.3.251
- 4,138,714 total tokens, including 1,118,159 monitor-role tokens
- Preliminary human audit complete for 10/50 attack trajectories (one reviewer)
- Portable StandardTrajectory JSONL export complete (100 trajectories)
- Team matrix export complete (600 long-format rows)
- Cross-attack `general_v2` defense interface implemented and locally validated

## Frozen results

| Measure | Result |
|---|---:|
| Valid collusion / target adherence | 0.280 (95% CI 0.175--0.417) |
| End-to-end attack success | 0.060 (95% CI 0.021--0.162) |
| Target selection under collusion | 0.100 |
| Target selection under control | 0.020 |
| Paired uplift | 0.080 (95% CI -0.020--0.180) |
| Exact McNemar p-value | 0.219 |

| Monitor scope | AUROC | AUPRC | Recall | FPR | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Public action only | 0.427 | 0.454 | 0.460 | 0.540 | 0.460 |
| Observable rationale only | 0.352 | 0.420 | 0.400 | 0.620 | 0.390 |
| Public action + rationale (primary) | 0.316 | 0.416 | 0.300 | 0.520 | 0.390 |

## Interpretation

The combined action+rationale monitor did not outperform action-only. This is
a negative result, not a pipeline failure. Attack-condition agents frequently
refused collusion instructions, while honest agents sometimes produced highly
similar arguments and unanimous votes that the monitor treated as suspicious.
The experiment therefore provides no evidence that elicited rationales improve
detection under this model and protocol.

In the independent 10-trajectory human subset, 3/10 runs satisfied both
automatic target adherence and semantic evidence of coordination. Seven of ten
were rated as containing refusal or noncompliance. These counts are preliminary
because only one reviewer labeled a fixed 20% subset.

## Required caveats

- The same underlying checkpoint was used for attacker and monitor calls,
  although the roles, prompts, temperatures, and contexts were separated.
- The run recorded 152 non-JSON attacker responses, 38 collusion abstentions,
  and 5 control abstentions. They remain in the primary end-to-end analysis.
- Elicited rationales are observable post-hoc explanations, not guaranteed
  faithful hidden chain of thought.
- The preliminary semantic audit covers 10/50 trajectories with one reviewer.
  No inter-rater agreement or Cohen's kappa is available, and the result must
  not be described as a complete human audit.
- Activation-probing compatibility with Slava's defense is unresolved because
  ordinary Bedrock text trajectories do not contain hidden activations.
- The completed NARCBench score used the narrower `narcbench_v1` monitor prompt.
  A final comparable 4x4 defense column must rescore saved trajectories with
  the attack-agnostic `general_v2` prompt; this requires monitor calls but no
  attacker regeneration.

## Files to integrate after team review

- `paper/methodology_narcbench_cot.tex`
- `paper/narcbench_refs.bib`
- `paper/manual_audit_protocol.md`
- `results/paper_main/attack_summary.csv`
- `results/paper_main/monitor_summary.csv`
- `results/paper_main/paper_results.md`
- `results/paper_main/audit_labels_r1.csv`
- `results/paper_main/audit_summary_r1.md`
- `results/paper_main/team_matrix_rows.csv`
- `results/paper_main/standard_trajectories.jsonl`
- Relevant `src`, `tasks`, `scripts`, `tests`, config, and official-data
  provenance files

Do not upload `.env`, `.venv`, temporary traces, diagnostic rescore logs, or
credentials.

## Additional completed attack × defense cell

The same 100 saved trajectories were scored by a **G-Safeguard-inspired
compositional graph monitor** without rerunning the attacker. This is a
detection-only adaptation of Atharv's `GraphMonitor` from commit `c5eea2a`; it
is not the published trained GNN and performs no topological intervention.

- Condition-label AUROC: 0.714.
- Fixed-threshold recall: 3/50 attack attempts (0.060).
- Matched-control false-positive rate: 0/50 (0).
- Strict successful attacks captured: 2/3, reported as exploratory because
  only three attacks achieved the end-to-end success definition.
- Final-run overhead: 101 monitor calls and 215,154 provider-reported tokens,
  including one truncated response that was retried.

The matrix-ready rows are in
`results/g_safeguard_inspired/team_rows.csv`; paper-ready details are in
`results/g_safeguard_inspired/RESULTS.md` and
`paper/methodology_g_safeguard_inspired.tex`. The cell must be named
**G-Safeguard-inspired graph monitor**, not G-Safeguard.

## Additional completed Sentinel Agents cell

Brad Wong confirmed that NARCBench may use Sentinel Agents in degraded
message-only mode, with a separate message-based Tier-3 library because his
default seen signatures are specific to SIF. The public-only primary condition
is now complete over the same 50 attack trajectories and 50 controls.

- Display name for this cell: **Sentinel Agents (message-only adaptation)**.
- Recall: 47/50 (0.940; 95% CI 0.838--0.979).
- False-positive rate: 45/50 (0.900; 95% CI 0.786--0.957).
- Precision: 0.511; balanced accuracy: 0.520; AUROC: 0.552.
- Paired flag-rate difference: 0.040 (95% CI -0.060--0.160; McNemar
  `p=0.727`).
- Strict successful attacks captured: 2/3 (exploratory).
- Every graph was degraded because NARCBench recorded no tool calls; Tier 1
  was not evaluated.
- Final overhead: 3,518 valid checker calls and 1,815,226
  provider-reported tokens.

The high recall is not useful in isolation because the defense also flagged
almost every benign deliberation. This is a negative cross-attack-transfer
result, not a failure of Brad's full tool-aware Sentinel configuration. The
matrix-ready rows are in `results/sentinel_message_only/team_rows.csv`, with
details in `results/sentinel_message_only/RESULTS.md` and
`paper/methodology_sentinel_message_only.tex`.
