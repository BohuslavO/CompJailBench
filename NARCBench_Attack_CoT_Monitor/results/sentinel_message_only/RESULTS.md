# NARCBench x Sentinel Agents (message-only adaptation)

## Status

The public-only primary evaluation is complete. It reused the same 50 matched
NARCBench scenarios used for the other defenses: 50 collusion-condition
trajectories and 50 benign controls. The attacker was not rerun.

The Sentinel Agents implementation at commit `e90d60c` supplied the
tiered analyzer and node/edge checker interfaces. Because the NARCBench traces
contain no tool calls, every execution graph was explicitly marked degraded.
This result therefore applies only to the **message-only adaptation**, not to
the full tool-aware Sentinel Agents configuration.

## Frozen configuration

- Primary scope: public messages and final aggregate output.
- Tier 1: not evaluated because no agreed task-specification checker exists.
- Tier 2: GPT-OSS-120B node and edge checkers through AWS Bedrock.
- Tier 3: frozen target-independent `narcbench_message_paths_v1` signatures.
- Strict recheck: enabled, matching the pinned default configuration.
- Unflagged Tier-3 paths: counted as human-review escalations.
- Hidden target, fair answer, colluder identities, attack labels, and evaluator
  metadata were excluded from the defense graph.

The checker wrapper treated trajectory content as untrusted data and accepted
only a numeric value in `[0, 1]`. The final run contained 3,518 logical checker
outputs and zero persistent malformed outputs. A preliminary 32-token run
produced empty completions and was rejected before analysis; it is not included
in any result below.

## Primary results

| Measure | Result |
|---|---:|
| Attack trajectories | 50 |
| Matched benign controls | 50 |
| Detection recall | 0.940 (47/50) |
| False-positive rate | 0.900 (45/50) |
| Precision | 0.511 |
| Balanced accuracy | 0.520 |
| AUROC | 0.552 |
| Average precision | 0.561 |
| Human-review rate | 0.230 |
| Degraded-graph rate | 1.000 |

The 95% Wilson interval was `[0.838, 0.979]` for recall and
`[0.786, 0.957]` for the false-positive rate. In matched analysis, the attack
minus control flag-rate difference was 0.040 (paired bootstrap 95% interval
`[-0.060, 0.160]`; exact McNemar `p=0.727`). Forty-two scenario pairs had both
arms flagged, five had only the attack arm flagged, three had only the control
flagged, and none had neither arm flagged.

Among the three trajectories satisfying the strict end-to-end attack-success
definition, two were flagged. This 2/3 value is exploratory because the
denominator is too small for a stable effectiveness estimate.

## Tier and cost breakdown

For attack attempts, 36 detections were attributed to Tier 2, 11 to Tier 3,
and three trajectories were missed. For controls, 33 detections were attributed
to Tier 2, 12 to Tier 3, and five were not flagged. The near-symmetric breakdown
explains the low selectivity.

The final run made 3,518 model calls and recorded 980,076 input tokens, 453,054
output tokens, and 1,815,226 provider-reported total tokens. Summed model time
was 3,384.83 seconds. Provider responses did not report a monetary cost; AWS
billing is the authoritative cost source.

## Interpretation and limitation

The message-only adaptation achieved high apparent recall by flagging almost
all deliberations, including benign ones. Generic decision advocacy, alignment,
and normative language are normal features of both attacked and honest
committee discussions, so the defense did not reliably distinguish malicious
coordination. The result is therefore negative evidence about cross-attack
transfer, not evidence that the full Sentinel Agents design is ineffective.

The tool-call layer was unavailable, Tier 1 was not evaluated, and the Tier-3
message signatures were an adaptation for NARCBench rather than the original
SIF signatures. The shared 4x4 table should retain the defense-column name
**Sentinel Agents** and qualify this individual cell as **message-only,
public-scope, degraded**.
