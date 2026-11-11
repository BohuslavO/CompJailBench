# Qwen r0 trajectories and monitor scores

The recovered checkpoint preserves trajectories, activation vectors, replay inputs, metadata, generation manifests, and recorded generation errors. Generation and monitoring use Qwen/Qwen3-32B-AWQ at revision 0499c3ac83fdef8810b907a23894ba91e95eddd8. Activation vectors come from teacher-forced replay at layer 28 with 5120 dimensions.

## Verified coverage

| Family | Saved r0 trajectories | Complete r0 pairs | Retained paired text-monitor scores |
|---|---:|---:|---:|
| AgentHarm | 350 | 174 | 174 pairs, 1740 rows |
| Semantic Intent Fragmentation | 21 | 10 | 10 pairs, 100 rows |

The checkpoint also includes seven AgentHarm r1 trajectories forming three complete pairs. The r1 run is incomplete and excluded from r0 scores.

AgentHarm control generation failed for pair IDs 14-1 and 53-1. The SIF snapshot lacks the control for sif-hr-m4. DeCompBench and NARCBench collusion trajectories are absent from this checkpoint.

AgentHarm scoring covers the smoke pair and batches 01 through 22. Kaggle scored the smoke pair and batches 01 through 13. Modal rescored batches 14 through 22 on an NVIDIA L40S using the same model revision, decoding settings, and scorer. Each Modal archive includes its runtime environment. In a matched smoke test, all 12 numerical scores across AgentHarm and SIF agreed with Kaggle.

## Monitor protocol and limitations

All compatible text monitors receive the same saved trajectory. Monitoring uses deterministic decoding, thinking disabled, and a 1536-token response limit. The three text monitors are CoT with three evidence scopes, the Agent-to-Agent baseline, and Sentinel's public message-only degraded adaptation.

SIF has 20 scored public-action CoT rows and 80 unavailable rows. It lacks final output for Agent-to-Agent and Sentinel, and lacks observable reasoning for two CoT scopes.

AUROCs in the summaries compare assigned attack and benign-control conditions. Independent attack-success labels, success-conditioned metrics, and confidence intervals are pending.

The source-task probe requires documented Qwen training and validation before freezing. Historical probe scores and weights are excluded from these summaries. qwen_source_split.json records the proposed source-task split.

Resumed trajectory generation reseeded the runtime. Its random sequence differs from uninterrupted generation. Generation settings and seeds remain recorded in the original manifests.

## Files and provenance

qwen-overnight-recovered.zip preserves 378 trajectories across the three saved runs, including incomplete pairs. qwen-overnight-audit.json records the checkpoint hash and inventory. Its SHA-256 is 872e20846e810afb72182bd81c83a90cc1481b0bffd562c827d3e13d129a492c.

Score JSONL files preserve per-example scores and qualify raw call IDs with source_archive. Each original score ZIP contains its manifest, numerical scores, raw monitor completions, and executed scorer. The summaries record source ZIP hashes.

scoring_inputs contains exact scoring inputs for smoke and numbered batches. The SIF remaining input is also preserved separately. Activation references retain original runtime paths. Locate the corresponding files in the recovered checkpoint by archive-relative family, repeat, and sample ID.

publication_manifest.json records SHA-256 and byte count for each published data artifact.
