# CoT/observable-rationale monitor column status

All four attack rows have a reported result. Prompt profiles and evidence scopes
are cell-specific because the retained trajectories do not expose the same
fields for every attack; these qualifications are recorded in the matrix and
paper rather than treated as missing work.

| Attack row | Reported result | Profile and evidence scope |
|---|---|---|
| NARCBench | AUROC 0.352 on 50 attack and 50 matched control trajectories | Fixed `narcbench_v1`; observable-rationale-only result reported in the paper |
| AgentHarm | AUROC 0.278 on 24 matched pairs | `general_v2`; observable-rationale-only result reported in the paper |
| Semantic Intent Fragmentation | Delta 0.00 | Action/message-only; no separate rationale field or retained per-sample scores for AUROC recovery |
| DeCompBench | AUROC 0.519 on 45 matched pairs | `general_v2`; public action/message-only because no separate rationale field was recorded |

## Interpretation constraints

- "CoT" means evaluator-visible, model-generated rationale, never hidden model
  reasoning or a guaranteed faithful account of computation.
- Results from `narcbench_v1` and `general_v2` are not pooled or presented as a
  common-prompt comparison.
- Missing rationale evidence is never silently imputed. Action/message-only
  cells are labeled with that scope explicitly.
- The fixed threshold is `0.5`; the paper reports threshold-free AUROC where
  retained per-sample scores make it available, and recall-minus-FPR otherwise.
