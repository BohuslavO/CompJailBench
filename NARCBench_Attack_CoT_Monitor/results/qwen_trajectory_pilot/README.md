# Qwen trajectory pilot

This pilot contains one attack/control pair each for AgentHarm (`1-1`) and Semantic Intent Fragmentation (`sif-fin-m1`). All four trajectories were generated on two Kaggle T4 GPUs with `Qwen/Qwen3-32B-AWQ` at revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, starting from repository commit `bdb6301c1cdd1943af1b786a7f472f8b112933c2`. Each family's manifest records its generation settings. AgentHarm requests `<output>` before `<reasoning>` in both conditions. SIF records planned tools in this pilot.

Each family directory contains `standard_trajectories.jsonl` for downstream processing, the individual JSON trajectories, and the corresponding activation files. The `.npz` files contain finite vectors of dimension 5120 from layer 28, extracted by replaying the saved response through the same model. Resolve each activation reference relative to the family's `activations/` directory.

The compatibility JSONL files record adapter checks. These runs did not compute attack success, defense scores, or AUROC. The pilot covers two pairs and is intended to verify the generation and export pipeline.
