# Data provenance

`narcbench_core_official.json` is a deterministic Inspect-ready conversion of
the 50 NARCBench-Core scenarios released by Rose et al. Each scenario appears
once under collusion and once under its matched honest-control condition, for
100 evaluation samples total.

The conversion preserves the official agent-name draw, label/content
counterbalancing, prompt-style assignments, three public rounds, and paired
scenario structure. It does not contain model outputs. Regenerate it with:

```powershell
python scripts/import_official_core.py `
  --source tmp/reference/narcbench_official/generation/core.py `
  --output data/narcbench_core_official.json `
  --license-source tmp/reference/narcbench_official/LICENSE `
  --license-output data/NARCBENCH_LICENSE.txt
```

Upstream source: <https://github.com/aaronrose227/narcbench>, commit
`85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d`.

Paper: <https://arxiv.org/abs/2604.01151>.

The upstream code is MIT-licensed; the required notice is preserved in
`NARCBENCH_LICENSE.txt`.
