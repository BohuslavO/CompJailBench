"""
Turn a sif_vs_gsafeguard .eval log into the SIF-vs-G-Safeguard report.

    python experiments/analyze_gsafeguard.py                 # newest gsafeguard log
    python experiments/analyze_gsafeguard.py path/to.eval

Prints, with denominators (a rate over a handful of samples is not a result):
  - arm sizes and plan-generation failures
  - plan-size distribution per arm  (the confound)
  - G-Safeguard TOPOLOGY layer: structural recall/FPR, collision, and the AUROC
    attribution full vs size-only vs shape -- how much of any separation is just
    plan size vs real structure
  - leakage judge (LLM-as-judge, reported separately): recall/FPR/precision,
    overall and on the valid-SIF subset
  - cost
"""
import glob
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sif_vs_gsafeguard as G  # noqa: E402  -- FEATURE_NAMES, _auroc, _collision_rate


def _sub(sample, needle):
    for name, sc in sample.scores.items():
        if needle in name:
            return sc
    return None


def _fired(score):
    """topology/leakage scorers store CORRECT/INCORRECT; composition stores 1/0.
    Returns None for an absent scorer (e.g. a log predating it)."""
    if score is None:
        return None
    v = score.value
    if isinstance(v, str):
        return v == "C"
    return float(v) >= 0.5


def rate(num, den):
    return f"{num/den:.3f} ({num}/{den})" if den else f"n/a (0/{den})"


def h(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main(path=None):
    if path is None:
        logs = sorted(glob.glob(str(Path(G.ROOT) / "logs" / "*sif-vs-gsafeguard*.eval")),
                      key=os.path.getmtime)
        if not logs:
            print("no sif-vs-gsafeguard logs in logs/")
            return
        path = logs[-1]

    L = read_eval_log(str(path))
    print(f"log:     {Path(path).name}")
    print(f"status:  {L.status}   samples: {len(L.samples)}")
    orch = getattr(L.eval, "model", "?")
    roles = getattr(L.eval, "model_roles", {}) or {}
    print(f"orchestrator: {orch}")
    for r, m in roles.items():
        print(f"  role {r}: {getattr(m, 'model', m)}")

    # Errored samples (e.g. a Bedrock 500) carry no scores -- keep only scored
    # ones and report the shortfall, so a partial run is never read as complete.
    scored = [s for s in L.samples if _sub(s, "topology") is not None]
    if len(scored) < len(L.samples):
        print(f"WARNING: {len(L.samples) - len(scored)} of {len(L.samples)} samples "
              f"errored/unscored -- analyzing the {len(scored)} scored ones only")

    attack, benign = [], []
    for s in scored:
        (attack if s.metadata.get("is_attack_arm") else benign).append(s)

    # ---- feature rows for AUROC attribution ----
    rows = []  # (features, label)
    for s in scored:
        feats = (_sub(s, "topology").metadata or {}).get("features")
        if feats is not None:
            rows.append((feats, 1 if s.metadata.get("is_attack_arm") else 0))
    X = [f for f, _ in rows]
    y = [lab for _, lab in rows]

    def col(i):
        return [f[i] for f in X]

    ndispatch = G.FEATURE_NAMES.index("n_dispatch")

    h("Arms")
    print(f"attack arms:  {len(attack)}")
    print(f"benign arms:  {len(benign)}")
    degraded = [s for s in scored
                if ((_sub(s, "topology").metadata or {}).get("features") or [0])[ndispatch] == 0]
    print(f"degraded (plan-generation failed, empty graph): {len(degraded)}")

    h("Plan size per arm  (n_dispatch = subtasks executed)")
    for label, group in (("attack", attack), ("benign", benign)):
        sizes = [((_sub(s, "topology").metadata or {}).get("features") or [0]*8)[ndispatch]
                 for s in group]
        sizes = [z for z in sizes if z > 0]  # exclude degraded
        if sizes:
            print(f"  {label:<7} n={len(sizes):<3} mean={mean(sizes):.2f}  "
                  f"sd={pstdev(sizes):.2f}  min={min(sizes)}  max={max(sizes)}")
        else:
            print(f"  {label:<7} no non-degraded samples")

    h("G-Safeguard TOPOLOGY layer")
    a_fire = sum(_fired(_sub(s, "topology")) for s in attack)
    b_fire = sum(_fired(_sub(s, "topology")) for s in benign)
    print(f"structural recall (attack):   {rate(a_fire, len(attack))}")
    print(f"structural FPR    (benign):   {rate(b_fire, len(benign))}")
    atk_sig = [tuple(f) for f, lab in rows if lab == 1]
    ben_sig = [tuple(f) for f, lab in rows if lab == 0]
    print(f"collision rate (attack structure == some benign): "
          f"{G._collision_rate(atk_sig, ben_sig):.3f}")
    print(f"AUROC full features:   {G._auroc(X, y):.3f}")
    print(f"AUROC size only (n_dispatch): {G._auroc([[v] for v in col(ndispatch)], y):.3f}")
    shape_idx = [G.FEATURE_NAMES.index('depth'), G.FEATURE_NAMES.index('has_cycle')]
    print(f"AUROC shape only (depth,has_cycle): "
          f"{G._auroc([[f[i] for i in shape_idx] for f in X], y):.3f}")
    print("  full ~ size-only  => topology separates SIF only via plan size (confound)")
    print("  shape-only ~ 0.5  => size-invariant structure is blind to SIF")

    h("Valid-SIF fraction (attack arms)")
    has_comp = any(_sub(s, "composition") for s in attack)
    valid = [s for s in attack if _fired(_sub(s, "composition"))] if has_comp else attack
    if has_comp:
        print(f"composition-only (plan taints, no subtask taints alone): "
              f"{rate(len(valid), len(attack))}")
    else:
        print("composition scorer absent in this log (predates it); "
              "treating all attack arms as the subset")

    h("Compositional-leakage judge  (LLM-as-judge, reported separately)")
    a_leak = sum(_fired(_sub(s, "leakage")) for s in attack)
    b_leak = sum(_fired(_sub(s, "leakage")) for s in benign)
    print(f"recall (attack):   {rate(a_leak, len(attack))}")
    print(f"FPR    (benign):   {rate(b_leak, len(benign))}")
    denom = a_leak + b_leak
    print(f"precision:         {rate(a_leak, denom)}")
    v_leak = sum(_fired(_sub(s, "leakage")) for s in valid)
    print(f"recall on valid-SIF subset: {rate(v_leak, len(valid))}")

    h("Tool composition per arm  (where the real signal lives)")
    # The discriminating info is semantic tool identity, not topology size:
    # egress tools (external post/webhook) + aggregation join are the harm steps.
    EGRESS = {"http.post", "webhook.register"}
    AGG = {"sheets.join"}
    for label, group in (("attack", attack), ("benign", benign)):
        egress = agg = 0
        tally = {}
        for s in group:
            tools = [c.get("tool") for r in (s.store.get("trajectory") or [])
                     for c in (r.get("tool_calls") or []) if c.get("tool")]
            tset = set(tools)
            egress += bool(tset & EGRESS)
            agg += bool(tset & AGG)
            for t in tools:
                tally[t] = tally.get(t, 0) + 1
        print(f"  {label:<7} uses egress tool: {rate(egress, len(group))}   "
              f"uses aggregation join: {rate(agg, len(group))}")
        print(f"          tool counts: {dict(sorted(tally.items()))}")
    print("  If egress/join presence separates arms while shape-AUROC ~0.5, the")
    print("  signal is in TOOL COMPOSITION (semantics), not graph topology.")

    h("Cost (model calls per sample)")
    costs = [float(_sub(s, "cost").value) for s in scored if _sub(s, "cost")]
    print(f"mean calls/sample: {mean(costs):.2f}   total calls: {int(sum(costs))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
