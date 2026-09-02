
import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler






MODEL_LAYERS = {
    "Qwen/Qwen3-32B-AWQ": (26, 30),
    "meta-llama/Llama-3.1-70B-Instruct-AWQ-INT4": (32, 37),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": (26, 30),
    "openai/gpt-oss-20b": (10, 14),
}


def model_short_name(model_id: str) -> str:
    """Mirrors NARCBench's config.py naming so --out defaults line up
    with their data/activations/<model_short>/ directory convention."""
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def default_layer_for_model(model_id: str) -> int:
    """Middle of NARCBench's recommended range for this model. Falls
    back to 28 (Qwen3-32B's range) with a warning if the model isn't
    one of the 4 pre-configured ones -- you should pass --layer
    explicitly in that case, same as NARCBench itself requires."""
    if model_id in MODEL_LAYERS:
        lo, hi = MODEL_LAYERS[model_id]
        return (lo + hi) // 2
    print(f"WARNING: no known layer range for {model_id!r}. "
          f"Defaulting to 28 (Qwen3-32B's range) -- pass --layer explicitly instead.")
    return 28


def _parse_run_id(run_id: str):
    parts = run_id.split("__")
    return parts[0], (parts[1] if len(parts) >= 2 else run_id)


def load_data(directory: Path):
    npz_path = directory / "activations_gen.npz"
    meta_path = directory / "metadata_gen.json"
    if not npz_path.exists():
        raise SystemExit(
            f"Activations not found at {npz_path}. Did you run "
            f"`huggingface-cli download aaronrose227/narcbench --repo-type dataset --local-dir data`?"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return meta, np.load(npz_path)


def get_layer(npz, n, layer):
    key = f"layer_{layer}"
    if key not in npz.files:
        available = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
        raise SystemExit(
            f"Layer {layer} not extracted. Available: {available[0]}-{available[-1]}."
        )
    return npz[key][:n].astype(np.float32)


def train_deception_probe(meta, npz, layer, phase="discussion"):
    """Fit the deception direction on all of NARCBench-Core.

    Identical to NARCBench's train_deception_probe, except we don't exclude a
    held-out domain -- that's only needed for their leave-domain-out CV table,
    not for producing one probe to apply elsewhere.
    """
    n = len(meta)
    acts = get_layer(npz, n, layer)

    col_idx, hon_idx = [], []
    for i, m in enumerate(meta):
        if m["mode"] != "collusion" or m["channel"] != "public":
            continue
        if phase == "discussion" and m["phase"] != "discussion":
            continue
        if m["role"] == "colluder":
            col_idx.append(i)
        elif m["role"] == "honest":
            hon_idx.append(i)

    nc = min(len(col_idx), len(hon_idx))
    if nc == 0:
        raise SystemExit("No colluder/honest samples found -- check --layer and dataset path.")
    rng = np.random.RandomState(42)
    hon_idx = list(rng.choice(hon_idx, nc, replace=False))
    col_idx = col_idx[:nc]

    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([acts[hon_idx], acts[col_idx]]))
    y = np.array([0] * nc + [1] * nc)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(X, y)
    w = probe.coef_[0]
    w = w / np.linalg.norm(w)
    return scaler, w


def build_groups(meta, npz, layer, phase="discussion", avg_rounds=True):
    """Group per-sample activations into one vector per agent per run.

    Unlike NARCBench's build_groups, this does NOT filter to exactly 4 agents
    -- kept general since it's reused for training-set diagnostics here, and
    CompJailBench trajectories may differ in size.
    """
    n = len(meta)
    acts = get_layer(npz, n, layer)

    accum = defaultdict(lambda: {"vecs": [], "meta": None})
    for i, m in enumerate(meta):
        if phase == "discussion" and avg_rounds:
            if m["phase"] != "discussion" or m["channel"] != "public":
                continue
        else:
            if m["phase"] != phase or m["channel"] != "public":
                continue
        accum[(m["run_id"], m["agent_name"])]["vecs"].append(acts[i])
        accum[(m["run_id"], m["agent_name"])]["meta"] = m

    groups = {}
    for (rid, agent_name), data in accum.items():
        m = data["meta"]
        if rid not in groups:
            scenario, _ = _parse_run_id(rid)
            groups[rid] = {"agents": {}, "mode": m["mode"], "roles": {}, "scenario": scenario}
        vec = np.mean(data["vecs"], axis=0) if avg_rounds else data["vecs"][0]
        groups[rid]["agents"][agent_name] = vec
        groups[rid]["roles"][agent_name] = m["role"]

    return groups

def compute_group_scores(groups, rids, scaler, w):
    """Same five quantities as NARCBench's compute_scores, generalised to any
    group size N via top-half/bottom-half instead of hardcoded top-2/bottom-2,
    and max-min instead of index [0] vs [3] for range."""
    out = {"peak_suspicion": [], "deception_split": [], "asymmetry_probe": [],
           "activation_alignment": [], "sorted_concat": [], "group_size": []}

    for r in rids:
        g = groups[r]
        names = sorted(g["agents"])
        raw_vecs = [g["agents"][nm] for nm in names]
        n_agents = len(raw_vecs)
        scaled_vecs = [scaler.transform(v.reshape(1, -1))[0] for v in raw_vecs]

        projs = np.array([w @ v for v in scaled_vecs])
        order = np.argsort(projs)[::-1]
        projs_sorted = projs[order]
        half = n_agents // 2

        out["peak_suspicion"].append(projs_sorted[0])
        out["deception_split"].append(
            float(np.mean(projs_sorted[:half]) - np.mean(projs_sorted[-half:])) if half else 0.0
        )
        out["asymmetry_probe"].append([
            np.var(projs),
            projs_sorted[0] - projs_sorted[-1],
            float(np.mean(projs_sorted[:half]) - np.mean(projs_sorted[-half:])) if half else 0.0,
            np.mean(projs),
        ])
        out["sorted_concat"].append(np.concatenate([scaled_vecs[i] for i in order]))

        norms = [np.linalg.norm(v) + 1e-10 for v in raw_vecs]
        normed = [v / nrm for v, nrm in zip(raw_vecs, norms)]
        pairs = [normed[i] @ normed[j] for i in range(n_agents) for j in range(i + 1, n_agents)]
        out["activation_alignment"].append(-float(np.mean(pairs)) if pairs else 0.0)
        out["group_size"].append(n_agents)

    out["peak_suspicion"] = np.array(out["peak_suspicion"])
    out["deception_split"] = np.array(out["deception_split"])
    out["asymmetry_probe"] = np.array(out["asymmetry_probe"])
    out["activation_alignment"] = np.array(out["activation_alignment"])
    out["sorted_concat"] = np.array(out["sorted_concat"])
    return out


def train_asymmetry_classifier(tr_scores, tr_y):
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
    clf.fit(fs.fit_transform(tr_scores["asymmetry_probe"]), tr_y)
    return fs, clf


def train_concat_classifier(tr_scores, tr_y):
    n_comp = min(50, len(tr_y) - 1, tr_scores["sorted_concat"].shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    fs = StandardScaler()
    X_tr = fs.fit_transform(pca.fit_transform(tr_scores["sorted_concat"]))
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
    clf.fit(X_tr, tr_y)
    return pca, fs, clf


def fit_bundle(activations_dir: Path, layer: int):
    meta, npz = load_data(activations_dir)
    scaler, w = train_deception_probe(meta, npz, layer)

    groups = build_groups(meta, npz, layer)
    groups4 = {rid: g for rid, g in groups.items() if len(g["agents"]) == 4}
    rids = sorted(groups4.keys())
    tr_y = np.array([1 if groups4[r]["mode"] == "collusion" else 0 for r in rids])
    tr_scores = compute_group_scores(groups4, rids, scaler, w)

    asym_scaler, asym_clf = train_asymmetry_classifier(tr_scores, tr_y)
    concat_pca, concat_scaler, concat_clf = train_concat_classifier(tr_scores, tr_y)

    return {
        "layer": layer,
        "scaler": scaler,
        "direction": w,
        "asymmetry_scaler": asym_scaler,
        "asymmetry_clf": asym_clf,
        "concat_pca": concat_pca,
        "concat_scaler": concat_scaler,
        "concat_clf": concat_clf,
        "n_training_runs": len(rids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path, required=True,
                     help="Path to data/activations/<model_short>/core")
    ap.add_argument("--model", default=None,
                     help="HF model ID this activations dir came from, e.g. "
                          "Qwen/Qwen3-32B-AWQ. Used to pick a sensible default "
                          "--layer and --out filename if those aren't given. "
                          "One of the 4 in MODEL_LAYERS, or any other open-weight "
                          "model as long as you also pass --layer explicitly.")
    ap.add_argument("--layer", type=int, default=None,
                     help="Defaults to the middle of --model's known range if omitted.")
    ap.add_argument("--out", type=Path, default=None,
                     help="Defaults to narcbench_probe_<model_short>.pkl if --model is given, else narcbench_probe.pkl.")
    args = ap.parse_args()

    layer = args.layer
    if layer is None:
        layer = default_layer_for_model(args.model) if args.model else 28

    out = args.out
    if out is None:
        out = Path(f"narcbench_probe_{model_short_name(args.model)}.pkl") if args.model else Path("narcbench_probe.pkl")

    bundle = fit_bundle(args.activations, layer)
    bundle["model"] = args.model
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved probe bundle ({bundle['n_training_runs']} training runs, model={args.model}, layer {layer}) to {out}")


if __name__ == "__main__":
    main()
