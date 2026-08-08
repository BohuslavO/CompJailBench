

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from train_narcbench_probe import fit_bundle
from score_pipeline import LiveScorer, post_hoc_scores, early_auroc

HIDDEN_DIM = 128
LAYER = 28
N_RUNS_PER_MODE = 15
RNG = np.random.RandomState(0)


def make_fake_core_dataset(tmpdir: Path):
    """Fabricate metadata_gen.json + activations_gen.npz shaped like a real
    NARCBench-Core extraction: 4 agents/run, discussion phase (3 rounds) +
    final phase, public channel, collusion vs control, with colluders'
    vectors shifted along one axis so the probe has something to find."""
    meta = []
    vecs = []

    def add_row(run_id, mode, role, phase, round_, agent_name):
        base = RNG.normal(size=HIDDEN_DIM).astype(np.float32)
        if mode == "collusion" and role == "colluder":
            base[0] += 4.0
        meta.append({
            "run_id": run_id, "mode": mode, "role": role, "channel": "public",
            "phase": phase, "round": round_, "agent_name": agent_name,
            "n_tokens": 100, "scenario_domain": "synthetic",
        })
        vecs.append(base)

    agents = ["A1", "A2", "A3", "A4"]
    for mode in ["collusion", "control"]:
        for i in range(N_RUNS_PER_MODE):
            run_id = f"synthetic__run_{i:02d}__{mode}"
            for round_ in [1, 2, 3]:
                for j, agent in enumerate(agents):
                    role = "colluder" if (mode == "collusion" and j < 2) else "honest"
                    add_row(run_id, mode, role, "discussion", round_, agent)
            for j, agent in enumerate(agents):
                role = "colluder" if (mode == "collusion" and j < 2) else "honest"
                add_row(run_id, mode, role, "final", 4, agent)

    acts_dir = tmpdir / "activations" / "synthetic_model" / "core"
    acts_dir.mkdir(parents=True, exist_ok=True)
    (acts_dir / "metadata_gen.json").write_text(json.dumps(meta))
    np.savez_compressed(acts_dir / "activations_gen.npz",
                        **{f"layer_{LAYER}": np.stack(vecs).astype(np.float32)})
    return acts_dir


def main():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        print("1. Fabricating fake NARCBench-Core dataset...")
        acts_dir = make_fake_core_dataset(tmpdir)

        print("2. Fitting probe bundle (train_narcbench_probe.fit_bundle)...")
        bundle = fit_bundle(acts_dir, LAYER)
        print(f"   n_training_runs={bundle['n_training_runs']}, "
              f"direction shape={bundle['direction'].shape}")
        assert bundle["direction"].shape == (HIDDEN_DIM,)
        assert bundle["n_training_runs"] == 2 * N_RUNS_PER_MODE

        print("3. post_hoc_scores on a synthetic 4-agent CompJailBench trajectory...")
        traj4 = [RNG.normal(size=HIDDEN_DIM).astype(np.float32) for _ in range(4)]
        traj4[0][0] += 4.0
        traj4[1][0] += 4.0
        scores4 = post_hoc_scores(traj4, bundle)
        print(f"   {scores4}")
        for k in ["peak_suspicion", "deception_split", "activation_alignment",
                  "asymmetry_probe", "sorted_concat"]:
            assert k in scores4
        assert scores4["asymmetry_probe"] is not None
        assert scores4["sorted_concat"] is not None

        print("4. post_hoc_scores on a synthetic 3-agent trajectory (OOD for asym/concat)...")
        traj3 = [RNG.normal(size=HIDDEN_DIM).astype(np.float32) for _ in range(3)]
        scores3 = post_hoc_scores(traj3, bundle)
        print(f"   {scores3}")
        assert scores3["asymmetry_probe"] is None
        assert scores3["sorted_concat"] is None
        assert scores3["peak_suspicion"] is not None

        print("5. LiveScorer incremental updates...")
        live = LiveScorer(bundle)
        for i, v in enumerate(traj4):
            state = live.add_agent(v)
            print(f"   after agent {i+1}: {state}")
        assert live.current()["n_agents_seen"] == 4

        print("6. early_auroc over a fabricated labeled set...")
        labels = [1] * 10 + [0] * 10
        checkpoint_scores = [RNG.normal(loc=2.0) for _ in range(10)] + \
                             [RNG.normal(loc=0.0) for _ in range(10)]
        auroc = early_auroc(checkpoint_scores, labels)
        print(f"   Early-AUROC (fabricated, separable by construction): {auroc:.3f}")
        assert auroc is not None and auroc > 0.7

        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
