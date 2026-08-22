"""Generate a model/layer-specific probe bundle with the released trainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.narcbench_probe import DEFAULT_PROBE_LAYER, DEFAULT_PROBE_MODEL


DEFAULT_TRAINER = REPOSITORY_ROOT / "Defence_NARCBench Probing" / "train_narcbench_probe.py"


def load_trainer(path: Path) -> ModuleType:
    """Load the trainer directly from the sibling defense directory."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NARCBench probe trainer not found: {path}")
    spec = importlib.util.spec_from_file_location("compjailbreak_narcbench_probe_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "fit_bundle", None)):
        raise ImportError(f"{path} does not expose fit_bundle()")
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Training activation directory expected by fit_bundle()",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination .pkl bundle"
    )
    parser.add_argument("--model", default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_PROBE_LAYER)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=DEFAULT_TRAINER,
        help="Path to the unchanged released train_narcbench_probe.py",
    )
    args = parser.parse_args()

    trainer = load_trainer(args.trainer)
    bundle = trainer.fit_bundle(args.activations.resolve(), args.layer)
    bundle["model"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(bundle, handle)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"Saved probe bundle: path={args.output}, model={args.model}, "
        f"layer={args.layer}, training_runs={bundle.get('n_training_runs')}, "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
