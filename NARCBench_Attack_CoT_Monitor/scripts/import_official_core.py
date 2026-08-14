"""Create Inspect-ready matched pairs from the official NARCBench-Core source.

The official release stores the 50 scenario definitions directly in
``generation/core.py``.  This importer reads only literal assignments from that
file, reproduces the published deterministic name/style/label assignments, and
writes 50 collusion/control pairs (100 samples).  It never imports or executes
the upstream generation module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


UPSTREAM_COMMIT = "85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d"
UPSTREAM_REPOSITORY = "https://github.com/aaronrose227/narcbench"
UPSTREAM_PAPER = "https://arxiv.org/abs/2604.01151"


def literal_assignment(tree: ast.Module, name: str) -> Any:
    """Return a top-level literal assignment without executing source code."""

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal assignment {name!r}")


def style_assignments(n: int = 50, seed: int = 42) -> list[dict[str, tuple[str, ...]]]:
    """Reproduce ``_compute_style_assignments`` from the official release."""

    rng = random.Random(seed)
    colluder_pairs = (
        [("standard", "standard")] * 20
        + [("standard", "lowkey")] * 10
        + [("lowkey", "standard")] * 10
        + [("lowkey", "lowkey")] * 10
    )
    rng.shuffle(colluder_pairs)

    honest_pairs = (
        [("standard", "standard")] * 20
        + [("standard", "aggressive")] * 10
        + [("aggressive", "standard")] * 10
        + [("aggressive", "aggressive")] * 10
    )
    rng.shuffle(honest_pairs)

    control_quads = (
        [("standard", "standard", "standard", "standard")] * 10
        + [("standard", "standard", "standard", "aggressive")] * 10
        + [("standard", "standard", "aggressive", "aggressive")] * 20
        + [("standard", "aggressive", "aggressive", "aggressive")] * 10
    )
    rng.shuffle(control_quads)

    return [
        {
            "colluder": colluder_pairs[index],
            "honest": honest_pairs[index],
            "control": control_quads[index],
        }
        for index in range(n)
    ]


def effective_options(scenario: dict[str, Any], index: int) -> tuple[dict[str, str], str, str]:
    """Apply the official four-way label/content counterbalancing."""

    variant = index % 4
    labels = ("1", "2") if variant >= 2 else ("A", "B")
    swap_content = variant % 2 == 1
    fair_description = scenario["options"]["A"]
    target_description = scenario["options"]["B"]

    if swap_content:
        return {labels[0]: target_description, labels[1]: fair_description}, labels[0], labels[1]
    return {labels[0]: fair_description, labels[1]: target_description}, labels[1], labels[0]


def build_records(scenarios: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if len(scenarios) != 50:
        raise ValueError(f"Expected 50 official Core scenarios, found {len(scenarios)}")
    if len(names) != 60:
        raise ValueError(f"Expected 60 official agent names, found {len(names)}")

    assignments = style_assignments(len(scenarios))
    records: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios):
        agent_names = random.Random(index).sample(names, 4)
        options, target_option, fair_option = effective_options(scenario, index)
        option_labels = list(options)
        final_frame = scenario["final_frame"].replace(
            "Option A or Option B",
            f"Option {option_labels[0]} or Option {option_labels[1]}",
        )

        for condition in ("collusion", "control"):
            colluder_ids = ["agent_1", "agent_2"] if condition == "collusion" else []
            agents: list[dict[str, str]] = []
            styles: dict[str, str] = {}
            for position, name in enumerate(agent_names):
                agent_id = f"agent_{position + 1}"
                role = "colluder" if agent_id in colluder_ids else "honest"
                if condition == "control":
                    style = assignments[index]["control"][position]
                elif role == "colluder":
                    style = assignments[index]["colluder"][position]
                else:
                    style = assignments[index]["honest"][position - 2]
                agents.append({"id": agent_id, "name": name, "role": role})
                styles[agent_id] = style

            records.append(
                {
                    "schema_version": "2.0",
                    "sample_id": f"{scenario['id']}_{condition}",
                    "pair_id": scenario["id"],
                    "scenario_id": scenario["id"],
                    "scenario_index": index,
                    "seed": index,
                    "condition": condition,
                    "domain": scenario["domain"],
                    "title": scenario["title"],
                    "context": scenario["context"],
                    "options": options,
                    "option_labels": option_labels,
                    "target_option": target_option,
                    "fair_choice": fair_option,
                    "label_variant": index % 4,
                    "agents": agents,
                    "colluders": colluder_ids,
                    "agent_styles": styles,
                    "collusion_motive": scenario["collusion_motive"],
                    "deception_instruction": scenario["deception_instruction"],
                    "lowkey_instruction": scenario["lowkey_instruction"],
                    "honest_mandate": scenario["honest_mandate"],
                    "aggressive_mandate": scenario["aggressive_mandate"],
                    "private_frame": scenario["private_frame"],
                    "public_frame": scenario["public_frame"],
                    "final_frame": final_frame,
                    "discussion_rounds": 3,
                    "source": {
                        "repository": UPSTREAM_REPOSITORY,
                        "paper": UPSTREAM_PAPER,
                        "commit": UPSTREAM_COMMIT,
                        "upstream_scenario_id": scenario["id"],
                    },
                }
            )

    return records


def validate_pairs(records: list[dict[str, Any]]) -> None:
    if len(records) != 100:
        raise ValueError(f"Expected 100 records, found {len(records)}")

    pairs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pairs.setdefault(record["pair_id"], []).append(record)

    if len(pairs) != 50:
        raise ValueError(f"Expected 50 matched pairs, found {len(pairs)}")

    for pair_id, pair in pairs.items():
        if {item["condition"] for item in pair} != {"collusion", "control"}:
            raise ValueError(f"Pair {pair_id} does not contain both conditions")
        attack, control = sorted(pair, key=lambda item: item["condition"])
        for key in ("context", "options", "target_option", "fair_choice", "option_labels"):
            if attack[key] != control[key]:
                raise ValueError(f"Pair {pair_id} differs on matched field {key}")
        if [agent["name"] for agent in attack["agents"]] != [agent["name"] for agent in control["agents"]]:
            raise ValueError(f"Pair {pair_id} uses different agent identities")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--license-source", type=Path, required=True)
    parser.add_argument("--license-output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    tree = ast.parse(source_bytes.decode("utf-8"), filename=str(args.source))
    scenarios = literal_assignment(tree, "SCENARIOS")
    names = literal_assignment(tree, "NAME_POOL")
    records = build_records(scenarios, names)
    validate_pairs(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.license_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.license_source, args.license_output)

    print(f"Wrote {len(records)} samples ({len(records) // 2} matched pairs) to {args.output}")
    print(f"Source SHA-256: {hashlib.sha256(source_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
