"""
Reads every CSV in raw/ (shared schema, see README.md) and builds the canonical
4x4 attack-by-defense matrix. Writes MATRIX.md.

Usage:
  python build_matrix.py
  python build_matrix.py --raw-dir raw --out MATRIX.md
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FIELDS = [
    "contributor",
    "attack_method",
    "defense_method",
    "model",
    "task_id",
    "strategy",
    "condition",
    "metric_name",
    "metric_value",
    "n_samples",
    "run_id",
    "notes",
]

EXPECTED_ATTACK_METHODS = [
    "agentharm",
    "semantic_intent_fragmentation",
    "narcbench_collusion",
    "decompbench_routing",
]
EXPECTED_DEFENSE_METHODS = [
    "g_safeguard_inspired_graph_monitor",
    "sentinel_agents_message_only",
    "cot_observable_reasoning_monitor",
    "narcbench_probing",
]
DISPLAY_NAMES = {
    "agentharm": "AgentHarm",
    "semantic_intent_fragmentation": "Semantic Intent Fragmentation",
    "narcbench_collusion": "NARCBench",
    "decompbench_routing": "DeCompBench",
    "g_safeguard_inspired_graph_monitor": "G-Safeguard*",
    "sentinel_agents_message_only": "Sentinel Agents*",
    "cot_observable_reasoning_monitor": "CoT monitor*",
    "narcbench_probing": "NARCBench Probing",
}


def load_raw(raw_dir):
    rows = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                missing = [c for c in FIELDS if c not in r]
                if missing:
                    print(f"WARNING: {path.name} missing columns {missing}, skipping file")
                    break
                rows.append(r)
    return rows


def mean(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def build_pivots(rows):
    """
    cells[(attack_method, defense_method)][metric_name][condition] -> list of values
    """
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        key = (r["attack_method"], r["defense_method"])
        cells[key][r["metric_name"]][r["condition"]].append(r["metric_value"])
    return cells


def ordered_methods(found, expected):
    """Keep the agreed 4x4 axes first, then preserve any unrecognized methods."""
    extras = sorted(set(found) - set(expected))
    return list(expected) + extras


def render_coverage_table(cells, attack_methods, defense_methods):
    lines = ["## 4x4 coverage", ""]
    header = "| attack \\ defense | " + " | ".join(
        DISPLAY_NAMES.get(method, method) for method in defense_methods
    ) + " |"
    sep = "|---" * (len(defense_methods) + 1) + "|"
    lines += [header, sep]
    for attack in attack_methods:
        row = [DISPLAY_NAMES.get(attack, attack)]
        for defense in defense_methods:
            data = cells.get((attack, defense), {})
            row.append(f"data ({len(data)} metric(s))" if data else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "`—` means no result has been submitted for that attack-defense cell.",
        "The asterisk marks a cell-specific adaptation: the G-Safeguard cell is an inspired detection-only baseline; Sentinel Agents is message-only/public/degraded; and the current CoT-monitor result uses the NARCBench-specific `narcbench_v1` prompt.",
        "",
    ]
    return "\n".join(lines)


def render_pivot_table(cells, attack_methods, defense_methods, metric_name, label):
    lines = [f"### {label}", ""]
    header = "| attack \\ defense | " + " | ".join(
        DISPLAY_NAMES.get(method, method) for method in defense_methods
    ) + " |"
    sep = "|---" * (len(defense_methods) + 1) + "|"
    lines += [header, sep]
    for am in attack_methods:
        row = [DISPLAY_NAMES.get(am, am)]
        for dm in defense_methods:
            data = cells.get((am, dm), {}).get(metric_name)
            if not data:
                row.append("—")
                continue
            m_attack = mean(data.get("attack", []))
            m_benign = mean(data.get("benign_control", []))
            if m_attack is None and m_benign is None:
                row.append("—")
            else:
                gap = (
                    m_attack - m_benign
                    if m_attack is not None and m_benign is not None
                    else None
                )
                cell_str = f"atk {m_attack:.2f}" if m_attack is not None else "atk —"
                cell_str += f" / ben {m_benign:.2f}" if m_benign is not None else " / ben —"
                if gap is not None:
                    cell_str += f" (Δ{gap:+.2f})"
                row.append(cell_str)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=BASE_DIR / "raw")
    ap.add_argument("--out", default=BASE_DIR / "MATRIX.md")
    args = ap.parse_args()

    rows = load_raw(args.raw_dir)
    if not rows:
        print(f"No valid rows found in {args.raw_dir}/ -- nothing to build.")
        return

    found_attacks = {r["attack_method"] for r in rows}
    found_defenses = {r["defense_method"] for r in rows}
    attack_methods = ordered_methods(found_attacks, EXPECTED_ATTACK_METHODS)
    defense_methods = ordered_methods(found_defenses, EXPECTED_DEFENSE_METHODS)
    contributors = sorted({r["contributor"] for r in rows})
    metric_names = sorted({r["metric_name"] for r in rows})

    print(f"Loaded {len(rows)} rows from {args.raw_dir}/")
    print(
        "Attack methods seen "
        f"({len(found_attacks & set(EXPECTED_ATTACK_METHODS))}/4 expected): "
        f"{sorted(found_attacks)}"
    )
    print(
        "Defense methods seen "
        f"({len(found_defenses & set(EXPECTED_DEFENSE_METHODS))}/4 expected): "
        f"{sorted(found_defenses)}"
    )
    print(f"Contributors: {contributors}")

    cells = build_pivots(rows)

    out_lines = [
        "# CompJailBench 4x4 Results Matrix",
        "",
        f"Auto-generated by `build_matrix.py` from {len(rows)} rows across "
        f"{len(contributors)} contributor(s).",
        f"Attack methods present: "
        f"{len(found_attacks & set(EXPECTED_ATTACK_METHODS))}/4. "
        f"Defense methods present: "
        f"{len(found_defenses & set(EXPECTED_DEFENSE_METHODS))}/4.",
        "",
        "Each cell shows mean(metric | attack) / mean(metric | benign_control), and the gap (Δ) between them.",
        "For binary flag metrics, the attack mean is detection recall and the benign-control mean is the false-positive rate.",
        "Gap is a placeholder signal, not a validated AUROC -- see README.md.",
        "",
        render_coverage_table(cells, attack_methods, defense_methods),
        "<details>",
        "<summary>Detailed metric tables</summary>",
        "",
        "## Detailed metrics",
        "",
    ]
    for metric in metric_names:
        label = f"Metric: {metric}"
        out_lines.append(render_pivot_table(cells, attack_methods, defense_methods, metric, label))
    out_lines.extend(["</details>", ""])

    Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
