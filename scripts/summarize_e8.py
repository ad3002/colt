#!/usr/bin/env python3
"""Collapse results/ablate6_*_seed*_{std,hard}.json into the E8 summary table.

Reports, per arm: accuracy mean +/- range across seeds on both slices, and the
fraction of the A->F gap (standard slice, per matching seed) each single
component accounts for, evaluated against the two pre-registered decision
rules (H-E8-graph: B >= 80% of gap; H-E8-additive: no component >= 50%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ["A", "B", "C", "D", "E", "F"]
DESC = {
    "A": "positional tables only (LDT-style; no rel bias, no coord MLP, no policy loss)",
    "B": "+ graph bias only",
    "C": "+ coord MLP only",
    "D": "CoLT minus policy loss",
    "E": "CoLT minus coord MLP",
    "F": "CoLT full",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/ablate6_summary.json"))
    args = ap.parse_args()

    acc: dict[str, dict[str, dict[int, float]]] = {a: {"std": {}, "hard": {}} for a in ARMS}
    for f in Path("results").glob("ablate6_*_seed*_*.json"):
        arm, seed_s, tag = f.stem.split("_")[1], f.stem.split("_")[2][4:], f.stem.split("_")[3]
        d = json.loads(f.read_text())
        acc[arm][tag][int(seed_s)] = float(d["accuracy"])

    rows, gap_fracs = {}, {}
    for a in ARMS:
        rows[a] = {"description": DESC[a]}
        for tag in ("std", "hard"):
            vals = sorted(acc[a][tag].values())
            if vals:
                rows[a][tag] = {
                    "seeds": {str(k): v for k, v in sorted(acc[a][tag].items())},
                    "mean": round(sum(vals) / len(vals), 4),
                    "min": vals[0], "max": vals[-1],
                }

    seeds_done = sorted(set(acc["A"]["std"]) & set(acc["F"]["std"]))
    for comp, arm in (("graph_bias(B)", "B"), ("coord_mlp(C)", "C")):
        fr = []
        for s in seeds_done:
            gap = acc["F"]["std"][s] - acc["A"]["std"][s]
            if abs(gap) > 1e-9 and s in acc[arm]["std"]:
                fr.append((acc[arm]["std"][s] - acc["A"]["std"][s]) / gap)
        if fr:
            gap_fracs[comp] = {"per_seed": [round(x, 3) for x in fr],
                               "mean": round(sum(fr) / len(fr), 3)}

    verdict = None
    if "graph_bias(B)" in gap_fracs:
        g = gap_fracs["graph_bias(B)"]["mean"]
        c = gap_fracs.get("coord_mlp(C)", {}).get("mean", 0.0)
        if g >= 0.8:
            verdict = "H-E8-graph holds: graph bias accounts for >=80% of the A->F gap"
        elif max(g, c) < 0.5:
            verdict = "H-E8-additive holds: no single component reaches 50%; the gain is joint"
        else:
            verdict = f"intermediate: graph bias {g:.0%}, coord MLP {c:.0%} of the gap"

    out = {"protocol": "REVISION_EXPERIMENTS.md E8 (CPU edition)",
           "arms": rows, "seeds_complete": seeds_done,
           "gap_fraction_of_A_to_F": gap_fracs, "pre_registered_verdict": verdict}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
