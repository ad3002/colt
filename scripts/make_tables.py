#!/usr/bin/env python3
"""Render the paper's result tables from results/*.json (mechanical fill).

Prints LaTeX rows for Table 2 (the 6×6 ablation grid) and Table 3 (one
checkpoint across boards) plus a Markdown version for README. Reads only
committed eval JSONs — no hand-typed numbers.

Usage:
    python scripts/make_tables.py --results results --tag colt6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float | None) -> str:
    return f"{100*x:.1f}" if x is not None else r"\pending{pending}"


def grid_rows(results: Path, tag: str) -> list[str]:
    rows = []
    for s in ("restart", "dfs"):
        for p in ("random", "mrv", "learned"):
            d = load(results / f"{tag}_{s}_{p}.json")
            label = f"{s} $\\times$ {p}"
            if s == "dfs" and p == "learned":
                label = r"\textbf{dfs $\times$ learned (full \colt{})}"
            if d is None:
                rows.append(f"\\colt{{}} ckpt & {label} & \\pending{{}} & & & \\\\")
                continue
            r50 = d["rounds"]["p50"]
            rows.append(
                f"\\colt{{}} ckpt & {label} & {pct(d['accuracy'])} & {pct(d['soundness'])} & "
                f"{r50:.0f} & {d['suppressed_unsound']:,} \\\\".replace(",", r"{,}")
            )
    return rows


def markdown_grid(results: Path, tag: str) -> list[str]:
    out = ["| search × policy | accuracy | soundness | rounds p50 | suppressed | backjumps |",
           "|---|---|---|---|---|---|"]
    for s in ("restart", "dfs"):
        for p in ("random", "mrv", "learned"):
            d = load(results / f"{tag}_{s}_{p}.json")
            if d is None:
                out.append(f"| {s} × {p} | _pending_ | | | | |")
                continue
            out.append(
                f"| {s} × {p} | **{100*d['accuracy']:.1f}%** | {100*d['soundness']:.0f}% | "
                f"{d['rounds']['p50']:.0f} | {d['suppressed_unsound']:,} | {d['backjumps']:,} |"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--tag", default="colt6")
    args = ap.parse_args()

    print("==== LaTeX: Table 2 (6x6 grid) rows ====")
    for r in grid_rows(args.results, args.tag):
        print(r)

    print("\n==== Markdown grid ====")
    for r in markdown_grid(args.results, args.tag):
        print(r)

    print("\n==== GRAM baselines ====")
    for n in (4, 6):
        d = load(args.results / f"eval_gram_sudoku{n}.json")
        if d:
            acc = d.get("accuracy_valid_solution")
            wrong = 1.0 - acc if acc is not None else None
            print(f"GRAM {n}x{n}: acc_valid={pct(acc)}%  exact={pct(d.get('accuracy_target_exact_match'))}%  "
                  f"wrong-answer-rate={pct(wrong)}% (no abstention)")
        else:
            print(f"GRAM {n}x{n}: pending")

    print("\n==== Multi-size + transfer ====")
    for tag, boards in [("colt_multi", (4, 6))]:
        for n in boards:
            d = load(args.results / f"{tag}_sudoku{n}_dfs_learned.json")
            print(f"multi ckpt {n}x{n} dfs×learned: acc={pct(d['accuracy']) if d else 'pending'}%")
    t = load(args.results / "transfer9_multi.json")
    if t:
        print(f"9x9 zero-shot: precision={t['elimination_precision']} recall={t['elimination_recall']} "
              f"AUC={t['conflict_auc']} policy_margin={t['policy_margin']}")
    else:
        print("9x9 zero-shot: pending")
    return 0


if __name__ == "__main__":
    main()
