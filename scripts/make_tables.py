#!/usr/bin/env python3
"""Regenerate every results table in paper/main.tex from committed raw JSONs.

This is the script the paper's Reproducibility section points at: no number in
a results table enters the manuscript by hand. Run it and diff the output
against the paper. Rows whose artifact lives in a different repository (the
LDT 6x6 published baseline, from ad3002/LTD) are labeled [external].

Usage:
    python scripts/make_tables.py             # all tables
    python scripts/make_tables.py --only h2   # one table by key
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

R = Path("results")


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def wilson(k: int, n: int, z: float = 1.96) -> str:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"[{100 * (c - h):.1f}, {100 * (c + h):.1f}]"


def t_grid6():
    """Table 2 (tab:grid6): 6x6 head-to-head, 32 chains x 60 rounds."""
    for f, lab in (("eval_gram_sudoku6_phase12.json", "run 1"),
                   ("eval_gram_sudoku6.json", "run 2")):
        g = load(f)
        print(f"GRAM {lab}: valid-answer rate {pct(g['accuracy_valid_solution'])}% "
              f"(wrong {pct(1 - g['accuracy_valid_solution'])}% of emitted answers)")
    print("LDT published baseline: acc 49.4 sound 100 p50 3 suppressed 148,567 [external: ad3002/LTD]")
    for s in ("restart", "dfs"):
        for p in ("random", "mrv", "learned"):
            d = load(f"colt6_{s}_{p}.json")
            print(f"CoLT {s} x {p:8s}: acc {pct(d['accuracy'])} sound {pct(d['soundness'])} "
                  f"p50 {d['rounds']['p50']:.0f} suppressed {d['suppressed_unsound']:,}")
    d = load("colt6_dfs_learned.json")
    print(f"counts: CoLT {d['n_answered']}/{d['n_examples']} Wilson95 "
          f"{wilson(d['n_answered'], d['n_examples'])}; LDT 89/180 Wilson95 {wilson(89, 180)}")
    accs = [load("colt6_dfs_learned.json")["accuracy"],
            load("colt6_seed43_dfs_learned.json")["accuracy"],
            load("colt6_seed44_dfs_learned.json")["accuracy"]]
    print(f"seeds {{42,43,44}}: {[pct(a) for a in accs]} -> mean {pct(statistics.mean(accs))} "
          f"+/- {pct(statistics.stdev(accs))} (sample sd)")


def t_hard():
    """Table 3 (tab:hard): hard slice (6x6, 14 clues, zero-shot) + budget sweep."""
    for s in ("restart", "dfs"):
        accs, sups, bts = [], [], []
        for p in ("random", "mrv", "learned"):
            d = load(f"colt6hard_{s}_{p}.json")
            accs.append(d["accuracy"]); sups.append(d["suppressed_unsound"]); bts.append(d.get("backjumps", 0))
        print(f"{s}: acc {sorted({pct(a) for a in accs})} suppressed {min(sups):,}--{max(sups):,} "
              f"backtracks {max(bts):,}")
    for b in (5, 15):
        for s in ("restart", "dfs"):
            d = load(f"colt6hard_budget{b}_{s}.json")
            print(f"budget {b} rounds, {s}: acc {pct(d['accuracy'])}")


def t_anatomy():
    """Contingency table (sec 5.3) + theta sweep."""
    d = load("anatomy6hard.json")
    print(f"contingency: {d['contingency']} (poisoned {d['n_first_pass_poisoned']}/{d['n']})")
    for th in ("0.02", "0.05"):
        print(f"theta={th}: poisoned {load(f'anatomy6hard_th{th}.json')['n_first_pass_poisoned']}")


def t_h1():
    """H1 one-shot rates (sec 5.4)."""
    for f, lab in (("h1_colt6.json", "6x6 standard"), ("h1_colt6hard.json", "6x6 hard"),
                   ("h1_colt9aug.json", "9x9 augmented")):
        d = load(f)
        print(f"{lab}: singleton {d['singleton_rate']:.3f} full-commit {d['full_commit_frac']:.3f} "
              f"commit-precision {d['commit_correct']:.4f}")


def t_h2():
    """H2 union symmetry frames (sec 5.4)."""
    for s in load("h2_colt6hard_union.json")["settings"]:
        print(f"K={s['frames']} {s['agg']:6s}: poisoning {pct(s['poisoning_rate'])} acc {pct(s['accuracy'])}")


def t_2x2():
    """2x2 structure x augmentation interaction (sec 5.4)."""
    print("LDT -aug: 49.4 [external: ad3002/LTD]")
    print(f"LDT +aug: {pct(load('eval_ldt6aug.json')['accuracy'])}")
    print(f"CoLT -aug: {pct(load('colt6_dfs_learned.json')['accuracy'])}")
    print(f"CoLT +aug: {pct(load('colt6aug_dfs_learned.json')['accuracy'])}")


def t_multi():
    """Table 4 (tab:multi) + zero-shot 9x9 transfer."""
    print(f"single 4x4 {pct(load('colt4_dfs_learned.json')['accuracy'])}, "
          f"single 6x6 {pct(load('colt6_dfs_learned.json')['accuracy'])}")
    print(f"multi  4x4 {pct(load('colt_multi_sudoku4_dfs_learned.json')['accuracy'])}, "
          f"multi  6x6 {pct(load('colt_multi_sudoku6_dfs_learned.json')['accuracy'])}")
    t = load("transfer9_multi.json")
    print(f"9x9 transfer: precision {t['elimination_precision']:.3f} recall {t['elimination_recall']:.3f} "
          f"conflict AUC {t['conflict_auc']:.2f} policy margin {t['policy_margin']:+.2f}")


def t_phase4():
    """Section 6 table: 9x9 at 25 clues, 64 chains x 200 rounds."""
    rows = [("CoLT no-aug restart x random", "colt9_restart_random.json"),
            ("CoLT no-aug dfs x learned   ", "colt9_dfs_learned.json"),
            ("LDT reimpl no-aug (b256)    ", "eval_sudoku9_ldt.json"),
            ("CoLT-aug dfs x learned      ", "colt9aug_dfs_learned.json"),
            ("CoLT-aug restart x random   ", "colt9aug_restart_random.json")]
    for lab, f in rows:
        d = load(f)
        print(f"{lab} acc {pct(d['accuracy'])} sound {pct(d['soundness'])} "
              f"p50 {d['rounds']['p50']} suppressed {d['suppressed_unsound']:,}")
    d = load("colt9aug_dfs_learned.json")
    print(f"residual: {d['n_answered']}/{d['n_examples']} Wilson95 {wilson(d['n_answered'], d['n_examples'])}")
    r = load("colt9aug_dfs_learned_repro.json")
    print(f"cross-machine retrain count: {r['n_answered']}/{r['n_examples']} "
          f"(matches: {r['n_answered'] == d['n_answered']})")


def t_coloring():
    """Section 7: 4-coloring G(20, 0.25)."""
    d = load("coloring20.json")
    print(f"acc {pct(d['accuracy'])} in-enumerated-set {pct(d['answered_in_enumerated_solution_set'])} "
          f"suppressed {d['suppressed_unsound']} K_max {d['k_max']}")


def t_boundary():
    """Section 8: phase-transition sweep + crossing arms."""
    d = load("phase_transition.json")
    for p in d["points"]:
        if "error" in p:
            print(f"c={p['avg_degree']}: not run ({p['error']}; {p['unsat']}/{p['tried']} generated graphs unsat)")
            continue
        gen = p.get("gen")
        unsat = f"{pct(gen['unsat'] / gen['tried'], 0)}%" if gen else "-"
        print(f"c={p['avg_degree']}: restart {pct(p['acc_restart'])} dfs {pct(p['acc_dfs'])} "
              f"singleton {p['h1_singleton_rate']:.3f} gen-unsat {unsat}")
    b = load("boundary_cross.json")
    for arm, v in b["arms"].items():
        if arm.startswith("F_diag"):
            print(f"{arm}: precision {v['elim_precision']:.4f} recall {v['elim_recall']:.4f}")
        else:
            print(f"{arm}: restart {pct(v['restart'])} dfs {pct(v['dfs'])}")


def t_classical():
    """Classical MRV reference timings."""
    for k, v in load("classical_reference.json")["suites"].items():
        print(f"{k}: median {v['time_ms_median']}ms max {v['time_ms_max']}ms "
              f"decisions median {v.get('decisions_median', '-')}")


def t_leakage():
    """Leakage audit."""
    for k, v in load("leakage_audit.json").items():
        print(f"{k}: puzzle {v['exact_puzzle_overlap']} solution {v['exact_solution_overlap']} "
              f"digit-orbit {v['digit_orbit_overlap']} full-orbit {v['full_orbit_overlap']}/{v['n_test']}")


TABLES = {"grid6": t_grid6, "hard": t_hard, "anatomy": t_anatomy, "h1": t_h1,
          "h2": t_h2, "2x2": t_2x2, "multi": t_multi, "phase4": t_phase4,
          "coloring": t_coloring, "boundary": t_boundary, "classical": t_classical,
          "leakage": t_leakage}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(TABLES), default=None)
    args = ap.parse_args()
    for key, fn in TABLES.items():
        if args.only and key != args.only:
            continue
        print(f"==== {key}: {fn.__doc__.splitlines()[0]}")
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[missing artifact] {e}")
        print()
    return 0


if __name__ == "__main__":
    main()
