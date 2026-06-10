# Frozen benchmark suite (Phase 0)

Everything below was fixed **before** any CoLT model was trained. Datasets are
byte-identical to the ad3002/LTD runs (copied, MD5-pinned); budgets equal the
published LDT baseline numbers' budgets, so every comparison is at matched
search compute on identical puzzles.

## Datasets (deterministic, MD5s in each `data/*/MANIFEST.md5`)

| id | board | clues | train / test | builder command |
|---|---|---|---|---|
| `sudoku4` | 4×4 (box 2×2) | 4–6 (minimal-ish) | 1700 / 300 | `build_sudoku_dataset.py --n 4 --num-puzzles 2000 --seed 42` |
| `sudoku6` | 6×6 (box 2×3) | 20 | 1020 / 180 | `build_sudoku_dataset.py --n 6 --num-puzzles 1200 --target-clues 20 --seed 42` |
| `sudoku9_small` | 9×9 (box 3×3) | 30 | 102 / 198 | `build_sudoku_dataset.py --n 9 --num-puzzles 300 --target-clues 30 --split 0.34 --seed 42` (transfer probes only, no training) |

## Metrics

- **accuracy** — fraction of puzzles answered with a verified solution.
- **soundness** — fraction of emitted answers that are correct (must stay 1.0;
  any emitted wrong grid is a hard failure).
- **abstain rate** — 1 − accuracy (answered-or-abstain regime).
- **rounds p50/p95** — forward passes per solved puzzle (compute profile).
- **suppressed_unsound / restarts / backjumps** — search-diagnostics.
- Transfer (9×9, zero-shot): **elimination precision/recall** per forward pass
  (a removed candidate is correct iff it is not the solution value) and
  **conflict-head AUC** on consistent-vs-⊥ states (`scripts/transfer_probe.py`).

## Frozen evaluation budgets

| board | chains × rounds | θ_CLS | seed |
|---|---|---|---|
| 4×4 | 32 × 60 | 0.6 | 0 |
| 6×6 | 32 × 60 | 0.6 | 0 |

(These equal the budgets behind the published LDT baseline numbers below.)

## Baselines (shared data, published before CoLT training)

| system | 4×4 acc / sound | 6×6 acc / sound | source |
|---|---|---|---|
| **LDT** (reimpl, ad3002/LTD) | 96.3 % / 100 % | **49.4 % / 100 %** | `LTD/results/eval_test_sudoku{4,6}.json` |
| **GRAM** (reimpl, ad3002/gram; matched-CPU small config, honest caveat: GRAM's recipe wants ~10–100× more compute) | `results/eval_gram_sudoku4.json` | `results/eval_gram_sudoku6.json` | this repo |
| MRV arm (CoLT ckpt, hand-coded policy) | — | ablation grid | this repo |

GRAM has no abstention: its "wrong answer rate" = 1 − accuracy_valid_solution
among emitted grids — the soundness contrast is a primary claim, reported
alongside the budget caveat.

## The Phase 1/2 ablation grid (one 6×6 checkpoint, inference-side switches)

search ∈ {restart, dfs} × policy ∈ {random, mrv, learned} at 32 × 60:

- `restart × random` = LDT inference exactly (control).
- `dfs × learned` = full CoLT.

**Pre-registered success criteria** (set before the runs):

1. Phase 1: some CoLT arm > 49.4 % on `sudoku6` at the frozen budget.
2. Policy: `learned` > `mrv` > `random` ordering on at least the dfs row.
3. Phase 2: `dfs × p` > `restart × p` for each policy p.
4. Soundness = 100 % in every cell of the grid.
5. Phase 3: the multi-size checkpoint is within 5 pp of the single-size
   checkpoint on each board at matched per-task steps, with positive 9×9
   zero-shot transfer (elimination precision > 0.9 at recall > 0.1, AUC > 0.8).

Failures are reported as failures (DESIGN.md §4 maps each to a design
falsification).
