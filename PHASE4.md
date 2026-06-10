# Phase 4 — GPU head-to-head: requirements and protocol

*Everything below is pre-registered before the GPU run. CPU phases 0–3 are
complete and reported in `results/` + `paper/`; Phase 4 fills the 9×9 cells.*

## 1. Hardware requirements

| item | requirement | rationale |
|---|---|---|
| GPU | **1× RTX 4090 24 GB** (fallback: 3090 / L4 / A10; ≥ 12 GB) | models are 0.3–2 M params; the bottleneck is *sequential* recurrent forwards (L = 10–16 unrolls of small matmuls), so clock rate dominates — a 4090 beats an A100 on $/wall-clock here |
| image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | same image as the ad3002/gram reproduction; known-good |
| disk | ≥ 15 GB | torch + datasets + checkpoints |
| wall-clock | ~3–5 h total | table below |
| budget | ≈ $2–5 at $0.35–0.45/h community cloud | — |

## 2. Setup (copy-paste)

```bash
git clone https://github.com/ad3002/colt && cd colt
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q          # all tests must pass

# Frozen 9×9 dataset (deterministic, seed-pinned; ~25-clue ⇒ search-required)
python scripts/build_sudoku_dataset.py --n 9 --num-puzzles 1200 \
    --target-clues 25 --split 0.85 --seed 42 --out data/sudoku9
```

## 3. Runs (in priority order)

| # | run | command sketch | est. time |
|---|---|---|---|
| 1 | **CoLT 9×9** | `python -m colt.train --config configs/colt9.yaml --dataset data/sudoku9 --output-dir runs/colt9 --device cuda --eval-every 500` — start from `configs/colt6.yaml` scaled to `d_model=128, n_iters=16, batch=256, steps=20000, tau_age=100` | ~1.5 h |
| 2 | **LDT 9×9 control** | same data, `ad3002/LTD` `configs/sudoku9.yaml` (paper recipe) | ~1 h |
| 3 | **Ablation grid 9×9** | `colt.eval` over {restart,dfs} × {random,mrv,learned}, frozen budget 64 chains × 200 rounds, seed 0 | ~0.5 h |
| 4 | **Multi-size + 9×9** | `colt.train --dataset data/sudoku4 data/sudoku6 data/sudoku9` (one checkpoint, three boards) + per-board evals | ~2 h |
| 5 | (stretch) 3-seed repeats of #1/#2 headline cells | mean ± std | ~2 h |

## 4. Pre-registered success criteria

1. **Headline**: CoLT (dfs × learned) accuracy on the 9×9 test split ≥ LDT
   (restart × random) at the identical frozen budget, soundness 1.0 both.
2. **Compute profile**: CoLT median rounds-per-solve ≤ LDT's at equal accuracy
   (learned ordering + backjumping should solve in fewer forwards).
3. **Policy ordering**: learned > mrv > random on the dfs row (replicating the
   6×6 Phase 2 finding at 9×9).
4. **One-checkpoint claim**: multi-size checkpoint within 5 pp of single-size
   on every board at matched per-task steps.
5. Soundness = 1.0 in **every** cell; any emitted wrong grid invalidates the
   run (and the architecture claim).

## 5. Risks / mitigations

- *9×9 from-scratch may need more than 20 k steps on generated 25-clue data*
  → curriculum reveal is already in the config; if probe accuracy is < 20 % at
  10 k steps, raise `target-clues` to 30 (easier split) and report both.
- *DFS python-side loop may underutilize GPU at large chain counts* → batch
  whole slot×chain block per forward (already implemented); if profiling shows
  < 50 % GPU util, raise `--slot-batch`.
- *Sudoku-Extreme corpus comparability* — our generated 9×9 ≠ the
  Tdoku/RRN-derived Sudoku-Extreme used in the LDT paper. The head-to-head is
  internally valid (same data both systems); cross-paper comparability is
  explicitly NOT claimed. Optional follow-up: vendor Sudoku-Extreme and rerun
  rows 1–3 on it.

## 6. Deliverables

- `results/eval_sudoku9_{ldt,colt}_*.json` (full grid), training histories,
  transfer JSONs; paper §6 table filled; README headline updated.
