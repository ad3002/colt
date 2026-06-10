#!/usr/bin/env bash
# GPU runbook: phases 1-4 end-to-end on a single CUDA box (RunPod 4090).
# Stages are idempotent-ish and ordered; run all or pass a stage name:
#   bash scripts/pod_runbook.sh            # everything
#   bash scripts/pod_runbook.sh phase4     # just the 9×9 head-to-head
# Assumes: repo at ~/colt (this script's parent), ad3002/LTD cloned at ~/LTD,
# gram repo at ~/gram (rsynced or cloned), torch+cuda working.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEV="${DEV:-cuda}"
STAGE="${1:-all}"

run() { echo; echo "######## $* ########"; "$@"; }

stage_deps() {
  pip install -q pyyaml pytest 2>&1 | tail -1 || true
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$ROOT" python3 -m pytest tests/ -q | tail -2
}

stage_phase1() {  # CoLT 6×6 training (frozen config) + the full ablation grid
  PYTHONPATH="$ROOT" python3 -m colt.train --config configs/colt6.yaml \
      --dataset data/sudoku6 --output-dir runs/colt6_seed42 --device "$DEV" \
      --seed 42 --eval-every 1000 2>&1 | tail -4
  bash scripts/run_grid.sh runs/colt6_seed42/final.pt data/sudoku6/test.tsv results colt6
}

stage_gram() {    # GRAM baselines on shared data (matched small recipe)
  cd ../gram
  for N in 4 6; do
    SEQ=$((N*N)); VOC=$((N+2)); BS=$([ "$N" = 4 ] && echo 256 || echo 128)
    python3 -m gram.train --config configs/nqueens_8x8.yaml \
        --dataset "$ROOT/data/gram_sudoku$N" --output-dir "$ROOT/runs/gram${N}_seed42" \
        --device "$DEV" --seed 42 --override task.seq_len=$SEQ task.vocab_size=$VOC \
        model.d_model=64 model.ff_hidden=64 model.n_heads=4 model.K_inner=2 \
        model.T_outer=2 model.nsup=6 train.epochs=300 train.global_batch_size=$BS \
        logging.log_every=100 2>&1 | tail -2
    python3 -m gram.eval --checkpoint "$ROOT/runs/gram${N}_seed42/final.pt" \
        --split "$ROOT/data/gram_sudoku$N/test.tsv" --n-samples 4 --device "$DEV" \
        --batch-size $BS --seed 0 --output "$ROOT/results/eval_gram_sudoku$N.json" 2>&1 | tail -3
  done
  cd "$ROOT"
}

stage_phase3() {  # multi-size training + per-board evals + 9×9 zero-shot transfer
  PYTHONPATH="$ROOT" python3 -m colt.train --config configs/colt_multi.yaml \
      --dataset data/sudoku4 data/sudoku6 --output-dir runs/colt_multi_seed42 \
      --device "$DEV" --seed 42 --eval-every 1000 2>&1 | tail -4
  for N in 4 6; do
    for cell in "dfs learned" "restart random"; do
      set -- $cell
      PYTHONPATH="$ROOT" python3 -m colt.eval --checkpoint runs/colt_multi_seed42/final.pt \
          --split data/sudoku$N/test.tsv --search "$1" --policy "$2" \
          --n-chains 32 --max-rounds 60 --seed 0 \
          --output "results/colt_multi_sudoku${N}_$1_$2.json" 2>&1 | tail -2
    done
  done
  PYTHONPATH="$ROOT" python3 scripts/transfer_probe.py \
      --checkpoint runs/colt_multi_seed42/final.pt \
      --split data/sudoku9_small/test.tsv --out results/transfer9_multi.json
  # single-size 4×4 reference for the one-checkpoint table
  PYTHONPATH="$ROOT" python3 -m colt.train --config configs/colt6.yaml \
      --dataset data/sudoku4 --output-dir runs/colt4_seed42 --device "$DEV" \
      --seed 42 --override train.steps=4000 2>&1 | tail -3
  PYTHONPATH="$ROOT" python3 -m colt.eval --checkpoint runs/colt4_seed42/final.pt \
      --split data/sudoku4/test.tsv --search dfs --policy learned \
      --n-chains 32 --max-rounds 60 --seed 0 --output results/colt4_dfs_learned.json 2>&1 | tail -2
}

stage_phase4() {  # 9×9 head-to-head (PHASE4.md): CoLT vs LDT, frozen budget 64×200
  [ -d data/sudoku9 ] || python3 scripts/build_sudoku_dataset.py --n 9 \
      --num-puzzles 1200 --target-clues 25 --split 0.85 --seed 42 --out data/sudoku9
  # CoLT 9×9 (scaled config via overrides)
  PYTHONPATH="$ROOT" python3 -m colt.train --config configs/colt6.yaml \
      --dataset data/sudoku9 --output-dir runs/colt9_seed42 --device "$DEV" --seed 42 \
      --eval-every 2000 --override model.d_model=128 model.n_iters=16 \
      train.batch_size=256 train.steps=20000 solve.tau_age=100 2>&1 | tail -4
  CHAINS=64 ROUNDS=200 bash scripts/run_grid.sh runs/colt9_seed42/final.pt \
      data/sudoku9/test.tsv results colt9
  # LDT 9×9 control on the same data (paper recipe, ad3002/LTD)
  cd ../LTD
  PYTHONPATH="$(pwd)" python3 -m ltd.train --config configs/sudoku9.yaml \
      --dataset "$ROOT/data/sudoku9" --output-dir "$ROOT/runs/ldt9_seed42" \
      --device "$DEV" --seed 42 --eval-every 2000 2>&1 | tail -4
  PYTHONPATH="$(pwd)" python3 -m ltd.eval --checkpoint "$ROOT/runs/ldt9_seed42/final.pt" \
      --split "$ROOT/data/sudoku9/test.tsv" --n-chains 64 --max-rounds 200 --seed 0 \
      --output "$ROOT/results/eval_sudoku9_ldt.json" 2>&1 | tail -2
  cd "$ROOT"
}

stage_tables() { PYTHONPATH="$ROOT" python3 scripts/make_tables.py --results results --tag colt6; }

case "$STAGE" in
  all)    stage_deps; stage_phase1; stage_gram; stage_phase3; stage_phase4; stage_tables ;;
  deps)   stage_deps ;;
  phase1) stage_phase1 ;;
  gram)   stage_gram ;;
  phase3) stage_phase3 ;;
  phase4) stage_phase4 ;;
  tables) stage_tables ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
echo "######## runbook stage '$STAGE' done ########"
