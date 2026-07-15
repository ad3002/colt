#!/usr/bin/env bash
# Round-8 CPU queue (E10-E15 minus GPU-bound 9x9 seeds). Ordered fast->slow.
set -u
cd "$(dirname "$0")/.."
LOG=runs/round8_logs
mkdir -p "$LOG" results
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CANON=runs/colt6_seed42_cpu/final.pt

step() { echo "[$(date +%H:%M)] $*"; }

# --- E15 verifier audit (minutes) --------------------------------------------
step E15 verifier
python3 scripts/independent_verifier.py --out results/verifier_audit.json \
  > "$LOG/e15.log" 2>&1 && step "E15 done (pass=$(python3 -c "import json;print(json.load(open('results/verifier_audit.json'))['pass'])"))" || step "E15 FAIL"

# --- E12b classical reference incl. fill densities ---------------------------
step E12b classical
python3 scripts/classical_reference.py --out results/classical_reference.json \
  > "$LOG/e12b.log" 2>&1 && step "E12b done" || step "E12b FAIL"

# --- E10 H2 factorial (approx 2-3 h) ------------------------------------------
step E10 factorial
OMP_NUM_THREADS=8 python3 scripts/h2_factorial.py --checkpoint "$CANON" \
  --split data/sudoku6_hard/test.tsv --out results/h2_factorial_cpu.json \
  > "$LOG/e10.log" 2>&1 && step "E10 done" || step "E10 FAIL"

# --- E13 canonical core on seeds 43/44 ----------------------------------------
for S in 43 44; do
  CK=runs/ablate6_F_seed${S}/final.pt
  step "E13 seed $S"
  python3 -m colt.eval --checkpoint "$CK" --split data/sudoku6_hard/test.tsv \
    --search dfs --policy learned --n-chains 32 --max-rounds 60 \
    --output results/colt6cpu_s${S}_hard_dfs_learned.json > "$LOG/e13_${S}_hard.log" 2>&1 &
  python3 -m colt.eval --checkpoint "$CK" --split data/sudoku6/test.tsv \
    --search dfs --policy learned --n-chains 32 --max-rounds 60 \
    --output results/colt6cpu_s${S}_std_dfs_learned.json > "$LOG/e13_${S}_std.log" 2>&1 &
  wait
  python3 scripts/failure_anatomy.py --checkpoint "$CK" --split data/sudoku6_hard/test.tsv \
    --eval-json results/colt6cpu_s${S}_hard_dfs_learned.json \
    --out results/anatomy6hard_cpu_s${S}.json > "$LOG/e13_${S}_anat.log" 2>&1
  python3 scripts/h2_symmetry_frames.py --checkpoint "$CK" --split data/sudoku6_hard/test.tsv \
    --frames 1 8 --agg union --out results/h2_colt6cpu_s${S}_union.json \
    > "$LOG/e13_${S}_h2.log" 2>&1
  step "E13 seed $S done"
done

# --- multi-size retrain + transfer + E12a floor -------------------------------
step multi retrain
[ -f runs/colt_multi_cpu/final.pt ] || \
python3 -m colt.train --config configs/colt_multi.yaml --dataset data/sudoku4 data/sudoku6 \
  --output-dir runs/colt_multi_cpu --device cpu > "$LOG/multi_train.log" 2>&1
step multi transfer probes
python3 scripts/transfer_probe.py --checkpoint runs/colt_multi_cpu/final.pt \
  --split data/sudoku9_small/test.tsv --out results/transfer9_multi_cpu.json \
  > "$LOG/transfer.log" 2>&1 || step "transfer FAIL"
python3 scripts/transfer9_floor.py --checkpoint runs/colt_multi_cpu/final.pt \
  --split data/sudoku9_small/test.tsv --out results/transfer9_floor.json \
  > "$LOG/floor.log" 2>&1 || step "floor FAIL"
for BOARD in 4 6; do
  python3 -m colt.eval --checkpoint runs/colt_multi_cpu/final.pt \
    --split data/sudoku${BOARD}/test.tsv --search dfs --policy learned \
    --n-chains 32 --max-rounds 60 --output results/multi_cpu_sudoku${BOARD}.json \
    > "$LOG/multi_eval${BOARD}.log" 2>&1 &
done; wait
step "multi block done"

# --- E11 policy grid at c=3.0, 3 seeds (parallel) ------------------------------
step E11 policy grid
for S in 42 43 44; do
  python3 scripts/policy_grid.py --density 3.0 --seed $S \
    --out results/policy_grid_c30_seed${S}.json > "$LOG/e11_${S}.log" 2>&1 &
done; wait
step "E11 done"

# --- E14 positional tables at 50k, 3 seeds (parallel, the long tail) -----------
step E14 50k sweep
ARMA="model.use_rel_bias=false model.use_coord_mlp=false model.pos_table_size=36 loss.lambda_policy=0 train.pool_policy=random"
for S in 42 43 44; do
  ( [ -f runs/armA_50k_s${S}/final.pt ] || \
    python3 -m colt.train --config configs/colt6.yaml --dataset data/sudoku6 \
      --output-dir runs/armA_50k_s${S} --device cpu --eval-every 5000 \
      --override $ARMA train.steps=50000 train.seed=$S > "$LOG/e14_train_${S}.log" 2>&1
    python3 -m colt.eval --checkpoint runs/armA_50k_s${S}/final.pt \
      --split data/sudoku6/test.tsv --search dfs --policy random \
      --n-chains 32 --max-rounds 60 --output results/budget_sweep_pos50k_s${S}.json \
      > "$LOG/e14_eval_${S}.log" 2>&1 ) &
done; wait
step "E14 done"
echo "[all done]"
