#!/usr/bin/env bash
# E8 (training-side component ablation) + E4 (probe reconciliation), CPU edition.
#
# Runs the six pre-registered arms of REVISION_EXPERIMENTS.md E8 at 6x6 for
# seeds 42/43/44 on CPU (one arm ~= 45 min at 4 threads; the box has 12 cores,
# so trainings go in waves of <=3). Arm F seed 42 is the checkpoint retrained
# by the E4 step (same recipe as the paper's primary run). lambda_policy=0 arms
# also set train.pool_policy=random: with no policy loss the head is untrained
# noise, and random pool branching is the LDT-faithful control (documented in
# the results JSONs via the resolved configs).
#
# Usage: bash scripts/run_e8_cpu.sh [logdir]   # blocks for hours; run under nohup
set -u
cd "$(dirname "$0")/.."
LOG=${1:-runs/e8_cpu_logs}
mkdir -p "$LOG" results
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

declare -A OV=(
  [A]="model.use_rel_bias=false model.use_coord_mlp=false model.pos_table_size=36 loss.lambda_policy=0 train.pool_policy=random"
  [B]="model.use_coord_mlp=false loss.lambda_policy=0 train.pool_policy=random"
  [C]="model.use_rel_bias=false loss.lambda_policy=0 train.pool_policy=random"
  [D]="loss.lambda_policy=0 train.pool_policy=random"
  [E]="model.use_coord_mlp=false"
  [F]=""
)
declare -A POL=([A]=random [B]=random [C]=random [D]=random [E]=learned [F]=learned)

train_arm() {  # $1 arm  $2 seed
  local arm=$1 seed=$2 dir="runs/ablate6_${1}_seed${2}"
  [ -f "$dir/final.pt" ] && { echo "[skip] $dir exists"; return 0; }
  # shellcheck disable=SC2086
  python3 -m colt.train --config configs/colt6.yaml --dataset data/sudoku6 \
    --output-dir "$dir" --device cpu --eval-every 1000 \
    --override ${OV[$arm]} train.seed="$seed" \
    > "$LOG/train_${arm}_${seed}.log" 2>&1
  echo "[done] train $arm seed $seed"
}

eval_arm() {  # $1 arm  $2 seed  $3 ckpt-dir
  local arm=$1 seed=$2 dir=$3 pol=${POL[$1]}
  for split in std:data/sudoku6/test.tsv hard:data/sudoku6_hard/test.tsv; do
    local tag=${split%%:*} tsv=${split#*:}
    local out="results/ablate6_${arm}_seed${seed}_${tag}.json"
    [ -f "$out" ] && continue
    python3 -m colt.eval --checkpoint "$dir/final.pt" --split "$tsv" \
      --search dfs --policy "$pol" --n-chains 32 --max-rounds 60 \
      --output "$out" > "$LOG/eval_${arm}_${seed}_${tag}.log" 2>&1
  done
  echo "[done] eval $arm seed $seed"
}

# ---- stage 0: wait for the E4 retrain (arm F, seed 42), then reconcile ------
echo "[wait] runs/colt6_seed42_cpu/final.pt"
while [ ! -f runs/colt6_seed42_cpu/final.pt ]; do sleep 30; done
sleep 5
if [ ! -f results/reconcile_anatomy_h2.json ]; then
  OMP_NUM_THREADS=8 python3 scripts/reconcile_probes.py \
    --checkpoint runs/colt6_seed42_cpu/final.pt \
    --split data/sudoku6_hard/test.tsv \
    --eval-json results/colt6hard_dfs_learned.json \
    --out results/reconcile_anatomy_h2.json > "$LOG/e4.log" 2>&1 \
    && echo "[done] E4 reconciliation" || echo "[FAIL] E4 (see $LOG/e4.log)"
fi
mkdir -p runs/ablate6_F_seed42
cp runs/colt6_seed42_cpu/final.pt runs/ablate6_F_seed42/final.pt 2>/dev/null || true

# ---- stage 1: seed 42 arms, then evals; F42 eval reuses the retrain ---------
train_arm A 42 & train_arm B 42 & train_arm C 42 & wait
train_arm D 42 & train_arm E 42 & eval_arm A 42 runs/ablate6_A_seed42 & wait
eval_arm B 42 runs/ablate6_B_seed42 & eval_arm C 42 runs/ablate6_C_seed42 & \
  eval_arm D 42 runs/ablate6_D_seed42 & wait
eval_arm E 42 runs/ablate6_E_seed42 & eval_arm F 42 runs/ablate6_F_seed42 & wait
echo "[stage] seed 42 complete"

# ---- stage 2: seeds 43, 44 --------------------------------------------------
for seed in 43 44; do
  train_arm F "$seed" & train_arm A "$seed" & train_arm B "$seed" & wait
  train_arm C "$seed" & train_arm D "$seed" & train_arm E "$seed" & wait
  for arm in A B C D E F; do eval_arm "$arm" "$seed" "runs/ablate6_${arm}_seed${seed}" & done
  wait
  echo "[stage] seed $seed complete"
done

python3 scripts/summarize_e8.py --out results/ablate6_summary.json || true
echo "[all done]"
