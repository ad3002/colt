#!/usr/bin/env bash
# Review-7 follow-ups, CPU edition, run sequentially overnight.
#
# Part 1 (reviewer point 5): densify the 3-coloring sweep between the last
# solvable point and the first dead point — c in {3.25, 3.5, 3.75} at n=40,
# same generator, seed, budgets as results/phase_transition.json.
#
# Part 2 (reviewer point 2): can the positional-table arm learn AT ALL in this
# codebase given more budget or a different learning rate? lr x steps sweep on
# arm-A configuration (no rel bias, no coord MLP, pos_table=36, lambda_pol=0,
# random pool policy), evaluated dfs x random on the standard slice.
set -u
cd "$(dirname "$0")/.."
LOG=runs/review7_logs
mkdir -p "$LOG" results
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[part1] 3-col density fill-in"
python3 scripts/phase_transition.py --densities 3.25 3.5 3.75 \
  --out results/phase_transition_fill.json > "$LOG/pt_fill.log" 2>&1 \
  && echo "[done] phase_transition_fill" || echo "[FAIL] pt_fill"

echo "[part2] arm-A lr x steps sweep"
ARMA="model.use_rel_bias=false model.use_coord_mlp=false model.pos_table_size=36 loss.lambda_policy=0 train.pool_policy=random"
run_a() {  # $1 tag  $2 extra overrides
  local dir="runs/armA_$1"
  [ -f "$dir/final.pt" ] || \
  python3 -m colt.train --config configs/colt6.yaml --dataset data/sudoku6 \
    --output-dir "$dir" --device cpu --eval-every 1000 \
    --override $ARMA $2 > "$LOG/train_$1.log" 2>&1
  [ -f "results/armA_$1_std.json" ] || \
  python3 -m colt.eval --checkpoint "$dir/final.pt" --split data/sudoku6/test.tsv \
    --search dfs --policy random --n-chains 32 --max-rounds 60 \
    --output "results/armA_$1_std.json" > "$LOG/eval_$1.log" 2>&1
  echo "[done] armA_$1: $(python3 -c "import json;print(json.load(open('results/armA_$1_std.json'))['accuracy'])" 2>/dev/null)"
}
run_a lr1e3_5k  "train.lr=0.001" &
run_a lr3e4_5k  "train.lr=0.0003" &
wait
run_a lr3e3_15k "train.steps=15000" &
run_a lr1e3_15k "train.lr=0.001 train.steps=15000" &
wait
echo "[all done]"
