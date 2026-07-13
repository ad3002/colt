#!/usr/bin/env bash
# Canonical 6x6 measurement set on the living CPU checkpoint
# (runs/colt6_seed42_cpu — same recipe/seed/data as the paper's original run).
# Produces every number the paper's Sections 5.2-5.4 cite, from ONE checkpoint
# in ONE environment: standard + hard ablation grids, budget sweep, H1 probes,
# H2 symmetry-frame sweep (union and mean), and the anatomy contingency.
# The E4 artifact (results/reconcile_anatomy_h2.json) already carries the
# bitwise probe check, theta sweep, and margin histogram for this checkpoint.
set -u
cd "$(dirname "$0")/.."
CKPT=runs/colt6_seed42_cpu/final.pt
LOG=runs/canonical_cpu_logs
mkdir -p "$LOG" results
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

ev() {  # $1 split-tag $2 tsv $3 search $4 policy $5 extra-args $6 out
  [ -f "$6" ] && { echo "[skip] $6"; return; }
  python3 -m colt.eval --checkpoint "$CKPT" --split "$2" --search "$3" --policy "$4" \
    --n-chains 32 --max-rounds 60 $5 --output "$6" > "$LOG/$(basename $6 .json).log" 2>&1
  echo "[done] $6"
}

# standard slice, full 2x3 grid (three at a time)
for p in random mrv learned; do
  ev std data/sudoku6/test.tsv restart $p "" results/colt6cpu_std_restart_${p}.json &
done; wait
for p in random mrv learned; do
  ev std data/sudoku6/test.tsv dfs $p "" results/colt6cpu_std_dfs_${p}.json &
done; wait
echo "[stage] standard grid done"

# hard slice, full 2x3 grid
for p in random mrv learned; do
  ev hard data/sudoku6_hard/test.tsv restart $p "" results/colt6cpu_hard_restart_${p}.json &
done; wait
for p in random mrv learned; do
  ev hard data/sudoku6_hard/test.tsv dfs $p "" results/colt6cpu_hard_dfs_${p}.json &
done; wait
echo "[stage] hard grid done"

# budget sweep on hard (dfs x learned at 5 and 15 rounds; 60 is above)
for r in 5 15; do
  [ -f results/colt6cpu_hard_budget${r}.json ] || \
  python3 -m colt.eval --checkpoint "$CKPT" --split data/sudoku6_hard/test.tsv \
    --search dfs --policy learned --n-chains 32 --max-rounds $r \
    --output results/colt6cpu_hard_budget${r}.json > "$LOG/budget${r}.log" 2>&1 &
done; wait
echo "[stage] budget sweep done"

# H1 probes (single forwards)
python3 scripts/h1_commit_rate.py --checkpoint "$CKPT" --split data/sudoku6/test.tsv \
  --out results/h1_colt6cpu.json > "$LOG/h1_std.log" 2>&1 && echo "[done] h1 std"
python3 scripts/h1_commit_rate.py --checkpoint "$CKPT" --split data/sudoku6_hard/test.tsv \
  --out results/h1_colt6cpu_hard.json > "$LOG/h1_hard.log" 2>&1 && echo "[done] h1 hard"

# H2 symmetry frames, union and mean
python3 scripts/h2_symmetry_frames.py --checkpoint "$CKPT" --split data/sudoku6_hard/test.tsv \
  --frames 1 4 8 --agg union --out results/h2_colt6cpu_hard_union.json > "$LOG/h2_union.log" 2>&1 \
  && echo "[done] h2 union"
python3 scripts/h2_symmetry_frames.py --checkpoint "$CKPT" --split data/sudoku6_hard/test.tsv \
  --frames 8 --agg mean --out results/h2_colt6cpu_hard_mean.json > "$LOG/h2_mean.log" 2>&1 \
  && echo "[done] h2 mean"

# anatomy contingency against the canonical dfs x learned eval
python3 scripts/failure_anatomy.py --checkpoint "$CKPT" --split data/sudoku6_hard/test.tsv \
  --eval-json results/colt6cpu_hard_dfs_learned.json \
  --out results/anatomy6hard_cpu.json > "$LOG/anatomy.log" 2>&1 && echo "[done] anatomy"

echo "[all done]"
