#!/usr/bin/env bash
# The frozen ablation grid (BENCHMARKS.md): {restart,dfs} × {random,mrv,learned}
# from ONE checkpoint at the frozen budget. Usage:
#   bash scripts/run_grid.sh runs/colt6_seed42/final.pt data/sudoku6/test.tsv results colt6
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?checkpoint}"
SPLIT="${2:?split tsv}"
OUTDIR="${3:-results}"
TAG="${4:-grid}"
CHAINS="${CHAINS:-32}"
ROUNDS="${ROUNDS:-60}"

for s in restart dfs; do
  for p in random mrv learned; do
    echo "=== ${s} × ${p} ==="
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" PYTHONPATH="$(pwd)" \
    python3 -m colt.eval --checkpoint "${CKPT}" --split "${SPLIT}" \
        --search "$s" --policy "$p" --n-chains "${CHAINS}" --max-rounds "${ROUNDS}" \
        --seed 0 --output "${OUTDIR}/${TAG}_${s}_${p}.json" 2>&1 | tail -2
  done
done
echo "=== grid done → ${OUTDIR}/${TAG}_*.json ==="
