# Pre-registered revision experiments (paper 1, review round 1)

Frozen **2026-06-11**, before any of the runs below were executed and before any
GPU was provisioned. Each experiment answers a specific referee point, states
its hypotheses and decision rules **in advance**, and names the artifact it
will produce. Text changes contingent on outcomes are written down here so the
revision cannot quietly move the goalposts.

Status legend: `[CPU-done]` already executed locally (no GPU needed),
`[GPU]` waiting on hardware, `[CODE]` needs implementation first.

Hardware assumption: one 24 GB-class GPU (RTX 4090 or better). Total GPU
budget for E1–E6: **~29 GPU-hours** (incl. the E2 batch-512-effective
amendment). Optional E7 adds ~30–40.

---

## E0 — Dataset construction + leakage audit `[CPU-done]`

**Referee point 6** (no generation/dedup/leakage documentation).

Protocol: `scripts/leakage_audit.py` reports, for every (train, test) pair the
paper evaluates, (a) exact puzzle overlap, (b) exact solution-grid overlap,
(c) overlap up to digit relabeling, (d) overlap up to the full Sudoku symmetry
group (digit relabeling x row/band x column/stack permutations x transpose for
square boxes), computed exactly via a digit-permutation-invariant hash plus
exhaustive enumeration of the cell subgroup (128 / 3,456 / 3,359,232
transforms at 4x4 / 6x6 / 9x9) with exact verification of every hash hit.
`data/sudoku9` (Phase 4) is regenerated deterministically (builder seed 42)
and audited the same way.

Pre-registered interpretation:
- 4x4 has only 288 valid solution grids; heavy overlap there is a property of
  the space, not a bug. Decision rule: reclassify 4x4 as an *integration/sanity
  tier* in the paper regardless of measured numbers; its 100% cells are not
  evidence of generalization.
- 6x6 / 9x9: report the measured full-orbit overlap fraction in the protocol
  section. If the 6x6 fraction exceeds 10% of test, add a deduplicated-test
  re-evaluation arm to E3 (same checkpoints, overlap-free test subset).

Artifact: `results/leakage_audit.json`.

---

## E1 — LDT + augmentation at extended budget `[GPU]`

**Referee point 1** ("destroys" may be slow convergence; headline baseline is
LDT-minus-its-own-calibration-mechanism).

Protocol: train the positional-table LDT reimplementation **with**
digit-permutation augmentation on `data/sudoku6` at 1x, 3x, 10x the frozen
budget (5k / 15k / 50k steps, all other hyperparameters as
`configs/colt6.yaml`-matched LDT baseline, seed 42). Evaluate on the frozen
6x6 test split at the frozen inference budget (32 chains x 60 rounds, dfs x
random for the LDT solver semantics = restart x random; use its native
inference).

Hypotheses and decision rules (pre-registered):
- **H-E1-fast**: if LDT+aug at 10x budget recovers to >= 49.4% (its own
  unaugmented score), the verb "destroys" is wrong; replace everywhere with
  "stalls at the frozen budget" and report the recovery curve in the paper.
- **H-E1-slow**: if LDT+aug at 10x stays < 25%, the 2x2 interaction stands as
  a structural claim at any practical budget; keep the cell but still qualify
  with "at 1-10x the frozen budget" and show the curve.
- Either way the abstract must continue to flag that the headline LDT row
  omits the LDT paper's own augmentation (its documented top deviation).

Budget: 3 runs x <= 1.5 h = **~4 GPU-h**.
Artifacts: `results/ldt6_aug_budget{1,3,10}x.json`.

---

## E2 — The 2x2 structure-x-augmentation factorial at 9x9 `[GPU]`

**Referee point 1** (2x2 shown only at 6x6; 9x9 LDT row additionally weakened
by batch 256 vs paper's 512).

Protocol: 4 cells at 9x9/25-clue (`data/sudoku9`, regenerated, audited in E0):
{LDT positional tables, CoLT graph bias} x {no aug, digit-permutation aug},
all at `d_model=128, n_iters=16, batch=256, steps=20000` (the Phase-4 recipe),
seed 42, frozen eval budget 64 chains x 200 rounds.

Pre-registered expectations: CoLT+aug ~ 96%, CoLT-no-aug ~ 0% (already
measured); LDT+aug and LDT-no-aug are the new cells. If LDT+aug >= 80%, the
"structure is what makes augmentation affordable" claim is **falsified at
9x9** and must be retracted to a 6x6-only observation. If LDT+aug <= 10%, the
interaction generalizes across board sizes and the claim strengthens.

**Amendment (2026-06-11, before any run; prompted by re-reading referee point
1):** the LDT cells additionally run at *effective* batch 512 via gradient
accumulation (2 x 256), which removes the reproduction's batch-size deviation
from the LDT recipe on a 24 GB card at the cost of ~2x wall-clock. The
batch-256 LDT row is kept as a secondary arm so the deviation's own effect is
measured rather than assumed. Decision rule unchanged; it applies to the
batch-512-effective cells.

Budget: 2 new cells x ~6 h, LDT cells at ~2x wall-clock for accumulation
(+2 reruns if seeds from E3 are folded in) = **~16 GPU-h**.
Artifacts: `results/grid9_{ldt,colt}_{aug,noaug}.json`,
`results/grid9_ldt_b512eff_{aug,noaug}.json`.

---

## E3 — Seeds and confidence intervals for every headline cell `[GPU]`

**Referee point 2** (single seeds, no intervals, "statistically inert" without
a test, criterion-5 "positive transfer" = 1 puzzle in 180).

Protocol: 3 seeds (42, 43, 44) for: (a) the hard-slice 2x3 ablation grid,
(b) 9x9 CoLT +/- aug, (c) the 6x6 2x2 cells, (d) H2 union K in {1, 4, 8}.
Tables in the paper gain mean +/- sd over seeds and Wilson 95% intervals on
the per-seed binomials (n = 180).

Pre-registered decision rules:
- The policy-head effect is declared nonzero only if |mean difference| >
  2 x SE across seeds on the same puzzles (paired). Otherwise the paper says
  "no detectable effect; the design is powered to ~ +/- 5 pp", not
  "statistically inert".
- Criterion-5 wording is downgraded **now** (before reruns) to
  "indistinguishable from the single-size checkpoint (one puzzle in 180)";
  reruns can only upgrade it back if the multi-size advantage replicates in
  all 3 seeds.

Budget: ~15 short trainings/evals = **~8 GPU-h**.
Artifacts: `results/seeds/*.json`, aggregated by `scripts/make_tables.py`.

---

## E4 — Single-environment reconciliation of the anatomy/H2 probe discrepancy `[GPU]`

**Referee point 5** (76.7%/42 poisoned in section 5.2-5.3 vs 74.4%/46 at H2
K=1; both probes are deterministic forwards of the same checkpoint).

The two JSONs were produced by separate scripts in separate sessions; the
forward passes are code-identical (verified by reading both scripts), so the
4/180-puzzle disagreement is an environment effect (device/dtype/kernel
nondeterminism at the elimination margin) and currently unexplained. That is
not acceptable for a paper about measurement.

Protocol: on one device, one torch build, one dtype: run
`scripts/failure_anatomy.py` and `scripts/h2_symmetry_frames.py --frames 1`
back-to-back on `runs/colt6_seed42/final.pt`; assert bitwise-equal first-pass
elimination masks; additionally report the margin histogram
|sigma(b) - theta_elim| for solution values, and the count of puzzles whose
poisoned status flips within +/- 0.02 of theta.

Pre-registered decision rules:
- If the masks agree (expected): one canonical number set replaces both in the
  paper; the old discrepancy is documented in the reproducibility section as a
  cross-environment sensitivity with the measured margin histogram.
- If they disagree on one device: there is a real bug; find it before any
  other experiment is trusted.

Budget: **< 0.5 GPU-h** (two forward passes + two solves).
Artifacts: `results/reconcile_anatomy_h2.json`.

---

## E5 — Value-permutation equivariance as the architectural alternative `[CODE, GPU]`

**Referee point 4** (if poisoning is value-specific bias, weight sharing along
the value axis removes it structurally; augmentation becomes unnecessary).

Implementation: a value-equivariant variant of the candidate head and value
embedding: shared per-value weights + a value-axis attention/DeepSets pooling
so that permuting the value axis of the input permutes the output identically
(exact equivariance, the NeuroSAT-style literal symmetry). Constraint-graph
bias unchanged (it is value-agnostic already).

Protocol: train equivariant CoLT on 6x6 (frozen budget) and 9x9 (Phase-4
recipe), **without** augmentation. Measure first-pass poisoning and solve
accuracy on the standard and hard slices.

Pre-registered hypotheses:
- **H-E5-strong**: equivariant model without augmentation matches the
  augmented baseline (9x9 >= 90%, poisoning <= 5%). Then "augmentation is
  calibration" gets the stronger architectural form and the paper's design
  rule is updated: prefer equivariance to a 2,880x data factor.
- **H-E5-weak**: equivariance alone underperforms augmentation (e.g. because
  clue digits must still break the symmetry through context). Then the paper
  reports why exact equivariance is not free on clue-rich tasks: the
  interesting outcome either way.

Budget: 2 trainings = **~8 GPU-h** + ~1-2 days of implementation.
Artifacts: `colt/equivariant.py`, `results/equi6.json`, `results/equi9.json`.

---

## E6 — Union frames on top of the augmented 9x9 model `[GPU]`

**Referee minor** (the residual 3.9% at 9x9+aug: still poisoning? does
test-time union close it?).

Protocol: `scripts/h2_symmetry_frames.py --frames 1 4 8 --agg union` on the
augmented 9x9 checkpoint, plus `failure_anatomy.py` on its 7/180 failures.

Pre-registered expectation: the residual failures are still first-pass
poisonings (the contingency held everywhere so far) and union-K8 closes most
of them; if it does not, the residual is a *different* failure mode and gets
its own paragraph: that would be a genuinely new finding.

Budget: **< 1 GPU-h**. Artifacts: `results/h2_colt9aug_union.json`,
`results/anatomy9aug.json`.

---

## E7 (optional, raises the paper's tier) — Cross-architecture anatomy `[CODE, GPU]`

**Generality limitation** (all claims shown on one architecture family).

Protocol: replicate the two probes (H1 singleton/commit rate, first-pass
poisoning contingency) on one non-lattice reasoner reproduced from the same
ecosystem: TRM (preferred; smallest) on the same 6x6/9x9 data with a
candidate-extraction shim (treat its per-cell output distribution as the
propagator; threshold to candidate sets for the probe only).

Pre-registered hypotheses: if TRM shows the same one-shot amortization and a
poisoning-failure contingency, the anatomy is a property of the *training
objective class* (amortized full-solution supervision), not of the lattice;
the paper's title claim upgrades from "a sound neural reasoner" to the class.
If TRM differs, the contrast localizes the mechanism in the lattice/verifier
loop: also reportable.

Budget: **~30-40 GPU-h** + implementation. Decision: run only if E1-E6 leave
headroom in the compute grant.

---

## Execution order on GPU day 1

```
E4 (30 min, gates everything)  ->  E1 (4 h)  ->  E6 (1 h)  ->  E3 (8 h)
->  E2 (12 h)  ->  E5 / E7 as compute allows
```

Every run lands in `results/` and is folded into the paper by
`scripts/make_tables.py`; no number enters the text by hand.
