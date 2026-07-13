# Prospectively specified revision experiments (paper 1, review round 1)

Frozen **2026-07-11**, before any of the runs below were executed and before any
GPU was provisioned. Each experiment answers a specific referee point, states
its hypotheses and decision rules **in advance**, and names the artifact it
will produce. Text changes contingent on outcomes are written down here so the
revision cannot quietly move the goalposts.

Status legend: `[CPU-done]` already executed locally (no GPU needed),
`[GPU]` waiting on hardware, `[CODE]` needs implementation first.

Hardware assumption: one 24 GB-class GPU (RTX 4090 or better). Total GPU
budget for E1–E6 + E8 + E9: **~45 GPU-hours** (incl. the E2
batch-512-effective amendment and the E3 five-seed/dataset-seed amendment).
Optional E7 adds ~30–40.

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

Pre-specified interpretation:
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

Hypotheses and decision rules (pre-specified):
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

Pre-specified expectations: CoLT+aug ~ 96%, CoLT-no-aug ~ 0% (already
measured); LDT+aug and LDT-no-aug are the new cells. If LDT+aug >= 80%, the
"structure is what makes augmentation affordable" claim is **falsified at
9x9** and must be retracted to a 6x6-only observation. If LDT+aug <= 10%, the
interaction generalizes across board sizes and the claim strengthens.

**Amendment (2026-07-11, before any run; prompted by re-reading referee point
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

**Amendment (2026-07-11, before any run; prompted by review 2):**
- central claims (9x9 +/- aug, the 2x2 interaction) extended to **5 training
  seeds** (42-46); peripheral cells stay at 3.
- **dataset-generation seeds**: regenerate `sudoku6` and `sudoku9` with
  builder seeds 43 and 44 (leakage-audited the same way) and retrain the
  headline arms once per dataset seed, so the conclusions are not specific to
  one generated corpus.
- arm comparisons on identical puzzles are reported with **paired McNemar
  tests**, and the poisoning-accuracy contingency gains a **bootstrap CI**
  over puzzles.
- all seeds are reported, including failures.

Pre-specified decision rules:
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

## E4 — Single-environment reconciliation of the anatomy/H2 probe discrepancy `[CPU-done]`

> **Result (2026-07-11):** decision rule 1 fired. Both probe code paths are
> **bitwise identical** (logits and masks) in one environment; the contingency
> replicates exactly (43/43 failed-poisoned vs 0/137, accuracy 76.1%);
> poisoned count invariant at theta in {0.10, 0.05, 0.02}; zero status flips
> within +/-0.02 of theta; solution-value probabilities bimodal (6,242/6,480
> values at margin >= 0.5, exactly one value within 0.02). Per review 4, the
> paper reports the source of the historical 42-vs-46 gap as unresolved
> cross-environment or checkpoint sensitivity: the original checkpoint was
> unavailable, so E4 rules out a code-path bug and replicates the contingency
> but cannot attribute the historical difference.
> Artifact: `results/reconcile_anatomy_h2.json`. Paper updated (S5.3, S5.4
> footnote, Reproducibility, claim-audit row).

> **Execution note (2026-07-11, before results were inspected):** the original
> `runs/colt6_seed42/final.pt` was lost with the rented pod, so E4 runs on a
> same-recipe, same-seed, same-data CPU retrain (`runs/colt6_seed42_cpu`,
> `runs_colt6_cpu_retrain.log`), executed by `scripts/reconcile_probes.py`
> (both probe code paths verbatim, back to back, one process). Deviation from
> the frozen text: checkpoint identity; everything else per protocol. E4 is
> CPU-feasible (inference only), so the `[GPU]` tag was pessimistic.

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

Pre-specified decision rules:
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

Pre-specified hypotheses:
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

Pre-specified expectation: the residual failures are still first-pass
poisonings (the contingency held everywhere so far) and union-K8 closes most
of them; if it does not, the residual is a *different* failure mode and gets
its own paragraph: that would be a genuinely new finding.

Budget: **< 1 GPU-h**. Artifacts: `results/h2_colt9aug_union.json`,
`results/anatomy9aug.json`.

---

## E8 — Training-side component ablation `[CPU-done, 3 seeds]` *(added 2026-07-11 per review 2, before any run)*

> **Final result, seeds 42/43/44 (2026-07-12):** decision rule **H-E8-graph
> fires on every seed** — gap fraction of arm B (graph bias only) = 0.994 /
> 1.006 / 1.000, mean 1.000; arms A (positional tables) and C (coord MLP
> only) score 0.0% on both slices on all seeds. Std means: B 99.4, D 99.6,
> E 99.3, F 99.4; hard means: B 78.1, D 78.5, E 73.9, F 75.9. Probe curves
> bimodal on the graph-bias axis on every seed (1.0 by step 1000 iff graph
> bias present). The seed-42 hint that no-policy arms beat F on hard (+4.5pp)
> did NOT replicate (seed 44 reverses it); across seeds the graph-bias arms
> are indistinguishable on hard. Notable: arm A collapses below the historical
> ltd-package baseline (49.4%), so attribution is within-codebase. Artifacts:
> `results/ablate6_*_seed4{2,3,4}_*.json`, `results/ablate6_summary.json`.
> Paper updated (abstract, S5.1 Table tab:e8, 3-seed mean(min-max)).

> **Execution note (2026-07-11, queued before any arm finished):** running on
> CPU via `scripts/run_e8_cpu.sh` (6 arms x 3 seeds, waves of 3 at 4 threads;
> ~45 min/arm), evaluated with `dfs x learned` where the policy loss is on and
> `dfs x random` otherwise, per the frozen table. One protocol detail the
> frozen text left open: arms with `lambda_policy=0` also set
> `train.pool_policy=random`, since an untrained policy head branching the
> training pool is neither the LDT control nor CoLT; this is recorded here
> before results. Summary artifact: `results/ablate6_summary.json`
> (`scripts/summarize_e8.py`), decision rules unchanged.

**Review-2 point 1** (the paper's strongest causal claim, "the training-side
delta closed the gap", bundles three components: graph bias, coordinate MLP,
policy loss; curriculum/data/optimizer are matched by construction).

Protocol: 6x6, frozen budget and data, 3 seeds each:

| arm | graph bias | coord MLP | policy loss |
|---|---|---|---|
| A LDT baseline (positional tables) | no | no | no |
| B + graph bias only | yes | no | no |
| C + coord MLP only | no | yes | no |
| D CoLT minus policy loss | yes | yes | no |
| E CoLT minus coord MLP | yes | no | yes |
| F CoLT full | yes | yes | yes |

Each arm evaluated on the standard and hard slices (dfs x learned where the
policy head exists, dfs x random otherwise) plus the held-out probe curve.

Pre-specified hypotheses and decision rules:
- **H-E8-graph**: B accounts for >= 80% of the A->F gap (paper keeps
  "constraint-graph conditioning carries the gain", now measured).
- **H-E8-additive**: if no single component reaches 50% of the gap, the
  claim is rewritten as "the gain is joint; no single component suffices",
  which is itself a finding about why the architecture works.
- Policy-loss arms (D vs F) double as a *training-side* test of the policy
  head, separate from its inference-side inertness.

Budget: 6 arms x 3 seeds x ~10 min = **~3 GPU-h**.
Artifacts: `results/ablate6_{A..F}_seed*.json`.

---

## E9 — External corpus evaluation `[GPU]` *(added 2026-07-11 per review 2, before any run)*

**Review-2 point 6** (external validity: all data is our generator; LDT's
own corpus is Sudoku-Extreme).

Protocol: evaluate, without retraining, the augmented 9x9 checkpoint (and the
E3 seed checkpoints) on an external 9x9 corpus: the Sudoku-Extreme test set
if obtainable, otherwise the RRN/Kaggle hardest-17-clue split, cleanly
labeled as out-of-distribution (clue count 17-24 vs our 25). Report accuracy,
soundness, abstention, first-pass poisoning, and the classical-reference
timing on the same instances. Optionally one fine-tuning arm (<= 2k steps) to
separate distribution shift from capability.

Pre-specified framing (written before seeing any number): this is an OOD
generalization measurement, not a comparison with published LDT numbers; the
paper's conclusions stay restricted to generated CSPs unless the external
numbers are strong, in which case the restriction is relaxed to "and
transfers to the external corpus at X%".

Budget: eval ~0.5 GPU-h; optional fine-tune ~2 GPU-h.
Artifacts: `results/external9_{eval,finetune}.json`.

---

## E7 (optional, raises the paper's tier) — Cross-architecture anatomy `[CODE, GPU]`

**Generality limitation** (all claims shown on one architecture family).

Protocol: replicate the two probes (H1 singleton/commit rate, first-pass
poisoning contingency) on one non-lattice reasoner reproduced from the same
ecosystem: TRM (preferred; smallest) on the same 6x6/9x9 data with a
candidate-extraction shim (treat its per-cell output distribution as the
propagator; threshold to candidate sets for the probe only).

Pre-specified hypotheses: if TRM shows the same one-shot amortization and a
poisoning-failure contingency, the anatomy is a property of the *training
objective class* (amortized full-solution supervision), not of the lattice;
the paper's title claim upgrades from "a sound neural reasoner" to the class.
If TRM differs, the contrast localizes the mechanism in the lattice/verifier
loop: also reportable.

Budget: **~30-40 GPU-h** + implementation. Decision: run only if E1-E6 leave
headroom in the compute grant.

---

## Execution order on GPU days 1–2

```
E4 (30 min, gates everything)  ->  E8 (3 h, gates the paper's main causal
claim)  ->  E1 (4 h)  ->  E6 + E9-eval (1.5 h)  ->  E3 (15 h with the
amendment)  ->  E2 (16 h)  ->  E5 / E9-finetune / E7 as compute allows
```

Every run lands in `results/` and is folded into the paper by
`scripts/make_tables.py`; no number enters the text by hand.
