# CoLT — Conflict-driven Lattice Transformer: design document

*Phase 0 artifact. This is the architecture rationale; the frozen experimental
protocol lives in [`BENCHMARKS.md`](BENCHMARKS.md); the GPU plan in
[`PHASE4.md`](PHASE4.md); the write-up in [`paper/`](paper/).*

## 1. Position

Two recent architectures attack small-model reasoning from opposite ends:

- **GRAM** (arXiv:2605.19376) wraps a deterministic recursive reasoner
  (HRM/TRM lineage) in a *probabilistic shell*: a learned stochastic guidance
  term, trained variationally, makes the latent trajectory explore; auxiliary
  heads (halt, LPRM) rank trajectories. GRAM learns *how to propose* — but it
  is blindly generative: nothing prevents a confidently wrong answer.
- **LDT** (arXiv:2605.08605) keeps an explicit *lattice of candidate sets* and
  learns sound deduction: monotone elimination, conflict (⊥) detection,
  branching, with the same search loop at train and inference. A grid is only
  emitted after passing an external constraint check, so LDT *answers
  correctly or abstains* — but its search is blind: branch cells are chosen
  uniformly at random, a conflict throws away the whole trajectory (restart
  from scratch), and parallel chains share nothing.

We reimplemented both (ad3002/gram, ad3002/LTD) and measured the failure modes
directly. The decisive empirical observation (LTD repo, `results/`): on the
puzzles it failed, the LDT-style solver **re-generated the same wrong
completions thousands of times** (35,200 suppressed wrong grids concentrated
on 11/300 puzzles at 4×4; 148,567 on 6×6). The soundness guard fires
correctly — but all that search compute is wasted re-discovering known dead
ends.

## 2. The core idea: learned search inside a sound envelope

LDT's soundness is enforced *at emission* (verify-or-abstain), **not** by any
property of the search policy. Therefore any search policy — however
aggressive, learned, or heuristic — preserves the guarantee. This licenses
putting the classical playbook of intelligent search (CDCL: conflict-driven
learning, backjumping, variable-ordering heuristics) *inside* the envelope,
with neural components in the roles classical solvers fill with hand-coded
logic:

| Classical CSP/SAT solver | CoLT |
|---|---|
| unit propagation | recurrent transformer over the lattice (from LDT) |
| variable-ordering heuristic (e.g. MRV/VSIDS) | **policy head** π (learned, §3.1) |
| value-ordering heuristic | branch distribution softmax(b/τ) (from LDT) |
| conflict ⇒ backtrack/backjump | **decision-stack DFS with value exclusion** (§3.3) |
| learned clauses (nogoods) | **per-puzzle ban set of failed completions** (§3.3) |
| restarts | stack-exhaustion restart with sampled orderings |
| solver state heuristics | **value function** V(x) = 1 − σ(conflict head) (§3.2) |

GRAM's contribution to the synthesis is the *perspective*: search decisions are
a learned, trainable distribution, not a fixed heuristic — our policy head is
the (simpler, supervised) analogue of GRAM's guidance for the one decision
that matters in lattice search: *where to commit next*.

## 3. Components

### 3.1 Branch-policy head (replaces uniform-random cell choice)

Per cell, a scalar logit π_i trained with dense supervision available for free
from the α-operator machinery: the **branch-survival probability**

    p*_i = Σ_{d : ŷ_i[d]=1} softmax(b_i/τ over alive candidates)[d]

— the probability that committing a value sampled from the *actual* branch
distribution at cell i keeps a known solution alive. Loss: BCE(σ(π_i), p*_i)
over multi-candidate cells in non-⊥ states, every unroll iteration. At
inference, branch at the (sampled-)argmax of π over multi-candidate cells.

This strictly generalizes MRV: fewer alive candidates ⇒ more probability mass
per candidate ⇒ higher p*, but the head also sees constraint structure through
the network. **MRV is kept as a hand-coded control arm** — the learned policy
must beat it to justify itself (BENCHMARKS.md).

### 3.2 Value function (reused, repurposed)

LDT already trains the CLS conflict head on 1[ŷ = ⊥]; V(x) = 1 − σ(c) is a
calibrated "is this state still consistent?" estimate. CoLT uses it as-is for
conflict detection and (Phase 4+) for fringe prioritization / budget
allocation across chains.

### 3.3 DFS with backjumping + nogood ban set (replaces restart-on-conflict)

Each chain keeps a stack of decision frames (pre-pin state, cell, branch
distribution, tried-value mask). On conflict: pop to the deepest decision with
untried alive values, exclude the failed value, re-pin, resume — chronological
backtracking with value exclusion. A completed grid that fails verification is
recorded in a **per-puzzle ban set shared across all chains**: any chain that
re-derives it treats it as an immediate conflict (prune) instead of
re-verifying/re-counting. Stack exhaustion ⇒ fresh restart (diversity through
sampled value orderings).

Soundness note: a "conflict" here is *neural evidence*, not proof (an
elimination can be wrong), so value exclusion is heuristic pruning — and that
is fine, because emission is still verify-gated. The exact-ban-set component
*is* exact: a grid that failed the verifier is provably not a solution.

### 3.4 Constraint-graph conditioning (one checkpoint, many boards)

All size-specific parameters are removed:

- structure enters as a **relational attention bias**: per-head learned scalar
  per relation id, rel(i,j) = bit-packed {same row, same column, same box}
  (+ CLS ids) — the Graphormer pattern over the CSP constraint graph;
- cell content gets an MLP over *normalized* coordinates (r/n, c/n, in-box
  offsets, 1/n);
- the value dimension is padded to a fixed v_max = 9 (padded slots are
  permanently dead candidates — semantically consistent with the lattice).

One set of weights therefore serves 4×4, 6×6, 9×9, and any box shape; training
keeps one solve-pool per task and round-robins optimizer steps. This converts
"one model per benchmark" (HRM/TRM/GRAM/LDT all retrain per task) into "one
reasoner, task given by its constraint graph".

### 3.5 Training

Identical to our LDT reproduction otherwise: pool-based parallel solve
(Algorithm 1), abstraction-operator targets, asymmetric BCE (w⁺=4, w⁻=0.5),
conflict BCE, singleton CE, plus the policy BCE (λ_pol = 0.25); curriculum
over deduction depth (partial-solution pool seeding annealed to clues-only —
the bootstrap fix discovered and documented in ad3002/LTD). The training pool
*branches with the learned policy* as it trains, keeping training states
in-distribution for the search actually run at inference.

## 4. What would falsify the design

1. **Policy ≤ MRV** at equal budget ⇒ the learned ordering is not extracting
   anything beyond candidate counts (kill §3.1, keep MRV).
2. **DFS+nogoods ≤ restart** at equal budget ⇒ neural propagation is too noisy
   for stack-based search to pay (kill §3.3).
3. **Multi-size ≤ single-size** on each size at matched per-task steps ⇒ the
   shared-weights claim costs accuracy (scale model or revert).

Each is a single row in the frozen ablation grid (BENCHMARKS.md): search
∈ {restart, dfs} × policy ∈ {random, mrv, learned}, all from one checkpoint.

## 5. Out of scope (this cycle), tracked for the paper's future-work

- Learned conflict *analysis* (which decision to blame — true CDCL backjumps;
  we do chronological).
- Generalized (lifted) nogoods beyond exact failed grids.
- Value-guided fringe scheduling across chains (best-first).
- GRAM-style latent ε for chain diversity.
- Non-Sudoku CSPs (random binary CSPs, graph coloring) via the same
  relational-bias interface; Maze/ARC.
