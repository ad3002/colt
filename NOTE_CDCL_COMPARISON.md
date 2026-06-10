# Note for the CoLT owner: should paper1 add a CDCL comparison?

*Context: discussion of `colt/paper/main.tex` (CoLT) and `colt/paper2/main.tex` (hybrid
follow-up), June 2026.*

## Recommendation in one line

Add a **classical-solver reference paragraph** (not an ablation arm), sharpen limitation
(iii), and fix the "backjumping" terminology. Do **not** implement CDCL inside CoLT — it is
structurally impossible with a neural propagator, and the exact-engine version is paper2.

## 1. Yes: add a classical reference point (cheap, defensive)

A paper titled "Conflict-Driven Search" that cites GRASP and Chaff will draw a SAT/CP
reviewer whose first question is "what does an actual solver do on these instances?" The
answer — everything solved in milliseconds — is already half-admitted for 3-COL in §7
(`paper/main.tex:632-634`) but never stated for Sudoku.

- Add one paragraph in §4 (protocol) or a footnote on Table 2: a classical exact solver
  (the repo's own MRV backtracker suffices) solves 100% of every test set in negligible
  time; all comparisons in the paper are internal to learned systems at matched neural
  budgets. The claims concern *learned amortization inside a sound envelope*, not
  competitiveness with classical solving.
- Keep it **out of the main ablation grid**. The grid's logic is matched-budget comparison,
  and "32 chains × 60 rounds" has no meaningful conversion to a classical solver's node
  count. A "minisat: 100%" row inside Table 2 invites the wrong reading.

This costs a paragraph and strengthens the paper: it shows the authors know where the
system sits and sets up paper2 as the constructive response.

## 2. No: don't add a CDCL arm to the ablation grid

**Structural reason.** True CDCL requires *explainable propagation*: every implied literal
needs an antecedent so conflict analysis can resolve back to a 1UIP cut and learn a lifted
clause. A transformer's eliminations have no antecedents and may simply be wrong (this is
exactly the first-pass poisoning of §5.3). The only sound nogoods a black-box propagator
admits are verifier-certified complete grids — i.e., precisely the leaf-level ban set CoLT
already has. "CDCL × policy" is not a missing ablation arm; it is an impossible one.

**Strategic reason.** An exact-propagation engine with real conflict analysis under a
learned heuristic *is* paper2. Pulling any of it into paper1 cannibalizes the companion and
muddies paper1's actual finding: search machinery of any sophistication is irrelevant when
the propagator either one-shots the puzzle or poisons it (the perfect solved/poisoned
contingency, §5.3).

**Suggested edit.** Upgrade limitation (iii) from "true CDCL conflict analysis is future
work" to the sharper claim: *leaf-level nogoods are the strongest sound nogoods a black-box
propagator admits; lifted nogoods require explainable propagation.* Cite lazy clause
generation (Ohrimenko, Stuckey & Codish 2009) as the technique that becomes available once
propagation is exact — this sentence is also the cleanest motivation for paper2.

## 3. Terminology fix: "backjumping" → "backtracking"

What §3.2 implements — pop to the deepest frame with untried values — is **chronological
backtracking with value exclusion**. "Backjumping" specifically means non-chronological
backtracking to the conflict's assertion level, which CoLT does not do. The paper says the
honest phrase once (`paper/main.tex:215-216`) but uses "backjump" in the §3.2 title,
Table 3's column header, and the search diagnostics. A SAT reviewer will flag every
instance. Rename throughout, or concede the loose usage in a footnote at first use.

## Background: how CoLT's machinery maps to real CDCL

| CDCL component | CoLT status |
|---|---|
| Conflict analysis (implication graph, 1UIP) | Absent — no antecedents from neural propagation |
| Learned clauses (lifted nogoods) | Leaf-level only: complete wrong grids, each pruning one leaf |
| VSIDS (conflict-driven, dynamic ordering) | Policy head is a static generalized MRV; no conflict feedback |
| Non-chronological backjumping | Chronological backtracking with value exclusion |
| Restarts retaining learned state | Restarts retain the ban set only |

The empirical irony: CoLT built search machinery for a solver that doesn't search
(singleton rate 1.0 — the grid is committed in one forward pass). CDCL is a technology for
large search trees under weak-but-sound propagation; CoLT has strong-but-occasionally-
unsound propagation and essentially no tree. Unsound propagation also destroys
completeness: a confident wrong elimination makes the true solution unreachable by every
chain, so exhaustion carries no information (confirmed by boundary arms B and E, §7).

## Concrete to-do for paper1 (~3 edits, no new experiments)

1. Classical-reference paragraph in §4 or footnote on Table 2.
2. Sharpened limitation (iii) + lazy clause generation citation in `refs.bib`.
3. Terminology pass replacing "backjump(ing)" with "backtracking with value exclusion"
   (or a first-use footnote).

## Pointer for paper2 (separate discussion)

Paper2's engine is also pre-GRASP search (chronological backtracking,
`paper2/main.tex:95-97`), but there AC propagation *is* explainable, so real 1UIP nogood
learning is implementable — lazy clause generation territory. That changes the right
supervision target too: predicting which variables appear in learned clauses is a learned
VSIDS prior (the NeuroCore recipe at CSP level), a third candidate alongside pin survival
and expected subtree size.
