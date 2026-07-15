# Verification audit

What was independently checked, by what, and where the evidence lives.
Context: the pipeline is agent-written (see the paper's AI-Assisted Research
Statement); this file targets the residual risk that a single shared bug in
the verifier or table generation could silently corrupt results.

| what | how | artifact |
|---|---|---|
| Soundness gate (exact verifier) | Second verifier written from the rules alone, importing nothing from `colt/` (`scripts/independent_verifier.py`); cross-agreement with `colt.tasks.sudoku.satisfies_constraints` on 25,231 grids: every dataset solution across all splits (accepted), single-cell corruptions (rejected), row-permuted grids, and 10,000 uniformly random 6x6 grids (all rejected). 0 disagreements. | `results/verifier_audit.json` |
| Verifier unit properties | Committed unit tests (constraint-violation taxonomy: row/col/box duplicates, blanks; batch semantics) | `tests/test_colt.py` |
| Emitted-grid re-verification | Every emitted grid passes an independent re-check at evaluation time inside `colt/eval.py` (`answered_match_known_solution` cross-checks against the known solution where unique) | eval JSONs, field `soundness` |
| Dataset integrity | MD5 manifests per split; construction-time exact uniqueness check; leakage audit up to the full symmetry group | `data/*/MANIFEST.md5`, `results/leakage_audit.json` |
| Table generation | Tables regenerate from committed raw JSONs by script; no hand transcription | `scripts/make_tables.py`, `results/` |
| Probe code paths | The two first-pass poisoning probes verified bitwise-identical (logits and masks) on the canonical checkpoint | `results/reconcile_anatomy_h2.json` |
| Cross-hardware replication | GPU-era runs replicate canonical counts (std within 2/180, hard within 1/180, dichotomy exact per environment); independent 9x9 retrain reproduced 173/180 | committed GPU-era JSONs |

Audited at monorepo commit: see `git log --oneline -1 -- AUDIT.md`.
Auditor: the research agents (automated checks above are reproducible by
`python scripts/independent_verifier.py --out /tmp/verifier_audit.json`);
the human author reviewed the reports, not the code line-by-line.
