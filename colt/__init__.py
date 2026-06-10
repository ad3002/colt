"""CoLT — Conflict-driven Lattice Transformer.

A sound neural reasoner that puts *learned search* (policy + value heads,
DFS backjumping, nogood memory, constraint-graph conditioning) inside the
sound-by-construction lattice envelope of the Lattice Deduction Transformer
(arXiv:2605.08605), borrowing the learned-stochastic-guidance perspective of
GRAM (arXiv:2605.19376).

This is new research code by the authors of the ad3002/gram and ad3002/LTD
reimplementations; see DESIGN.md for the architecture rationale.
"""

__version__ = "0.1.0"
