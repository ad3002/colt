"""Tasks for CoLT."""

from .sudoku import (
    SudokuDataset,
    SudokuGeometry,
    TaskContext,
    context_for,
    coord_features,
    factor_box,
    geometry_for,
    relation_ids,
    satisfies_constraints,
)

__all__ = [
    "SudokuDataset",
    "SudokuGeometry",
    "TaskContext",
    "context_for",
    "coord_features",
    "factor_box",
    "geometry_for",
    "relation_ids",
    "satisfies_constraints",
]
