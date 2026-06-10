"""Sudoku task for CoLT: geometry, constraint graph, verifier, TSV loader.

Extends the LDT-repo task module (ad3002/LTD) with the two pieces CoLT's
size-agnostic model needs:

  * :func:`relation_ids` — the (S, S) constraint-relation matrix that drives the
    relational attention bias (S = 1 CLS + C cells).
  * :func:`coord_features` — per-cell *normalized* coordinate features for the
    content MLP (size-agnostic replacement for learned positional tables).

TSV schema is identical to ad3002/LTD (`puzzle_id  clues  solution  n_clues
n_solutions`, human 1-indexed; loader converts to 0-indexed, -1 = blank), so
the frozen datasets are shared byte-for-byte between the two repos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Relation ids (see colt.blocks docstring).
REL_CLS_CELL = 8
REL_CLS_CLS = 9


def factor_box(n: int) -> tuple[int, int]:
    best = (1, n)
    for br in range(1, int(n**0.5) + 1):
        if n % br == 0:
            best = (br, n // br)
    return best


@dataclass
class SudokuGeometry:
    n: int
    box_rows: int
    box_cols: int

    @property
    def n_cells(self) -> int:
        return self.n * self.n

    @property
    def n_cand(self) -> int:
        return self.n

    def coords(self) -> torch.Tensor:
        rc = [(i // self.n, i % self.n) for i in range(self.n_cells)]
        return torch.tensor(rc, dtype=torch.long)

    def box_ids(self) -> torch.Tensor:
        n, br, bc = self.n, self.box_rows, self.box_cols
        ids = []
        for i in range(self.n_cells):
            r, c = i // n, i % n
            ids.append((r // br) * (n // bc) + (c // bc))
        return torch.tensor(ids, dtype=torch.long)


def geometry_for(n: int, box_rows: int | None = None, box_cols: int | None = None) -> SudokuGeometry:
    if box_rows is None or box_cols is None:
        box_rows, box_cols = factor_box(n)
    if box_rows * box_cols != n:
        raise ValueError(f"box_rows*box_cols ({box_rows}*{box_cols}) must equal n ({n})")
    return SudokuGeometry(n=n, box_rows=box_rows, box_cols=box_cols)


# ----------------------------------------------------------------------------
# Constraint graph → relation ids + coordinate features
# ----------------------------------------------------------------------------


def relation_ids(geom: SudokuGeometry) -> torch.Tensor:
    """(S, S) long relation-id matrix, S = 1 + n_cells (CLS first).

    Cell-cell ids bit-pack the shared constraint groups:
    bit0 same-row | bit1 same-col | bit2 same-box  (id 7 ⇔ self).
    """
    C = geom.n_cells
    coords = geom.coords()
    box = geom.box_ids()
    row = coords[:, 0]
    col = coords[:, 1]
    same_row = (row.unsqueeze(0) == row.unsqueeze(1)).long()
    same_col = (col.unsqueeze(0) == col.unsqueeze(1)).long()
    same_box = (box.unsqueeze(0) == box.unsqueeze(1)).long()
    cell_rel = same_row + 2 * same_col + 4 * same_box        # (C, C) in 0..7

    S = C + 1
    rel = torch.full((S, S), REL_CLS_CELL, dtype=torch.long)
    rel[1:, 1:] = cell_rel
    rel[0, 0] = REL_CLS_CLS
    return rel


def coord_features(geom: SudokuGeometry) -> torch.Tensor:
    """(C, 5) float: [r/n, c/n, (r % br)/br, (c % bc)/bc, 1/n].

    Normalized so the same MLP serves every board size; the within-box offsets
    let the model distinguish box-local roles, and 1/n exposes the scale.
    """
    n, br, bc = geom.n, geom.box_rows, geom.box_cols
    feats = []
    for i in range(geom.n_cells):
        r, c = i // n, i % n
        feats.append([r / n, c / n, (r % br) / br, (c % bc) / bc, 1.0 / n])
    return torch.tensor(feats, dtype=torch.float32)


@dataclass
class TaskContext:
    """Everything size-specific the size-agnostic model needs at forward time."""

    name: str
    geom: SudokuGeometry
    rel_ids: torch.Tensor       # (S, S) long
    coord_feats: torch.Tensor   # (C, 5) float
    n_cand: int

    def to(self, device: torch.device) -> "TaskContext":
        return TaskContext(self.name, self.geom, self.rel_ids.to(device),
                           self.coord_feats.to(device), self.n_cand)


def context_for(geom: SudokuGeometry) -> TaskContext:
    return TaskContext(
        name=f"sudoku{geom.n}",
        geom=geom,
        rel_ids=relation_ids(geom),
        coord_feats=coord_features(geom),
        n_cand=geom.n_cand,
    )


# ----------------------------------------------------------------------------
# Batched constraint verifier (soundness check) — same as ad3002/LTD
# ----------------------------------------------------------------------------


def _group_ok(board_oh: torch.Tensor, group_id: torch.Tensor, n_groups: int, V: int) -> torch.Tensor:
    n = board_oh.shape[0]
    counts = torch.zeros(n, n_groups, V, device=board_oh.device, dtype=board_oh.dtype)
    idx = group_id.view(1, -1, 1).expand(n, -1, V)
    counts.scatter_add_(1, idx, board_oh)
    return (counts == 1).all(dim=2).all(dim=1)


def satisfies_constraints(grids: torch.Tensor, geom: SudokuGeometry) -> torch.Tensor:
    """(n, C) long grids (values in [0, N), -1 = blank) → (n,) bool."""
    n_grids, C = grids.shape
    V = geom.n_cand
    device = grids.device
    filled = (grids >= 0).all(dim=1)
    safe = grids.clamp_min(0)
    board_oh = torch.zeros(n_grids, C, V, device=device).scatter_(2, safe.unsqueeze(-1), 1.0)
    coords = geom.coords().to(device)
    ok = (
        _group_ok(board_oh, coords[:, 0], geom.n, V)
        & _group_ok(board_oh, coords[:, 1], geom.n, V)
        & _group_ok(board_oh, geom.box_ids().to(device), geom.n, V)
    )
    return ok & filled


# ----------------------------------------------------------------------------
# Dataset loader — TSV format shared with ad3002/LTD
# ----------------------------------------------------------------------------


@dataclass
class SudokuItem:
    puzzle_id: int
    clues: torch.Tensor      # (C,) long, -1 blank else value in [0, N)
    solution: torch.Tensor   # (C,) long


def _parse_ints(s: str) -> list[int]:
    return [int(t) for t in s.split(" ") if t]


class SudokuDataset(Dataset):
    EXPECTED_HEADER = ["puzzle_id", "clues", "solution", "n_clues", "n_solutions"]

    def __init__(self, tsv_path: str | Path, geom: SudokuGeometry):
        self.tsv_path = Path(tsv_path)
        if not self.tsv_path.exists():
            raise FileNotFoundError(self.tsv_path)
        self.geom = geom
        C = geom.n_cells
        self._items: list[SudokuItem] = []
        with self.tsv_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            if header != self.EXPECTED_HEADER:
                raise ValueError(f"unexpected TSV header in {self.tsv_path}: {header}")
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    raise ValueError(f"malformed row in {self.tsv_path}: {fields!r}")
                clues_raw = _parse_ints(fields[1])
                sol_raw = _parse_ints(fields[2])
                if len(clues_raw) != C or len(sol_raw) != C:
                    raise ValueError(f"cell-count mismatch in {self.tsv_path}")
                clues = torch.tensor([(v - 1) if v > 0 else -1 for v in clues_raw], dtype=torch.long)
                solution = torch.tensor([v - 1 for v in sol_raw], dtype=torch.long)
                self._items.append(SudokuItem(int(fields[0]), clues, solution))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> SudokuItem:
        return self._items[idx]

    def clues_tensor(self) -> torch.Tensor:
        return torch.stack([it.clues for it in self._items])

    def solution_tensor(self) -> torch.Tensor:
        return torch.stack([it.solution for it in self._items])
