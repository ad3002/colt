#!/usr/bin/env python3
"""Trimmed tail of boundary_cross: arms C (train scale), D (K=48 supervision),
E (no-learning stub control) and F (elimination diagnostics), all at the base
budget 16x40. The slow 32x240 variants of G and E are dropped: arm B already
measured the budget axis (1.9% at 12x budget) and arm G at base budget showed
the CLS-calibration hypothesis does not explain the cliff. Results are merged
into results/boundary_cross.json next to the A/B/G arms.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "bc", Path(__file__).resolve().parent / "boundary_cross.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-instances", type=int, default=700)
    ap.add_argument("--out", type=Path, default=Path("results/boundary_cross.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(args.seed + 40)
    print("[gen] regenerating the identical seeded instance set", file=sys.stderr)
    inst = bc.gen_instances(bc.DENSITY, args.n_instances, 48, rng)
    n_train = int(0.85 * len(inst))
    train_i, test_i = inst[:n_train], inst[n_train:]

    out = json.loads(args.out.read_text()) if args.out.exists() else {
        "density": bc.DENSITY, "n_train": len(train_i), "n_test": len(test_i), "arms": {}}

    def record(name, val):
        out["arms"][name] = val
        print(json.dumps({name: val}), file=sys.stderr)
        args.out.write_text(json.dumps(out, indent=2) + "\n")

    t0 = time.time()
    print("[C] scale d128/12k/batch128", file=sys.stderr)
    model_c, loss_c = bc.train_model(train_i, 12, 128, 12000, 128, args.seed, device)
    record("C_scale_16x40", {**bc.evaluate(model_c, test_i, 16, 40, args.seed, device),
                             "final_loss": round(loss_c, 4)})
    record("F_diag_C", bc.elim_profile(model_c, test_i, 12, args.seed, device))

    print("[D] supervision K=48", file=sys.stderr)
    model_d, loss_d = bc.train_model(train_i, 48, 64, 4000, 64, args.seed, device)
    record("D_K48_16x40", {**bc.evaluate(model_d, test_i, 16, 40, args.seed, device),
                           "final_loss": round(loss_d, 4)})

    print("[E] stub (no learning), base budget", file=sys.stderr)
    record("E_stub_16x40", bc.evaluate(bc.StubPropagator(), test_i, 16, 40, args.seed, device))

    print("[F] diag for a fresh arm-A model", file=sys.stderr)
    model_a, _ = bc.train_model(train_i, 12, 64, 4000, 64, args.seed, device)
    record("F_diag_A", bc.elim_profile(model_a, test_i, 12, args.seed, device))

    out["tail_wall_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
