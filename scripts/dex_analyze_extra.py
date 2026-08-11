#!/usr/bin/env python
"""Follow-up conditions: mirror-init noise floor, sign test with lambda pinned on,
annealed-vs-always-on lambda, and the differential module alone."""

from __future__ import annotations

import glob
import json

import numpy as np
from scipy import stats

CONDS = {
    "dex_minus(anneal)": "out_dex/dex_dex_minus_lr2e-5_s*.json",
    "dex_plus(anneal)": "out_dex/dex_dex_plus_lr2e-5_s*.json",
    "mirror_plus(-W_D)": "out_dex/dexx_mirror_plus_lr2e-5_s*.json",
    "fix_minus(lam on)": "out_dex/dexx_fix_minus_lr2e-5_s*.json",
    "fix_plus(lam on)": "out_dex/dexx_fix_plus_lr2e-5_s*.json",
    "fix_adapteronly": "out_dex/dexx_fix_adapteronly_lr2e-5_s*.json",
    "base": "out_dex/dex_base_lr2e-5_s*.json",
}
PAIRS = [
    ("E", "mirror_plus(-W_D)", "dex_minus(anneal)", "noise floor: a run that is the exact mirror of dex_minus"),
    ("F", "fix_minus(lam on)", "fix_plus(lam on)", "sign test with the differential branch pinned on"),
    ("G", "fix_minus(lam on)", "dex_minus(anneal)", "always-on lambda vs the paper's Eq. (4) annealing"),
    ("H", "fix_adapteronly", "base", "active differential module alone, attention frozen"),
]


def load(pattern: str) -> dict:
    out = {}
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        if d.get("final"):
            out[int(d["args"]["seed"])] = d
    return out


def main() -> None:
    conds = {k: load(v) for k, v in CONDS.items()}
    print(f"{'condition':20s}{'n':>4s}{'F1':>9s}{'std':>8s}{'valCE':>8s}{'lam_end':>9s}{'|dO|/|O|':>10s}  per-seed F1")
    for k, v in conds.items():
        if not v:
            continue
        seeds = sorted(v)
        f1 = [v[s]["final"]["qa"]["F1"] for s in seeds]
        ce = [v[s]["final"]["val_loss"] for s in seeds]
        fs = [(v[s].get("final_dex_stats") or {}) for s in seeds]
        lam = [x["lambda"] for x in fs if "lambda" in x]
        cr = [x["corr_ratio"] for x in fs if "corr_ratio" in x]
        std = np.std(f1, ddof=1) if len(f1) > 1 else 0.0
        print(f"{k:20s}{len(f1):4d}{np.mean(f1):9.4f}{std:8.4f}{np.mean(ce):8.4f}"
              f"{(np.mean(lam) if lam else float('nan')):9.4f}{(np.mean(cr) if cr else float('nan')):10.4f}"
              f"  {[round(x, 4) for x in f1]}")
    print()
    for cid, a, b, why in PAIRS:
        A, B = conds[a], conds[b]
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        da = np.array([A[s]["final"]["qa"]["F1"] for s in shared])
        db = np.array([B[s]["final"]["qa"]["F1"] for s in shared])
        d = da - db
        ea = np.array([[r["f1"] for r in A[s]["final"]["qa"]["per_example"]] for s in shared]).mean(0)
        eb = np.array([[r["f1"] for r in B[s]["final"]["qa"]["per_example"]] for s in shared]).mean(0)
        dex = ea - eb
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(dex), size=(10000, len(dex)))
        lo, hi = np.percentile(dex[idx].mean(1), [2.5, 97.5])
        p_ex = stats.ttest_rel(dex, np.zeros_like(dex)).pvalue
        p_seed = (stats.ttest_rel(d, np.zeros_like(d)).pvalue
                  if len(d) > 1 and not np.allclose(d, d[0]) else float("nan"))
        print(f"{cid}: {a} − {b}  ({why})")
        print(f"    seed-level n={len(d)} meanD={d.mean():+.4f} p={p_seed:.3f} | "
              f"example-level n={len(dex)} meanD={dex.mean():+.4f} CI95[{lo:+.4f},{hi:+.4f}] "
              f"p={p_ex:.4f} dz={dex.mean() / dex.std(ddof=1):+.3f}")


if __name__ == "__main__":
    main()
