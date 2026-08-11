#!/usr/bin/env python
"""Final tables: corrected (v2) runs for every lambda-dependent condition,
v1 runs for the lambda-independent ones (base / attn_only / ungated_adapter),
plus the v1-vs-v2 robustness check."""

from __future__ import annotations

import glob
import json

import numpy as np
from scipy import stats

SOURCES = {  # label -> (file glob, provenance)
    "base":             ("out_dex/dex_base_lr2e-5_s*.json", "v1 (lambda-independent)"),
    "attn_only":        ("out_dex/dex_attn_only_lr2e-5_s*.json", "v1 (lambda-independent)"),
    "ungated_adapter":  ("out_dex/dex_residual_adapter_lr2e-5_s*.json", "v1 (lambda == 1)"),
    "dex_minus":        ("out_dex/v2_dex_dex_minus_lr2e-5_s*.json", "v2"),
    "dex_plus":         ("out_dex/v2_dex_dex_plus_lr2e-5_s*.json", "v2"),
    "adapter_only":     ("out_dex/v2_dex_adapter_only_lr2e-5_s*.json", "v2"),
    "mirror_plus":      ("out_dex/v2_dexx_mirror_plus_lr2e-5_s*.json", "v2"),
    "fix_minus":        ("out_dex/v2_dexx_fix_minus_lr2e-5_s*.json", "v2"),
    "fix_plus":         ("out_dex/v2_dexx_fix_plus_lr2e-5_s*.json", "v2"),
    "fix_adapteronly":  ("out_dex/v2_dexx_fix_adapteronly_lr2e-5_s*.json", "v2"),
    "dex_minus@v1":     ("out_dex/dex_dex_minus_lr2e-5_s*.json", "v1 (shifted lambda_init)"),
    "dex_plus@v1":      ("out_dex/dex_dex_plus_lr2e-5_s*.json", "v1 (shifted lambda_init)"),
}

COMPARISONS = [
    ("A", "dex_minus", "dex_plus", "does the minus sign beat the plus sign?"),
    ("B", "dex_minus", "ungated_adapter", "vs an ungated fixed-scale adapter"),
    ("C", "dex_minus", "attn_only", "how much is just W_K/W_V/W_O finetuning?"),
    ("D", "dex_minus", "adapter_only", "is the adapter alone enough?"),
    ("E", "mirror_plus", "dex_minus", "NOISE FLOOR: an exact mirror of dex_minus"),
    ("F", "fix_minus", "fix_plus", "sign test with the branch pinned on"),
    ("G", "fix_minus", "dex_minus", "always-on lambda vs Eq. (4) annealing"),
    ("H", "fix_adapteronly", "base", "active differential module alone"),
    ("V", "dex_minus", "dex_minus@v1", "v2 vs v1: effect of the lambda_init fix"),
]


def load(pattern: str) -> dict:
    out = {}
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        if d.get("final"):
            out[int(d["args"]["seed"])] = d
    return out


def main() -> None:
    runs = {k: load(v[0]) for k, v in SOURCES.items()}
    print(f"{'condition':18s}{'src':26s}{'n':>3s}{'F1':>9s}{'std':>8s}{'valCE':>8s}"
          f"{'lam_end':>9s}{'|D|/|O|':>9s}  per-seed F1")
    for k, (_, src) in SOURCES.items():
        v = runs[k]
        if not v:
            continue
        seeds = sorted(v)
        f1 = [v[s]["final"]["qa"]["F1"] for s in seeds]
        ce = [v[s]["final"]["val_loss"] for s in seeds]
        fs = [(v[s].get("final_dex_stats") or {}) for s in seeds]
        lam = [x["lambda"] for x in fs if "lambda" in x]
        cr = [x["corr_ratio"] for x in fs if "corr_ratio" in x]
        std = np.std(f1, ddof=1) if len(f1) > 1 else 0.0
        g = lambda a: f"{np.mean(a):9.4f}" if a else "        -"
        print(f"{k:18s}{src:26s}{len(f1):3d}{np.mean(f1):9.4f}{std:8.4f}{np.mean(ce):8.4f}"
              f"{g(lam)}{g(cr)}  {[round(x, 4) for x in f1]}")
    print()
    for cid, a, b, why in COMPARISONS:
        A, B = runs.get(a, {}), runs.get(b, {})
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
        p_sd = (stats.ttest_rel(d, np.zeros_like(d)).pvalue
                if len(d) > 1 and not np.allclose(d, d[0]) else float("nan"))
        print(f"{cid}: {a} - {b}   ({why})")
        print(f"    seeds n={len(d)} meanD={d.mean():+.4f} p={p_sd:.3f} | "
              f"examples n={len(dex)} meanD={dex.mean():+.4f} "
              f"CI95[{lo:+.4f},{hi:+.4f}] p={p_ex:.4f} dz={dex.mean()/dex.std(ddof=1):+.3f}")


if __name__ == "__main__":
    main()
