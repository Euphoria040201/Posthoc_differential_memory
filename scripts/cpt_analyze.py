#!/usr/bin/env python
"""Analyse the CPT matrix from `*_eval.json` artifacts and emit the report tables.

Every number in the final report comes from here; nothing is transcribed.

Statistics
----------
* Arms are compared PAIRED on the same val windows (same order, same file,
  sha256 recorded), so the between-window variance that dominates raw NLL is
  differenced out.
* CIs are percentile bootstrap over val windows, and — where >1 seed exists —
  a hierarchical bootstrap that resamples seeds as well, because a CI computed
  over windows alone understates the uncertainty a reader cares about.
* The seed-noise floor is the within-arm spread across seeds.  Any effect
  smaller than it is reported as indistinguishable from noise, not as a result.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_evals(d: Path):
    runs = []
    for p in sorted(d.glob("*_eval.json")):
        r = json.loads(p.read_text())
        if r.get("tag", "").startswith("smoke"):
            continue
        r["_path"] = str(p.relative_to(REPO))
        runs.append(r)
    return runs


def boot_mean_diff(a, b, n=20000, seed=0):
    """Percentile bootstrap on the PAIRED per-window difference a-b."""
    a, b = np.asarray(a), np.asarray(b)
    n_min = min(len(a), len(b))
    d = a[:n_min] - b[:n_min]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return {"delta": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
            "n_windows": int(len(d))}


def hier_boot_diff(arm_a_runs, arm_b_runs, n=20000, seed=0):
    """Resample seeds AND windows: the interval a multi-seed claim needs."""
    rng = np.random.default_rng(seed)
    A = [np.asarray(r["per_seq_nll"]) for r in arm_a_runs]
    B = [np.asarray(r["per_seq_nll"]) for r in arm_b_runs]
    if not A or not B:
        return None
    L = min(min(len(x) for x in A), min(len(x) for x in B))
    means = np.empty(n)
    for i in range(n):
        ia = rng.integers(0, len(A), len(A))
        ib = rng.integers(0, len(B), len(B))
        w = rng.integers(0, L, L)
        ma = np.mean([A[j][:L][w].mean() for j in ia])
        mb = np.mean([B[j][:L][w].mean() for j in ib])
        means[i] = ma - mb
    obs = np.mean([x[:L].mean() for x in A]) - np.mean([y[:L].mean() for y in B])
    return {"delta": float(obs), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
            "n_seeds": [len(A), len(B)], "n_windows": int(L)}


def paired_seed_boot(arm_a_runs, arm_b_runs, n=20000, seed=0):
    """Bootstrap that RESPECTS the seed pairing.

    Continuation arms with the same seed fork from the same T0 file (sha256
    asserted) and consume the same token stream, so seed i of arm A and seed i
    of arm B differ only by method.  Resampling seeds independently from each
    arm, as `hier_boot_diff` does, throws that pairing away and inflates the
    interval by the full between-seed variance -- which here is an order of
    magnitude larger than the effect.  This resamples seed PAIRS and windows
    jointly, which is the correct interval for a matched design.
    """
    rng = np.random.default_rng(seed)
    A = [np.asarray(r["per_seq_nll"]) for r in arm_a_runs]
    B = [np.asarray(r["per_seq_nll"]) for r in arm_b_runs]
    k = min(len(A), len(B))
    if k == 0:
        return None
    L = min(min(len(x) for x in A), min(len(x) for x in B))
    D = np.stack([A[i][:L] - B[i][:L] for i in range(k)])      # [seeds, windows]
    means = np.empty(n)
    for i in range(n):
        s = rng.integers(0, k, k)
        w = rng.integers(0, L, L)
        means[i] = D[np.ix_(s, w)].mean()
    per_seed = [float(d.mean()) for d in D]
    return {"delta": float(D.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
            "per_seed_delta": [round(x, 6) for x in per_seed],
            "per_seed_sd": round(float(np.std(per_seed, ddof=1)), 6) if k > 1 else None,
            "same_sign_all_seeds": bool(all(x < 0 for x in per_seed)
                                        or all(x > 0 for x in per_seed)),
            "n_seed_pairs": k, "n_windows": int(L), "design": "paired by seed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "out_cpt_20260817"))
    ap.add_argument("--out-json", default=str(REPO / "out_cpt_20260817" / "cpt_analysis.json"))
    ap.add_argument("--out-md", default=str(REPO / "out_cpt_20260817" / "CPT_RESULTS.md"))
    args = ap.parse_args()

    d = Path(args.dir)
    runs = load_evals(d)
    by_arm: dict[str, list] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    for v in by_arm.values():
        v.sort(key=lambda r: r.get("seed", 0))

    res = {"n_runs": len(runs), "arms": {}, "provenance": [r["_path"] for r in runs]}
    for arm, rs in sorted(by_arm.items()):
        nlls = [r["val_nll"] for r in rs]
        res["arms"][arm] = {
            "seeds": [r.get("seed") for r in rs], "n_runs": len(rs),
            "per_seed_nll": [round(x, 5) for x in nlls],
            "mean_nll": round(float(np.mean(nlls)), 5),
            "seed_sd": round(st.stdev(nlls), 5) if len(nlls) > 1 else None,
            "seed_spread": round(max(nlls) - min(nlls), 5) if len(nlls) > 1 else None,
            "tokens_seen": rs[0].get("tokens_seen"),
            "nll_by_position": {k: round(float(np.mean([r["nll_by_position"][k] for r in rs])), 5)
                                for k in rs[0]["nll_by_position"]},
        }

    # ---- seed-noise floor: the largest within-arm spread we actually measured
    spreads = [v["seed_spread"] for v in res["arms"].values() if v["seed_spread"]]
    floor = max(spreads) if spreads else None
    res["seed_noise_floor"] = floor
    res["seed_noise_floor_note"] = (
        "largest within-arm seed spread across all arms; an effect smaller than "
        "this is not distinguishable from initialization noise at this seed count")

    # ---- the native gap (the denominator of any recovery ratio)
    cmp_ = {}
    if "vanilla" in by_arm and "native_diffv2" in by_arm:
        cmp_["native_diffv2_minus_vanilla"] = hier_boot_diff(
            by_arm["native_diffv2"], by_arm["vanilla"])
        cmp_["native_diffv2_minus_vanilla_perseed"] = [
            round(a["val_nll"] - b["val_nll"], 5)
            for a, b in zip(by_arm["native_diffv2"], by_arm["vanilla"])]

    # ---- continuation arms vs their controls
    base = "vanilla_continue"
    for arm in ("lowrank", "lowrank_unfreeze", "localreader", "additive"):
        if arm in by_arm and base in by_arm:
            cmp_[f"{arm}_minus_{base}"] = hier_boot_diff(by_arm[arm], by_arm[base])
        if arm in by_arm and "additive" in by_arm and arm != "additive":
            cmp_[f"{arm}_minus_additive"] = hier_boot_diff(by_arm[arm], by_arm["additive"])
    # the direct method-vs-method comparison: faithful split vs the LocalRead
    # split it replaces, at identical trainable budget and identical T0/stream
    if "lowrank" in by_arm and "localreader" in by_arm:
        cmp_["lowrank_minus_localreader"] = hier_boot_diff(
            by_arm["lowrank"], by_arm["localreader"])
        cmp_["lowrank_minus_localreader_perseed"] = [
            round(a["val_nll"] - b["val_nll"], 5)
            for a, b in zip(by_arm["lowrank"], by_arm["localreader"])]
    if "lowrank_unfreeze" in by_arm and "lowrank" in by_arm:
        cmp_["lowrank_unfreeze_minus_lowrank"] = hier_boot_diff(
            by_arm["lowrank_unfreeze"], by_arm["lowrank"])
    res["comparisons"] = cmp_

    # ---- paired-by-seed intervals for the arms that genuinely share a T0
    CONT = ("vanilla_continue", "lowrank", "additive", "localreader",
            "lowrank_unfreeze")
    paired = {}
    for a in CONT:
        for b in CONT:
            if a >= b or a not in by_arm or b not in by_arm:
                continue
            paired[f"{a}_minus_{b}"] = paired_seed_boot(by_arm[a], by_arm[b])
    res["paired_comparisons"] = paired
    res["paired_note"] = (
        "Continuation arms with the same seed load the SAME T0 file (sha256 "
        "asserted) and consume the same token stream, so they are a matched "
        "design and these are the intervals that apply to them. The unpaired "
        "hierarchical intervals above are reported too because they are the "
        "correct ones for the from-scratch arms, which share no checkpoint.")

    # ---- recovery ratio, GATED
    rec = {"computed": False, "value": None, "reason": ""}
    gap = cmp_.get("native_diffv2_minus_vanilla")
    if gap is None:
        rec["reason"] = "native_diffv2 and/or vanilla arms missing"
    else:
        denom = -gap["delta"]          # positive when DiffV2 has LOWER nll
        per_seed = cmp_.get("native_diffv2_minus_vanilla_perseed", [])
        consistent = bool(per_seed) and all(x < 0 for x in per_seed)
        if not consistent:
            rec["reason"] = (f"native DiffV2 does not beat vanilla consistently across "
                             f"seeds (per-seed deltas {per_seed}); a recovery ratio "
                             f"would divide by an unstable denominator")
        elif floor is not None and denom <= floor:
            rec["reason"] = (f"denominator {denom:.5f} does not exceed the measured "
                             f"seed-noise floor {floor:.5f}; ratio would be division on noise")
        elif gap["ci_hi"] >= 0:
            rec["reason"] = (f"native-vs-vanilla CI [{gap['ci_lo']:.5f}, {gap['ci_hi']:.5f}] "
                             f"includes zero")
        elif "lowrank" in by_arm and base in by_arm:
            num = (np.mean([r["val_nll"] for r in by_arm[base]])
                   - np.mean([r["val_nll"] for r in by_arm["lowrank"]]))
            rec.update(computed=True, value=round(float(num / denom), 4),
                       numerator=round(float(num), 5), denominator=round(float(denom), 5),
                       reason="gate passed")
        else:
            rec["reason"] = "lowrank/vanilla_continue arms missing"
    res["recovery_ratio"] = rec

    Path(args.out_json).write_text(json.dumps(res, indent=1))

    # ------------------------------------------------------------------ markdown
    L = ["# Continue-pretraining results (PG19, seq 4096)", "",
         f"Generated by `scripts/cpt_analyze.py` from {len(runs)} `*_eval.json` artifacts.",
         "All arms re-scored by the single evaluator in `scripts/cpt_eval.py` on the same",
         "validation windows (disjoint books), so comparisons are paired.", "",
         "## Arms", "",
         "| arm | seeds | per-seed val NLL | mean | seed sd | tokens |",
         "|---|---|---|---|---|---|"]
    for arm, v in res["arms"].items():
        L.append(f"| `{arm}` | {v['n_runs']} | {v['per_seed_nll']} | **{v['mean_nll']}** | "
                 f"{v['seed_sd']} | {v['tokens_seen']:,} |" if v["tokens_seen"] else
                 f"| `{arm}` | {v['n_runs']} | {v['per_seed_nll']} | **{v['mean_nll']}** | {v['seed_sd']} | - |")

    L += ["", f"**Seed-noise floor: {floor}** ({res['seed_noise_floor_note']}).", "",
          "## NLL by within-document position", "",
          "Every val window lies inside one book, so position p carries p tokens of",
          "genuine same-book context. A method that uses long context should widen its",
          "advantage as p grows.", ""]
    pos_keys = list(next(iter(res["arms"].values()))["nll_by_position"])
    L += ["| arm | " + " | ".join(pos_keys) + " |", "|---" * (len(pos_keys) + 1) + "|"]
    for arm, v in res["arms"].items():
        L.append(f"| `{arm}` | " + " | ".join(f"{v['nll_by_position'][k]:.4f}" for k in pos_keys) + " |")

    L += ["", "## Comparisons (paired, hierarchical bootstrap over seeds x windows)", "",
          "| comparison | delta NLL | 95% CI | p | seeds |", "|---|---|---|---|---|"]
    for k, v in cmp_.items():
        if not isinstance(v, dict):
            continue
        L.append(f"| {k} | **{v['delta']:+.5f}** | [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}] | "
                 f"{v['p_two_sided']:.4f} | {v.get('n_seeds')} |")
    if "native_diffv2_minus_vanilla_perseed" in cmp_:
        L += ["", f"native DiffV2 - vanilla, per seed: "
              f"{cmp_['native_diffv2_minus_vanilla_perseed']} (negative = DiffV2 better)"]

    if res.get("paired_comparisons"):
        L += ["", "### Paired-by-seed comparisons (continuation arms share a T0)", "",
              res["paired_note"], "",
              "| comparison | delta NLL | 95% CI | p | per-seed | same sign |",
              "|---|---|---|---|---|---|"]
        for k, v in res["paired_comparisons"].items():
            if not v:
                continue
            L.append(f"| {k} | **{v['delta']:+.5f}** | [{v['ci_lo']:+.5f}, "
                     f"{v['ci_hi']:+.5f}] | {v['p_two_sided']:.4f} | "
                     f"{v['per_seed_delta']} | {v['same_sign_all_seeds']} |")

    L += ["", "## Recovery ratio", ""]
    if rec["computed"]:
        L += [f"`Recovery = ({rec['numerator']}) / ({rec['denominator']}) = "
              f"**{rec['value']}**`", "", "Gate passed: " + rec["reason"]]
    else:
        L += [f"**N/A** — {rec['reason']}", "",
              "Per the pre-registered rule, a recovery ratio is not reported when its",
              "denominator is unstable or noise-sized; quoting one would be division on noise."]
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
