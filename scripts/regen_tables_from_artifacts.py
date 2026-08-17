#!/usr/bin/env python
"""Regenerate every headline table of the 2026-08-14 study FROM ITS ARTIFACTS.

Audit issues 7 and 8: the committed report prints Hotpot numbers that do not
match the committed Hotpot JSONs, and quotes a seed-0 LoCoMo delta (+0.0423)
where the two-seed mean (+0.0364) belongs.  Rather than trusting either, this
script recomputes all three benchmarks straight from the artifact files and
records the path, sha256 and size of every file it read.

Historical JSONs are NEVER rewritten.  Output is a NEW derived artifact:
    out_cpt_20260817/corrected_tables.json
    out_cpt_20260817/CORRECTED_TABLES.md
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "out_diffsplit_20260814"
OUT = REPO / "out_cpt_20260817"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


PROV: list[dict] = []


def load(p: Path):
    d = json.loads(p.read_text())
    PROV.append({"path": str(p.relative_to(REPO)), "sha256": sha256(p),
                 "bytes": p.stat().st_size})
    return d


def qasper():
    """Qasper val F1 per arm, mean over seeds, straight from run JSONs."""
    arms = {
        "split_local256_fixed": ["split_local256_fixed_s0", "split_local256_fixed_s1",
                                 "split_local256_fixed_s2"],
        "param_matched_additive": ["param_matched_additive_s0", "param_matched_additive_s1",
                                   "param_matched_additive_s2"],
        "param_matched_lora": ["param_matched_lora_s0", "param_matched_lora_s1",
                               "param_matched_lora_s2"],
    }
    rows = {}
    for arm, tags in arms.items():
        vals, tr = [], None
        for t in tags:
            p = OLD / f"{t}.json"
            if not p.exists():
                continue
            d = load(p)
            vals.append(float(d["final"]["qa"]["F1"]))
            tr = (d.get("trainable") or {}).get("trainable_param_count", tr)
        rows[arm] = {"seeds": len(vals), "per_seed": [round(v, 4) for v in vals],
                     "mean": round(sum(vals) / len(vals), 4),
                     "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None,
                     "trainable": tr}
    return rows


def hotpot():
    """HotpotQA screening subset: artifacts store SUMS over n, not means."""
    rows = {}
    for tag, arm in (("hotpot_split_s0", "split"), ("hotpot_additive_s0", "additive")):
        p = OLD / f"{tag}.json"
        if not p.exists():
            continue
        d = load(p)
        for key, label in (("base", "base"), ("ours", arm)):
            r = d["res"][key]
            rows.setdefault(label, []).append(
                {"f1": r["f1"] / r["n"], "em": r["em"] / r["n"], "n": r["n"],
                 "from": tag})
    out = {}
    for k, v in rows.items():
        f1s = [x["f1"] for x in v]
        out[k] = {"f1": round(sum(f1s) / len(f1s), 4),
                  "em": round(sum(x["em"] for x in v) / len(v), 4),
                  "n": v[0]["n"], "runs": [x["from"] for x in v],
                  "f1_per_run": [round(x, 6) for x in f1s]}
    out["_base_bit_identical_across_runs"] = len(set(out["base"]["f1_per_run"])) == 1
    return out


def locomo():
    """Full LoCoMo, 2 seeds; report the SEED MEAN, not seed 0."""
    per = {}
    for arm in ("split", "additive"):
        for s in (0, 1):
            p = OLD / f"locomo_{arm}_s{s}.json"
            if not p.exists():
                continue
            d = load(p)
            bc = d["by_cat"]
            for cond, label in (("base", "base"), ("ours", arm)):
                if cond not in bc:
                    continue
                o = bc[cond]["overall"]
                per.setdefault(label, {})[s] = o["sum"] / o["n"]
                per.setdefault(f"{label}_bycat", {})[s] = {
                    c: bc[cond][c]["sum"] / bc[cond][c]["n"]
                    for c in ("1", "2", "3", "4") if c in bc[cond]}
    rows = {}
    for k in ("base", "split", "additive"):
        if k not in per:
            continue
        vals = [per[k][s] for s in sorted(per[k])]
        rows[k] = {"per_seed": [round(v, 4) for v in vals],
                   "mean": round(sum(vals) / len(vals), 4), "seeds": len(vals)}
    if "split" in rows and "base" in rows:
        rows["split_minus_base_mean"] = round(rows["split"]["mean"] - rows["base"]["mean"], 4)
        rows["additive_minus_base_mean"] = round(rows["additive"]["mean"] - rows["base"]["mean"], 4)
        rows["split_minus_additive_mean"] = round(rows["split"]["mean"] - rows["additive"]["mean"], 4)
        rows["_note"] = ("seed-0-only split-base is "
                         f"{round(per['split'][0] - per['base'][0], 4)}; the two-seed "
                         f"mean is {rows['split_minus_base_mean']} and is the number that "
                         "belongs in prose")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"qasper": qasper(), "hotpot_screening_subset": hotpot(), "locomo_full": locomo()}
    res["provenance"] = PROV
    res["_disclaimer"] = (
        "Derived from the 2026-08-14 artifacts; no historical JSON was modified. "
        "HotpotQA is a 300-example SCREENING SUBSET scored with the in-repo metric, "
        "not the official hotpot_evaluate_v1.py, and must not be called an official "
        "benchmark result.")
    (OUT / "corrected_tables.json").write_text(json.dumps(res, indent=1))

    L = ["# Corrected tables, regenerated from artifacts (2026-08-17)", "",
         "Generated by `scripts/regen_tables_from_artifacts.py`. Every number below is",
         "computed from the committed JSON artifacts listed under provenance; none is",
         "transcribed by hand. Historical JSONs were not modified.", "",
         "## Qasper (internal 187-example split, val F1)", "",
         "| arm | seeds | per-seed | mean | sd | trainable |", "|---|---|---|---|---|---|"]
    for k, v in res["qasper"].items():
        L.append(f"| `{k}` | {v['seeds']} | {v['per_seed']} | **{v['mean']}** | "
                 f"{v['sd']} | {v['trainable']:,} |" if v["trainable"] else
                 f"| `{k}` | {v['seeds']} | {v['per_seed']} | **{v['mean']}** | {v['sd']} | - |")

    h = res["hotpot_screening_subset"]
    L += ["", "## HotpotQA — 300-example SCREENING SUBSET (in-repo metric, NOT official)", "",
          "| arm | F1 | EM | n |", "|---|---|---|---|"]
    for k in ("base", "split", "additive"):
        if k in h:
            L.append(f"| {k} | **{h[k]['f1']}** | {h[k]['em']} | {h[k]['n']} |")
    L += ["", f"base bit-identical across runs: **{h['_base_bit_identical_across_runs']}** "
          "(this is the check that the condition switch really toggles the split)",
          "", "The 2026-08-14 report printed base 0.5959 / split 0.5848 / additive 0.6034.",
          f"The artifacts give base {h['base']['f1']} / split {h['split']['f1']} / "
          f"additive {h['additive']['f1']}. The artifact values are authoritative."]

    lc = res["locomo_full"]
    L += ["", "## LoCoMo (full, 1540 questions, 2 seeds)", "",
          "| arm | per-seed | mean |", "|---|---|---|"]
    for k in ("base", "split", "additive"):
        if k in lc:
            L.append(f"| {k} | {lc[k]['per_seed']} | **{lc[k]['mean']}** |")
    L += ["", f"- split - base (2-seed mean): **{lc['split_minus_base_mean']}**",
          f"- additive - base (2-seed mean): **{lc['additive_minus_base_mean']}**",
          f"- split - additive (2-seed mean): **{lc['split_minus_additive_mean']}**",
          "", lc["_note"], ""]
    L += ["## Provenance", "", "| artifact | sha256 (first 16) | bytes |", "|---|---|---|"]
    for p in PROV:
        L.append(f"| `{p['path']}` | `{p['sha256'][:16]}` | {p['bytes']:,} |")
    (OUT / "CORRECTED_TABLES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print(f"\nwrote {OUT/'corrected_tables.json'} and {OUT/'CORRECTED_TABLES.md'}")


if __name__ == "__main__":
    main()
