#!/usr/bin/env python
"""Paired statistics for benchmark arms sharing the same sample IDs.

- paired bootstrap (default 10,000 resamples) on the per-example metric
- McNemar exact test on the binary metric (EM / correct-incorrect)
- optional cluster bootstrap (resample CLUSTERS, e.g. LoCoMo conversations)
- optional hierarchical bootstrap over training seeds:
  resample examples first, then resample the seed set, so repeated examples
  across seeds are never treated as independent observations.

Reads result JSONs produced by the eval scripts and emits one summary JSON.
"""
from __future__ import annotations

import argparse, json, math, random, sys
from pathlib import Path


def _scored_rows(d):
    """Schema C: {"records": [{"id","gold", <arm>: prediction_string, ...}]}.

    Predictions are scored HERE with the same metric functions the eval used, so
    the statistics never depend on a metric the generator happened to store.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from deltamem.eval.benchmark_compare import hotpotqa_f1, hotpotqa_exact_match
    recs = d["records"]
    arms = [k for k in recs[0] if k not in ("id", "gold", "cluster", "category",
                                            "conversation_id", "question")]
    out = {}
    for arm in arms:
        out[arm] = {}
        for k, r in enumerate(recs):
            sid = str(r.get("id", k))
            pred, gold = r.get(arm), r.get("gold")
            if pred is None or gold is None:
                continue
            out[arm][sid] = {
                "f1": hotpotqa_f1(pred, gold),
                "em": float(hotpotqa_exact_match(pred, gold)),
                "cluster": r.get("cluster") or r.get("conversation_id"),
                "category": r.get("category"),
            }
    return out


def load_rows(path, arm_key=None):
    """-> {arm: {sample_id: {"f1": float, "em": float, "cluster": str}}}"""
    d = json.load(open(path))
    out = {}
    # schema A: {"conds": {arm: {"per_example": [...]}}}
    if "records" in d and isinstance(d["records"], list) and d["records"]:
        return _scored_rows(d), d
    conds = d.get("conds") or d.get("results") or {}
    if isinstance(conds, dict) and conds:
        for arm, payload in conds.items():
            rows = payload.get("per_example") or payload.get("rows") or []
            out[arm] = {str(r.get("id", r.get("i", k))): r for k, r in enumerate(rows)}
    # schema B: {"per_example": [{"id":..., "base_f1":..., "ours_f1":...}]}
    if not out and "per_example" in d:
        rows = d["per_example"]
        arms = sorted({k.rsplit("_", 1)[0] for r in rows for k in r
                       if k.endswith("_f1") or k.endswith("_em")})
        for arm in arms:
            out[arm] = {}
            for k, r in enumerate(rows):
                sid = str(r.get("id", r.get("i", k)))
                out[arm][sid] = {"f1": r.get(f"{arm}_f1"), "em": r.get(f"{arm}_em"),
                                 "cluster": r.get("cluster") or r.get("conversation_id")}
    return out, d


def paired_bootstrap(a, b, n_boot, rng, clusters=None):
    """mean(a) - mean(b) with a percentile CI; resamples clusters when given."""
    assert len(a) == len(b)
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    point = sum(diff) / n
    if clusters is None:
        idx_pool = [[i] for i in range(n)]
    else:
        groups = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        idx_pool = list(groups.values())
    k = len(idx_pool)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        m = 0
        for _ in range(k):
            grp = idx_pool[rng.randrange(k)]
            for i in grp:
                s += diff[i]
                m += 1
        boots.append(s / max(m, 1))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return {"delta": point, "ci95": [lo, hi], "n": n, "n_clusters": k,
            "significant": lo > 0 or hi < 0}


def mcnemar_exact(a_bin, b_bin):
    """Exact binomial McNemar on discordant pairs (a correct/b wrong vs reverse)."""
    b01 = sum(1 for x, y in zip(a_bin, b_bin) if x > y)   # a right, b wrong
    b10 = sum(1 for x, y in zip(a_bin, b_bin) if y > x)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(b01, b10)
    # log-space: math.comb(n, i) * 0.5**n overflows float for n in the thousands
    def log_pmf(i):
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + n * math.log(0.5))
    terms = [log_pmf(i) for i in range(k + 1)]
    mx = max(terms)
    p = 2.0 * math.exp(mx) * sum(math.exp(t - mx) for t in terms)
    return {"b01": b01, "b10": b10, "p_value": min(1.0, p)}


def hierarchical_bootstrap(per_seed_pairs, n_boot, rng):
    """per_seed_pairs: [ [(a_i, b_i), ...] per seed ].  Resample examples then seeds."""
    n_seeds = len(per_seed_pairs)
    boots = []
    for _ in range(n_boot):
        seed_means = []
        for _ in range(n_seeds):
            pairs = per_seed_pairs[rng.randrange(n_seeds)]
            n = len(pairs)
            s = 0.0
            for _ in range(n):
                a, b = pairs[rng.randrange(n)]
                s += a - b
            seed_means.append(s / n)
        boots.append(sum(seed_means) / n_seeds)
    boots.sort()
    point = sum(sum(a - b for a, b in p) / len(p) for p in per_seed_pairs) / n_seeds
    return {"delta": point, "ci95": [boots[int(0.025 * n_boot)],
                                     boots[int(0.975 * n_boot) - 1]],
            "n_seeds": n_seeds,
            "significant": boots[int(0.025 * n_boot)] > 0 or boots[int(0.975 * n_boot) - 1] < 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="result JSONs; multiple files = multiple training seeds")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--binary-metric", default="em")
    ap.add_argument("--cluster-key", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    per_seed, report = [], {"files": args.files, "arm": args.arm,
                            "baseline": args.baseline, "metric": args.metric,
                            "n_boot": args.n_boot, "per_file": []}
    for f in args.files:
        arms, raw = load_rows(f)
        if args.arm not in arms or args.baseline not in arms:
            raise SystemExit(f"{f}: arms {sorted(arms)} lack {args.arm}/{args.baseline}")
        ids = sorted(set(arms[args.arm]) & set(arms[args.baseline]))
        if len(ids) != len(arms[args.arm]):
            print(f"WARNING {f}: {len(arms[args.arm])} arm rows vs {len(ids)} shared ids",
                  file=sys.stderr)
        a = [float(arms[args.arm][i][args.metric]) for i in ids]
        b = [float(arms[args.baseline][i][args.metric]) for i in ids]
        clusters = None
        if args.cluster_key:
            clusters = [arms[args.arm][i].get(args.cluster_key) or
                        arms[args.arm][i].get("cluster") for i in ids]
        entry = {"file": f, "n": len(ids),
                 f"{args.arm}_mean": sum(a) / len(a),
                 f"{args.baseline}_mean": sum(b) / len(b),
                 "paired_bootstrap": paired_bootstrap(a, b, args.n_boot, rng, clusters)}
        try:
            ab = [float(arms[args.arm][i][args.binary_metric]) for i in ids]
            bb = [float(arms[args.baseline][i][args.binary_metric]) for i in ids]
            entry["mcnemar"] = mcnemar_exact(ab, bb)
            entry[f"{args.arm}_{args.binary_metric}"] = sum(ab) / len(ab)
            entry[f"{args.baseline}_{args.binary_metric}"] = sum(bb) / len(bb)
        except (KeyError, TypeError):
            entry["mcnemar"] = "binary metric unavailable"
        report["per_file"].append(entry)
        per_seed.append(list(zip(a, b)))

    if len(per_seed) > 1:
        report["hierarchical_bootstrap"] = hierarchical_bootstrap(per_seed, args.n_boot, rng)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
