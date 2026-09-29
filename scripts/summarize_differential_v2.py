#!/usr/bin/env python
"""Matched comparisons; bootstrap training seeds and paper clusters, never pick best seeds."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

PAIRS = [("sub", "add"), ("add_aug", "add"), ("sub_aug", "add_aug"),
         ("sub_nui", "sub_aug"), ("sub_nui", "add_aug")]


def paired_effect(left, right, metric, draws=2000, seed=42):
    if set(left) != set(right):
        raise ValueError("paired comparisons require the same complete set of training seeds")
    differences = {}
    paper_ids = None
    for training_seed in sorted(left):
        a, b = left[training_seed], right[training_seed]
        if a["manifest_sha256"] != b["manifest_sha256"]:
            raise ValueError("cannot compare different data manifests")
        for field in ("split", "perturb_seed", "group_k", "max_new_tokens"):
            if a["args"][field] != b["args"][field]:
                raise ValueError(f"evaluation protocols differ: {field}")
        aa = {x["sample_id"]: x for x in a["per_example"]}
        bb = {x["sample_id"]: x for x in b["per_example"]}
        if set(aa) != set(bb):
            raise ValueError("paired sample IDs differ")
        by_paper = {}
        for sample in sorted(aa):
            if aa[sample]["gold"] != bb[sample]["gold"]:
                raise ValueError("paired references differ")
            if [p["order"] for p in aa[sample]["predictions"]] != [p["order"] for p in bb[sample]["predictions"]]:
                raise ValueError("paired context permutations differ")
            paper = aa[sample]["paper_id"] or sample
            by_paper.setdefault(paper, []).append(aa[sample][metric] - bb[sample][metric])
        if paper_ids is not None and set(by_paper) != set(paper_ids):
            raise ValueError("evaluation paper sets differ across seeds")
        paper_ids = sorted(by_paper)
        differences[training_seed] = by_paper
    means = [statistics.mean(x for values in row.values() for x in values)
             for row in differences.values()]
    rng = random.Random(seed)
    keys = sorted(differences)
    bootstrap = []
    for _ in range(draws):
        sampled_papers = rng.choices(paper_ids, k=len(paper_ids))
        sampled_seeds = rng.choices(keys, k=len(keys))
        bootstrap.append(statistics.mean(
            statistics.mean(x for paper in sampled_papers for x in differences[s][paper])
            for s in sampled_seeds))
    bootstrap.sort()
    return {"mean_delta": statistics.mean(means), "per_seed_delta": dict(zip(keys, means)),
            "bootstrap_95_interval": [bootstrap[int(.025 * draws)], bootstrap[int(.975 * draws)]],
            "n_seeds": len(keys), "n_papers": len(paper_ids),
            "interpretation": "single-seed pilot; no training-seed uncertainty" if len(keys) == 1
                              else "descriptive hierarchical bootstrap; not proof of mechanism"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    runs = {}
    counts = {}
    for path in sorted(Path(args.directory).glob("*_robust.json")):
        arm, raw_seed = path.stem.removesuffix("_robust").rsplit("_s", 1)
        status_path = path.with_name(f"{arm}_s{raw_seed}.status.json")
        if not status_path.exists() or json.loads(status_path.read_text())["status"] != "COMPLETE":
            continue
        runs.setdefault(arm, {})[int(raw_seed)] = json.loads(path.read_text())
        train = json.loads(path.with_name(f"{arm}_s{raw_seed}.json").read_text())
        count = train.get("trainable_param_count", train.get("trainable", {}).get("trainable_param_count"))
        counts.setdefault(arm, set()).add(count)
    matched_counts = {n for arm, values in counts.items() if arm in {x for pair in PAIRS for x in pair} for n in values}
    if len(matched_counts) > 1 or None in matched_counts:
        raise ValueError(f"sidecar parameter counts do not match: {counts}")
    result = {"arms": {}, "paired_comparisons": {}}
    for arm, seeds in runs.items():
        result["arms"][arm] = {"n_seeds": len(seeds), "trainable_parameters": sorted(counts[arm]),
            **{metric: {"mean": statistics.mean(values),
                        "seed_std": statistics.stdev(values) if len(values) > 1 else None}
               for metric in ("clean_f1", "clean_em", "permuted_f1", "worst_f1", "eos_rate")
               for values in [[run["summary"][metric] for run in seeds.values()]]}}
    for left, right in PAIRS:
        if left in runs and right in runs:
            result["paired_comparisons"][f"{left}_minus_{right}"] = {
                metric: paired_effect(runs[left], runs[right], metric)
                for metric in ("clean_f1", "permuted_f1", "worst_f1")}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
