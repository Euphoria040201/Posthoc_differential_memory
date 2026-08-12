#!/usr/bin/env python
"""Score our saved predictions with the ORIGINAL upstream scorers.

Nothing is re-implemented here: HotpotQA is scored by running the official
`hotpot_evaluate_v1.py`, LoCoMo by importing the official
`task_eval/evaluation.py` functions.  This file only converts our result JSONs
into the exact input each official scorer expects and records the pinned SHAs.

HotpotQA note: the official gold file (curtis.ml.cmu.edu) is offline, so gold is
materialised from the HuggingFace mirror `hotpotqa/hotpot_qa` (distractor,
validation) into the official schema. The *scorer* is untouched official code.
Our method has no supporting-fact head, so `sp` is submitted EMPTY and the
Support/Joint metrics are reported as the ~0 they truly are.

Usage:
  official_score.py hotpot --results a.json b.json --output out.json
  official_score.py locomo --results locomo.json --output out.json
"""
from __future__ import annotations

import argparse, json, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = Path("/work/mingze/official_evaluators")
SHAS = {
    "hotpot": "3635853403a8735609ee997664e1528f4480762a",
    "locomo": "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
    "ruler": "e8bbff677ca2c239640dc90f93310dcf32408c93",
}


def load_records(path):
    d = json.load(open(path))
    recs = d.get("records") or d.get("per_example") or []
    arms = [k for k in recs[0]
            if k not in ("id", "gold", "question", "category", "conversation_id",
                         "cluster", "i", "n_ctx_tok")]
    return recs, arms, d


# ----------------------------------------------------------------- hotpot
def hotpot_gold(ids):
    """Official-schema gold for the requested ids, from the HF distractor mirror."""
    # reuse the repo's own cached loader so gold comes from the SAME rows the eval saw
    sys.path.insert(0, str(REPO))
    from pathlib import Path as _P
    from deltamem.eval.benchmark_compare import load_hotpotqa
    ds = load_hotpotqa(cache_dir=_P.home() / ".cache/huggingface/datasets",
                       max_samples=None, seed=42, local_files_only=True)
    want = set(ids)
    gold = []
    for r in ds:
        if r["id"] not in want:
            continue
        sf = r["supporting_facts"]
        gold.append({
            "_id": r["id"],
            "answer": r["answer"],
            "question": r["question"],
            "supporting_facts": [[t, int(s)] for t, s in zip(sf["title"], sf["sent_id"])],
            "type": r.get("type", ""), "level": r.get("level", ""),
        })
    missing = want - {g["_id"] for g in gold}
    return gold, sorted(missing)


def score_hotpot(args):
    evaluator = OFFICIAL / "hotpot" / "hotpot_evaluate_v1.py"
    if not evaluator.exists():
        raise SystemExit(f"official evaluator missing: {evaluator}")
    out = {"benchmark": "hotpotqa", "evaluator": str(evaluator),
           "evaluator_sha": SHAS["hotpot"],
           "gold_source": "HF hotpotqa/hotpot_qa distractor validation "
                          "(official curtis.ml.cmu.edu file is offline)",
           "sp_submitted": "EMPTY - method has no supporting-fact head; "
                           "Support/Joint are therefore ~0 by construction",
           "runs": []}
    for path in args.results:
        recs, arms, raw = load_records(path)
        ids = [r["id"] for r in recs]
        gold, missing = hotpot_gold(ids)
        with tempfile.TemporaryDirectory() as td:
            gold_path = Path(td) / "gold.json"
            json.dump(gold, open(gold_path, "w"))
            for arm in arms:
                pred = {"answer": {r["id"]: str(r.get(arm, "")) for r in recs},
                        "sp": {r["id"]: [] for r in recs}}
                pred_path = Path(td) / f"pred_{arm}.json"
                json.dump(pred, open(pred_path, "w"))
                proc = subprocess.run(
                    [sys.executable, str(evaluator), str(pred_path), str(gold_path)],
                    capture_output=True, text=True)
                metrics = None
                for line in proc.stdout.strip().splitlines()[::-1]:
                    try:
                        metrics = json.loads(line.replace("'", '"'))
                        break
                    except Exception:
                        continue
                out["runs"].append({
                    "file": path, "arm": arm, "n": len(recs),
                    "n_gold_missing": len(missing),
                    "official_metrics": metrics,
                    "stdout": proc.stdout.strip()[-400:],
                    "stderr": proc.stderr.strip()[-400:],
                })
                print(f"[official-hotpot] {Path(path).name} {arm}: {metrics}", flush=True)
    return out


# ----------------------------------------------------------------- locomo
def score_locomo(args):
    sys.path.insert(0, str(OFFICIAL / "locomo"))
    from task_eval.evaluation import (  # noqa: E402
        exact_match_score, f1_score, rougel_score,
    )
    out = {"benchmark": "locomo", "evaluator": str(OFFICIAL / "locomo/task_eval/evaluation.py"),
           "evaluator_sha": SHAS["locomo"],
           "note": "official exact_match/f1/rouge on QA; no LLM-judge metric is run "
                   "(no API key) and none is substituted",
           "runs": []}
    for path in args.results:
        recs, arms, raw = load_records(path)
        for arm in arms:
            per_cat, rows = {}, []
            for r in recs:
                gold = r.get("gold")
                pred = str(r.get(arm, ""))
                if gold is None:
                    continue
                golds = gold if isinstance(gold, list) else [gold]
                em = max(float(exact_match_score(pred, str(g))) for g in golds)
                f1 = max(float(f1_score(pred, str(g))) for g in golds)
                try:
                    rl = max(float(rougel_score(pred, str(g))) for g in golds)
                except Exception:
                    rl = None
                cat = str(r.get("category", "?"))
                per_cat.setdefault(cat, []).append((em, f1))
                rows.append({"id": r.get("id"), "cluster": r.get("conversation_id"),
                             "category": cat, "em": em, "f1": f1, "rougeL": rl})
            n = len(rows)
            summary = {
                "file": path, "arm": arm, "n": n,
                "official_em": sum(x["em"] for x in rows) / max(n, 1),
                "official_f1": sum(x["f1"] for x in rows) / max(n, 1),
                "by_category": {c: {"n": len(v),
                                    "em": sum(e for e, _ in v) / len(v),
                                    "f1": sum(f for _, f in v) / len(v)}
                                for c, v in sorted(per_cat.items())},
            }
            out["runs"].append(summary)
            print(f"[official-locomo] {Path(path).name} {arm}: "
                  f"EM={summary['official_em']:.4f} F1={summary['official_f1']:.4f} n={n}",
                  flush=True)
            if args.dump_rows:
                json.dump(rows, open(f"{args.output}.{arm}.rows.json", "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["hotpot", "locomo"])
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dump-rows", action="store_true")
    args = ap.parse_args()
    payload = {"hotpot": score_hotpot, "locomo": score_locomo}[args.benchmark](args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
