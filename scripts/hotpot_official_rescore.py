#!/usr/bin/env python
"""Re-score the stored HotpotQA predictions with the OFFICIAL scorer.

The 2026-08-14 report carried a caveat: the pinned `hotpot_evaluate_v1.py` was
not present on the machine, so the numbers used the in-repo metric and could not
be checked against the official one.  The file is now vendored at
`third_party/hotpot/hotpot_evaluate_v1.py`
(sha256 d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0,
from hotpotqa/hotpot@master), so the stored per-example predictions can be
re-scored properly.

The official answer metric differs from a naive token-F1 in one way that matters
here: if either the prediction or the gold is exactly `yes`/`no`/`noanswer` and
they are not equal, the example scores a hard zero rather than partial token
overlap.  A model that answers "Yes" to a span question is penalised fully.

This scores ANSWERS only.  The official script also scores supporting facts and
a joint metric; those need retrieved sp pairs, which these runs never produced.
Reporting answer EM/F1 alone is therefore correct, and it is still a 300-example
SCREENING SUBSET, not the official dev benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "hotpot"))

from hotpot_evaluate_v1 import exact_match_score, f1_score  # noqa: E402

OLD = REPO / "out_diffsplit_20260814"
OUT = REPO / "out_cpt_20260817"


def score(records, key):
    em = f1 = 0.0
    for r in records:
        pred, gold = r[key], r["gold"]
        em += float(exact_match_score(pred, gold))
        f1 += f1_score(pred, gold)[0]
    n = len(records)
    return {"em": em / n, "f1": f1 / n, "n": n}


def main():
    res = {"scorer": "third_party/hotpot/hotpot_evaluate_v1.py (official, answers only)",
           "sha256": "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0",
           "arms": {}, "note": "300-example screening subset, NOT the official dev set"}
    for tag, arm in (("hotpot_split_s0", "split"), ("hotpot_additive_s0", "additive")):
        p = OLD / f"{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        recs = d["records"]
        official_base = score(recs, "base")
        official_arm = score(recs, "ours")
        inrepo_base = {"f1": d["res"]["base"]["f1"] / d["res"]["base"]["n"],
                       "em": d["res"]["base"]["em"] / d["res"]["base"]["n"]}
        inrepo_arm = {"f1": d["res"]["ours"]["f1"] / d["res"]["ours"]["n"],
                      "em": d["res"]["ours"]["em"] / d["res"]["ours"]["n"]}
        res["arms"].setdefault("base", {})[tag] = {
            "official": official_base, "in_repo": inrepo_base,
            "f1_delta_official_minus_inrepo": official_base["f1"] - inrepo_base["f1"]}
        res["arms"][arm] = {
            "official": official_arm, "in_repo": inrepo_arm,
            "f1_delta_official_minus_inrepo": official_arm["f1"] - inrepo_arm["f1"],
            "official_minus_base_f1": official_arm["f1"] - official_base["f1"]}
    (OUT / "hotpot_official_rescore.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
