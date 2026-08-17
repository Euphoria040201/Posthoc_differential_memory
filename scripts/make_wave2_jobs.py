#!/usr/bin/env python
"""Build the Wave-2 job list: continuation arms forked from the shared T0s.

Ordering matters.  Jobs are emitted seed-major (all arms of seed 0, then all
arms of seed 1, ...) so that if the queue is cut short there is a COMPLETE arm
matrix at fewer seeds rather than an incomplete matrix at more seeds.  Within a
seed the parameter-matched controls come first, because a missing control is
worse than a missing variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out_cpt_20260817"

# arm -> extra args.  batch 1 x accum 32 everywhere: LocalRead materialises a
# dense [B,T,T] score matrix and OOMs at batch 2 / seq 4096, and giving one arm a
# different batch shape would be a confound.  Same 32 sequences per step, equal
# lengths, so the averaged gradient matches batch 2 x accum 16 exactly.
ARM_ARGS = {
    "vanilla_continue": [],
    "lowrank": ["--rank", "96"],
    "additive": ["--read-dim", "64", "--window", "256"],
    "localreader": ["--read-dim", "64", "--window", "256"],
    "lowrank_unfreeze": ["--rank", "96", "--stage1-frac", "0.15",
                         "--backbone-lr", "1e-5"],
}
ORDER = ["vanilla_continue", "lowrank", "additive", "localreader", "lowrank_unfreeze"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--continue-tokens", type=int, default=200_000_000)
    ap.add_argument("--out", default=str(OUT / "wave2_jobs.json"))
    ap.add_argument("--t0-prefix", default="van_s")
    ap.add_argument("--layer-screen-seed", type=int, default=0)
    ap.add_argument("--layer-layouts", default="",
                    help="JSON: {name: [layers]} parameter-matched placements")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = []
    common = ["--batch-size", "1", "--grad-accum", "32",
              "--continue-tokens", str(args.continue_tokens),
              "--log-every", "50", "--eval-every", "500"]

    for s in seeds:
        t0 = OUT / f"{args.t0_prefix}{s}_T0.pt"
        if not t0.exists():
            print(f"[skip] seed {s}: {t0} missing")
            continue
        sha = sha256_file(t0)
        for arm in ORDER:
            jobs.append({
                "tag": f"{arm}_s{s}",
                "args": ["--arm", arm, "--seed", str(s),
                         "--init-from", str(t0), "--init-sha256", sha]
                + common + ARM_ARGS[arm]})

    # ---- parameter-matched layer-placement screen (seed of --layer-screen-seed)
    if args.layer_layouts:
        layouts = json.loads(Path(args.layer_layouts).read_text()
                             if Path(args.layer_layouts).exists() else args.layer_layouts)
        s = args.layer_screen_seed
        t0 = OUT / f"{args.t0_prefix}{s}_T0.pt"
        if t0.exists():
            sha = sha256_file(t0)
            for name, layers in layouts.items():
                # keep the TOTAL trainable budget fixed at 786,432 by scaling rank
                # with the number of layers: r * (hidden + H*hd) * n_layers = const
                per_layer_dim = 512 + 512
                rank = 786432 // (len(layers) * per_layer_dim)
                assert rank * len(layers) * per_layer_dim == 786432, (name, rank)
                jobs.append({
                    "tag": f"layout_{name}_s{s}",
                    "args": ["--arm", "lowrank", "--seed", str(s),
                             "--init-from", str(t0), "--init-sha256", sha,
                             "--rank", str(rank),
                             "--split-layers", ",".join(str(x) for x in layers)]
                    + common})

    Path(args.out).write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    for j in jobs:
        print(" ", j["tag"])


if __name__ == "__main__":
    main()
