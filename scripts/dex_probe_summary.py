#!/usr/bin/env python
"""Merge per-variant nuisance-probe JSONs into one comparison table."""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

ORDER = ["base", "attn_only", "dex_minus", "dex_plus", "adapter_only"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out_dex/nuisance_main_*.json")
    ap.add_argument("--out", default="out_dex/nuisance_summary.json")
    args = ap.parse_args()

    merged = {}
    for path in sorted(glob.glob(args.glob)):
        payload = json.load(open(path))
        merged.update(payload["results"])
    if not merged:
        raise SystemExit(f"no probe results matched {args.glob}")

    names = [n for n in ORDER if n in merged] + [n for n in merged if n not in ORDER]
    base = merged.get("base", {})
    print(f"{'variant':14s}{'V_nuis':>9s}{'S_evid':>10s}{'NSR':>9s}{'acc':>7s}"
          f"{'hid.var':>9s}{'VRR':>9s}{'AntiAlign':>11s}{'MeanShift':>11s}{'dNSR%':>8s}")
    for n in names:
        r = merged[n]
        h = r["hidden_mean"]
        d = ((r["NSR"] / base["NSR"] - 1) * 100) if base else float("nan")
        print(f"{n:14s}{r['V_nuis']:9.3f}{r['S_evid']:10.2f}{r['NSR']:9.5f}"
              f"{r['readout_accuracy']:7.3f}{h['hidden_var']:9.5f}{h['VRR']:+9.4f}"
              f"{h['AntiAlign']:+11.4f}{h['MeanShift']:11.5f}{d:+8.1f}")

    print("\nper-layer VRR / AntiAlign (adapted heads), first and last thirds:")
    for n in names:
        per = merged[n]["hidden_per_layer"]
        if not per:
            continue
        li = sorted(per, key=int)
        vrr = [per[i]["VRR"] for i in li]
        aa = [per[i]["AntiAlign"] for i in li]
        third = max(1, len(li) // 3)
        print(f"  {n:14s} VRR early={np.mean(vrr[:third]):+.4f} late={np.mean(vrr[-third:]):+.4f}"
              f" | AntiAlign early={np.mean(aa[:third]):+.4f} late={np.mean(aa[-third:]):+.4f}"
              f" | max|VRR|={max(abs(v) for v in vrr):.4f}")

    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
