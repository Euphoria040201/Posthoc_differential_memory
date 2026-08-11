#!/usr/bin/env python
"""Phase 6/8: statistics, result tables and figures for the DEX control study."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

COMPARISONS = [
    ("A", "dex_minus", "dex_plus", "does the minus sign beat the plus sign?"),
    ("B", "dex_minus", "residual_adapter", "does DEX beat a plain residual adapter?"),
    ("C", "dex_minus", "attn_only", "how much is just W_K/W_V/W_O finetuning?"),
    ("D", "dex_minus", "adapter_only", "is the adapter alone enough?"),
]


def load_runs(pattern: str) -> dict:
    runs = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            d = json.load(fh)
        if not d.get("final"):
            continue
        runs[d["variant"]][int(d["args"]["seed"])] = d
    return runs


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_t(diff: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats

    if len(diff) < 2 or np.allclose(diff, diff[0]):
        return float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_rel(diff, np.zeros_like(diff))
    dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return float(t), float(p), dz


def per_example_matrix(runs: dict, variant: str, key: str = "f1") -> np.ndarray | None:
    if variant not in runs:
        return None
    seeds = sorted(runs[variant])
    rows = []
    for s in seeds:
        pe = runs[variant][s]["final"]["qa"].get("per_example")
        if not pe:
            return None
        rows.append([r[key] for r in pe])
    return np.array(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out_dex/dex_*.json")
    ap.add_argument("--out-md", default="out_dex/dex_results.md")
    ap.add_argument("--out-json", default="out_dex/dex_stats.json")
    ap.add_argument("--fig-dir", default="out_dex/figs")
    args = ap.parse_args()

    runs = load_runs(args.glob)
    if not runs:
        raise SystemExit(f"no finished runs matched {args.glob}")

    order = [v for v in ("base", "dex_minus", "dex_plus", "residual_adapter",
                         "attn_only", "adapter_only") if v in runs]
    seeds_all = sorted({s for v in runs for s in runs[v]})

    lines = ["# DEX control study — results", ""]
    lines.append("## Per-seed results (Qasper val F1, 187 examples, greedy)")
    lines.append("")
    head = "| Variant | Trainable Params | " + " | ".join(f"Seed {s}" for s in seeds_all) + " | Mean | Std |"
    lines.append(head)
    lines.append("|" + "---|" * (3 + len(seeds_all)))
    summary = {}
    for v in order:
        vals = [runs[v][s]["final"]["qa"]["F1"] if s in runs[v] else None for s in seeds_all]
        present = [x for x in vals if x is not None]
        n_par = next(iter(runs[v].values()))["trainable"]["trainable_param_count"]
        mean = float(np.mean(present)) if present else float("nan")
        std = float(np.std(present, ddof=1)) if len(present) > 1 else 0.0
        summary[v] = {"F1": present, "mean": mean, "std": std, "params": n_par}
        cells = " | ".join(f"{x:.4f}" if x is not None else "—" for x in vals)
        lines.append(f"| {v} | {n_par:,} | {cells} | {mean:.4f} | {std:.4f} |")
    lines.append("")

    lines.append("## Per-seed results (EM and final validation CE)")
    lines.append("")
    lines.append("| Variant | " + " | ".join(f"EM s{s}" for s in seeds_all)
                 + " | EM mean | " + " | ".join(f"val CE s{s}" for s in seeds_all) + " | CE mean |")
    lines.append("|" + "---|" * (2 + 2 * len(seeds_all)))
    for v in order:
        ems = [runs[v][s]["final"]["qa"]["EM"] if s in runs[v] else None for s in seeds_all]
        ces = [runs[v][s]["final"].get("val_loss") if s in runs[v] else None for s in seeds_all]
        em_p = [x for x in ems if x is not None]
        ce_p = [x for x in ces if x is not None]
        summary[v]["EM"] = em_p
        summary[v]["val_loss"] = ce_p
        lines.append(
            f"| {v} | " + " | ".join(f"{x:.4f}" if x is not None else "—" for x in ems)
            + f" | {np.mean(em_p):.4f} | "
            + " | ".join(f"{x:.4f}" if x is not None else "—" for x in ces)
            + f" | {np.mean(ce_p):.4f} |"
        )
    lines.append("")

    stats_out = {"summary": summary, "comparisons": {}}
    lines.append("## Comparisons")
    lines.append("")
    lines.append("| Comparison | Mean Difference (F1) | 95% CI | p-value | Effect Size | Level |")
    lines.append("|---|---:|---|---:|---:|---|")
    for cid, a, b, question in COMPARISONS:
        if a not in runs or b not in runs:
            continue
        shared = sorted(set(runs[a]) & set(runs[b]))
        if not shared:
            continue
        da = np.array([runs[a][s]["final"]["qa"]["F1"] for s in shared])
        db = np.array([runs[b][s]["final"]["qa"]["F1"] for s in shared])
        diff = da - db
        lo, hi = paired_bootstrap(diff) if len(diff) > 1 else (float("nan"), float("nan"))
        t, p, dz = paired_t(diff)
        lines.append(
            f"| {cid}: {a} − {b} | {diff.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{p:.4f} | dz={dz:+.3f} | seed-level (n={len(diff)}) |"
        )
        entry = {
            "question": question, "a": a, "b": b, "seeds": shared,
            "seed_level": {"mean_diff": float(diff.mean()),
                           "std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                           "ci95": [lo, hi], "t": t, "p": p, "cohens_dz": dz,
                           "per_seed_diff": diff.tolist()},
        }

        ma, mb = per_example_matrix(runs, a), per_example_matrix(runs, b)
        if ma is not None and mb is not None and ma.shape[1] == mb.shape[1]:
            ea, eb = ma.mean(axis=0), mb.mean(axis=0)     # average over seeds per example
            d_ex = ea - eb
            lo2, hi2 = paired_bootstrap(d_ex)
            t2, p2, dz2 = paired_t(d_ex)
            lines.append(
                f"| {cid}: {a} − {b} | {d_ex.mean():+.4f} | [{lo2:+.4f}, {hi2:+.4f}] | "
                f"{p2:.4f} | dz={dz2:+.3f} | example-level (n={len(d_ex)}) |"
            )
            entry["example_level"] = {
                "mean_diff": float(d_ex.mean()),
                "std": float(d_ex.std(ddof=1)),
                "ci95": [lo2, hi2], "t": t2, "p": p2, "cohens_dz": dz2,
                "n_examples": int(len(d_ex)),
                "n_better": int((d_ex > 0).sum()), "n_worse": int((d_ex < 0).sum()),
                "n_tied": int((d_ex == 0).sum()),
            }
        stats_out["comparisons"][cid] = entry
    lines.append("")

    # ---- figures --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {v: c for v, c in zip(order, plt.cm.tab10.colors)}

    def series(variant, key):
        out = []
        for s in sorted(runs[variant]):
            rows = [(r["step"], r[key]) for r in runs[variant][s]["log"] if key in r]
            if rows:
                out.append(np.array(rows, dtype=float))
        return out

    panels = [
        ("train_loss", "training loss", "train_loss_vs_step.png"),
        ("val_loss", "validation CE", "val_loss_vs_step.png"),
        ("grad_norm", "gradient norm", "grad_norm_vs_step.png"),
        ("dex_lambda", "learned lambda(t)", "lambda_vs_step.png"),
        ("dex_corr_ratio", "||lambda f_D(O)|| / ||O||", "corr_ratio_vs_step.png"),
    ]
    for key, ylabel, fname in panels:
        plt.figure(figsize=(6, 4))
        drew = False
        for v in order:
            ss = series(v, key)
            if not ss:
                continue
            n = min(len(x) for x in ss)
            arr = np.stack([x[:n] for x in ss])
            plt.plot(arr[0, :, 0], arr[:, :, 1].mean(axis=0), label=v, color=colors[v])
            if arr.shape[0] > 1:
                plt.fill_between(arr[0, :, 0], arr[:, :, 1].min(axis=0),
                                 arr[:, :, 1].max(axis=0), color=colors[v], alpha=0.15)
            drew = True
        if not drew:
            plt.close()
            continue
        plt.xlabel("optimizer update")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.title(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=150)
        plt.close()

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(order))
    means = [summary[v]["mean"] for v in order]
    stds = [summary[v]["std"] for v in order]
    plt.bar(xs, means, yerr=stds, capsize=4, color=[colors[v] for v in order])
    for i, v in enumerate(order):
        for f in summary[v]["F1"]:
            plt.plot(i, f, "k.", ms=4)
    plt.xticks(xs, order, rotation=20, ha="right", fontsize=8)
    plt.ylabel("Qasper val F1")
    plt.title("final F1 (bars: mean +- std over seeds, dots: seeds)")
    plt.tight_layout()
    plt.savefig(fig_dir / "final_f1.png", dpi=150)
    plt.close()

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    Path(args.out_json).write_text(json.dumps(stats_out, indent=2) + "\n")
    print("\n".join(lines))
    print(f"[analyze] wrote {args.out_md}, {args.out_json}, figures in {fig_dir}")


if __name__ == "__main__":
    main()
