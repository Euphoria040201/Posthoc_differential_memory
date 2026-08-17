# TODO — faithful low-rank split + continue pretraining (2026-08-17)

Branch `agent/faithful-lowrank-split-cpt-2026-08-17`, forked from
`agent/diff-head-split-2026-08-14` @ `8115e6d`.  The LocalRead split and every
2026-08-14 artifact are preserved untouched and serve as comparison arm E.

## Scientific question

Can a pretrained dense Transformer be converted post hoc into a differential-head
model and then recover some of a native Differential Transformer's advantage
through continue pretraining?

The gating problem inherited from 2026-08-14: the native-DiffV2 advantage was
only ~0.005 nats, the same size as seed noise, so any recovery ratio was
division on noise.  **First establish a real denominator, then measure recovery.**

## Status

| # | item | state |
|---|---|---|
| 1 | token-local low-rank split module (`deltamem/core/lowrank_split.py`) | **done** |
| 2 | 25 unit gates (`tests/test_lowrank_split.py`) | **done, 25 pass** |
| 3 | Qwen3-4B real gate (`scripts/lowrank_realgate.py`) | **done, PASS** |
| 4 | PG19 long-document corpus at seq 4096 | **done** (899M train / 8.4M val) |
| 5 | HF-matched native DiffV2 (`deltamem/core/diffv2_native.py`) | **done** |
| 6 | CPT trainer, 7 arms (`scripts/cpt_train.py`) | **done, all arms smoked** |
| 7 | Wave 1: vanilla x3 + native_diffv2 x3 @ 800M tokens | **running** |
| 8 | Wave 2: continuation arms A/C/D/E/F from shared T0 | queued |
| 9 | layer-selection screening (4 parameter-matched layouts) | queued |
| 10 | analysis + bootstrap + final report from artifacts | queued |
| 11 | 4B port of the best defensible configuration | stretch, compute permitting |

## Audit items carried in from 2026-08-14

| # | issue | state |
|---|---|---|
| 1 | "KV-cache-free" language overstated; LocalRead keeps `_read_h`/`_read_v` | **fixed**: measured — LocalRead adds 21,934,080 B/sequence (14.5% of its KV cache) at seq 1024; low-rank split adds **0 B**. `out_cpt_20260817/inference_state.json` |
| 2 | shuffle ablation permuted probabilities after causal masking (could move past mass onto future keys) | **fixed**: invalid path now raises; valid within-window permutation implemented and re-measured |
| 3 | LocalRead requires `read_dim == head_dim` but did not enforce it | **fixed**: enforced at construction |
| 4 | LocalRead ignores `attention_mask`/padding | **quarantined**: documented in-module; all reported runs used packed unpadded sequences. The new method is padding-safe by construction and tested on left/right padding |
| 5 | diff checkpoint loader was fail-open | **fixed**: fails closed in `eval_ours_hotpotqa.load_ours` and in the probe; the new loader is fail-closed and unit-tested |
| 6 | dynamic-gate parameter-count assertion inconsistent | **fixed**: gate term added |
| 7 | Hotpot report numbers disagree with Hotpot artifacts | **fixed**: regenerated — base 0.5939 / split 0.5796 / additive 0.6012 |
| 8 | LoCoMo prose used seed-0 +0.0423 instead of the 2-seed mean | **fixed**: 2-seed mean is +0.0364 |
| 9 | small-model DiffV2 init not matched to the baseline | **fixed**: measured old std 0.02553 (uniform) vs vanilla 0.02004 (normal); new module matches at 0.01997 |

## Design decisions and why

**Delta goes PRE-norm.** The task specifies a low-rank parameterization of
`Wq_minus = Wq_plus + DeltaW`.  Adding `dQ` to the already-normalized query (what
LocalRead does) is *not* that: it is an additive patch on a normalized vector.
Adding before `q_norm` makes the module exactly `q_norm((Wq + BA)h)`, i.e. one
query matrix followed by the same norm — the native DiffV2 query path.  A
`delta_pre_norm=False` switch preserves the old behaviour for ablation.

**PG19, not C4.** The chain-B C4 corpus was packed at seq_len 1024, where
essentially no sequence carries a dependency longer than a few hundred tokens.
A differential transformer's claimed benefit is cancellation of attention noise
over long context, so that corpus is the one setting where it provably cannot
show an advantage.  PG19 books are ~70k tokens; every 4096-token window is drawn
from ONE book, so position p carries p tokens of genuine same-book context and
position-stratified NLL is meaningful.

**Rank 96 for the small model.** `96*(512+512) = 98,304` per layer x 8 layers =
786,432 — *exactly* the LocalRead and additive trainable budget, not approximately.
(For Qwen3-4B, r=177 gives 14,137,344, within 0.13% of 14,155,776.)

**Batch 1 x accum 32 for every continuation arm.** LocalRead materializes a dense
`[B,T,T]` score matrix before applying its window mask — O(T^2) memory despite
being a windowed reader — and OOMs at seq 4096 with batch 2.  Rather than give
one arm a different batch shape, all continuation arms use batch 1 x accum 32.
Each step consumes the same 32 sequences in the same order, all of equal length,
so the averaged gradient is identical to batch 2 x accum 16 in exact arithmetic.

## Gate for the recovery ratio

Compute
`Recovery = (L_vanilla_continue - L_posthoc) / (L_vanilla_continue - L_native_diff)`
**only if** native DiffV2 beats vanilla consistently across seeds and the
denominator exceeds the measured seed-noise floor.  Otherwise report **N/A** and
say why.  A negative or noise-sized denominator is not a result.
