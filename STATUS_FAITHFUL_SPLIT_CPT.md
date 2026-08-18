# STATUS — faithful low-rank split + continue pretraining

Branch `agent/faithful-lowrank-split-cpt-2026-08-17` (from `8115e6d`).
Last updated 2026-08-18 00:15 UTC.

## One-paragraph state

The token-local low-rank split is implemented, gated and trained.  It is a
**faithful** post-hoc DiffV2 approximation (delta added pre-`q_norm`, so the
module is exactly `q_norm((Wq + BA)h)`), it is function-preserving to the bit in
both fp32 and bf16 on the real Qwen3-4B, and it carries **zero** per-sequence
inference state where the LocalRead method it replaces carries 21.9 MB.  On the
continue-pretraining matrix it **beats LocalRead consistently** but **ties the
parameter-matched additive control exactly**, and every adapter arm loses to
simply continuing to train the dense model by ~0.07 nats.  The native-DiffV2
reference advantage **fails its pre-registered gate** (−0.0071 nats against a
0.0123 seed-noise floor), so the recovery ratio is **N/A** — and the advantage
turns out to *decay with training*, which explains the 2026-08-14 result rather
than contradicting it.

## Completed

| phase | result | artifacts |
|---|---|---|
| implementation + 25 unit gates | pass | `deltamem/core/lowrank_split.py`, `tests/test_lowrank_split.py` |
| Qwen3-4B real gate | **PASS** | `out_cpt_20260817/realgate_lowrank.json` |
| PG19 corpus | 899M train / 8.4M val tokens, intra-book windows | `out_cpt_20260817/pg19_manifest.json` |
| Wave 1: vanilla x3, native_diffv2 x3 @ 800M tokens | native gap fails gate | `van_s*.json`, `diffv2_s*.json` |
| Wave 2: 5 continuation arms x 3 seeds @ +200M tokens | 15/15, 0 failed | `{arm}_s{seed}.json` |
| full-val paired re-score, 21 checkpoints | 2048 windows each | `*_final_eval.json` |
| analysis + bootstrap | see below | `cpt_analysis.json`, `CPT_RESULTS.md` |
| audit items 1-9 | all fixed or explicitly quarantined | `SOURCE_AUDIT_FAITHFUL_SPLIT.md` |

## Running (node16, launched 2026-08-18 00:0x UTC)

| GPUs | jobs | ETA |
|---|---|---|
| 4-7 | layer-placement screen: `evidence4` [0,2,4,6], `last4`, `midlate4`, `first4`, all r=192 = 786,432 params | ~00:50 |
| 0-3 | 4B arms on Qwen3-4B-Instruct-2507: `lowrank` (r=177), `additive`, `localreader`, `lowrank_unfreeze`, 50M tokens each @ 8.6k tok/s | ~01:50 |

## Headline numbers (full 2048-window val, 3 seeds, paired hierarchical bootstrap)

**Native reference gap — the denominator**

| comparison | delta NLL | 95% CI | p |
|---|---|---|---|
| native_diffv2 − vanilla | **−0.00708** | [−0.01461, +0.00010] | 0.054 |

per-seed −0.00961 / −0.00265 / −0.00898; seed-noise floor **0.01230**.
Direction is consistent but the magnitude is inside noise and the CI grazes zero.

**It decays with training** (256-window in-training eval, matched steps):

| tokens | native_diffv2 − vanilla |
|---|---|
| 210M | −0.0359 (complete separation) |
| 420M | −0.0070 |
| 800M | −0.0048 |

A convergence-speed effect, not an asymptotic quality difference.  The
2026-08-14 measurement (~0.005 at 480M tokens on C4 @ seq 1024) sits on this
same curve, so the two studies agree.

**Continuation matrix** (shared T0, identical token stream, +200M tokens)

| arm | mean val NLL | vs additive | vs vanilla_continue |
|---|---|---|---|
| A `vanilla_continue` (106.5M trainable) | **3.45297** | — | — |
| D `lowrank_unfreeze` (5.24M) | 3.52344 | −0.00519 (p=.15) | +0.07047 (p=.0000) |
| F `additive` (786,432) | 3.52863 | — | +0.07566 (p=.0000) |
| C `lowrank` (786,432) | 3.52866 | **+0.00003 (p=.96)** | +0.07569 (p=.0000) |
| E `localreader` (786,432) | 3.53309 | +0.00446 (p=.23) | +0.08012 (p=.0000) |

`lowrank − localreader = −0.00443`, negative in all three seeds
(−0.00454/−0.00444/−0.00431): the faithful parameterization is a real
improvement over the method it replaces.  `lowrank − additive = +0.00003` is the
cleanest null in the study.

**Position-stratified NLL contradicts the long-range story.**  native DiffV2 −
vanilla by within-book position: −0.0245 (0-128), −0.0063 (128-512), −0.0037
(512-1024), −0.0057 (1024-2048), −0.0077 (2048-4096).  The advantage is largest
where context is *shortest*, on a corpus built specifically to give long-range
dependencies the best possible chance.

## Recovery ratio

**N/A.**  Denominator 0.00708 < seed-noise floor 0.01230.  Per the pre-registered
rule this is not reported; quoting a ratio would be division on noise.

## Honest limitations

* One model scale for the controlled matrix (106.5M params, 28.8M non-embedding
  — the 151,936-token Qwen vocab makes embeddings 73% of the parameter count).
  A convergence-speed effect that vanishes by 800M tokens at this scale may not
  vanish at a scale where 800M tokens is early training.
* 3 seeds. Enough to establish a noise floor, not enough to resolve a 0.007
  effect against it.
* Continue-pretraining budget is 200M tokens (25% of T0's 600M). A longer
  continuation could change the adapter-vs-full-finetune gap.
* The 4B runs continue-pretrain on PG19, which is public-domain Gutenberg text
  and almost certainly inside Qwen3-4B's own pretraining corpus. Those arms are
  compared under identical conditions to each other, but absolute gains there
  must not be read as learning new material.
* FlashAttention is untested; only eager and sdpa were run.
