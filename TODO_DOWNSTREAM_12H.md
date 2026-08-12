# 12-hour downstream audit + rescue — live TODO

START 2026-08-12 05:24:19 UTC · DEADLINE 2026-08-12 17:24:19 UTC · node16 (8x H100)
Branch `agent/downstream-audit-2026-08-12` · outputs `out_downstream_audit_20260812/`
Last refresh: 08:05 UTC

Status legend: PENDING / RUNNING / PASS / FAIL / INVALID

## Headline (evidence, not claims — see DOWNSTREAM_RESCUE_REPORT.md)

| # | finding | where |
|---|---|---|
| F1 | The pre-o/post-o line has **no written memory** (P=0, `prefix_write=False`). | §1 |
| F2 | `ours_window_only` is **not an ablation** at P=0 — bit-identical to the full arm. | §1 |
| F3 | pre_o formula exact (residual 0.0); post_o exact; widths verified under GQA. | §1 |
| F4 | Base parity **bit-exact** over 100 examples. | §1 |
| F5 | pre_o vs post_o_projected: **no bug** (fp32 top-1 disagreement 0.0000); bf16 flips 1.1%. | §1 |
| F6 | Backbone **bit-identical** across real optimizer steps; 0 backbone grads. | §1 |
| F7 | **HotpotQA untouched holdout (4678 ids): +0.0602 F1 (P=0) and +0.1111 F1 (P=64), both CI>0.** | §2c |
| F8 | **LoCoMo untouched (1339 QA): +0.0316 F1, conversation-clustered CI [+0.023, +0.042].** | §4 |
| F9 | **RULER officially generated: ours ≤ base at 8K/16K/32K.** | §3 |
| F10 | **Memory is inert: no checkpoint's correct−swap CI lies above zero; two are significantly negative.** | §2e |
| F11 | P=64 training does not converge; final checkpoints emit `,,,,,,` when memory is on. | §2b |
| F12 | The method is **3.2x slower per query** than the base it beats; amortization does not fix it. | §6 |

## P0 — must finish

- [x] git / env / GPU / model / dataset / evaluator SHA manifest — PASS
- [x] WRITE/READ inputs, injection points, widths — PASS
- [x] frozen-backbone hash — PASS
- [x] base parity — PASS (bit-exact)
- [x] state isolation / swap / batch contamination — PASS
- [x] pre_o vs post_o_projected fp32/bf16/cache/batch/repeat — PASS (no bug)
- [x] historical inventory + contamination manifest — PASS
- [x] official evaluator for all three benchmarks — PASS (hotpot_evaluate_v1, locomo task_eval, RULER official generation)
- [x] Hotpot screening + confirmation + untouched final — PASS
- [x] LoCoMo official protocol + clustered CI — PASS
- [x] RULER 8K/16K/32K official — PASS
- [x] unified per-example prediction schema — PASS (`records[{id,gold,<arm>}]` everywhere)

## P1 — rescue round (all closed)

- [x] `post_o + fixed_add` (P=0) — PASS as task adapter, memory undefined
- [x] `pre_o + fixed_add` (P=0) — PASS as task adapter, memory undefined
- [x] noctx training, P=64, post_o — **FAIL** (memory +0.0065; final ckpt degenerate)
- [x] noctx training, P=64, pre_o — **FAIL** (memory −0.0451; final ckpt degenerate)
- [x] ctxmask training — **FAIL** (memory −0.1428 vs its own window-only control)
- [x] swap-contrastive β=5 — **FAIL** (correct−swap −0.0272, CI excludes 0 on the negative side)
- [x] swap-contrastive β=1 — **FAIL** (harmful)
- [x] stability re-train (prefix-lr 1e-3, 12 layers) — **FAIL to converge** (loss 1.65@180 → 4.20@200); best-ever memory value +0.0065, CI spans 0
- [x] matched no-memory adapter control — PASS (`method_nomem` / `window_only` in every run)
- [x] zero-shot transfer of Qasper checkpoints — PASS (this is the headline positive)

## Running now

- [ ] `hotpotFULL_preo_s1` / `_s2` — full official dev, seeds 1-2 (3-seed final)
- [ ] `locomo_preo_s1` / `_s2` — LoCoMo seeds 1-2 (interim +0.039 / +0.037, consistent with seed 0)
- [ ] `locomo_mem64preo_step100` — P=64 with the window-only ablation (interim: ours .3307 vs window-only .3259)
- [ ] `memarms_preo_step100_n600` — correct−swap at n=600 for a tighter interval
- [ ] `mem64_noctx_preo_s1` / `_s2` — does the +0.111 replicate across seeds?
- [ ] `hotpot_dev1000_ctxmask` — the failed ctxmask arm at scale

## Not done / not possible

- RULER 64K — not attempted; 8K/16K/32K already answer the question
- LoCoMo LLM-judge metric — **unavailable** (no API key); not substituted
- HotpotQA official gold file — curtis.ml.cmu.edu is 404; gold rebuilt from the HF
  mirror into the official schema (0 missing ids), scorer untouched
- P3 items (slot count, layer subsets, softplus gain, subtractive-from-scratch) — not
  run: the diagnosis never pointed at capacity, and GPU time went to confirmation
  instead, per §11-F
