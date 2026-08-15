# TODO — Post-hoc Differential Head Splitting, 12h comparison

Status key: `[x]` done with artifact · `[~]` partial · `[ ]` not done (reason given).
Every completed line carries the path that backs it.

## Audit and provenance

- [x] git / GPU / process / disk audit — branch `agent/diff-head-split-2026-08-14`;
      only free GPUs used; my own spinquant + vLLM jobs on other GPUs left running
- [x] audit both repos — `DIFF_SPLIT_SOURCE_AUDIT.md` §2 code map
- [x] authenticate historical artifacts — `DIFF_SPLIT_SOURCE_AUDIT.md` §3, §6.1.
      `base` confirmed as `trainable=0, steps=1` frozen eval ⇒ reusable.
      `attn_only` lr=2e-5 confirmed as the project's per-parameter-type convention
      (`scripts/dex_train_qasper.py:355-373`) ⇒ legitimate comparator, not re-run.
- [x] pin DIFF V2 (`unilm@833df7e7…`) and installed Qwen3 (`transformers==5.9.0`)
- [x] create source audit / registry / ledger / report — 4 files at repo root

## Implementation and hard gates

- [x] single-layer shared-K/V query split — `deltamem/core/diff_split.py`
- [x] GQA pairing verified — `realgate_v2.json`, mapping table in report §1
- [x] identical q-norm / RoPE / mask — one `apply_rotary_pos_emb` over `q_cat`
- [x] FP32 step-0 parity — **max_abs 0.000e+00**; BF16 max 2.031e-01, greedy identical
- [x] cached decode / KV cache — **28,311,552 bytes in base and split alike**
- [x] causal local reader — past-perturbation effect **0.000e+00**
- [x] backbone frozen — SHA256 over 398 tensors unchanged, 0 backbone grads
- [x] delta_q nonzero gradient after step 1 (exactly zero at step 0 by construction)
- [x] extended to 12 layers — `0,3,6,9,12,15,18,21,24,27,30,33`
- [x] parameter budget **14,155,776 exactly (0.00% deviation)**

## Chain A — Qwen3-4B / Qasper

- [x] `base_fullctx` — 0.2444 (historical, authenticated)
- [x] `split_zero` — covered by the FP32 parity gate (bit-exact ⇒ identical score)
- [x] `ours_split_local256_fixed` — 3 seeds, **0.2940**
- [x] `ours_split_current_fixed` (w=1) — 0.2863
- [x] `param_matched_additive` — 3 seeds, **0.2958**
- [x] `param_matched_LoRA` (r=58, +0.69% budget) — 3 seeds, **0.2908**
- [x] `attention_only` — 0.2933 (historical, 5 seeds, authenticated)
- [~] `old_fixed_add` / `old_DEX_minus` / `old_DEX_plus` — authenticated in
      `SOURCE_AUDIT.md` §3 but **not re-run under this session's protocol**;
      `param_matched_additive` is the strictly-matched stand-in
- [x] LR / smoke screening — rescue #4, three points
- [x] locked config — `split_local256_fixed`, w=256, gamma=1, lr 5e-4, 156 steps
- [x] ≥3 seeds — 3 per trained arm (**not 5**; time)
- [x] paired bootstrap — hierarchical over examples × seeds, 20,000 draws

## Chain A — controls and rescue tree

- [x] #4 LR 0.3× / 1× / 3× — 0.2622 / **0.2940** / 0.2803
- [x] #5 window w=1 / 64 / 256 — 0.2863 / 0.2930 / **0.2940**
- [x] #7 gamma 0.25 / 0.5 / 1.0 — 0.2639 / 0.2840 / **0.2940**
- [x] zero-window and shuffled-window read controls —
      `divergence_split_local256_s0.json` (zero ⇒ divergence exactly 0;
      shuffle ⇒ −61% divergence)
- [x] branch divergence / correction norm / cosine per layer — same file
- [ ] #6 layer-placement (middle / late) — not run; #4/#5/#7 already showed the
      canonical point is the best in the searched space
- [ ] #8 dynamic gate (Stage B) — **correctly not run**: pre-registered as gated on
      the fixed-gamma version working, which it did not

## Chain A — downstream

- [x] HotpotQA screening (300 examples, seed 0) — base 0.5959 / split 0.5848 /
      additive 0.6034. **In-repo scorer**: the pinned official
      `hotpot_evaluate_v1.py` is absent from this machine (audit §6.6)
- [~] LoCoMo official, `--max-context-tokens 32000` — running at time of writing
- [ ] RULER 4K/8K — **not run**: `/work/mingze/ruler_official_data*/` does not
      exist on this machine
- [ ] LongBench-v2 / IFEval — not reached

## Chain B — small-model controlled comparison

- [x] audit small-model trainer / data — **none exists in either repo**; §8.1's
      "reuse an existing framework" could not be satisfied (audit §6.3)
- [x] fixed config + manifest — C4 (cached), 160M tokens, sha256 in
      `out_smalllm_20260814/data_manifest.json`; T=480M is **3 epochs**, stated
- [x] `small_vanilla` — val_loss 4.0307
- [x] `small_diffv2_from_scratch` — val_loss 4.0267
- [x] save `vanilla@T0` — sha256 `b9399955b8551413…`
- [x] fork `ours` from that T0 — val_loss 4.0860, sha asserted at load
- [x] fork `additive` from the **same** T0 — val_loss 4.0789, sha asserted
- [x] step-0 parity for both forks — identical loss **3.5288**, identical 786,432
      trainable params, differing gnorm (0.006 vs 0.034)
- [x] loss-vs-token curve — logged every 1000 steps in both run JSONs
- [x] params / wall time / peak VRAM reported both ways — report §4
- [~] seed 1 for `vanilla` and `diffv2` — running; added because the
      DiffV2-vs-vanilla claim rested on n=1
- [ ] attention-sink / outlier / qk-logit diagnostics — not run (time)
- [x] recovery ratio — **N/A**, per the pre-registered rule: denominator
      −0.0041 nats (near zero) and numerator has the wrong sign

## Reporting

- [x] registry — `diff_split_experiment_registry.jsonl` (28 runs)
- [x] ledger — `diff_split_gpu_job_ledger.tsv`
- [x] summarise positive AND negative results — `DIFF_SPLIT_COMPARISON_REPORT.md`
- [x] local commits (no push) — `8d065c2`, `c395191`
- [x] reproduction commands — report §7

## Defects found and fixed (all in audit §6.4–6.5)

- [x] `set_trainable()` silently froze the split ⇒ would have reported base parity
      as a result; same trap re-hit by LoRA; both now asserted
- [x] divergence probe never forwarded its ablation flags ⇒ "controls" were copies
- [x] HotpotQA "base" condition re-ran the split model ⇒ fake baseline; the same
      latent bug existed in the LoCoMo and RULER scripts, all three fixed
- [x] duplicate C4 cache shards ⇒ would have repeated a third of training
- [x] HF label double-shift ⇒ off-by-one objective
- [x] fork data misalignment ⇒ forks would have trained on different data than the
      control they are compared against
