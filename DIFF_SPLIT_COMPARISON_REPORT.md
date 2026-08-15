# Post-hoc Function-Preserving Differential Head Splitting — final comparison report

Branch `agent/diff-head-split-2026-08-14`. Artifacts: `out_diffsplit_20260814/`
(chain A, Qwen3-4B) and `out_smalllm_20260814/` (chain B, small-model controlled).
Registry: `diff_split_experiment_registry.jsonl` (28 runs). Ledger:
`diff_split_gpu_job_ledger.tsv`. Every number below traces to a metrics JSON.

## Headline

The method is **implemented correctly**, **preserves the base function exactly**
at initialisation, and **beats the frozen base by a wide margin** on Qasper
(+0.0496) and LoCoMo (+0.0423). It **never beats a parameter-matched control** —
on either evidence chain, on any benchmark measured — and on LoCoMo, the only
comparison in the study with enough power to resolve the difference, it is
**significantly worse** than the additive control (−0.0312, CI [−0.0429,
−0.0199], p<0.0001).

The mechanism demonstrably works: the branches diverge, and the reader is
content-dependent. That is what makes the null informative rather than a
debugging story. The divergence simply does not convert into accuracy, and
routing the correction through a second attention pass appears to cost something
relative to applying the same parameters directly to the output.

---

## 1. Construction

```
Q+ = q_norm(q_proj(H))                    # frozen
R  = LocalRead_phi(H[t-w:t])              # causal, w=256, values = frozen backbone V
Q- = Q+ + delta_q(R)                      # delta_q ZERO-INIT  => Q- == Q+ at step 0
q_cat = interleave(Q+, Q-)                # pair (2i,2i+1) shares head i's kv group
O+, O- = split(Attn(q_cat, K, V))         # ONE attention call, SAME K/V, same RoPE
O~ = O+ + gamma*(O+ - O-)                 # gamma=1, BEFORE o_proj
Y  = o_proj(O~)
```

**GQA mapping (H=32, G=8).** Doubling queries takes the repeat factor 4→8, so new
head `j` maps to kv group `j//8`. For the pair: `2i//8 = i//4` and
`(2i+1)//8 = i//4` — exactly original head `i`'s group. A front/back split would
map head `i` and `i+32` to groups `i//8` and `(i+32)//8`, *different* groups,
breaking both GQA and function preservation. The interleaving is load-bearing.

**Parameter budget matched exactly, not approximately.** The split reuses the old
sidecar's tensor shapes (`2560→128`, `2560→128`, `128→4096`) and changes only the
*destination* of the third projection. `1,179,648 × 12 = 14,155,776` — a 0.00%
deviation.

## 2. Hard gates — all passed before any training

`realgate_v2.json`, plus 23 unit tests in `tests/test_diff_split.py`.

| gate | result |
|---|---|
| A GQA pairing | 32 q / 8 kv, repeat 4→8, pair shares kv group |
| B FP32 parity (delta_q=0) | **max_abs = 0.000e+00** (bit-exact) |
| B BF16 | max 2.031e-01, mean 1.280e-02, **greedy tokens identical** |
| C gradients / freeze | trainable **14,155,776**; 0 backbone grads; backbone SHA256 over 398 tensors unchanged |
| D KV cache | **28,311,552 bytes in both**; prefill vs cached-decode within base's own gap |
| E causality | past-perturbation effect **0.000e+00**; future 9.788e-01 |

## 3. Chain A — Qwen3-4B / Qasper (187 examples, identical protocol)

| arm | seeds | per-seed F1 | mean F1 | trainable |
|---|---|---|---:|---:|
| `base_fullctx` (historical, authenticated) | 1 | 0.2444 | **0.2444** | 0 |
| `split_local256_fixed` (**canonical**) | 3 | .2928 / .2987 / .2905 | **0.2940** | 14,155,776 |
| `param_matched_additive` | 3 | .2950 / .2981 / .2943 | **0.2958** | 14,155,776 |
| `param_matched_LoRA` (r=58, q/k/v/o) | 3 | .2876 / .2960 / .2887 | **0.2908** | 14,254,080 (+0.69%) |
| `attn_only` (historical, 5 seeds) | 5 | .2980/.2997/.2872/.2946/.2872 | **0.2933** | — |

Hierarchical bootstrap (examples × seeds, 20,000 draws):

| comparison | delta F1 | 95% CI | p |
|---|---:|---|---:|
| **split − base** | **+0.0496** | [+0.0259, +0.0743] | **<0.001** |
| split − additive | −0.0018 | [−0.0205, +0.0157] | 0.85 |
| split − LoRA | +0.0032 | [−0.0188, +0.0248] | 0.77 |
| split − attn_only | +0.0007 | [−0.0200, +0.0210] | 0.95 |

Within-arm seed spread (0.2905–0.2987) exceeds every between-arm gap.

### Pre-registered rescue tree — exhausted, nothing recovered

| rescue step | setting | F1 |
|---|---|---:|
| #4 LR 0.3× | 1.5e-4 | 0.2622 |
| #4 LR 1× | 5e-4 | **0.2940** |
| #4 LR 3× | 1.5e-3 | 0.2803 |
| #5 window | w=1 | 0.2863 |
| #5 window | w=64 | 0.2930 |
| #5 window | w=256 | **0.2940** |
| #7 gamma | 0.25 | 0.2639 |
| #7 gamma | 0.5 | 0.2840 |
| #7 gamma | 1.0 | **0.2940** |

The canonical configuration is already the best point in the searched space.
Step #8 (dynamic gate) was **not** run: it is gated on the fixed version working,
which it did not.

### Mechanism — the branches really do diverge

24 real Qasper val examples, measured immediately before `o_proj`
(`divergence_split_local256_s0.json`):

| condition | cos(O+,O−) | ‖O+−O−‖/‖O+‖ |
|---|---:|---:|
| zero-init | **1.000000** | **0.000000** |
| trained | 0.971521 | **0.185427** |
| trained + zeroed window | 1.000000 | 0.000000 |
| trained + **shuffled** window | 0.994267 | **0.071712** |

Function equivalence at init holds **on real data**, exactly, at all 12 layers.
Divergence grows with depth (layer 0 = 0.0036 → layer 33 = 0.766). Shuffling the
window destroys 61% of it, so the reader uses window content and order — it has
not collapsed to a constant bias.

### LoCoMo (official protocol, full 1540 questions, `--max-context-tokens 32000`)

Scored with `deltamem/eval/locomo_protocol.py::score_locomo_prediction` on the
canonicalised predictions; an independent re-score reproduces the evaluator's own
`by_cat` figure exactly (0.4045, match to 1e-9). Both runs' base predictions are
bit-identical, which is the check that the condition switch works.

| arm | F1 | vs base | 95% CI | p |
|---|---:|---:|---|---:|
| base | 0.4045 | — | — | — |
| split | 0.4468 | **+0.0423** | [+0.0319, +0.0530] | <0.0001 |
| additive | 0.4780 | **+0.0735** | [+0.0595, +0.0875] | <0.0001 |

**split − additive = −0.0312, 95% CI [−0.0429, −0.0199], p<0.0001.**

This is the **only comparison in the whole study whose confidence interval
excludes zero**, and it runs *against* the differential split. It is consistent
across all four question categories:

| category | n | base | split | additive |
|---|---:|---:|---:|---:|
| 1 (multi-hop) | 282 | 0.3799 | 0.4801 | **0.5089** |
| 2 (temporal) | 321 | 0.3206 | 0.3424 | **0.3791** |
| 3 (open-domain) | 96 | 0.1073 | 0.1038 | **0.1231** |
| 4 (single-hop) | 841 | 0.4787 | 0.5146 | **0.5459** |

So LoCoMo upgrades the finding from "no evidence of a differential-specific
benefit" to "significant evidence of a differential-specific *disadvantage*"
against the parameter-matched additive sidecar. Caveat: questions are nested
within 10 conversations and this bootstrap resamples questions independently, so
a conversation-clustered interval would be wider than the one quoted.

### HotpotQA zero-shot transfer (300-example screening subset, seed 0)

| arm | F1 | vs base | 95% CI | p | predictions changed |
|---|---:|---:|---|---:|---:|
| base | 0.5959 | — | — | — | — |
| split | 0.5848 | **−0.0111** | [−0.0440, +0.0217] | 0.51 | 81/300 |
| additive | 0.6034 | +0.0075 | [−0.0293, +0.0445] | 0.69 | 129/300 |

split − additive = **−0.0186** [−0.0535, +0.0162], p=0.29. Same ordering as
Qasper. Both `base` columns are bit-identical across the two runs, which is the
check that the condition switch works (see §6 defect 3).

**Scoring caveat:** the pinned official `hotpot_evaluate_v1.py` recorded in
SOURCE_AUDIT is **not present on this machine**, so these use the in-repo metric.
The previous session recorded that the two agree to 4 decimals, but that could not
be re-verified here. Labelled a **screening subset**, not a full benchmark result.

## 4. Chain B — small-model controlled comparison

No LM pretraining trainer exists in either repo, so §8.1's "reuse a validated
framework" **could not be satisfied**. The model is HuggingFace's own
`Qwen3ForCausalLM` plus the audited DIFF V2 module; only the data pipeline and
training loop are new code.

**Config derived, not preset:** smallest local Qwen3 is 4B and the box is offline,
so the config preserves Qwen3-4B's ratios — GQA 4:1 (8 q / 2 kv, as 32/8) and
intermediate/hidden = 3.75 (Qwen3-4B: 3.80). Deviation: head_dim 64, not 128.

**Data:** cached `allenai/c4:en`, on-disk order, Qwen3-4B tokenizer, 160M tokens,
sha256 in `data_manifest.json`. T = 480M tokens is therefore **3 epochs** — stated,
not implied.

| arm | val_loss | ppl | trainable | attention params | wall | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| `small_vanilla` | **4.0307** | 56.30 | 106,500,096 | 5,243,904 | 70.5m | 34.0 GB |
| `small_diffv2_from_scratch` | **4.0267** | 56.07 | 108,630,016 | **7,373,824 (+40.6%)** | 78.2m | 34.9 GB |
| `small_ours_posthoc` | 4.0860 | 59.50 | 786,432 | 6,030,336 | 20.3m | 32.5 GB |
| `small_param_matched_additive` | **4.0789** | 59.08 | 786,432 | 6,030,336 | 19.5m | 32.6 GB |

Both post-hoc arms forked from the same `small_vanilla_T0.pt`
(sha256 `b9399955b8551413dedbfbebdfbb4c79a72d67e485958da74313c4ab3176c509`,
asserted at load) and both start at **exactly loss 3.5288** with identical
786,432 trainable parameters — a free step-0 parity gate — while their step-1
gradient norms differ (0.006 vs 0.034), confirming genuinely different paths.

Three results:

1. **DIFF V2 from scratch barely beats vanilla: −0.0041 nats**, for +40.6%
   attention parameters and +10.9% wall time. At this scale the architecture is
   essentially a wash. (It is also undertrained — 480M tokens over a 28.8M-param
   body — so this is a small-scale pilot, not a verdict on DIFF V2.)
2. **Both post-hoc arms are worse than vanilla** (+0.0553 and +0.0482 nats). This
   is expected and is not an architecture comparison: vanilla kept training all
   106M parameters through T1 while the forks trained 786K with a frozen backbone.
3. **ours − additive = +0.0071 nats (ours worse)** — the decisive within-chain-B
   control, and it agrees with chain A.

### Recovery ratio: **N/A**

`(M_ours − M_vanilla) / (M_DiffV2 − M_vanilla) = +0.0553 / −0.0041 = −13.6`.
The denominator is 0.0041 nats (near zero) **and** the numerator has the wrong
sign. Per the pre-registered rule this must be reported as **N/A**; quoting −13.6
or any percentage would be division on noise.

## 5. The ten questions

1. **Implemented correctly?** Yes — every hard gate passes, including bit-exact
   FP32 parity, exact GQA pairing, and an unchanged KV cache.
2. **Step-0 base preserved?** Yes, exactly. FP32 max_abs 0.000e+00 on Qwen3-4B;
   cos(O+,O−)=1.000000 on real data; identical step-1 loss for both chain B forks.
3. **Beats Qwen3 base?** Yes — +0.0496 F1, CI excludes zero, 3 seeds consistent.
4. **Beats attention-only / additive / LoRA?** **No.** −0.0018 vs additive
   (p=0.85), +0.0032 vs LoRA (p=0.77), +0.0007 vs attn_only (p=0.95) — every CI
   spans zero and every gap is smaller than seed noise.
5. **Across Qasper / HotpotQA / LoCoMo / RULER?** Qasper, a HotpotQA screening
   subset, and **full LoCoMo (1540 questions)**. The split beats base on Qasper
   (+0.0496) and LoCoMo (+0.0423) and is level on HotpotQA (−0.0111, n.s.). It
   loses to the additive control on all three, and on LoCoMo that loss is
   **significant**: −0.0312, CI [−0.0429, −0.0199], p<0.0001. **RULER was not
   run** — its generated data is absent from this machine.
6. **Small DIFF V2 > small vanilla?** Only by **0.0041 nats**, for +40.6%
   attention parameters and +10.9% wall time. Effectively a tie at this scale.
7. **How much of the from-scratch gain does ours recover?** **N/A** — see §4.
8. **Is that ratio statistically credible?** No. It is not reportable at all.
9. **Extra cost.** Inference: KV cache **unchanged** (28,311,552 bytes, identical
   to base) and no new K/V heads — the real selling point of this construction.
   Compute: one attention call over 2H query heads instead of H. Chain B measured
   +40.6% attention parameters and −9.6% throughput for full DIFF V2; the post-hoc
   split adds 14.16M trainable parameters (0.35% of a 4B model).
10. **Worth continuing?** On this evidence, **not in this form.** The engineering
    is sound and the KV-cache-free property is genuinely attractive, but across
    two independent evidence chains, four parameter-matched controls, three seeds,
    an exhausted rescue tree and two transfer benchmarks, the differential split
    never beats simply spending the same parameters additively — and on LoCoMo it
    is significantly *worse*. The honest reading is that on an already-trained
    model the gain comes from the added capacity, not from the differential
    structure, and that forcing the correction through a second attention pass
    costs something relative to applying it directly.

## 6. Defects found and fixed this session

All recorded in `DIFF_SPLIT_SOURCE_AUDIT.md` §6.4.

1. **`set_trainable()` silently froze the split** — 156 steps would have reported
   base parity as a result. Caught by checking that `delta_q` was still exactly 0
   in the saved checkpoint. Same trap re-appeared for LoRA and is now asserted for
   both.
2. **Divergence probe ablations were no-ops** — the sweep never forwarded
   `zero=True`/`shuffle=True`, so the "controls" reproduced the trained numbers to
   six decimals. Implementation was fine; the probe was wrong.
3. **HotpotQA "base" was not base for split checkpoints** —
   `set_steer_enabled()` only reaches `PrefixMemSteerAttention` and leaves
   `DiffSplitAttention` untouched, so the base condition silently re-ran the split
   model. Caught because base and ours matched to 4 decimals over 300 examples.
   Fixed; both runs now produce bit-identical base predictions. The bad artifact is
   kept as `hotpot_split_s0_INVALID_base_was_split.json`.
4. **Duplicate C4 cache shards** — the glob matched byte-identical copies,
   reporting 712,635 documents instead of 356,318; a 210M-token budget would have
   made the tail an exact repeat of the head.
5. **HF label double-shift** — HF shifts `labels` internally; passing pre-shifted
   targets trains an off-by-one objective.
6. **Fork data misalignment** — post-hoc arms resume past the manifest end in a
   multi-epoch run and would have restarted at token 0, training on different data
   than the control.

## 7. Reproduction

```bash
# hard gates
python scripts/diffsplit_realgate.py
pytest tests/test_diff_split.py

# chain A canonical (per seed)
python scripts/dex_train_qasper.py --variant diff_split \
  --diff-read-dim 128 --diff-window 256 --diff-gamma 1.0 --seed 0 \
  --steer-lr 5e-4 --steer-lr-schedule constant --steps 156 \
  --steer-layers 0,3,6,9,12,15,18,21,24,27,30,33 \
  --batch-size 1 --grad-accum 16 --grad-checkpointing true \
  --data-compose-seed 42 --train-papers 800 --val-papers 75 \
  --max-ctx-tok 4500 --train-target-n 935 --tag split_local256_fixed_s0

# chain B
python scripts/small_lm_data.py --train-tokens 160000000
python scripts/small_lm_train.py --arm vanilla --tag small_vanilla \
  --total-tokens 480000000 --t0-tokens 336000000 --batch-size 16 --grad-accum 2
python scripts/small_lm_train.py --arm ours --tag small_ours_posthoc \
  --init-from out_smalllm_20260814/small_vanilla_T0.pt --init-sha256 b9399955... \
  --total-tokens 480000000 --t0-tokens 336000000 --batch-size 16 --grad-accum 2

# diagnostics
python scripts/diffsplit_divergence_probe.py \
  --ckpt out_diffsplit_20260814/split_local256_fixed_s0_diff.pt --out /tmp/div.json
```
