# DEX control study — does the minus sign do the work?

Target paper: **Understanding Differential Transformer Unchains Pretrained
Self-Attentions**, Kong, Jang & Kwak, arXiv:2505.16333 (NeurIPS 2025), Sec. 3
("Differential Extension"), Eq. (3)–(5), Fig. 9, App. B.3/B.4/E.1.

Status: phases 1–4 complete (implementation, unit tests, smoke test, LR probe).
Phase 5 (formal runs) and 6 (statistics) are filled in at the bottom as runs land.

---

## 1. What the paper actually specifies

Read from the PDF, not from memory:

| Item | Paper | Where |
|---|---|---|
| Core op | `O = softmax(QK^T/sqrt d) V`, `O' = O − λ f_D(O)` | Eq. (3), Fig. 9 |
| Per-head form | `O' = O − λ(t)·1[h ∈ H]·f_D(O)`, then concat → `W_O` | Eq. (5) |
| `f_D` | learnable projection with weight `W_D ∈ R^{dv×dv}`; Fig. 9 hands **one** `f_D` to every head of a layer | Sec. 3.1, App. E.3 |
| λ | learnable scalar, annealed | Sec. 3.3 |
| λ schedule | `λ(t) = (1−α)(t/T)λ_init + α·λ_learn`, `α = min(1, t/T)`; `λ_learn` init ≈ 0 | Eq. (4) |
| `λ_init` | depth-aware DIFF schedule `0.8 − 0.6·exp(−0.3·(l−1))` (best in App. B.3) | Sec. 4.1, App. B.3 |
| Head selection | top-k **highest-entropy** heads per layer (best); low-importance is the alternative; `k = H/2` | Sec. 3.2, App. B.4 |
| Trained params | `W_K, W_V, W_O` **plus** `W_D, λ_learn`; everything else (incl. `W_Q`, FFN) frozen | Sec. 3.4 |
| Optimiser | AdamW, cosine LR, peak 1e-4 for partial-FT methods, warmup ratio 0.03, 1 epoch | App. E.1 |
| Adaptation data | 887M tokens of Dolmino mix, 32k context, < 0.01% of pretraining | Sec. 4.1, App. E.1 |
| Models | Llama-3.1-8B, Llama-3.2-3B/1B, Qwen-2.5-1.5B/0.5B | Sec. 4.1 |
| Official code | **none released** — checked arXiv abs/HTML, the NeurIPS proceedings page and OpenReview | — |

Reported headline: +4.0 avg over 11 LM benchmarks on Llama-3B vs LoRA +1.6 / full FT +1.3.

## 2. The control the paper does not run

`f_D` is a free `d_v × d_v` projection, so

```
O − λ f_{W_D}(O)   ==   O + λ f_{−W_D}(O)
```

is an *identity*, not an approximation: the two parameterisations have exactly
the same function class, and (`W_D` ↦ `−W_D`) is a bijection between them. The
paper's ablations (Table 5) vary head selection and the λ mechanism, never the
sign, and never compare against a plain attention-output residual adapter or
against tuning `W_K,W_V,W_O` alone. Since DEX also unfreezes `W_K,W_V,W_O`
(566M params on a 4B model here) while its own adapter is ~0.6M, the reported
gain could come from (a) the subtraction, (b) extra adapter capacity, or
(c) continued attention finetuning. This study separates the three.

## 3. Where the code lives

| File | Role |
|---|---|
| `deltamem/core/dex.py` | `AttentionOutputAdapter` (Eq. 3/4/5), `DexOutputProjection`, `DexConfig` with the variant table, head-plan resolution, trainable-set control, diagnostics |
| `scripts/dex_select_heads.py` | per-layer/head attention **entropy** and gate-gradient **importance**; writes one plan JSON reused by every run |
| `scripts/dex_train_qasper.py` | single training/eval entry point for all variants |
| `scripts/dex_smoke.py` | phase-4 smoke test with the required diagnostics |
| `scripts/run_dex_matrix.sh` | sequential variant×seed runner for one GPU |
| `tests/test_dex.py` | 28 unit tests (sign-flip equivalence, head isolation, freezing, λ schedule, param-count matching) |

**Insertion point.** `attn_output.reshape(B, T, H·Dh)` — the tensor fed to
`o_proj` — is exactly the concatenation of the per-head `O_h` in head order
(`transformers/models/qwen3/modeling_qwen3.py`, mirrored at
`deltamem/core/prefix_steer.py:1162-1168`). Wrapping `o_proj` and viewing its
input as `[B, T, H, Dh]` therefore applies `f_D` to each head's `O_h`
individually, before concat+`W_O`, which is Fig. 9 / Eq. (5) verbatim, without
re-implementing the attention kernel. Non-selected heads are multiplied by a 0/1
head mask *before* the addition, so they are returned bit-identical.

Backbone: **Qwen3-4B-Instruct-2507**, 36 layers × 32 heads × `d_v = 128`,
GQA with 8 KV heads. `k = 16` heads per layer (half, per the paper).

## 4. Variants (all share one code path; only the config changes)

| Variant | Forward | Trainable tensors | Trainable params |
|---|---|---|---|
| `base` | `O' = O` | — | 0 |
| `dex_minus` (paper) | `O' = O − λ(t) f_D(O)` | 180 | 566,820,900 (adapter 589,860 + attn 566,231,040) |
| `dex_plus` | `O' = O + λ(t) f_D(O)` | 180 | 566,820,900 (identical) |
| `residual_adapter` | `O' = O + g_D(O)` (λ ≡ 1, not learnable) | 144 | 566,820,864 (36 fewer = the per-layer λ) |
| `attn_only` | `O' = O` | 108 | 566,231,040 |
| `adapter_only` | `O' = O − λ(t) f_D(O)`, attention frozen | 72 | 589,860 |

`dex_plus` and `residual_adapter` are *not* redundant: the former keeps the
paper's λ(t) annealing/learning, the latter is the plain residual adapter with a
constant unit scale. They coincide only if `dex_plus` is run with
`--lambda-anneal-steps 0` and a fixed `λ=1` (tested in
`test_residual_adapter_equals_plus_with_unit_lambda`).

## 5. Unit tests (28 passed)

Key results (`pytest tests/test_dex.py -q`):

* **Sign-flip equivalence.** `minus(W_D)` vs `plus(−W_D)`, float64 adapter:
  `max_abs_err = 0.000e+00`, `mean_abs_err = 0.000e+00`. Whole tiny-Qwen3 model
  logits: `max_abs_err = 0.000e+00`.
* λ(t) matches Eq. (4) at t = 0, T/4, T/2, T−1, T, 2.5T; λ(0) = 0 and λ(≥T) = λ_learn.
* `diff_lambda_init` matches `0.8 − 0.6 exp(−0.3 l)`.
* Non-selected heads bit-identical; selected heads changed.
* Per-variant trainable sets exactly as in the table; `W_Q`, MLP and embeddings
  never require grad in any variant; `adapter_only` produces **no** attention grads.
* `dex_minus` and `dex_plus` have identical parameter shapes, counts and (same
  seed) identical initial weights.

## 6. Smoke test on the real 4B backbone (`out_dex/smoke.json`)

One forward+backward+AdamW step per variant, one full Qasper example
(1024-token tail), λ evaluated mid-annealing (t = T/2 = 39, T = 78):

| variant | trainable | adapter | loss | grad norm | λ(layer 0) | ‖λf_D(O)‖/‖O‖ | cos(O, Δ) | selected heads changed | non-selected bit-identical | frozen params with grad | max |Δθ| after step |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|---:|
| base | 0 | 0 | 3.6577 | — | — | — | — | — | — | 0 | 0 |
| dex_minus | 566,820,900 | 589,860 | 3.5631 | 46.03 | 0.0889 | 0.0802 | +0.0001 | yes | yes | 0 | 2.0e-05 |
| dex_plus | 566,820,900 | 589,860 | 3.7339 | 47.89 | 0.0889 | 0.0802 | +0.0003 | yes | yes | 0 | 2.0e-05 |
| residual_adapter | 566,820,864 | 589,824 | 3.9412 | 54.71 | 1.0000 | 0.4219 | +0.0013 | yes | yes | 0 | 2.0e-05 |
| attn_only | 566,231,040 | 0 | 3.6577 | 47.28 | — | — | — | — | — | 0 | 2.0e-05 |
| adapter_only | 589,860 | 589,860 | 3.5631 | 0.83 | 0.0889 | 0.0802 | +0.0001 | yes | yes | 0 | 2.0e-05 |

Consistency checks that must hold, and do:

* `base` and `attn_only` have **identical** initial loss (3.6577) — the wrapper
  is a no-op without an adapter.
* `dex_minus` and `adapter_only` have identical initial loss (3.5631) — same
  adapter, same forward, different trainable set.
* λ(39) = 0.0889 = 0.25 · λ_init(layer 0) = 0.25 · (0.8 − 0.6e^{-0.3}), i.e. Eq. (4) exactly.
* λ gradient reaches all 36 layers for `dex_minus`/`dex_plus`/`adapter_only` and
  is absent for `residual_adapter` (λ fixed) — as designed.
* every trainable tensor has a finite non-zero gradient; no frozen tensor has one.

**Paired sign-flip diagnostic on the 4B model** (`W_D` for minus vs `−W_D` for
plus, same seed, λ mid-anneal, full 4.5k-token example):

```
loss(minus, W_D)      = 2.342708110809326
loss(plus,  -W_D)     = 2.342708110809326
|Δloss|               = 0.0
logits max_abs_err    = 0.0
logits mean_abs_err   = 0.0
```

So at initialisation the two variants are *the same function* up to a sign flip
of a free parameter. Any difference in final results can only come from
optimisation dynamics (initial-direction asymmetry under a shared init
distribution, interaction with the λ schedule), never from expressivity.

### Bugs found and fixed by the smoke test

1. `.to(dtype=float32)` on the head-gate wrapper silently converted the wrapped
   `o_proj` of the bf16 backbone → dtype mismatch at the first matmul.
2. Head-truncating a Qasper example left only `-100` labels → `loss = NaN` with
   zero gradients. Fixed by tail-truncation (answers live at the end).
3. bf16 master weights cannot absorb `lr ~ 1e-4` updates on pretrained weights
   (bf16 eps ≈ 4e-3 relative); trainable parameters are now kept in fp32 with
   bf16 autocast, so attention finetuning is not silently a no-op.

## 7. Experimental design for the formal runs

The paper's own setting (887M tokens, 32k context, 5 models, 11 lm-eval
benchmarks) is far out of budget here, and the instruction was to prefer a setup
this repo already runs stably. Vehicle:

* **Backbone** Qwen3-4B-Instruct-2507 (frozen except each variant's set).
* **Adaptation data** Qasper QA, 935 training examples composed exactly like the
  repo's existing runs (`--train-papers 800 --max-chunk-tok 256 --max-ctx-tok 4500
  --max-ans-tok 24 --train-target-n 935 --max-yesno-frac 0.03 --data-compose-seed 42`).
* **Budget** 156 optimizer updates × grad-accum 16 × batch 1 ≈ 11M supervised
  tokens — the same budget as the repo's memory-sidecar Qasper runs, so DEX
  numbers land on the same scale as `out_swa_sharedv/`.
* **Metric** Qasper validation F1/EM over all 187 val examples (75 papers),
  greedy, 24 new tokens — the repo's existing protocol; plus validation CE.
* **Deviation from the paper, stated plainly**: this is supervised QA adaptation,
  not LM-corpus adaptation, and the metric is one task rather than 11 zero-shot
  benchmarks. It tests the *causal* question (does the minus sign matter, given
  matched capacity and matched attention tuning) but not the paper's absolute
  claims.

Config checklist — identical across variants unless it is the variable:

| Item | Value |
|---|---|
| Pretrained checkpoint | Qwen3-4B-Instruct-2507 (same files, same dtype) |
| Training data + order | `random.Random(seed)` over the same 935 examples; order depends on seed only, never on variant |
| Examples / tokens | 156 × 16 = 2496 microbatches |
| Batch / grad accum | 1 / 16 |
| Optimiser | AdamW, wd 0.0, betas default |
| LR schedule | cosine to 0, warmup ratio 0.03 |
| Peak LR | one value for every variant (chosen on the `attn_only` control, see §8) |
| Steps | 156 optimizer updates |
| Layers / heads | all 36 layers, `k = 16` entropy-high heads per layer, from one shared plan file |
| Adapter dim | 128 × 128 per layer |
| Init scale | `nn.Linear` default, same distribution and same seed for minus/plus |
| λ init | depth-aware DIFF, `λ_learn = 0` |
| λ annealing | T = 78 (half the run) |
| Eval prompts | identical (repo's Qasper prompt) |
| Generation | greedy, 24 tokens, identical |
| Seeds | 0, 1, 2 |
| Grad checkpointing | on for every run (exact recompute) |

## 8. Learning-rate probe

`attn_only` (the largest trainable set, hence the most at-risk variant, and a
*control* rather than the method under test) was run for 30 updates at the
paper's 1e-4 and at 2e-5. Validation CE on the same 32 held-out examples:

| updates | lr 1e-4 | lr 2e-5 |
|---:|---:|---:|
| 0 (frozen) | 2.1065 | 2.1065 |
| 10 | 0.8269 | 0.7394 |
| 20 | 0.9281 | 0.7407 |
| 30 | 0.8858 | **0.7255** |

1e-4 overshoots and starts climbing again by update 20; 2e-5 decreases
monotonically. **lr = 2e-5 is used for every variant and every seed.** This is a
deliberate deviation from the paper's 1e-4, which was tuned for a global batch of
256 x 32k tokens (~8M tokens/update); here one update sees ~72k tokens. The value
was picked on the control variant, never on DEX, and is shared by all runs.

Wall-clock: 30 updates in 6.9 min while sharing a GPU, so a 156-update run plus
the 187-example greedy evaluation is ~35-40 min.

## 9. Formal results

57 runs in total: a first matrix (v1), then a full re-run (v2) of every
lambda-dependent condition after a code review found an off-by-one in
`lambda_init` (see §9.4). The tables below use **v2 for every condition whose
numbers depend on lambda** and v1 for the three that provably do not
(`base` and `attn_only` have no adapter; `ungated_adapter` pins lambda == 1).
The `set_trainable` rewrite was verified not to change any trainable-parameter
count, and `base`/`attn_only` initial losses are bit-identical before and after,
which is what licenses mixing the two batches.

### 9.1 Main table — Qasper val F1 (187 examples, greedy)

| Variant | Trainable | s0 | s1 | s2 | s3 | s4 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | .2444 | — | — | — | — | **.2444** | — |
| dex_minus (paper) | 566,820,900 | .2902 | .2933 | .2907 | .2879 | .2930 | **.2910** | .0022 |
| dex_plus (sign control) | 566,820,900 | .2959 | .3014 | .2936 | .2889 | .2950 | **.2950** | .0045 |
| attn_only | 566,231,040 | .2980 | .2997 | .2872 | .2946 | .2872 | **.2933** | .0059 |
| ungated_adapter | 566,820,864 | .2897 | .2814 | .2792 | .2928 | .2709 | **.2828** | .0087 |
| adapter_only | 589,860 | .2416 | .2423 | .2457 | — | — | **.2432** | .0022 |

Final validation CE: base 2.1065, dex_minus 0.7074, dex_plus 0.6788,
attn_only 0.6954, ungated_adapter 0.7400, adapter_only 2.1013. EM ~0 everywhere.

| Comparison | seed-level ΔF1 (n, p) | example-level ΔF1 | 95% CI | p | dz |
|---|---:|---:|---|---:|---:|
| A: dex_minus − dex_plus | −0.0039 (5, **0.039**) | −0.0040 | [−0.0098, +0.0016] | 0.168 | −0.10 |
| B: dex_minus − ungated_adapter | +0.0082 (5, 0.158) | +0.0082 | [−0.0035, +0.0194] | 0.166 | +0.10 |
| C: dex_minus − attn_only | −0.0023 (5, 0.465) | −0.0023 | [−0.0100, +0.0050] | 0.542 | −0.05 |
| D: dex_minus − adapter_only | +0.0482 (3, 0.001) | +0.0478 | [+0.0218, +0.0737] | 0.0005 | +0.26 |

### 9.2 Follow-up conditions

| condition | n | F1 | Std | val CE | λ end | ‖Δ‖/‖O‖ |
|---|---:|---:|---:|---:|---:|---:|
| mirror_plus (dex_plus initialised at −W_D) | 3 | .2959 | .0028 | 0.7251 | −0.0000 | 0.0001 |
| fix_minus (λ pinned at λ_init) | 3 | .2908 | .0112 | 0.6881 | 0.7357 | 0.3163 |
| fix_plus (λ pinned at λ_init) | 3 | .2914 | .0079 | 0.7535 | 0.7357 | 0.3169 |
| fix_adapteronly (λ on, attention frozen) | 3 | .2379 | .0043 | 1.0446 | 0.7357 | 0.3196 |

| Comparison | seed-level (n, p) | example-level Δ | 95% CI | p | dz |
|---|---:|---:|---|---:|---:|
| **E: mirror_plus − dex_minus (must be 0)** | +0.0045 (3, **0.022**) | +0.0045 | [−0.0035, +0.0132] | 0.293 | +0.08 |
| F: fix_minus − fix_plus (sign, λ on) | −0.0006 (3, 0.932) | −0.0006 | [−0.0130, +0.0117] | 0.926 | −0.01 |
| G: fix_minus − dex_minus (λ on vs Eq. 4) | −0.0006 (3, 0.942) | −0.0006 | [−0.0124, +0.0118] | 0.926 | −0.01 |
| H: fix_adapteronly − base | −0.0016 (1) | −0.0015 | [−0.0214, +0.0185] | 0.882 | −0.01 |

### 9.3 The calibration that decides how to read A

`mirror_plus` is `dex_plus` initialised at exactly `−W_D`. Because
`(W_D, λ) ↦ (−W_D, λ)` is a bijection between the minus and plus
parameterisations, and AdamW is elementwise and sign-equivariant, the two
trajectories are mirror images: **E must be exactly 0 in exact arithmetic.**

Measured: E = +0.0045 with a seed-level paired-t p of **0.022** — larger in
magnitude than comparison A (−0.0039) and with a smaller p-value. A paired
seed-level test at n=3–5 therefore reports "significant" differences between two
runs that are provably the same algorithm; bf16 kernel non-determinism and data
order alone produce that much spread. Nothing below roughly **0.005 F1** is
resolvable in this design, and the example-level tests (which do not treat 3–5
seeds as the sample) put A at p = 0.17 and E at p = 0.29.

### 9.4 The lambda schedule switches the branch off, and the lambda_init fix

| condition | λ peak | λ at end | ‖λ f_D(O)‖/‖O‖ at end |
|---|---:|---:|---:|
| dex_minus / dex_plus (Eq. 4, T = 78) | 0.157 avg (0.050 at layer 0) | ≈0.0000 | 0.0001 |
| adapter_only (Eq. 4) | same | 0.0008 | 0.0005 |
| fix_* (λ ≡ depth-aware λ_init) | 0.736 avg | 0.736 | 0.32 |
| ungated_adapter (λ ≡ 1) | 1.0 | 1.0 | 0.42 |

Eq. (4) hands control to `λ_learn` at `t ≥ T`, and `λ_learn` starts at 0; 156
updates at lr 2e-5 move it by ~1e-3. So under the paper's own schedule the
differential branch ends training effectively **off**, and the final DEX model is
the attention-finetuned model — which is exactly what C reports. G shows that
forcing the branch on all the way changes nothing (−0.0006, p = 0.93).

The v1 batch used `diff_lambda_init(layer_idx + 1)`, giving layer 0 a λ_init of
0.3555 instead of the official 0.2 (microsoft/unilm `lambda_init_fn(depth)` takes
the 0-based layer index). Everything lambda-dependent was re-run after the fix.
Robustness check V: `dex_minus` v2 − v1 = −0.0027 (example-level p = 0.41), i.e.
inside the E noise floor, and no comparison changes sign or verdict.

## 10. Answer to the proposition

> DEX's benefit comes from differential subtraction rather than ordinary adapter
> capacity or attention finetuning.

**Not supported** in this setting.

1. **The sign is not the source.** Under the paper's schedule the minus sign is
   if anything *worse* (A = −0.0039; seed p = 0.039, example p = 0.17), and with
   the branch pinned on the difference vanishes (F = −0.0006, p = 0.93). Both are
   at or below the mirror-run noise floor (E = +0.0045, seed p = 0.022 for a
   comparison that must be exactly zero). This is the expected outcome: `f_D` is
   a free `d_v × d_v` projection, so minus and plus are the same function class
   (verified to 0.0 max-abs-error on the 4B model) and their training
   trajectories are mirror images.
2. **Almost all of the gain is attention finetuning.** base → dex_minus is
   +0.0466 F1; base → attn_only alone is +0.0489. C = −0.0023 (p = 0.54).
   The adapter alone is worth nothing: adapter_only 0.2432 vs base 0.2444, and
   with λ forced on 0.2379 (H, p = 0.88) — even though its validation CE drops
   from 2.11 to 1.04.
3. **No demonstrated advantage over an ordinary adapter.** B = +0.0082,
   p = 0.16–0.17, and `ungated_adapter` differs from DEX in the λ treatment as
   well as the sign, so even that trend is not attributable to "differential"
   adaptation. The capacity-matched ordinary adapter is `dex_plus`, and that is
   comparison A.

Interpretation rules one, two and three all apply simultaneously.

What this does **not** show: that DEX fails in the paper's regime. The paper
adapts on 887M LM tokens at 32k context over thousands of updates and reports an
11-benchmark average; here it is 156 updates of supervised QA adaptation (~11M
tokens) on one task with one 4B backbone, at a budget where Eq. (4) never reaches
an interesting operating point.

## 11. Limitations

* One backbone (Qwen3-4B-Instruct-2507), one task (Qasper QA F1), one budget.
* Supervised QA adaptation, not the paper's LM-corpus adaptation; no lm-eval
  11-benchmark average, no needle-in-a-haystack, no ICL suite.
* LR 2e-5 (chosen on the `attn_only` control) instead of the paper's 1e-4,
  because one update here sees ~72k tokens instead of ~8M.
* **Resolution floor ~0.005 F1** (§9.3). Seed-level paired tests at n=3–5 are not
  trustworthy here; example-level paired tests are reported alongside and are the
  ones to read.
* `W_D` is per layer, shared across that layer's selected heads (Fig. 9 pseudocode
  + App. E.3). A per-head `W_D` is a defensible alternative reading and was not run.
* Head selection is entropy-based on 8 Qasper calibration sequences of 512 tokens;
  the importance-based plan is computed but was not run as a variant.
* Only Qwen3 attention is wrapped; the paper used Llama-3 and Qwen-2.5.
* EM is ~0 on this task, so only F1 and CE discriminate.
* Attention weights are not checkpointed (only adapters), so runs cannot be
  re-evaluated later without retraining.

## 12. Review findings fixed after the first matrix

A code review of `deltamem/core/dex.py` raised ten points. Seven were real:

| # | Finding | Real? | Affected v1 results? |
|---|---|---|---|
| 1 | `lambda_init` off-by-one (layer 0 got 0.3555, official is 0.2) | yes | **yes** — every λ-dependent run re-run as v2 |
| 2 | Library default is a no-annealing DEX | yes (footgun) | no — the trainer always passed T = 78 |
| 3 | `cfg.layers` subset still unfroze every layer's `W_K/W_V/W_O` | yes | no — all runs used every layer |
| 4 | Head-plan criterion check was a dead `pass`; `importance_low` would have read entropy scores | yes | no — all runs used entropy_high + the entropy plan |
| 5 | `residual_adapter` is not a capacity-matched control | yes | naming/claim only; renamed `ungated_adapter`, matched control is `dex_plus` |
| 6 | `adapter.to(dtype=...)` demoted λ to bf16 | yes | learnable λ was fp32 anyway (trainer promotes trainable params); non-learnable λ buffers were bf16 |
| 7 | `cos_delta_o` had the opposite sign to Fig. 11 | yes | diagnostics only; both `cos_raw_corr_o` and `cos_delta_o` are now logged |
| 8 | `.item()` in `current_lambda` forces a per-layer GPU sync | yes | performance only |
| 9 | Wrapping changes `state_dict` keys | valid caveat | no — adapters are saved separately and reloaded via `attach_dex` |
| 10 | `view`→`reshape`, missing `in_features` guard, Qwen3-only | yes | no |
| 11 | `plan.get(want, plan["scores"])` evaluates the default eagerly → KeyError on a named-block-only plan | yes | no — every plan carried a `scores` block |
| 12 | dead-config check ran before variant resolution, so `base`/`attn_only` with `fd_init="zeros"` were wrongly rejected | yes | no — no run used zero-init |

Tests grew from 28 to 43, with a regression for each fix (0-based λ_init,
paper-style variant refusing to run without annealing, dead-config rejection,
layer-subset freezing, head-plan mismatch, importance block, fp32 λ under a bf16
backbone, sign-free cosine, shape guard).

## 13. Reproduction

```bash
# 1. head plan (once, shared by every run)
python scripts/dex_select_heads.py --model-path /work/mingze/models/Qwen3-4B-Instruct-2507 \
  --num-samples 8 --seq-len 512 --output out_dex/head_plan_qwen3_4b.json

# 2. unit tests + smoke test
pytest tests/test_dex.py -q                       # 39 passed
python scripts/dex_smoke.py --head-plan out_dex/head_plan_qwen3_4b.json --seq-len 1024 \
  --anneal-steps 78 --output out_dex/smoke_v2.json

# 3. matrix and follow-ups (one GPU per call; DEX_TAG_PREFIX namespaces a re-run)
DEX_TAG_PREFIX=v2_ bash scripts/run_dex_matrix.sh 0 2e-5 "base:0 dex_minus:0 dex_plus:0 attn_only:0 adapter_only:0"
DEX_TAG_PREFIX=v2_ bash scripts/run_dex_extra.sh  0 2e-5 "mirror_plus:0 fix_minus:0 fix_plus:0 fix_adapteronly:0"

# 4. tables, comparisons, figures
python scripts/dex_analyze.py --glob 'out_dex/final/dex_*.json' \
  --out-md out_dex/dex_results_final.md --fig-dir out_dex/figs_final
python scripts/dex_report_tables.py            # v1/v2 provenance + comparisons A-H, V
```

Environment: Python 3.10.19, torch 2.9.0+cu128, CUDA 12.8, 8x NVIDIA H100 80GB,
repo `/work/mingze/delta-mem` (not a git repository; recorded as such in every run
JSON). 26-28 min per training run, 8.4 min for `base` (eval only). Artefacts:
`out_dex/*.json` (config, full log, per-example predictions, environment,
command), `out_dex/*_adapter.pt`, `out_dex/dex_results_final.md`,
`out_dex/dex_stats_final.json`, `out_dex/figs_final/*.png`, and the pre-fix v1
batch kept alongside for the robustness check.


## 14. Nuisance-variance probe (mechanism, not accuracy)

The F1 study answers "does the minus sign buy anything". It does not answer the
mechanistic question:

> **Does the DEX correction actually target long-context nuisance variance?**

`scripts/dex_nuisance_probe.py` is the diagnostic for that, and it is
deliberately run *before* any variance-conditioned method is implemented.

### 14.1 Construction

A grouped binding probe over real Qasper prose. For each group the query `q` and
the evidence content are fixed; only nuisance varies across the K
instantiations: which filler paragraphs surround the needles, the needle order,
and each needle's depth in the context.

Both candidate values are always present. The queried subject owns `v1` in one
evidence condition and `v2` in the other, while a foil subject owns the
complementary value, so the multiset of values in the context is identical in
both conditions — only the subject→value **binding** differs. That keeps the
evidence swap free of surface confounds.

### 14.2 Readout, and a deliberate deviation

The scalar readout is fixed across evidence conditions:

```
g(x) = log p(v1 | x) − log p(v2 | x)
```

The literal "gold logit margin" from the brief would compare *different* targets
in the two evidence conditions (gold is `v1` in one and `v2` in the other), so
`|E[s|e1] − E[s|e2]|²` would be near zero for a model that is equally confident
in both — it would measure confidence symmetry, not evidence sensitivity. The
log-odds readout keeps the target fixed, so

```
V_nuis = mean_g Var_k[ g ]                    (within one evidence condition)
S_evid = mean_g ( E_k[g | v1] − E_k[g | v2] )²
NSR    = V_nuis / (S_evid + eps)
```

is a genuine nuisance-to-signal ratio on one scalar.

### 14.3 Hidden-state metrics

Per layer, on the probe tokens (mean over the last 16 query positions), with
`Y` = per-head attention output at the DEX insertion point, `Δ` = the signed
update the adapter actually applies, `Ỹ = Y + Δ`, all centred **within a group**:

```
VRR       = 1 − Σ‖Ỹ − mean_k Ỹ‖² / Σ‖Y − mean_k Y‖²
AntiAlign = − ⟨r_Δ, r_Y⟩ / (‖r_Δ‖‖r_Y‖)
MeanShift = ‖mean_k Ỹ − mean_k Y‖ / (‖mean_k Y‖ + eps)
```

All three are computed **only over the heads DEX adapts** (16 of 32 per layer).
On the other heads `Δ ≡ 0`, so including them would mechanically halve VRR.
`base` and `attn_only` have no adapter; for them the same head set is taken from
the shared head plan so that `hidden_var` remains comparable.

### 14.4 Calibration, and one limitation stated up front

A single unique needle is retrieved perfectly by this backbone (readout accuracy
1.000, NSR 5e-4 at 1.5k tokens), so the first design had no nuisance variance to
study. The binding construction raises the nuisance variance of the readout by
roughly an order of magnitude (V_nuis 0.66 → 4.1 at 6k tokens, 10 distractor
needles), but **readout accuracy stays at 1.000**: the model still gets the
binding right every time, and the nuisance moves only the margin. So this probe
measures variance of a continuous readout, not error rate; a claim that nuisance
variance is behaviourally costly would need a harder regime (32k+, or a task the
backbone does not saturate).

Two bugs were found and fixed while building it: the forward hooks also fired
during the two candidate-scoring forwards, so captured rows outnumbered prompts
and the group index map silently misaligned (now gated to the canonical forward,
with an assertion that rows == prompts); and the head-mask restriction above.

### 14.5 What this will and will not license

If `VRR_DEX ≈ VRR_attn_only` and `VRR_adapter_only ≈ 0`, the F1 result upgrades
from "no performance benefit" to "the differential branch does not independently reduce
long-context nuisance variance either" — which is the honest motivation test for
any closed-form `λ* = Cov(Y,C)/Var(C)` successor. If instead the correction does
anti-align with the nuisance residual, the closed-form variant becomes worth
implementing, and the remaining design problem is where `μ_C` comes from at
inference time (per-layer/head EMA is the deployable first version; the group
mean is training-only).

### 14.6 Results

16 groups x K=4 nuisance instantiations x 2 evidence assignments = 128 contexts
of ~6.3k tokens, 12 needles each (target + foil + 10 distractors), probe tokens =
last 16 query positions, seed-0 checkpoints of each variant.

| variant | V_nuis | S_evid | NSR | readout acc | hidden var | VRR | AntiAlign | MeanShift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 3.332 | 821.9 | 0.00405 | 1.000 | 0.00355 | +0.0000 | +0.0000 | 0.00000 |
| attn_only | 1.883 | 308.7 | 0.00610 | 0.992 | 0.00771 | +0.0000 | +0.0000 | 0.00000 |
| dex_minus | 1.568 | 331.0 | 0.00474 | 0.992 | 0.00779 | −0.0000 | −0.0008 | 0.00001 |
| dex_plus | 1.535 | 331.2 | 0.00463 | 1.000 | 0.00717 | −0.0000 | −0.0011 | 0.00001 |
| adapter_only | 3.405 | 821.6 | 0.00415 | 1.000 | 0.00356 | −0.0000 | +0.0014 | 0.00030 |

Four things follow.

1. **The trained DEX correction does nothing at inference.** VRR = 0 to four
   decimals, |AntiAlign| ~ 1e-3 (a meaningful anti-alignment would be order
   1e-1), MeanShift ~ 1e-5. This is the mechanistic confirmation of §9.4: Eq. (4)
   drives λ back to ~0, so the branch is inert by the end of training. It
   therefore cannot be reducing nuisance variance — the answer to "does the DEX
   correction target long-context nuisance variance?" is, for DEX as the paper
   specifies it at this budget, **no, because it does not act at all**.
2. **`adapter_only` is numerically the base model** (V_nuis 3.405 vs 3.332,
   S_evid 821.6 vs 821.9, hidden var 0.00356 vs 0.00355). With attention frozen
   and λ → 0 there is nothing left.
3. **No variant improves the nuisance-to-signal ratio; attention tuning makes it
   worse.** Adaptation shrinks the readout scale, but the signal shrinks more
   than the noise: NSR goes 0.00405 (base) → 0.00610 (attn_only, +50%) →
   0.0047 (dex_minus, +17%). Absolute nuisance variance does fall (3.33 → 1.57),
   which a naive "variance went down" reading would have called success.
4. **Hidden-state nuisance sensitivity roughly doubles** at the adapted heads
   after attention finetuning (0.0036 → 0.0077), it does not fall.

Caveat that bounds all of this: readout accuracy is 0.992–1.000 everywhere, so
these are margin-scale effects, and every DEX number here is for a branch that
annealed itself off. The active-branch condition (λ pinned at the depth-aware
λ_init, ‖Δ‖/‖O‖ ≈ 0.32) is the one that can still say something about whether the
*form* `O − λ f_D(O)` is capable of cancelling nuisance; those two runs are
training now and will be probed with the same harness.

### 14.7 Consequence for the variance-conditioned successor

The closed-form `λ* = Cov(Y, C) / Var(C)` variant is **not yet motivated by
evidence**. Its premise is that `C = f_D(Y)` carries a stable, group-centred
covariance with the nuisance residual of `Y`. In the trained DEX models that
covariance is indistinguishable from zero (AntiAlign ~ 1e-3) — but only because
λ ≈ 0 makes the whole branch inert, so the measurement is uninformative about
the form itself rather than evidence against it. The λ-pinned runs are the
decisive test; if AntiAlign stays at ~1e-3 there too, then the control variable
`C` simply does not explain the nuisance residual and a different control
variable (not a per-head linear map of `O`) is needed before writing the
objective.
