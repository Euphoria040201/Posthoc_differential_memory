# Post-hoc Function-Preserving Differential Head Splitting — comparison report

Branch `agent/diff-head-split-2026-08-14`. Outputs in `out_diffsplit_20260814/`
(chain A) and `out_smalllm_20260814/` (chain B). Every number below has a metrics
JSON, log and checkpoint behind it; paths are given per row.

**Status: chain A main matrix complete (3 seeds), chain B running.**

---

## 1. What was built

`deltamem/core/diff_split.py` converts a *pretrained* Qwen3-4B into a
shared-K/V differential attention without changing its function at init:

```
Q+ = q_norm(q_proj(H))                      # untouched, frozen
R  = LocalRead_phi(H[t-w:t])                # causal, w=256, values = frozen backbone V
Q- = Q+ + delta_q(R)                        # delta_q ZERO-INIT  => Q- == Q+ at step 0
q_cat = interleave(Q+, Q-)                  # [+,-,+,-,...] so pair (2i,2i+1) shares
                                            #  the kv group of original head i
O+, O- = split(Attn(q_cat, K, V))           # ONE attention call, SAME K/V, same RoPE
O~ = O+ + gamma * (O+ - O-)                 # gamma=1 => 2O+ - O-, BEFORE o_proj
Y  = o_proj(O~)
```

Constraint compliance, each verified rather than asserted:

| requirement | status | evidence |
|---|---|---|
| P=0: no prefix slots / memory tokens / WRITE / persistent state | ✅ | reader is a pure function of `H[t-w:t]`; no state across examples |
| full context still enters the backbone | ✅ | backbone forward unchanged |
| same q-norm / RoPE / position ids / cache positions for Q+ and Q− | ✅ | both branches are rows of one `q_cat` passed through one `apply_rotary_pos_emb` |
| correction before `o_proj` | ✅ | `o_proj` is applied to `O~`, code above |
| pair shares K/V and GQA group | ✅ | gate A below |
| no new K/V heads, no KV-cache growth | ✅ | 28,311,552 bytes in base and split alike |
| backbone frozen | ✅ | SHA256 over 398 tensors unchanged; 0 backbone grads |
| negative branch really passes through attention | ✅ | it is a query row of the single attention call, not an output-space correction |

### GQA head-to-KV mapping (Qwen3-4B: H=32, G=8)

Doubling queries takes the repeat factor 4 → 8, so new head `j` maps to kv group
`j // 8`. For the pair `(2i, 2i+1)`: `2i // 8 = i // 4` and `(2i+1) // 8 = i // 4`,
which is exactly original head `i`'s group. A front/back split would instead send
head `i` and head `i+32` to groups `i//8` and `(i+32)//8` — **different groups** —
silently breaking both GQA and function preservation. This is why the
interleaving is not cosmetic.

---

## 2. Hard gates (all passed before any training)

`scripts/diffsplit_realgate.py` → `out_diffsplit_20260814/realgate_v2.json`,
plus 23 unit tests in `tests/test_diff_split.py`.

| gate | result |
|---|---|
| A shapes / GQA pairing | 32 q-heads, 8 kv-heads, repeat 4→8, pair shares kv group |
| B FP32 parity (split, delta_q=0) | **max_abs = 0.000e+00** (bit-exact) |
| B BF16 | max 2.031e-01, mean 1.280e-02, **greedy tokens identical** |
| C gradients | trainable = **14,155,776** (delta +0 vs old sidecar); 0 backbone grads; backbone SHA256 over 398 tensors unchanged |
| D KV cache | 28,311,552 bytes in both; prefill vs cached-decode within base's own gap |
| E causality | past-perturbation effect **0.000e+00**; future-perturbation effect 9.788e-01 |

Parameter budget is matched **exactly**, not approximately: the split reuses the
old sidecar's tensor shapes (`mem_q 2560→128`, `mem_k 2560→128`, `128→4096`) and
only changes the *destination* of the third projection (old: added to Z before
`o_proj`; new: added to the negative branch's query). Hence
`1,179,648 × 12 = 14,155,776`, a **0.00%** deviation.

---

## 3. Chain A — Qwen3-4B on Qasper

Identical across all arms: data compose seed 42, 800/75 papers, `train_target_n`
935, `max_ctx_tok` 4500, 156 steps, batch 1 × grad-accum 16, layers
`0,3,6,9,12,15,18,21,24,27,30,33`, greedy generation, all 187 validation
examples, same official evaluator.

| arm | seeds | F1 per seed | mean F1 | trainable |
|---|---|---|---:|---:|
| `base_fullctx` (historical, authenticated) | 1 | 0.2444 | **0.2444** | 0 |
| `split_current_fixed` (w=1) | 1 | 0.2863 | 0.2863 | 14,155,776 |
| `split_local256_fixed` (**canonical**) | 3 | 0.2928 / 0.2987 / 0.2905 | **0.2940** | 14,155,776 |
| `param_matched_additive` | 3 | 0.2950 / 0.2981 / 0.2943 | **0.2958** | 14,155,776 |

`base` was authenticated as a `trainable=0, steps=1` frozen-model evaluation with
the same data-compose seed and the same 187 examples, so it is directly reusable.

### Paired uncertainty (hierarchical bootstrap: examples × seeds, 20,000 draws)

| comparison | delta F1 | 95% CI | p |
|---|---:|---|---:|
| split − base | **+0.0496** | [+0.0256, +0.0742] | <0.0001 |
| additive − base | **+0.0515** | [+0.0256, +0.0781] | 0.0001 |
| **split − additive** | **−0.0018** | **[−0.0205, +0.0154]** | **0.85** |

Seed-to-seed spread within the split arm (0.2905–0.2987, range 0.0082) is
**larger than the split-vs-additive gap** (0.0018). The historical `attn_only`
(0.2933, 5 seeds) and `dex_plus` (0.2950, 5 seeds) also sit inside this band.

### Mechanism: the branches really do diverge

`scripts/diffsplit_divergence_probe.py` → `divergence_split_local256_s0.json`,
24 real Qasper validation examples, measured immediately before `o_proj`.

| condition | cos(O+,O−) | ‖O+−O−‖/‖O+‖ | ‖ΔQ‖/‖Q+‖ |
|---|---:|---:|---:|
| zero-init | **1.000000** | **0.000000** | 0.000000 |
| trained | 0.971521 | **0.185427** | 0.082351 |
| trained + zeroed window | 1.000000 | 0.000000 | 0.000000 |
| trained + **shuffled** window | 0.994267 | **0.071712** | 0.028910 |

- Function equivalence at init holds **on real data**, exactly, at all 12 layers.
- Divergence grows monotonically with depth: layer 0 = 0.0036 → layer 33 = 0.766.
- Shuffling the window destroys **61%** of the divergence, so the reader is using
  the window's content and order — it has not collapsed to a constant bias.

**So the mechanism works and does nothing useful here.** The branches diverge,
the reader is genuinely content-dependent, and none of it converts into a
measurable Qasper gain over spending the same 14.16M parameters additively.

---

## 4. Chain B — small-model controlled comparison (running)

No LM pretraining trainer exists in either repository (`grep -riE
"pretrain|small|scratch|lm_train|tiny"` over both `scripts/` trees returns
nothing), so §8.1's "reuse an existing validated framework" **could not be
satisfied**. The model implementation is HuggingFace's own `Qwen3ForCausalLM`
plus the audited DIFF V2 module; only the data pipeline and training loop are
new code, written for this experiment and listed here as such.

### Config (derived, since no small Qwen3 preset exists on disk)

The smallest local Qwen3 checkpoint is 4B, and the environment is offline, so the
small config is **derived by preserving Qwen3-4B's architectural ratios** rather
than taken from a preset: GQA 4:1 (8 query / 2 kv heads, as 32/8), and
intermediate/hidden = 3.75 (Qwen3-4B is 9728/2560 = 3.80). Deviation: head_dim is
64 (hidden/heads) rather than Qwen3's 128.

| | vanilla | diffv2 |
|---|---:|---:|
| total params | 106,500,096 | 108,630,016 |
| non-embedding | 28,845,568 | **30,975,488** |
| attention | 5,243,904 | **7,373,824** (+40.6%, doubled q_proj) |
| ffn | 23,592,960 | 23,592,960 |
| embedding (tied) | 77,654,528 | 77,654,528 |
| throughput | 114k tok/s | 103k tok/s (−9.6%) |

DIFF V2 is **not** parameter-matched to vanilla — the doubled query projection is
intrinsic to the architecture. Per §8.4 this is reported rather than papered over
by widening vanilla's FFN, and both token-matched and wall-clock views are given.

### Data

`allenai/c4:en` from the HF cache, on-disk order, no shuffle, Qwen3-4B tokenizer.
Manifest and sha256 in `out_smalllm_20260814/data_manifest.json`.
**Defect found and fixed:** several cache directories hold byte-identical copies
of the same shards, so the naive glob reported 712,635 documents instead of
356,318 and a large token budget would have made the tail of training an exact
repeat of the head. Deduplicated by basename; the real corpus is ~160M tokens.
T = 480M tokens is therefore **3 epochs**, which is stated rather than implied.

### Arms

`small_vanilla` 0→480M (T0 checkpoint at 336M) · `small_diffv2_from_scratch`
0→480M · `small_ours_posthoc` and `small_param_matched_additive` both forked from
the **same** `vanilla@T0` file with its sha256 asserted at load.

Results pending; recovery ratio will be reported as `N/A` unless DiffV2 actually
beats vanilla by a margin that does not put the denominator near zero.

---

## 5. Conclusions supported so far

1. **Implementation correct** — every hard gate passes, including bit-exact FP32
   parity and an exactly-preserved KV cache.
2. **Step-0 base preserved exactly**, verified on real data (cos 1.000000).
3. **Beats base** — +0.0496 F1, CI excludes zero, consistent across 3 seeds.
4. **Does not beat the parameter-matched additive control** — −0.0018, CI spans
   zero, p=0.85, and the gap is smaller than seed noise.

Per the pre-registered criteria, the required wording is:

> **新增可训练容量有效，但现有证据不支持 differential splitting 的特异性贡献。**
> The added trainable capacity helps; the present evidence does not support a
> differential-splitting-specific contribution.

Not yet answered: parameter-matched LoRA, HotpotQA / LoCoMo / RULER transfer, and
all of chain B.
