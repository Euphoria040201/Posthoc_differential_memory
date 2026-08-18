# Source audit — token-local low-rank split (2026-08-17)

What is implemented, what is verified by a test, and what is assumed.  Written
so a reader can tell the three apart without running anything.

## 1. The construction

`deltamem/core/lowrank_split.py`, class `LowRankSplitAttention`:

```
dQ_t   = lr_B(lr_A(h_t))                        A: d->r, B: r->H*hd, both bias-free
Q+     = q_norm(q_proj(h)          .view(B,L,H,hd))
Q-     = q_norm((q_proj(h) + dQ)   .view(B,L,H,hd))     # delta_pre_norm=True (default)
q_cat  = interleave(Q+, Q-)                     # [+,-,+,-,...] over 2H heads
O+,O-  = split(Attn(q_cat, K, V, same RoPE, same mask, same cache))
O~     = O+ + gamma*(O+ - O-)
Y      = o_proj(O~)
```

`lr_B` is zero-initialised, so `dQ = 0` and `Q- == Q+` at conversion.

**Why the delta is pre-norm.** With `delta_pre_norm=True` the module is exactly
`q_norm((Wq + BA)h)`: one query matrix, then the same per-head RMSNorm the base
model applies.  That is a low-rank reparameterization of a *second query
projection*, which is what native DiffV2 has.  Adding `dQ` after `q_norm`
(`delta_pre_norm=False`, and what LocalRead does) is an additive patch on an
already-normalized vector and is not equivalent to any single query matrix.  The
flag exists so the difference can be ablated rather than assumed.

## 2. Verified by test

`tests/test_lowrank_split.py` (25 tests) and `scripts/lowrank_realgate.py`
(Qwen3-4B, artifact `out_cpt_20260817/realgate_lowrank.json`).

| property | how it is verified | result |
|---|---|---|
| whole-construction correctness | independent two-pass reference: O+ and O- computed by two separate H-head attention calls through the frozen base module, then combined; compared to the interleaved 2H-head path, for **both** eager and sdpa | matches to 1e-6 |
| interleaved GQA mapping | arithmetic check that head 2i and 2i+1 both map to kv group i//(H/G), plus the reference above (a wrong repeat factor cannot match it) | pass |
| fp32 parity at zero init | `torch.equal` on logits, tiny model and Qwen3-4B | **bit-exact**, max_abs 0.0 |
| bf16 difference | measured per backend on CUDA at production shapes; NOT assumed from the fp32 result | **bit-exact on eager and sdpa** (max_abs 0.0, greedy match 1.0) |
| enabled/disabled switch | disabled must equal base bit-exactly *while lr_B is non-zero*, and the switch must report a non-zero module count | pass, 12 modules switched |
| KV cache unchanged | key/value shapes and total bytes compared against base | identical, 5,898,240 B |
| no private inference cache | scans module `vars()` for tensor attributes after prefill+decode | empty |
| decode == prefill | cached incremental decode vs full forward | tiny model 2e-5; on 4B the bf16 gap **equals the base model's own** (0.2578 both) |
| backbone frozen | sha256 over all 398 non-split tensors before/after a real backward; count of backbone params with grad | unchanged, 0 grads |
| gradient flow | `lr_B` non-zero at step 0; `lr_A` **exactly** zero at step 0 (chain rule through B=0) and non-zero once B moves | pass |
| padding safety | left and right padding with attention_mask and position_ids vs unpadded | matches to 2e-4 |
| batch > 1 | shapes and finiteness at b=1,3,5 | pass |
| strict checkpoint round trip | save/load reproduces logits exactly | pass |
| fail-closed loader | missing key, unexpected key, backbone key in a split-only ckpt, shape mismatch | all four rejected |
| parameter count | formula `r*(d + H*hd)*n_layers` vs measured | 14,137,344 on 4B (r=177, 12 layers), 786,432 on the small model (r=96, 8 layers) |
| output_attentions | raises unless `effective_attention=True`; then returns H-head `A_eff=(1+g)A+ - gA-` verified to satisfy `O~ = A_eff @ V` and to sum to 1 | pass |

## 3. Assumed, not verified

* **FlashAttention is untested.** The module routes through whatever
  `config._attn_implementation` selects, and the interleaving logic is
  backend-agnostic, but only eager and sdpa have been run. No FA claim is made.
* **`delta_pre_norm=False` is not the LocalRead module.** It is the low-rank
  module in post-norm mode; it shares LocalRead's placement, not its reader.
* **The effective-attention tensor is exact only for the output**, i.e.
  `O~ = A_eff @ V`. `A_eff` has negative entries and is not a probability
  distribution, though its rows sum to 1. It must not be fed to code that
  assumes non-negativity.
* Position-stratified NLL assumes each val window lies inside one book. That is
  enforced by the corpus builder, not re-checked at eval time.

## 4. Defects found in the preserved LocalRead baseline

These are recorded, not silently repaired: `diff_split.py` keeps its numerics so
arm E remains the same method that produced the 2026-08-14 artifacts.  Changes
made are guards and diagnostics that cannot alter a valid run.

1. **Per-sequence reader state.** `_read_h`/`_read_v` hold `w-1` hidden states
   and V rows per layer. Measured on Qwen3-4B at seq 1024:
   **21,934,080 B/sequence, 14.5% of its own KV cache**, versus **0 B** for the
   token-local split. The KV cache is genuinely unchanged in both; "KV-cache-free"
   was being read as "state-free", which was never true.
2. **Invalid shuffle ablation.** `probs[:, :, perm]` permuted the whole key axis
   *after* causal masking, so past probability mass could land on a future key.
   That path now raises. The replacement permutes only within each query's causal
   window and asserts that no mass falls outside it. Re-measured: the causal
   permutation destroys **61.4%** of trained branch divergence (div_rel
   .1854 -> .0716), so the original conclusion held; only its evidence was wrong.
3. **`read_dim == head_dim` was implicit.** The reader's values are backbone V
   averaged over kv heads, i.e. width `head_dim`, consumed by a `read_dim`-input
   projection. Now enforced at construction.
4. **`attention_mask` ignored by the reader.** Padded positions are attended by
   the local reader, so left-padded batches would change dQ at real positions.
   All reported runs used packed, unpadded sequences, so no published number is
   affected. Documented in-module; **quarantined, not fixed**, because fixing it
   would change arm E's numerics.
5. **Fail-open checkpoint loader.** `load_ours` verified that every key in the
   checkpoint existed in the model but never the converse, so a checkpoint
   missing diff tensors loaded cleanly and those tensors kept zero init — i.e. a
   base-parity run reported as a split result. Now fails closed, in both the
   evaluator and the divergence probe.
6. **Dynamic-gate parameter assertion.** The expected-count formula omitted the
   gate's `read_dim*n_heads`, so a `--diff-dynamic-gate` run would have died on
   its own sanity check. Fixed.
7. **O(T^2) reader memory.** The reader builds a dense `[B,T,T]` score matrix and
   only then applies the window mask, so a *windowed* reader costs full quadratic
   memory. Invisible at seq 1024; it OOMs at seq 4096 with batch 2 on an 80 GB
   card. Newly found in this session, not in the original list.

## 5. Corrected numbers from the 2026-08-14 artifacts

Regenerated by `scripts/regen_tables_from_artifacts.py` (never editing history):

* HotpotQA 300-example screening subset — artifacts give **base 0.5939 /
  split 0.5796 / additive 0.6012**; the old report printed .5959/.5848/.6034.
* LoCoMo full, 2 seeds — split-base is **+0.0364** as a seed mean; +0.0423 is
  seed 0 alone.
* Qasper 3-seed means are unchanged (.2940 / .2958 / .2908).

## 6. Init defect in the native DiffV2 reference

`small_diffv2.convert_to_diffv2` constructs fresh `nn.Linear` modules after HF
has initialised the model, so the arm under test received PyTorch's default
`kaiming_uniform` while the vanilla control received HF `normal_(0, 0.02)`.
Measured on the chain-B config: **q_proj std 0.02553 (uniform, bounded +-0.0442)
versus vanilla 0.02004 (normal)** — a 27% larger init on exactly the tensors
being compared, while the effect under measurement was 0.005 nats.
`deltamem/core/diffv2_native.py` initialises every new tensor the way HF
initialises the tensor it replaces (measured 0.01997) and is used for all new
runs. The old module is left untouched so its artifacts remain reproducible.

## 7. The native-DiffV2 arm is a reimplementation, not an official model

Added 2026-08-18 after verifying against the official source, now vendored at
`third_party/diff_transformer/` with `PROVENANCE.md`.

* **No official checkpoint exists.** The Diff-Transformer README publishes module
  code only, no weights. "native_diffv2" is therefore necessarily our own
  reimplementation trained from scratch (108.6M params, Qwen3 blocks, 800M PG19
  tokens), not an official model.
* **The official codebase was never run.** It requires `flash_attn` and the
  repo's own rotary kernel. We reimplemented the module on Qwen3 blocks. The
  core algebra was verified line-by-line against the official file: doubled query
  heads, unchanged KV heads, undoubled o_proj, interleaved `0::2`/`1::2` split,
  and `attn1 - sigmoid(lambda_proj(x)) * attn2`. All match.
* **Two deviations** (both applied to the vanilla control as well, so the
  contrast is clean while absolute numbers are not comparable to published ones):
  the official module has no q/k normalization whereas ours inherits Qwen3's
  per-head RMSNorm, and ours zero-inits `lambda_proj` where the official leaves
  PyTorch's default.
* **The evaluation is not an official benchmark.** Arm B was scored only on
  held-out PG19 language-modelling NLL (93 books disjoint from training, 2048
  windows x 4096 tokens) plus position-stratified NLL. No downstream task was
  run for it.
* **Scope of the null.** The Differential Transformer papers argue for gains in
  long-context retrieval, needle-in-a-haystack, hallucination and in-context
  learning robustness, at multi-billion parameters and ~1T tokens. This study
  measures LM NLL at 108.6M parameters and 800M tokens. Our result -- that the
  advantage is a convergence-speed effect that decays into seed noise -- is a
  statement about THIS scale and THIS metric. It is not a refutation of those
  papers' claims.
