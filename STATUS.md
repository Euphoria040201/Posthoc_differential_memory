# STATUS — post-hoc differential head splitting

Start 2026-08-14 05:59:13Z · branch `agent/diff-head-split-2026-08-14` (from `089ca9c`)
Hosts: **node12 GPUs 0–3 free** (4–7 = my own vLLM) · node16 fully occupied by user `peng` (untouched)

## Phase 1 complete — implementation + hard gates PASS on the real model

`out_diffsplit_20260814/realgate_v2.json`

| gate | result |
|---|---|
| FP32 end-to-end parity (split_zero vs base, 12 layers) | **max_abs = 0.000e+00** (bit-exact; threshold 1e-5) |
| trainable parameters | **14,155,776** — delta **+0** vs the old sidecar budget |
| GQA pair sharing | 32q/8kv, repeat 4→8; head `i` and pair `(2i, 2i+1)` all map to kv group `i//4` |
| causality of local reader | future perturbation changes past reads by **0.000e+00** |
| backbone frozen | SHA256 over 398 tensors **unchanged** across a real AdamW step; 0 backbone grads |
| sidecar gradients | delta_q non-zero at step 0; read_q/read_k non-zero once delta_q ≠ 0 |
| BF16 | max_abs 2.03e-01 / mean 1.28e-02 vs base; **greedy tokens identical** |
| KV cache | **28,311,552 bytes in both** — no extra KV, negative queries never cached |
| prefill vs cached decode (bf16) | base 9.38e-2, split_zero 9.38e-2, split_nonzero 7.81e-2 (tol 2.81e-1) |

Unit gates: `tests/test_diff_split.py` — **23 passed**.

### Real bugs the gates caught (all fixed, none shipped)
1. attention interfaces read `module.num_key_value_groups`; with 2H query heads the
   repeat must be 2H/G, not H/G → added `_GQAProxy`.
2. head/seq layout was inferred from shape and is **ambiguous when seq == 2H** →
   replaced with the documented `[B, seq, heads, dim]` contract + assertion.
3. the local reader had **no state during incremental decoding**, so it saw only the
   newest token and decode disagreed with prefill by 3.3e-3 → added a rolling
   (w−1)-hidden-state reader cache keyed to the KV-cache lifecycle.
4. (test-side) reader gradient is **exactly zero at step 0** by construction with a
   zero-init `delta_q`; the gate now asserts that and separately requires non-zero
   reader gradient at step 1.

## Next
Smoke + LR probe on Qasper dev, then the arm matrix.
