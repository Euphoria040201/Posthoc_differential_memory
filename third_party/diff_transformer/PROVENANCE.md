# Official Differential Transformer reference (fetched 2026-08-18)

Source: microsoft/unilm @ master
  Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py   (V2, the arm B reference)
  Diff-Transformer/multihead_flashdiff_1.py                        (V1, for contrast)

Fetched because the previous session cited this file in docstrings but never
vendored or verified against it. NOT vendored earlier = the faithfulness claim
was unchecked.

## No released weights

The Diff-Transformer README lists no checkpoint, download or HuggingFace link.
There is no official DiffV2 model to evaluate, so "native DiffV2" in this study
is necessarily a REIMPLEMENTATION trained from scratch, not an official model.

## Verified: what the official V2 module does

    num_q_heads = 2 * num_heads          # query heads doubled
    k_proj/v_proj = num_kv_heads * head_dim   # KV heads UNCHANGED
    o_proj input = num_heads * head_dim       # NOT doubled
    lambda_proj  = Linear(d_model, num_heads, bias=False)
    attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]     # INTERLEAVED
    attn = attn1 - sigmoid(lambda_val).unsqueeze(-1) * attn2

Confirmed: V2 has NO post-differential per-head norm (that was V1's GroupNorm),
and the gate is a plain input-dependent sigmoid. Both claims in our docstrings
are correct.

## Where deltamem/core/diffv2_native.py DEVIATES

1. **q_norm / k_norm.** The official module applies no normalization to q or k.
   Ours applies Qwen3's per-head RMSNorm to both, because it is built from Qwen3
   blocks. The vanilla control has the same norms, so the comparison stays
   symmetric, but the module is not byte-faithful to the official one.
2. **lambda_proj init.** Official leaves it at PyTorch default; ours zero-inits
   so sigmoid(0)=0.5 exactly at step 0. Both land near lambda=0.5 in practice.
3. **RoPE convention.** Official uses interleaved rotary; Qwen3 uses the
   half-rotation form. This is a backbone difference, applied identically to
   both arms.
4. **Attention kernel.** Official requires flash_attn; ours routes through
   sdpa/eager. Numerically equivalent up to kernel reduction order.

Deviations 1 and 3 apply equally to the vanilla control, so the *contrast*
between arms is clean; they mean the absolute numbers are not comparable to any
number published for the official implementation.
