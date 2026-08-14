# SOURCE_AUDIT — Post-hoc Function-Preserving Differential Head Splitting

Session start **2026-08-14 05:59:13 UTC**. Branch `agent/diff-head-split-2026-08-14`,
forked from `089ca9c` (`agent/downstream-audit-2026-08-12`).
Outputs: `out_diffsplit_20260814/`.

## 0. Environment (recorded, not assumed)

| item | value |
|---|---|
| hosts | **node12** (`lux-2-node-12`, 8xH100 80GB — GPUs **0–3 FREE**, 4–7 running my own vLLM workers) · **node16** (8xH100, **all 8 occupied by user `peng`**, ~52 GB each, 100% util — not touched) |
| python / torch / transformers | 3.10.20 / **2.6.0+cu124** / **5.9.0** |
| flash-attn | 2.7.4.post1 |
| CUDA | 12.4 |
| git HEAD at start | `089ca9c53fc386fa67349da8464c4c4d5abdb12a` |
| model | `/work/mingze/models/Qwen3-4B-Instruct-2507` (the checkpoint already locked by this project) |

**Blocker found and fixed during audit:** node12's `~/.local/lib/python3.10/site-packages`
had lost ~9 packages including `numpy`, which made the `deltamem` env unimportable on
node12 (it resolves numpy from user-site). Restored with `rsync --ignore-existing` from
the node16 copy; verified `numpy 2.2.6 / torch 2.6.0+cu124 / transformers 5.9.0 /
cuda True`. Nothing in the conda env itself was modified.

## 1. Official sources

| Component | Source | Commit/version | Reused or modified | Verification |
|---|---|---|---|---|
| DIFF V2 reference impl | `microsoft/unilm` → `Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py` | **`833df7e7832e5064a281131ee64a481afa8e5b95`** (sparse clone, `/work/mingze/official_refs/unilm`) | **read-only reference**; not imported (it is a fairseq-style from-scratch module, we do post-hoc conversion of a HF Qwen3) | line-by-line read, key points below |
| DIFF V1 paper | arXiv:2410.05258 | — | conceptual only | — |
| Qwen3 attention | installed `transformers==5.9.0` → `models/qwen3/modeling_qwen3.py::Qwen3Attention` | file size 23353 B, lines 222–290 read verbatim | **reused, wrapped** — we call the base module's own `q_proj/k_proj/v_proj/q_norm/k_norm/o_proj` and the same `apply_rotary_pos_emb` + `ALL_ATTENTION_FUNCTIONS` dispatch | see §3 |
| RULER | `NVIDIA/RULER` (`rulerv1-ns`) + NeMo-Skills generator | `e8bbff6…` / `f4a3fd8…` (pinned in previous session, data already generated at 8K/16K/32K with the Qwen3 tokenizer) | reused | `/work/mingze/ruler_official_data*/` |
| HotpotQA scorer | `hotpotqa/hotpot` `hotpot_evaluate_v1.py` | `3635853403a8735609ee997664e1528f4480762a` | reused unmodified | previous session: reproduces in-repo metric to 4 decimals |
| LoCoMo scorer | `snap-research/locomo` `task_eval/evaluation.py` | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | reused unmodified | ⚠ see §5 known defect |

### What the official DIFF V2 code actually does (verified, not remembered)

```python
self.num_q_heads = 2 * self.num_heads          # query heads doubled
self.q_proj = Linear(d_model, num_q_heads*head_dim)
self.k_proj = Linear(d_model, num_kv_heads*head_dim)   # KV heads UNCHANGED
self.o_proj = Linear(self.num_heads*head_dim, d_model) # o_proj input NOT doubled
self.lambda_proj = Linear(d_model, num_heads, bias=False)
...
attn = flash_attn_func(q, k, v, causal=True)
attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]      # INTERLEAVED pairing
attn = attn1 - sigmoid(lambda_val).unsqueeze(-1)*attn2 # sigmoid gate, BEFORE o_proj
```

Audit points, all confirmed against the source:
- query heads ×2, KV heads unchanged, no extra KV cache — ✅
- pairing is `0::2` / `1::2`, **not** a front/back split — ✅
- gate is `sigmoid(lambda_proj(x))`, per output head, input-dependent — ✅
- differential is applied **before** `o_proj` — ✅
- **no** post-differential per-head RMSNorm (that was V1) — ✅

**Why the interleaving matters for GQA (derived, then used):** with `num_q_heads = 2H`
and `num_kv_heads = G`, query head `j` maps to KV head `j // (2H/G)`. For Qwen3-4B
(H=32, G=8): new head `2i` → `(2i)//8 = i//4`, new head `2i+1` → `(2i+1)//8 = i//4`,
and the original head `i` → `i//4`. So **the `(2i, 2i+1)` pair lands in exactly the
original head `i`'s KV group**. A front/back split would map head `i` and head `i+32`
to KV groups `i//8` and `(i+32)//8` — different groups — and silently break both GQA
and function preservation. This is why our implementation must interleave.

**Scope caveat recorded up front:** official V2 is a *from-scratch pre-training*
architecture. Nothing in the official material demonstrates post-hoc conversion of a
trained dense model, nor long-context gains from post-training differentialisation.
This project's claim space is strictly post-hoc.

## 2. Existing project code map (real paths, verified by reading)

| role | file | notes |
|---|---|---|
| attention wrapper | `deltamem/core/prefix_steer.py::PrefixMemSteerAttention` | replaces `Qwen3Attention` via `attach_prefix_steer()`; mirrors the base forward and adds a sidecar |
| pre-`o_proj` correction | same file, `o_fusion_position="pre_o"` | `Z = attn_out.reshape(...); Z = fuse(Z, C); out = base.o_proj(Z)` — verified in the previous session with max-abs residual **0.0** |
| local reader | same file, `_memory_read()` with `num_prefix_tokens=0`, `sliding_window_size=256` | causal SWA read over the current sequence, no prefix/no write |
| training entry | `scripts/qasper_prefix_steer.py` | `--train-mode ctx/noctx/ctxmask`, freezes backbone via `freeze_backbone_keep_steer` |
| alternative entry | `scripts/dex_train_qasper.py` | `--variant base/dex_minus/dex_plus/attn_only/adapter_only/residual_adapter/swa_steer` |
| evaluators | `scripts/eval_ours_hotpotqa.py`, `eval_ours_locomo.py`, `eval_ours_ruler.py`, `scripts/official_score.py` | official scorers wired in the previous session |
| stats | `scripts/paired_stats.py` | paired bootstrap / McNemar / cluster / hierarchical |

**Naming mapping (prompt → repo).** The prompt's `delta_heads=o` refers to
`PrefixSteerConfig.delta_heads`, a string subset of `"qkvo"` selecting which of
q/k/v/o receive an *additive* correction from the sidecar read. It is **not** related
to differential head splitting. The new work introduces a separate module rather than
overloading that flag.

## 3. Historical Qasper claims — authentication result

Claims in the task prompt, checked against the original result JSONs in
`/work/mingze/delta-mem/out_dex/`:

| claim | value | artifact family | artifact mean | verdict |
|---|---:|---|---:|---|
| base | 0.2444 | `dex_base_lr2e-5_s0` (n=1) | 0.2444 | ✅ authenticated |
| dex_minus | 0.2910 | **`v2_dex_dex_minus_lr2e-5_s{0..4}`** | 0.2910 | ✅ authenticated |
| dex_plus | 0.2950 | **`v2_dex_dex_plus_lr2e-5_s{0..4}`** | 0.2950 | ✅ authenticated |
| attn_only | 0.2933 | `dex_attn_only_lr2e-5_s{0..4}` | 0.2933 | ✅ authenticated |
| residual_adapter | 0.2828 | `dex_residual_adapter_lr2e-5_s{0..4}` | 0.2828 | ✅ authenticated |
| adapter_only | 0.2432 | **`v2_dex_adapter_only_lr2e-5_s{0..2}`** (n=3) | 0.2432 | ✅ authenticated |

**Two caveats that must travel with these numbers:**

1. **The published table mixes two run families.** `dex_minus`, `dex_plus` and
   `adapter_only` come from the `v2_*` family; `attn_only`, `residual_adapter` and
   `base` come from the non-`v2` family. The only argument difference is
   `allow_no_anneal` (`None` vs `False`) plus the tag, so the protocols are otherwise
   identical (steps 156, lr 2e-5, bs 1, grad-accum 16, 800/75 papers,
   `max_ctx_tok` 4500, `train_target_n` 935, `data_compose_seed` 42, all 187 val).
2. **Same variant, same seed, different family ⇒ different score.** e.g. `dex_minus`
   seed 0: `dex_*` 0.2921 vs `v2_*` 0.2902. That is a **≈0.002 reproducibility floor**
   for this evaluator at fixed seed across code revisions. Any new effect smaller than
   ~0.002 is not resolvable by this harness, and the non-`v2` `dex_minus`/`dex_plus`
   means (0.2937 / 0.2932) *reorder* the two arms relative to the published table.

## 4. Parameter budget — exact match available

Old sidecar (`out_dex_fusion/preo_swa_steer_s0_steer.pt`), 12 layers
`[0,3,6,9,12,15,18,21,24,27,30,33]`, **14,155,776** trainable in 48 tensors:

| tensor | shape | per layer | ×12 |
|---|---|---:|---:|
| `mem_q` | 2560 → 128 | 327,680 | 3,932,160 |
| `mem_k` | 2560 → 128 | 327,680 | 3,932,160 |
| `delta_o` | 128 → 4096 | 524,288 | 6,291,456 |
| `prefix` | (P=0) | 0 | 0 |
| **total** | | **1,179,648** | **14,155,776** |

The differential split with local-read dim `d_r = 128` uses **the identical shapes** —
`mem_q(2560→128)`, `mem_k(2560→128)`, `delta_q(128→4096)` — giving
**1,179,648 × 12 = 14,155,776**, i.e. a **0.00%** deviation from the old budget, not
merely within 5%. The only change is the *destination* of the `128→4096` projection:
old = added to `Z` before `o_proj`; new = added to the **negative branch's query**.
This makes `param_matched_additive` literally the pre-existing architecture, so the
differential-specific comparison is exact by construction.

## 5. Known defect inherited from the previous session (must not be repeated)

`scripts/eval_ours_locomo.py` defaults to `--max-context-tokens 8000` while LoCoMo
conversations average ~19.3k tokens: measured **81,598 / 193,162 tokens = 42.2%** of
the conversation was actually shown, dropping the *earliest* turns. The previous
session's LoCoMo absolute numbers are therefore **NON-OFFICIAL (truncated input)**;
its paired deltas remain internally valid because base and ours were truncated
identically. Any LoCoMo run in this session must pass `--max-context-tokens 32000`.

---

## 6. Addendum — second session (chain A completion + chain B)

### 6.1 Historical artifact re-verification

| arm | file | verdict |
|---|---|---|
| `base` | `/work/mingze/delta-mem/out_dex/dex_base_lr2e-5_s0.json` | **`trainable_param_count=0`, `steps=1`** → a pure frozen-model evaluation, `data_compose_seed=42`, 187 val examples, same evaluator ⇒ **directly reusable as `base_fullctx`** |
| `attn_only` | `dex_attn_only_lr2e-5_s{0..4}.json` | runs at `lr=2e-5`, not the 5e-4 used by sidecar arms. Checked `scripts/dex_train_qasper.py:355-373`: params are grouped **by type** — pretrained weights get `args.lr` (2e-5), randomly-initialised sidecars get `args.steer_lr` (5e-4). This is the project's established convention, so the historical 5-seed `attn_only` is a legitimate comparator and was **not** re-run. |

### 6.2 Environment at this session

`nvidia-smi` (node12) was the only basis for scheduling. GPU 1 held another job
of mine (`sq_production.py`, spinquant) and GPUs 4–7 held my vLLM workers; none
were touched. node16's 8 GPUs remain fully occupied by user `peng` (~56 GB each,
100% util) and were likewise not touched.

### 6.3 New code in this session

| file | role | new? |
|---|---|---|
| `scripts/diffsplit_divergence_probe.py` | ‖O+−O−‖/‖O+‖, cos, zero/shuffle-window controls on real Qasper data | new |
| `scripts/run_diffsplit_queue.sh` | per-GPU serial job queue + ledger writer | new |
| `deltamem/core/small_diffv2.py` | official-style DIFF V2 attention for from-scratch training | new |
| `deltamem/core/small_additive.py` | parameter-matched additive control (same reader, no second attention) | new |
| `scripts/small_lm_data.py` | deterministic tokenized C4 manifest + sha256 | new |
| `scripts/small_lm_train.py` | chain B trainer (4 arms) | new |
| `deltamem/core/diff_split.py` | added `branch_cos` to the stats block | modified |

**Framework-reuse caveat (§8.1).** Neither repo contains an LM pretraining
trainer, so `small_lm_train.py` had to be written. It does not reimplement any
model: `Qwen3ForCausalLM` is HuggingFace's, and the DIFF V2 module follows the
audited official reference. Only data handling and the optimisation loop are new.

### 6.4 Defects found and fixed this session

1. **Divergence probe ablations were no-ops** — the sweep never forwarded
   `zero=True` / `shuffle=True`, so the "controls" reproduced the trained numbers
   exactly. Caught because identical-to-6-decimals output is not a plausible
   ablation result. The implementation was fine (verified by direct tiny-model
   test); the probe was fixed.
2. **Duplicated C4 cache shards** — `cache.glob(...)` matched byte-identical
   copies in several cache dirs, reporting 712,635 documents instead of 356,318.
   A 210M-token budget would have silently made the last ~47M tokens an exact
   repeat of the first 47M. Deduplicated by basename; budget re-derived from the
   true ~160M-token corpus.
3. **Chain B label double-shift** — HuggingFace shifts `labels` internally, so
   passing the already-shifted target would have trained on an off-by-one
   objective. Fixed to pass the unshifted sequence.
4. **Fork data misalignment** — post-hoc arms resume at T0 tokens, which exceeds
   the manifest length in a multi-epoch run; the batcher would have reset to
   token 0 and trained the forks on *different data* than the control saw next.
   Fixed with `start_token % len(arr)`.
5. **151k-vocab OOM** — a materialised fp32 logits tensor at batch 32 × seq 1024
   is ~20 GB, larger than the rest of training combined. Switched to HF's fused
   chunked cross-entropy.
