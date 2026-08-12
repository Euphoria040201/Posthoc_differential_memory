# delta-mem prefix-memory: investigation bundle

Frozen Qwen3-4B + attached SWA/prefix memory steering (delta_q/k/v/o corrections, gain 0.1),
trained on Qasper, evaluated zero-shot on HotpotQA / LoCoMo. This bundle is the **curated,
useful** subset — training/eval infra + the diagnostic scripts that produced real findings +
the checkpoints they reference. Dead-end sweeps and wrong configs are NOT included.

## Verified downstream results (2026-08-12 audit, branch `agent/downstream-audit-2026-08-12`)

Official evaluators, untouched holdouts, 3 training seeds. Full detail and every
failing arm in `DOWNSTREAM_RESCUE_REPORT.md`.

**The sidecar significantly beats the frozen full-context base:**

| benchmark | set | base | ours | Δ | 95% CI |
|---|---|---:|---:|---:|---|
| HotpotQA (official `hotpot_evaluate_v1.py`) | 4678 untouched dev ids | F1 0.5615 | F1 0.6216 | **+0.0582** (3-seed hierarchical) | [+0.0488, +0.0670] |
| LoCoMo (official `task_eval/evaluation.py`) | 1540 QA, conversation-clustered | F1 0.1833 | F1 0.2163 | **+0.0347** (3-seed hierarchical) | [+0.0294, +0.0406] |
| RULER (officially generated, 13/13 tasks) | 8K / 16K / 32K | 0.9363 / 0.9425 / 0.9157 | 0.9229 / 0.9047 / 0.9153 | **at or below base** | — |

**But the written memory contributes nothing.** Reading a state written from a
*different* document scores the same as the correct one, across three independently
trained checkpoints at n=600:

| checkpoint | correct − swap | 95% CI | ours_fullctx − base | 95% CI |
|---|---:|---|---:|---|
| post_o @step100 | +0.0027 | [−0.0066, +0.0125] | +0.0805 | [+0.0535, +0.1084] |
| pre_o @step100 | +0.0007 | [−0.0124, +0.0141] | +0.1099 | [+0.0793, +0.1415] |
| stable pre_o @step150 | +0.0014 | [−0.0089, +0.0122] | +0.0774 | [+0.0485, +0.1068] |

On LoCoMo, masking the written memory out of the read costs **+0.0034 F1, CI
[−0.0055, +0.0105]** — while the memory-free model still beats base by +0.1155.

**Therefore: this is a task/attention adapter, not a memory system.** It must not be
described as episodic memory, memory augmentation or context compilation. It also costs
**3.2x the base latency per query** (§6 of the report).

Audit outcomes: pre_o implementation correct (fusion residual 0.0); backbone
bit-identical across training; base parity bit-exact; pre_o ≡ post_o_projected in fp32
(top-1 disagreement 0.0000).

## Key findings (what the scripts here established)
1. **No-context is where the memory has value.** With context in the prompt (backbone = full
   attention) the memory is redundant; with-context prefix ≈ no-prefix. But drop the context and
   read from the written prefix: base_noctx 0.036 → ours_noctx 0.138 (**+0.10, 6/6 seeds**).
   Training FOR no-context (`--train-mode noctx`) lifts it to ~0.19; capacity (64/128/256 slots)
   does NOT help → the attention-pool WRITE is the ceiling.  → `noctx_ours_hotpot.py`
2. **prefix barely trains & is nearly inert.** init→trained cos ≈ 0.996 (prefix); the real
   learning is in delta_q (biggest, norm 11.4) / delta_o. Per-layer prefR/R (prefix's share of
   the read) is nonzero only at shallow L0/L3, exactly 0 at L12–L33.  → `vnorm_probe.py`,
   `prefr_alllayers.py`
3. **The seed=0 champion (0.6815) vs worst ms3 (0.618):** all seeds give mutually-orthogonal
   weight solutions of equal norm; no weight/init/spectral stat predicts the champion. The gap
   lives in ~31% divergent hard docs where the worst degenerates to Yes/No 6× more (781 vs 127).
4. **ms3's degeneration is a localizable, ablatable L33 pathology.** ms3 (and ONLY ms3 of 7
   seeds) saturates 100% of its final steer-layer (L33) memory-read onto the doc-independent
   prefix (prefR/R=1.0; injection 0.63, its largest layer). Masking L33 prefix on ms3 flips 6/8
   degeneration docs back to real answers (several correct). Harmless on easy docs (byte-identical),
   causal on the hard/degeneration ones.  → `l33_allseeds.py`, `deg_test.py`, `three_cmp.py`

## Delta-O fusion position (`o_fusion_position`)

Where the memory correction `C = delta_o(reads)` joins the frozen attention, with
`Z` the concat-head activation (`o_proj` input, width `n_query_heads * head_dim`
for MHA and GQA alike) and `W_O, b` the frozen `o_proj`:

| position | formula | C lives in |
|---|---|---|
| `post_o` (default, historical) | `Y = fuse(W_O Z + b, C)` | output space (`o_proj.out_features`) |
| `pre_o` (DEX-inspired) | `Y = W_O fuse(Z, C) + b` | concat-head space (`o_proj.in_features`) |
| `post_o_projected` (control) | `Y = fuse(W_O Z + b, W_O C)` | concat-head space, projected without bias |

`fuse` is any `output_fusion` mode (`fixed_add`, `fixed_sub`, `learned_diff`,
`variance_diff`, `rms_match`, `cosine`) — one shared `_fuse_delta_o`, no forked
code paths. For the linear fusions, `pre_o` and `post_o_projected` are
mathematically identical (`W_O(Z ± gC) + b == (W_O Z + b) ± g(W_O C)`), so the
control isolates implementation-path effects (bf16 rounding, kernel order) from
the mathematical position; for the nonlinear-coefficient fusions they differ
because the per-token statistics are computed in different bases. Old configs
and checkpoints deserialize to `post_o` (audited in
`deltamem/eval/steer_checkpoint.py`); `pre_o` checkpoints are not
shape-compatible with `post_o` ones and fail loudly on load. This is
**DEX-inspired pre-O prefix differential fusion** — `C` comes from the
prefix/SWA sidecar, not from `O` itself, so no equivalence with original DEX is
claimed. Experiments: `scripts/run_preo_fusion_matrix.sh`, results in
`preo_fusion_report.md`.

## Layout
```
deltamem/            core package (model = core/prefix_steer.py; eval = eval/benchmark_compare.py)
scripts/             training + eval + two repo diagnostics
  qasper_prefix_steer.py   TRAIN (ctx / noctx modes); build_examples; --delta-heads qkvo
  eval_ours_hotpotqa.py    HotpotQA eval (conds: base / ours / ours_window_only / ours_zero_read)
  eval_ours_locomo.py      LoCoMo eval (auto-restores backbone_window from ckpt)
  noctx_ours_hotpot.py     NO-CONTEXT test (write→drop→read); base_ctx/ours_ctx/base_noctx/ours_noctx
  diag_locomo_prefix.py    prefix-read decomposition (prefR/R, pfx_mass, cos_across) HotpotQA vs LoCoMo
investigation/       the L33 / prefix-contribution probes (see below)
ckpts/               spread12_swa (champion, seed0, 0.6815) ; s0d_ms1..6 (same data, init seeds 1-6) ;
                     nct_p64_s1 (noctx-trained example)
eval_records/        saved per-question predictions (champion + ms3) for find_divergent_ids.py
data/locomo10.json   LoCoMo eval data
requirements.txt     pip deps (torch 2.6.0+cu124 — works on the 4080 / Ada sm_89)
```

## Investigation scripts (run from repo root, all CPU-friendly, n small)
- `vnorm_probe.py`      per-layer Vp_RMS / read_RMS / prefR/R / injection (champion). Edit the last
                        line to point at any ckpt.
- `prefr_alllayers.py`  per-layer prefR/R for the middle seeds ms1/2/4/5/6 (12-layer table).
- `l33_allseeds.py`     L33 prefR/R + injection + deep-max-mass across all 7 seeds vs F1.
- `three_cmp.py`        raw outputs: champion vs worst vs worst-with-L33-masked, same docs.
- `find_divergent_ids.py` → deg_ids.json : ids where ms3 degenerates but champion doesn't.
- `deg_test.py`         L33-prefix ablation on those degeneration docs (needs deg_ids.json first).

Typical run:
```bash
cd delta-mem
source .venv/bin/activate
export PYTHONPATH=.:scripts
python investigation/l33_allseeds.py            # 7-seed L33 table (CPU ~25min, or GPU fast)
python investigation/find_divergent_ids.py      # regen deg_ids.json
python investigation/deg_test.py                # L33 causal ablation
```
NB: HotpotQA/Qasper load from the HF datasets cache (`~/.cache/huggingface/datasets`,
`local_files_only=True`). Populate that cache once on the 4080 (remove `local_files_only=True`
for the first download, or copy the cache over).

## Setup on the 4080
```bash
bash setup.sh        # creates .venv (python3.10), installs requirements
```
Training (38GB footprint) does NOT fit 16GB — this box is for the ANALYSIS / inference
diagnostics + small evals, which run on CPU or ~10GB GPU.

## Training on 16GB cards (RTX 4080 etc.) — VERIFIED
The single-card 38GB footprint is dominated by the 4500-token activation graph, not weights (8GB).
`scripts/qasper_prefix_steer.py --grad-checkpointing` recomputes the frozen backbone in the backward
(activations ~5-10x smaller); combined with `--max-ctx-tok 3500` peak is **~15.2GB** (tested on a
4080 SUPER 16GB). Use one seed per GPU:
```bash
bash train_16gb.sh                    # seeds 1,2 on GPU 0,1
GPUS="0 1" SEEDS="3 4" bash train_16gb.sh
```
Notes: needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the launcher sets it); ctx 4500
peaks ~16.4GB (just over) so 3500 is the fit; drop to 3000 for more margin. ~30% slower than
un-checkpointed. Qasper loads from the HF cache (transferred) — `datasets` can't run its loader
script, so keep `~/.cache/huggingface/datasets/allenai___qasper`.

## Keeping the FULL 4500 ctx on 2x16GB (pipeline) — VERIFIED
4500 ctx peaks ~16.4GB — just over one 16GB card. To keep 4500, split ONE run across BOTH cards
with `--device-map balanced` + `--grad-checkpointing`. The attached steer modules move their
seg/valid/prefix to the local layer device in forward, so the custom attach is device-map-safe.
```bash
bash train_16gb_4500.sh              # one run uses GPU 0+1 (both cards), 4500 ctx
SEEDS="1 2 3" bash train_16gb_4500.sh   # seeds run SEQUENTIALLY (both cards per run)
```
Tradeoff vs the single-card 3500 recipe: keeps full 4500 ctx but a run occupies BOTH cards
(no 2-seeds-in-parallel). Needs `accelerate` (in requirements). device_map='auto' overfills GPU0
(loss OOMs) — use 'balanced'.
