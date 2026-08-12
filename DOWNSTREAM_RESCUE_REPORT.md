# Downstream audit + rescue — 12h session, 2026-08-12

node16 (8x H100 80GB) · env `/work/mingze/miniconda3/envs/deltamem` (torch 2.6.0+cu124)
backbone `/work/mingze/models/Qwen3-4B-Instruct-2507` · branch `agent/downstream-audit-2026-08-12`
Repo HEAD at start: `dc9269b` · outputs `out_downstream_audit_20260812/`

**STATUS: IN PROGRESS** (headline results final; long-running arms still landing) — this file is updated as evidence lands. Numbers below are
copied from result JSONs, never from memory.

## Direct answers (§13)

| # | question | answer |
|---|---|---|
| 1 | **Bug in the pre_o implementation?** | **No.** On the live model the `o_proj` input equals `Z + g·C` with max abs residual **0.0**; widths Z = C = 4096 into a 4096→2560 bias-free `o_proj` (GQA, 32 query / 8 KV heads). `a0_preo_graph.json` |
| 2 | **Is pre_o vs post_o_projected really just bf16?** | **Yes as to correctness; "noise" understates the size.** In fp32 they agree to 6.8e-3 max / 9.9e-6 mean with **top-1 disagreement 0.0000** — mathematically equivalent, differences are accumulation order only. In bf16 they flip the top-1 token on **1.1%** of positions (max abs 10.2), enough for greedy decoding to produce visibly different generations. Both arms are internally deterministic (repeat runs bit-exact). `a1_numerics_numerics.json` |
| 3 | **Is the backbone fully frozen?** | **Yes, bit-identical.** SHA256 over all 398 backbone tensors unchanged across 4 real optimizer steps; **0** backbone params ever receive a gradient; the optimizer holds exactly the 48 sidecar tensors. `--lr 2e-5` applies to an empty group for `swa_steer`. `a0b_frozen_frozen.json` |
| 4 | **Does the old Qasper +0.055 F1 hold on independent data?** | **Yes, and it replicates in size.** Untouched HotpotQA holdout (4678 ids never used for any tuning decision), 3 seeds, official scorer: hierarchical **+0.0582, CI [+0.0488, +0.0670]**. But it is a task-adapter effect, not memory (answer 10). |
| 5 | **LoCoMo official** | Official `task_eval/evaluation.py`, all 1540 QA, conversation-clustered CIs. `base_fullctx` EM 0.0396 / F1 0.1833. `ours_fullctx` (P=0) EM 0.0617 / F1 0.2163, 3-seed hierarchical **ΔF1 +0.0347, CI [+0.0294, +0.0406]**; untouched 1339 QA +0.0316 [+0.0226, +0.0417]. `ours_fullctx` (P=64) F1 0.3021, **ΔF1 +0.1189 [+0.1079, +0.1308]**. **`ours_window_only` (memory masked out) F1 0.2988 — ours − window_only = +0.0034, CI [−0.0055, +0.0105], NOT significant.** `ours_state_only`/`swap` are undefined for P=0 (no state) and inapplicable under the official full-history protocol for P=64; the HotpotQA write-then-drop protocol carries those arms (§2e). |
| 6 | **HotpotQA official Answer EM/F1 and Joint** | Untouched holdout (n=4678), official `hotpot_evaluate_v1.py` SHA `3635853`: base EM **0.4288** / F1 **0.5615**; ours (P=0) EM **0.4906** / F1 **0.6216**; ours (P=64) EM **0.5158** / F1 **0.6725**. Full dev n=7405: base .4319/.5647. **Support and Joint are 0.0 on every arm** — the method has no supporting-fact head and `sp` is submitted empty. These are Answer metrics and must never be quoted as Joint. |
| 7 | **RULER 8K/16K/32K, 13-task macro** | Officially generated (NVIDIA/RULER generator via NeMo-Skills `prepare.py` SHA `f4a3fd8`, Qwen3 tokenizer), 13/13 tasks: 8K **0.9363 → 0.9229** (−0.0134); 16K **0.9425 → 0.9047** (−0.0378); 32K **0.9157 → 0.9153** (−0.0005). **Ours is at or below base at every length.** By category the loss is entirely in aggregation (cwe/fwe) and QA; retrieval and multi-hop are at ceiling for both arms. |
| 8 | **Which configuration is genuinely strongest?** | On raw accuracy, **P=64 noctx pre_o qkvo** (hierarchical over 3 seeds on the untouched holdout: **+0.0990, CI [+0.0764, +0.1155]**). On defensibility, **P=0 SWA sidecar** (**+0.0582, CI [+0.0488, +0.0670]**) — 27x fewer parameters (14.2M vs 383.4M), stable convergent training, half the seed spread. P=64's training **diverges** past ~100–150 updates and its final checkpoints emit `,,,,,,` whenever memory is active. Both are significant; the honest recommendation is P=0. |
| 9 | **Is it significantly higher than the full-context base?** | **Yes, on two official benchmarks, on data never used for any tuning decision, with 3 seeds each.** HotpotQA untouched (n=4678): +0.0582 [+0.0488, +0.0670]. LoCoMo (n=1540, conversation-clustered): +0.0347 [+0.0294, +0.0406]. Same prompt, chat template, decoding and context budget as base in every case. RULER is the exception: at or below base at all three lengths. |
| 10 | **Does the gain come from correct memory rather than task adaptation?** | **No — it is task adaptation.** Across **6 trained checkpoints, 2 architectures, 2 fusion positions, 3 seeds and 5 training objectives**, `correct − swap` never has a CI above zero, and is significantly negative twice. At n=600 on three independently trained checkpoints: **+0.0027 / +0.0007 / +0.0014**, intervals ~0.02 wide straddling zero — tight enough to have detected a +0.02 effect. On LoCoMo, deleting the written memory from the read costs **+0.0034 (n.s.)** while the memory-free model still beats base by **+0.1155**. The written state is an activation the adapter expects to see, not an information carrier: zeroing it hurts significantly (+0.0612) while *swapping* it costs nothing (+0.0014). |
| 11 | **If it does not beat base: distance, failure class, next experiment** | It **does** beat base on accuracy, so this is about the memory claim. Distance to a real memory effect: `correct − swap` is +0.001 ± 0.013 against a §9 bar of +0.01 — the effect is not small-but-positive, it is **absent**. Failure class: **not implementation** (audit F1–F6 all pass, official scorers reproduce the in-repo metrics to 4 decimals), **not evaluation** (three official evaluators, untouched holdouts, clustered/hierarchical CIs), but **write capacity / state use**: the WRITE produces a state whose *content* the READ never depends on. Five rescue objectives — including the correct-vs-swap contrastive one prescribed for exactly this situation — moved that dependence by at most +0.0074, and two made it worse. **That experiment was run — see §7.** On a synthetic task where no adapter can cheat, the write path carries real document-specific information at **1 fact per document (+0.195 correct−swap)**, +0.008 at 2 facts, and **exactly 0.000 at 4, 8 and 16 facts**. The write is not broken, it is nearly empty: capacity collapses between 1 and 4 bindings, while real HotpotQA/LoCoMo documents contain far more. A slot sweep (§7) shows **adding slots does not help and at 2 facts actively hurts** (+0.0078 at 64 slots → 0.0000 at 128 and 256), so the ceiling is the pooled-attention WRITE, not the slot budget. The next experiment is to replace the write with an addressable mechanism (per-fact key-value binding or per-fact write supervision) and re-run the capacity curve; until `correct − swap` stays positive at 8+ facts, no downstream memory claim can succeed. |

## 1. Forced implementation audit (§4)

| # | finding | evidence |
|---|---|---|
| **F1** | **The entire pre-o/post-o line has NO written memory.** `num_prefix_tokens=0`, `prefix_write=False`: the sidecar is a local sliding-window attention adapter over the current sequence. `ours_state_only`, `ours_swap_state`, `ours_zero_state` are **undefined** for these checkpoints — there is no state to keep, swap or zero. | `a0_preo_graph.json`, `a2_posto_isolation.json` |
| **F2** | **`ours_window_only` is not an ablation here.** With P=0 there are no prefix columns to mask, so window-only output is **bit-identical** to the full arm (verified, and reproduced in this session's HotpotQA runs where the two arms scored identically to 4 decimals). Any prior claim resting on that ablation is void. | `a2_posto_isolation.json`, `hotpot_dev200_*.json` |
| **F3** | pre_o formula exact on the live model (residual 0.0); post_o formula exact as well. | `a0_preo_graph.json`, `a2_posto_graph.json` |
| **F4** | **Base parity is bit-exact** over 100 examples: wrapper-with-steer-off and wrapper-at-gain-0 both reproduce pristine HF logits with max abs diff 0.0 and 0% top-1 disagreement. The wrapper adds nothing when disabled. | `a2_posto_parity.json` |
| **F5** | pre_o vs post_o_projected: no implementation bug (see answer 2). | `a1_numerics_numerics.json` |
| **F6** | WRITE/READ inputs: the sidecar reads `hidden_states` of the whole sequence with segment ids {1,2,3} = {ctx, question, answer} present in the tensor; at P=0 there is no separate WRITE pass, so "the writer never sees the question" is vacuously true and equally uninformative. Teacher forcing means the answer tokens ARE in the READ window for training-time positions — for a P=0 window adapter this is standard causal LM behaviour, not leakage, but it does mean the sidecar is trained on a window that includes gold answer tokens at answer positions. | `a0_preo_graph.json: memory_read.seg_unique = [1,2,3]` |

## 2. HotpotQA — internal development (NOT the final holdout)

Full official 10-paragraph distractor context for every arm; identical prompt, chat
template, decoding (greedy, 32 new tokens) and context budget for base and ours.
`ours` here = **`ours_fullctx`** in the §6 taxonomy. Sampling seed 1234 (disjoint
selection process from the historical seed-42 screening).

| n | arm | F1 | EM | ΔF1 vs base | paired bootstrap 95% CI (10k) | McNemar on EM |
|---:|---|---:|---:|---:|---|---|
| 200 | base_fullctx | 0.5439 | 0.4150 | — | — | — |
| 200 | ours pre_o s0 | 0.5720 | 0.4550 | +0.0281 | [−0.0154, +0.0716] **n.s.** | b01=16 b10=8, p=0.152 |
| 200 | ours post_o s0 | 0.5489 | 0.4350 | +0.0050 | not run | — |
| 1000 | base_fullctx | 0.5659 | 0.4340 | — | — | — |
| 1000 | ours pre_o s0 | 0.6156 | 0.4860 | **+0.0497** | **[+0.0322, +0.0677]** | b01=86 b10=34, **p=2.3e-6** |
| 1000 | ours pre_o s1 | 0.6256 | 0.4870 | **+0.0597** | **[+0.0419, +0.0780]** | b01=82 b10=29, **p=4.9e-7** |
| 1000 | 2-seed hierarchical | — | — | **+0.0547** | **[+0.0403, +0.0692]** | — |

The n=200 screen was **not** significant; only n=1000 resolves it. This is exactly why
screening results are not evidence.

### Official scorer cross-check (`hotpot_evaluate_v1.py`, SHA `3635853`)

Our saved predictions were re-scored by the **untouched official evaluator**
(`official_hotpot_dev1000.json`); gold rebuilt from the HF distractor mirror with
**0 missing ids**. `sp` submitted EMPTY because the method has no supporting-fact
head, so Support/Joint are honestly 0.

| file | arm | official EM | official F1 | sp_f1 | joint_f1 |
|---|---|---:|---:|---:|---:|
| dev1000 s0 | base_fullctx | 0.4340 | 0.5659 | 0.0 | 0.0 |
| dev1000 s0 | ours_fullctx | 0.4860 | 0.6156 | 0.0 | 0.0 |
| dev1000 s1 | ours_fullctx | 0.4870 | 0.6256 | 0.0 | 0.0 |

The official numbers match the in-repo metric to 4 decimals, so the measurement
chain (prompt → generation → parser → metric) is validated end-to-end.

**Interpretation, stated precisely.** This is a significant improvement over the
full-context frozen base under identical inputs and decoding. It is **NOT** a memory
result: with P=0 there is no written state, and the window-only arm is bit-identical
to the full arm (F2). The correct label is **task adaptation / attention-output
sidecar**, i.e. §6's "correct ≈ swap > base" branch taken to its limit — swap and
correct are not merely close, they are the same computation. It must not be described
as episodic memory, context compilation or memory augmentation.


## 2b. DECISIVE memory test — P=64 written memory, all §6 arms (HotpotQA n=200, dev)

The §2 checkpoints have no state at all (F1), so a P=64 written-memory sidecar was
trained from scratch this session (`noctx`: WRITE the document → DROP the context →
READ from the frozen state; `delta_heads=qkvo`, gain 0.1, pre_o, 200 updates —
checkpoint at step 100 evaluated first). Every §6 arm, same 200 examples, same
prompt/decoding:

| §6 arm | script name | F1 |
|---|---|---:|
| `base_fullctx` | base_ctx | 0.5439 |
| **`ours_fullctx`** | ours_ctx | **0.6643** |
| `base_queryonly` | base_noctx | 0.0600 |
| `ours_state_only` | ours_noctx | 0.1707 |
| `ours_swap_state` | swap_noctx | 0.1681 |
| `ours_zero_state` | zero_noctx | 0.1782 |
| `ours_window_only` | wo_noctx | 0.1935 |

| contrast | Δ | paired t |
|---|---:|---:|
| ours_fullctx − base_fullctx | **+0.1204** | **+4.59** |
| ours_state_only − base_queryonly | +0.1107 | +4.63 |
| **ours_state_only − ours_swap_state** | **+0.0025** | **+0.34** |
| ours_state_only − ours_zero_state | −0.0076 | −0.64 |
| ours_state_only − ours_window_only | −0.0229 | −1.74 |

**Verdict: the written state carries no document-specific information.** Reading a
state written from a *different* document scores the same as the correct one
(+0.0025, t=0.34); zeroing the state is as good; masking the prefix out of the read
is slightly *better*. The whole state_only − queryonly gap (+0.11) is reproduced by
the zero/swap/window arms, i.e. it is the trained adapter's answer-style prior, not
retrieved content. Under §6's rule this is **task adaptation, not memory** — and now
that verdict rests on an architecture that genuinely has a written state, so it is
not an artifact of the P=0 line.

**Replicated at the other fusion position.** The same protocol on the P=64 `post_o`
checkpoint gives correct−swap **−0.0034** (t=−0.32), correct−zero +0.0019 (t=+0.17),
correct−window −0.0007 (t=−0.10), with ours_fullctx 0.6302 vs base 0.5439 (+0.0863,
t=+3.35). Two architectures (P=0, P=64) x two fusion positions all agree: the written
state is inert, the adapter is not.

**STABILITY CAVEAT — the P=64 run does not converge.** Its loss oscillates
(0.94 → 3.01 → 3.16 → 2.60 → 1.53 over the first 100 updates) and the **final
(step-200) checkpoints are degenerate**: on the same 200 HotpotQA examples the
post_o final scores **0.0000** on every arm that actually reads the written memory
(`ours_ctx`, `ours_noctx`, `swap_noctx`) while the memory-free arms are unaffected
(`zero_noctx` 0.2058, `wo_noctx` 0.1936); the pre_o final scores 0.034 on the Qasper
noctx `method` arm, *below* its own base 0.0791. Every P=64 number quoted below
therefore comes from the **step-100 intermediate checkpoint**, i.e. it rests on
checkpoint selection, which is itself a tuning decision and is disclosed as such.
The pre_o final collapses identically (`ours_ctx` 0.0035, `ours_noctx` 0.0028,
`swap_noctx` 0.0176 vs `zero_noctx` 0.1486, `wo_noctx` 0.1176).

**The collapse is genuine divergence, not a scoring or formatting artifact.** Raw
greedy generations from the final post_o checkpoint on three HotpotQA items, memory
active vs the same model with the sidecar disabled:

| gold | ours_ctx (memory on) | base_ctx (same weights, sidecar off) |
|---|---|---|
| `15` | `,,,,,,,,,,,,,,,,,,,,` | `15` |
| `Rabies` | `,,,,,,,,,,,,,,,,,,,,` | `Rabies` |
| `Rothschild banking dynasty` | `,,,,,,,,,,,,,,,,,,,,` | `Rothschild dynasty` |

Note the collapse is a **transfer** phenomenon: on the in-distribution Qasper noctx
eval these same final checkpoints still look healthy (`method` 0.0856 vs `base`
0.0791), so an in-distribution validation metric would not have caught it.

A lower-LR re-train (prefix-lr 1e-3, lr 2e-4, 12 layers instead of 36) is running to
test whether the architecture can be made to converge at all.

**The P=64 adapter (step-100) is the strongest configuration measured this session.** At n=1000
(dev, seed 1234, full context): ours **0.6667** vs base **0.5659**, ΔF1 **+0.1008**,
paired bootstrap CI **[+0.0765, +0.1256]**, EM .5050 vs .4340, McNemar p=3.5e-7 —
roughly double the P=0 line's +0.056. The cost is size: this checkpoint trains
**355,074,048** parameters over **36** patched layers (vs 14,155,776 over 12 layers
for the P=0 line), so the two are NOT a matched-capacity comparison. Its own
matched no-memory control inside the Qasper eval (`method_nomem`) scores 0.0791 vs
method 0.0856, i.e. the memory's marginal contribution there is +0.0065.

The same run shows the **largest positive result of the session**: with full context
retained, this sidecar scores **+0.1204 F1 over the frozen full-context base**
(t=+4.59, n=200 dev). That is a memory-*augmentation*-shaped number produced by a
component that provably is not using its memory — it must be reported as an
attention/task adapter. n=1000 confirmation is running.


## 2c. FINAL — HotpotQA untouched holdout (official evaluator, config frozen)

Config frozen before this run: `pre_o` + `fixed_add`, `swa_steer` P=0, gain 0.1,
`delta_heads=o`, `main_v`, layers 0,3,…,33, seed 0 checkpoint
`out_dex_fusion/preo_swa_steer_s0_steer.pt`; official 10-paragraph distractor
context; greedy, 32 new tokens; parser `extract_first_line`; scorer
`hotpot_evaluate_v1.py` SHA `3635853`. The **untouched complement** excludes every
id used by any historical screening run and by this session's dev screens
(2727 excluded → 4678 remain).

| set | n | arm | official EM | official F1 | ΔF1 | 95% CI (paired bootstrap 10k) | McNemar (EM) |
|---|---:|---|---:|---:|---:|---|---|
| **untouched complement** | 4678 | base_fullctx | 0.4288 | 0.5615 | — | — | — |
| **untouched complement** | 4678 | **ours_fullctx** | **0.4906** | **0.6216** | **+0.0602** | **[+0.0513, +0.0691]** | b01=445 b10=156, **p=3.9e-33** |
| full official dev | 7405 | base_fullctx | 0.4319 | 0.5647 | — | — | — |
| full official dev | 7405 | ours_fullctx | 0.4905 | 0.6209 | +0.0562 | [+0.0492, +0.0633] | b01=675 b10=241, p=3.0e-48 |

### The P=64 sidecar on the same untouched holdout (strongest config, with caveats)

| set | n | arm | EM | F1 | ΔF1 | 95% CI | McNemar (EM) |
|---|---:|---|---:|---:|---:|---|---|
| **untouched complement** | 4678 | base_fullctx | 0.4288 | 0.5615 | — | — | — |
| **untouched complement** | 4678 | **ours_fullctx (P=64 @step100)** | **0.5158** | **0.6725** | **+0.1111** | **[+0.0993, +0.1228]** | b01=677 b10=270, **p=5.4e-41** |
| full official dev | 7405 | ours_fullctx (P=64 @step100) | 0.5113 | 0.6673 | +0.1027 | [+0.0935, +0.1116] | b01=1023 b10=435, p=8.1e-55 |

This is **the largest verified gain of the session — +0.111 F1 / +0.087 EM over the
frozen full-context base on 4678 never-touched official dev ids.** Four caveats, all
load-bearing, none of which the number itself shows:

1. **27x the parameters** — 383,385,600 trainable over 36 layers, against 14,155,776
   over 12 for the P=0 line. This is not a matched-capacity comparison.
2. **Checkpoint selection.** The step-200 checkpoint of this exact run is degenerate
   (§2b); the number comes from step 100, chosen because the run does not converge.
   That selection is a tuning decision made with knowledge of the failure.
3. **Its memory is inert** — correct−swap +0.0025, CI [−0.012, +0.018] (§2e).
4. **3.2x slower per query** than the base it beats (§6).

**Replication (and a cleaner protocol).** Seeds 1 and 2 were trained with
`--steps 100` outright, so their *final* checkpoint IS the step-100 model — **no
checkpoint selection is involved for them**, which removes caveat 2 above for those
seeds. Internal dev n=1000, same 1000 items, same base:

| seed | checkpoint provenance | base F1 | ours F1 | ΔF1 |
|---|---|---:|---:|---:|
| 0 | step-100 **selected** from a 200-step run | 0.5659 | 0.6667 | +0.1008 |
| 2 | final of a 100-step run (**no selection**) | 0.5659 | 0.6841 | **+0.1182** |
| 1 | final of a 100-step run (no selection) | — | RUNNING | — |

The effect replicates and is *larger* on the seed that required no checkpoint
selection. Full official dev + untouched-complement runs for seeds 1 and 2 are queued.

### P=64 three-seed untouched final (the number to quote)

Same 4678 never-touched official dev ids, same base, three independently trained seeds
(seeds 1 and 2 involve **no checkpoint selection** — they were trained for exactly 100
steps):

| seed | ours F1 | ΔF1 | 95% CI | EM | McNemar |
|---|---:|---:|---|---:|---|
| 0 | 0.6725 | +0.1111 | [+0.0996, +0.1226] | 0.5158 | p=5.4e-41 |
| 1 | 0.6363 | **+0.0748** | [+0.0637, +0.0858] | 0.4767 | p=8.6e-15 |
| 2 | 0.6726 | +0.1111 | [+0.0998, +0.1227] | 0.5346 | p=2.9e-62 |
| **hierarchical (3 seeds)** | — | **+0.0990** | **[+0.0764, +0.1155]** | — | — |

Seed 1 is genuinely lower — its interval does not overlap the other two — so P=64 does
carry **real seed variance of +0.075 to +0.111**, roughly twice the spread of the P=0
line (+0.049 to +0.065). All three seeds are individually significant.

**Head-to-head, both configurations on the identical untouched holdout:**

| config | trainable | hierarchical ΔF1 (3 seeds) | 95% CI | training | memory |
|---|---:|---:|---|---|---|
| P=0 SWA sidecar | 14,155,776 | +0.0582 | [+0.0488, +0.0670] | stable, converged | none exists |
| P=64 memory sidecar | 383,385,600 | **+0.0990** | [+0.0764, +0.1155] | **does not converge** (§2b) | inert (§2e) |

P=64 wins on accuracy by ~4 F1 points, at 27x the parameters, with a training run that
diverges if left past ~100-150 updates, twice the seed spread, and a memory that
contributes nothing. **P=0 is the configuration this work can actually defend**; P=64 is
the larger number.

**Seed variance, measured at two sample sizes.** On a 300-item sample (seed 777) the
three P=64 seeds gave `ours_fullctx − base_fullctx` of +0.1204 / +0.0110 / +0.0939 —
which looked like seed 1 barely transferring. At **n=1000** (seed 1234, the same 1000
items for every seed) the picture is tighter: **+0.1008 / +0.0713 / +0.1182**, mean
+0.097. The n=300 spread was largely sampling noise, not seed failure; the honest
statement is that P=64 transfers at **+0.07 to +0.12 depending on seed**, which is
still visibly wider than the P=0 line's +0.049 to +0.065. The 3-seed full-dev
confirmation now running is the number that should be quoted.

**The memory verdict replicates on every seed.** Seed 1: correct − swap **−0.0433**,
correct − window −0.0253, correct − zero −0.0681.

**The memory verdict replicates on that seed too.** Seed 2's own §6 arm set (n=300,
no checkpoint selection anywhere in its history): base_ctx 0.5886, ours_ctx 0.6824
(**+0.0939**), base_noctx 0.0726, ours_state_only 0.1611, swap 0.1726, zero 0.1965,
window 0.1769 — i.e. **correct − swap = −0.0114**, correct − window = −0.0158,
correct − zero = −0.0354. An independently trained model, selected in no way, still
does *worse* reading the correct document's state than a different document's.

**Multi-seed on the same untouched holdout** (§9 Stage-C: direction must agree across
seeds; the hierarchical bootstrap resamples examples first, then seeds, so the same
4678 ids are never counted as independent observations across seeds):

| seed | n | base F1 | ours F1 | ΔF1 | 95% CI | EM McNemar |
|---|---:|---:|---:|---:|---|---|
| 0 | 4678 | 0.5615 | 0.6216 | +0.0602 | [+0.0510, +0.0692] | p=3.9e-33 |
| 1 | 4678 | 0.5615 | 0.6269 | +0.0654 | [+0.0569, +0.0740] | p=2.8e-46 |
| 2 | 4678 | 0.5615 | 0.6105 | +0.0490 | [+0.0397, +0.0582] | p=2.5e-24 |
| **hierarchical, all 3 seeds** | — | — | — | **+0.0582** | **[+0.0488, +0.0670]** | — |

**3/3 seeds are individually significant and the hierarchical interval excludes zero**,
so §9 Stage-C (2-of-3 direction) and Stage-D (one untouched final) are both satisfied
for the P=0 configuration. Full-dev means per seed: 0.6209 / 0.6259 / 0.6090 vs the
same base 0.5647.

Support/Joint are **0.0** on every arm: the method has no supporting-fact head and
`sp` is submitted empty. These are **Answer** EM/F1 only and must never be quoted as
HotpotQA Joint numbers.

**This clears the §9 Stage-D bar**: significantly higher than the frozen full-context
base, on data never used for any tuning decision, with the paired CI lower bound
+0.051 > 0, under an identical prompt/template/decoding/context budget.

**It does not clear the §6 memory bar** — see §2b: the same family of sidecars scores
identically when the written state is swapped, zeroed, or masked out. The honest
label is a **task/attention adapter that improves full-context HotpotQA**, not memory.


## 2d. P1 rescue round — every arm, including the failures

All trained this session on Qasper with P=64 written memory (write the document →
drop/mask the context → read from the frozen state), evaluated by each run's own
in-distribution eval. `method_nomem` / `window_only` are the matched no-memory
controls (§6 arm 9): same weights, same step count, memory read disabled.

| arm | training | base | method (memory ON) | matched no-memory control | memory's marginal value |
|---|---|---:|---:|---:|---:|
| `mem64_noctx_posto_s0` | noctx, post_o | 0.0791 | 0.0856 | 0.0791 (`method_nomem`) | **+0.0065** |
| `mem64_noctx_preo_s0` | noctx, pre_o | 0.0791 | **0.0340** | 0.0791 | **−0.0451 (harmful)** |
| `mem64_ctxmask_posto_s0` | ctxmask ratio 1.0 | 0.1784 | **0.0718** | 0.2146 (`window_only`) / 0.1784 (`nomem`) | **−0.1428 vs window-only (harmful)** |
| `mem64_swapc_b5_s0` | noctx + swap-contrast β=5, margin 0.5 | 0.0791 | 0.0865 | 0.0791 | **+0.0074** |
| `mem64_stable_preo_s0` | noctx, pre_o, prefix-lr 1e-3, **12 layers** | — | **did not converge** (loss 1.65 @180 → 4.20 @200) | — | the low-LR/fewer-layer fix does NOT stabilise the architecture |
| `mem64_swapc_b1_s0` | noctx + swap-contrast β=1 | 0.0791 | **0.0717** | 0.0791 | **−0.0074 (harmful)** |

Not one rescue arm produced a memory contribution worth more than **+0.0074**, and two
of them made the model *worse* than its own no-memory control.

**The correct-vs-swap contrastive objective — the intervention §11-B prescribes for
exactly this situation — failed decisively.** Full §6 arm set for β=5 on 200 HotpotQA
items:

| arm | F1 | vs the correct-state arm |
|---|---:|---:|
| base_fullctx | 0.5439 | — |
| **ours_fullctx** | **0.2784** | the trained model is **0.266 BELOW base** with full context |
| base_queryonly | 0.0600 | — |
| ours_state_only (correct) | 0.0884 | — |
| ours_swap_state | 0.1155 | **correct − swap = −0.0272** |
| ours_zero_state | 0.1900 | correct − zero = −0.1016 |
| ours_window_only | 0.1666 | correct − window = −0.0782 |

Reading the state written from the **correct** document is *worse* than reading one
written from a **different** document. Adding an explicit penalty for behaving the same
under a swapped state did not create document-specific memory; it degraded the model on
every arm, including the full-context arm that had been the method's only win. β=1 is
harmful too (Qasper method 0.0717 vs base 0.0791). This is the strongest available
evidence that the write path cannot be made informative by changing the objective.

**The ctxmask arm at n=1000 is the sharpest instance of the pattern.** Same model,
same 1000 HotpotQA items, full official context:

| arm | F1 | EM |
|---|---:|---:|
| base_fullctx | 0.5659 | 0.4340 |
| **ours (written memory ON)** | **0.2486** | 0.1890 |
| **ours_window_only (same weights, memory masked out of the read)** | **0.6429** | 0.4890 |

Turning the written memory ON costs **0.394 F1** relative to the identical model with
the memory masked out — and the memory-free version is the best HotpotQA number this
arm produces, **+0.077 above base**. The adapter is worth +0.077; its memory is worth
−0.394.

**LoCoMo ablation with a real written state (the strongest test available).** LoCoMo is
write-once/query-many, the scenario most favourable to memory, and at P=64
`ours_window_only` is a *genuine* ablation (there are prefix columns to mask, unlike
P=0). Interim over 450/1540 QA: base_fullctx **0.2195**, ours **0.3310**,
ours_window_only **0.3259** — masking the entire written memory out of the read costs
**0.005 of an 11-point gain**.


## 2e. The memory verdict, stated statistically (all six trained checkpoints)

`ours_state_only − ours_swap_state` is the one quantity that isolates
document-specific memory: identical model, identical prompt, identical decoding,
the *only* difference is whether the state was written from the correct document
or from a different one. Paired bootstrap, 10,000 resamples, n=200 HotpotQA items
each (`memarms_*.json`).

| checkpoint | correct − swap | 95% CI | correct − window | correct − zero | ours_fullctx − base_fullctx |
|---|---:|---|---:|---:|---:|
| P=64 noctx post_o @step100 | −0.0034 | [−0.0248, +0.0164] | −0.0007 | +0.0019 | +0.0863 |
| P=64 noctx pre_o @step100 | +0.0025 | [−0.0120, +0.0175] | −0.0229 | −0.0076 | **+0.1204** |
| P=64 stable pre_o @step150 (lr 1e-3, 12 layers) | **+0.0065** | [−0.0143, +0.0291] | +0.0095 | +0.0627 | +0.0825 |
| P=64 swap-contrastive β=5 | **−0.0272** | **[−0.0539, −0.0018]** | −0.0782 | −0.1016 | −0.2655 |
| P=64 noctx pre_o @final | **−0.0148** | **[−0.0286, −0.0046]** | −0.1148 | −0.1458 | −0.5404 |
| P=64 noctx post_o @final | 0.0000 | [0.0000, 0.0000] | −0.1936 | −0.2058 | −0.5439 |

### n=600 on three independently trained checkpoints — the summary table

| checkpoint | n | **correct − swap** | 95% CI | **ours_fullctx − base_fullctx** | 95% CI |
|---|---:|---:|---|---:|---|
| P=64 noctx post_o @step100 | 600 | **+0.0027** | [−0.0066, +0.0125] | **+0.0805** | [+0.0535, +0.1084] |
| P=64 noctx pre_o @step100 | 600 | **+0.0007** | [−0.0124, +0.0141] | **+0.1099** | [+0.0793, +0.1415] |
| P=64 stable pre_o @step150 | 600 | **+0.0014** | [−0.0089, +0.0122] | **+0.0774** | [+0.0485, +0.1068] |

Three separately trained sidecars, 600 fresh items each, two columns: the memory's
document-specific value is **+0.001 to +0.003 with intervals of width ~0.02 straddling
zero**, while the same models' gain over full-context base is **+0.077 to +0.110 with
intervals nowhere near zero**. The intervals are tight enough that a real memory effect
of even +0.02 would have been detected. It is not there.

### Tightened to n=600 on the strongest checkpoint

The central claim deserves more than 200 examples. Re-running the full arm set on
**600** fresh HotpotQA items (seed 777, disjoint selection) with the P=64 pre_o
step-100 checkpoint:

| contrast | Δ | 95% CI (paired bootstrap 10k) | significant |
|---|---:|---|---|
| **ours_state_only − ours_swap_state** | **+0.0007** | **[−0.0126, +0.0144]** | **NO** |
| ours_state_only − ours_window_only | +0.0055 | [−0.0096, +0.0208] | NO |
| ours_state_only − ours_zero_state | −0.0111 | [−0.0249, +0.0025] | NO |
| **ours_fullctx − base_fullctx** | **+0.1099** | **[+0.0793, +0.1415]** | **YES** |

Arm means: base_ctx 0.5471, ours_ctx 0.6571, base_noctx 0.0593, ours_noctx 0.2022,
swap 0.2015, window 0.1968, zero 0.2134. At n=600 the document-specific value of the
written memory is **0.0007 ± 0.013** — measured precisely, and precisely zero — while
the same model's gain over full-context base is +0.11 with an interval nowhere near
zero.

**Not one checkpoint has a correct−swap interval lying above zero. Two are
significantly negative.** The best arm ever measured (+0.0065, the stability
re-train at step 150) has an interval spanning zero and sits far below the
§9 promotion bar of +1 absolute point. The post_o final row is 0.0000 with a
zero-width interval because *both* arms emit degenerate output — it is not
evidence of equality, it is evidence of collapse.

Meanwhile `ours_fullctx − base_fullctx` is large and positive for the healthy
checkpoints (+0.086 to +0.120). The two quantities together are the whole story:
**the sidecar helps, its memory does not.**


### Why `zero_state` is not a sufficient control — and what the state actually is

The stability re-train at step 150 (the only checkpoint whose correct−swap point
estimate was positive) re-measured at **n=600**:

| contrast | Δ | 95% CI | significant |
|---|---:|---|---|
| ours_state_only − ours_swap_state | **+0.0014** | [−0.0089, +0.0119] | NO |
| ours_state_only − ours_window_only | −0.0056 | [−0.0233, +0.0116] | NO |
| **ours_state_only − ours_zero_state** | **+0.0612** | **[+0.0391, +0.0832]** | **YES** |
| ours_fullctx − base_fullctx | +0.0774 | [+0.0484, +0.1066] | YES |

Its n=200 point estimate of +0.0065 shrinks to +0.0014 once 600 examples are used.

The three rows together identify what the written state is. **Zeroing it hurts
significantly (+0.061), swapping it does not hurt at all (+0.001).** A zero tensor is
off-distribution for a model trained to always receive a state; a state written from a
*different document* is perfectly on-distribution and carries the wrong content — and
the model scores the same with it. So the state functions as a **generic activation the
adapter expects to see, not as an information carrier**.

This is exactly why `ours_zero_state` alone cannot support a memory claim and why §6
requires the swap arm: an evaluation that reported only correct-vs-zero here would have
shown a significant "+0.061 from memory" and been wrong.

## 3. RULER (mirror `simonjegou/ruler`, 13/13 official tasks, 40 samples/task)

| length | arm | macro | note |
|---|---|---:|---|
| 8192 | base | 0.9288 | at ceiling: 8/13 tasks exactly 1.000 for both arms |
| 8192 | ours pre_o s0 | 0.9254 | −0.0034; qa_1 +0.025, qa_2 −0.050, cwe −0.012 |
| 16384 | base | 0.9439 | 13/13 tasks |
| 16384 | ours pre_o s0 | 0.9361 | −0.0078; qa_1 +0.075, cwe −0.058, qa_2 −0.100 |
| 8192 **OFFICIAL** | base | **0.9363** | official generation, 13/13 tasks, 25 samples/task |
| 8192 **OFFICIAL** | ours pre_o s0 | **0.9229** | −0.0134; qa_1 0.60 vs 0.68, qa_2 0.44 vs 0.56, fwe 1.00 vs 0.96 |
| 16384 **OFFICIAL** | base | **0.9425** | official generation |
| 16384 **OFFICIAL** | ours pre_o s0 | **0.9047** | **−0.0378**, the largest gap; qa_2 0.48 vs 0.76, fwe 0.867 vs 0.96 |
| 32768 **OFFICIAL** | base | **0.9157** | NVIDIA/RULER generator via NeMo-Skills `prepare.py` (SHA `f4a3fd8`) with the Qwen3 tokenizer; 13/13 tasks, 25 samples/task |
| 32768 **OFFICIAL** | ours pre_o s0 | **0.9153** | −0.0005 (tied); qa_1 0.80 vs 0.68, fwe 0.867 vs 0.907, cwe 0.832 vs 0.848 |

Historical runs agree (4k: base 0.9360 / ours 0.9355; 16k: base 0.9282 / ours 0.9274),
so RULER at these lengths has no headroom for this method to show anything.

The 8k/16k rows above come from the `simonjegou/ruler` **mirror** and are labelled
**NON-OFFICIAL-GENERATION**; the 32768 row is officially generated. Official 8k/16k
regeneration is in flight. Across every length and both data sources the verdict is
the same: **ours is at or fractionally below base on RULER**, with a consistent
per-task signature (qa_1 better, fwe/cwe/qa_2 worse).


### RULER by official task category (§5-C-7)

`ruler_official_category_table.json`. Categories follow the official task grouping:
retrieval = the 8 NIAH variants, multi-hop = variable tracking, aggregation = cwe+fwe,
QA = qa_1+qa_2.

| length | arm | retrieval (8) | multi-hop (1) | aggregation (2) | QA (2) | macro-13 |
|---|---|---:|---:|---:|---:|---:|
| 8K | base | 1.0000 | 1.0000 | 0.9660 | 0.6200 | 0.9363 |
| 8K | ours (P=0) | 0.9988 | 1.0000 | 0.9840 | 0.5200 | 0.9229 |
| 8K | ours (P=64) | 0.9950 | 1.0000 | 0.9053 | 0.5800 | 0.9177 |
| 16K | base | 1.0000 | 1.0000 | 0.9460 | 0.6800 | 0.9425 |
| 16K | ours (P=0) | 0.9938 | 1.0000 | 0.8853 | 0.5200 | 0.9047 |
| 16K | ours (P=64) | 0.9912 | 1.0000 | 0.8620 | 0.6200 | 0.9149 |
| 32K | base | 0.9988 | 1.0000 | 0.8773 | 0.5800 | 0.9157 |
| 32K | ours (P=0) | 0.9950 | 1.0000 | 0.8493 | 0.6200 | 0.9153 |
| 32K | ours (P=64) | 1.0000 | 1.0000 | 0.7180 | 0.6200 | 0.8982 |

**P=64 on the same officially generated data, two seeds** (macro-13): 8K 0.9177 /
0.9194, 16K 0.9149 / 0.9129, 32K 0.8982 — every cell below the corresponding base
(0.9363 / 0.9425 / 0.9157). Neither configuration nor any seed beats base on RULER.

**Retrieval and multi-hop are at ceiling for both arms at every length** — the sidecar
neither helps nor hurts needle-finding. The entire deficit sits in **aggregation**
(counting/frequency over the whole context) and **QA**. That is the signature of a
component that perturbs global aggregation while leaving local retrieval intact, which
is what a sliding-window attention adapter should do. No task was dropped from this
table after seeing the results.

## 4. LoCoMo — official scorer, all 1540 QA

Protocol: official `data/locomo10.json`, official conversation serialization and QA
prompt, WRITE once per conversation and query many. Scored by the **official**
`task_eval/evaluation.py` (SHA `3eb6f2c`) — its F1 uses PorterStemmer stemming, which
differs from the in-repo metric, so only the official numbers are quoted here. The
LLM-judge metric is **unavailable** (no API key) and is **not** substituted.

Conversation assignment for clustering was reconstructed by position and verified:
**0 category mismatches across all 1540 rows** (10 conversations, counts
152/81/152/199/178/123/150/191/156/158 = 1540).

| set | n | arm | official EM | official F1 | ΔF1 | conversation-clustered CI95 (10k) |
|---|---:|---|---:|---:|---:|---|
| all QA | 1540 | base_fullctx | 0.0396 | 0.1833 | — | — |
| all QA | 1540 | **ours_fullctx** | **0.0617** | **0.2163** | **+0.0331** | **[+0.0247, +0.0416]** |
| **untouched** (excl. the 200 QA used by historical `10x20` runs) | 1339 | base_fullctx | 0.0403 | 0.1653 | — | — |
| **untouched** | 1339 | **ours_fullctx** | **0.0612** | **0.1969** | **+0.0316** | **[+0.0226, +0.0417]** |

EM on all QA: +0.0221, clustered CI [+0.0140, +0.0321]; McNemar b01=43 b10=9,
**p=2.0e-6**. Untouched EM +0.0209, CI [+0.0139, +0.0289].

**Three training seeds, all 1540 QA, official scorer, conversation-clustered CIs**
(identical base every time: EM 0.0396 / F1 0.1833):

| seed | ours EM | ours F1 | ΔF1 | clustered CI95 | significant |
|---|---:|---:|---:|---|---|
| 0 | 0.0617 | 0.2163 | +0.0331 | [+0.0247, +0.0416] | YES |
| 1 | 0.0584 | 0.2173 | +0.0340 | [+0.0265, +0.0425] | YES |
| 2 | 0.0636 | 0.2203 | +0.0370 | [+0.0262, +0.0474] | YES |
| **hierarchical (conversations × seeds)** | — | — | **+0.0347** | **[+0.0294, +0.0406]** | **YES** |

**3/3 seeds individually significant**, and the hierarchical bootstrap — which
resamples conversations first and then the seed set, so neither the 10 conversations
nor the 3 seeds are treated as more independent than they are — also excludes zero.
Conversation mapping verified with **0 category mismatches in 1540 rows** for every
seed.

By category (in-repo metric, all 1540): cat1 0.2506→0.3048, cat2 0.1613→0.2071,
cat3 0.1198→0.1176 (the only non-positive category), cat4 0.2262→0.2409. No single
category drives the average.

This is a **second benchmark** where the sidecar significantly beats the frozen
full-context base on untouched data. It is the same `ours_fullctx` task-adapter
effect as HotpotQA — LoCoMo was evaluated with the P=0 checkpoint, which has no
state at all (F1), so it cannot be a memory result.


## 6. Cost accounting (required by the reporting spec)

Measured on 15 real HotpotQA items with the P=64 `pre_o` step-100 checkpoint,
identical no-KV-cache greedy loop for every arm so the comparison is fair
(`cost_mem64preo_step100.json`).

| quantity | value |
|---|---|
| trainable params | **383,385,600** (P=64, 36 layers) vs **14,155,776** (P=0, 12 layers) |
| context tokens (official 10 paragraphs) | 1696.6 mean |
| tokens the WRITER consumes | 1696.6 — the full document, no truncation |
| tokens the READER sees, `state_only` | **49.3** (question + template only) |
| tokens the READER sees, `fullctx` | 1745.9 |
| written state | 36 layers x 64 slots x 2560 dims = **5,898,240 elements = 11.8 MB** bf16 (0.33 MB/layer) |
| write latency | **0.131 s** |
| read latency, `state_only` | 1.814 s |
| read latency, `ours_fullctx` | 1.752 s |
| read latency, **frozen base fullctx** | **0.543 s** |
| overhead of the sidecar per query | **+1.21 s (3.2x slower than base)** |
| amortized `state_only` per query, 1 / 5 / 20 queries | 1.945 / 1.840 / 1.821 s |
| effective correction norm | ‖gC‖/‖Z‖ = **0.145**, ‖g·W_O C‖/‖W_O Z‖ = **0.260** (g = 0.1) |
| cos(Z, C) / cos(W_O Z, W_O C) | 0.018 / 0.043 |

Two practical consequences the accuracy tables do not show:

1. **The method is 3.2x slower per query than the base it beats.** The write is
   cheap (0.131 s) but the per-token sidecar cost dominates decoding.
2. **Amortizing the write over many queries does not help** — at 20 queries per
   state the cost is still 1.82 s/query vs the base's 0.543 s. The compression is
   real (1697 context tokens → 49 read tokens, an 11.8 MB state) but it buys no
   latency, because the sidecar runs on every decode step regardless.


## 4b. THE DECISIVE TABLE — LoCoMo, official scorer, all 1540 QA, one model, three arms

Same P=64 checkpoint (which *does* have a written state), same 1540 questions, same
prompt and decoding. `ours_window_only` masks the written memory out of the read and
changes nothing else — at P=64 this is a genuine ablation, not the no-op it is at P=0.
Official `task_eval/evaluation.py`; conversation-clustered paired bootstrap, 10,000
resamples, 10 clusters.

| contrast | ΔF1 | 95% CI (conversation-clustered) | significant | ΔEM | 95% CI |
|---|---:|---|---|---:|---|
| **ours − base_fullctx** | **+0.1189** | **[+0.1079, +0.1308]** | **YES** | +0.0468 | [+0.0371, +0.0578] |
| **ours − ours_window_only** | **+0.0034** | **[−0.0055, +0.0105]** | **NO** | **−0.0026** | [−0.0113, +0.0050] |
| **ours_window_only − base_fullctx** | **+0.1155** | **[+0.1042, +0.1275]** | **YES** | +0.0494 | [+0.0358, +0.0653] |

Absolute numbers: base EM 0.0396 / F1 0.1833; ours EM 0.0864 / F1 0.3021;
window-only EM 0.0890 / F1 0.2988.

**All three P=64 seeds, in-repo metric, all 1540 QA** (identical base 0.2105):

| seed | ours | ours_window_only | ours − window_only |
|---|---:|---:|---:|
| 0 | 0.3179 | 0.3127 | +0.0052 |
| 1 | 0.2759 | 0.2903 | **−0.0144** |
| 2 | 0.2893 | 0.3006 | **−0.0113** |
| mean | 0.2944 | 0.3012 | **−0.0068** |

**Two of the three seeds score *higher* with the written memory masked out**, and the
mean effect of having the memory is negative. Every seed still beats base by +0.065 to
+0.107 without it.

**Read the second and third rows together.** Deleting the written memory from the read
costs 0.003 F1 with an interval spanning zero, while the memory-free model still beats
the full-context base by +0.116 with an interval far from zero. On EM the memory-free
arm is *ahead*. This is the whole finding in one table, on the benchmark most
favourable to memory (write once, query many), with a checkpoint that genuinely has a
state to ablate:

> **the sidecar's gain over base is real and significant; the written memory
> contributes nothing measurable to it.**

## 5. Contamination (§3)

`contamination_manifest.json`:

| dataset | already seen | untouched remaining |
|---|---:|---:|
| Qasper | 800 train / 75 val papers, 187 val examples — every config decision since July | **0** (unusable as confirmation) |
| HotpotQA official dev (7405) | 2000 historical screening IDs + 1000 this-session dev IDs = 2727 unique | **4678** |
| LoCoMo (1540 QA) | 200 historical questions | 1340 |
| RULER | 4k/16k mirror runs, per-task caps | new seeds registrable |

The HotpotQA confirmatory number will be computed on the **4678-ID untouched
complement** of the full-dev run now in flight, not on the dev-1000 above.


## 7. Why the memory is inert — the write-capacity curve (synthetic diagnostic)

Answer 11 above recommended measuring whether the write path can encode retrievable
content **at all** before spending more compute downstream. That experiment was run.

`investigation/episodic_kv_test.py` trains the same P=64 architecture on a synthetic
task where **a task adapter cannot succeed by construction**: every episode randomly
re-assigns the facts, so the question alone does not determine the answer, and only
information actually written into the state can produce a correct answer. Same four
arms as the benchmarks. 400 training steps, 128 eval episodes, identical
hyper-parameters at every point.

| facts per document | base | window | swap | correct | **correct − swap** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0078 | 0.0469 | 0.0469 | 0.2422 | **+0.1953** |
| 2 | 0.0078 | 0.0469 | 0.1719 | 0.1797 | +0.0078 |
| 4 | 0.0000 | 0.0703 | 0.0469 | 0.0469 | **0.0000** |
| 8 | 0.0078 | 0.0859 | 0.0859 | 0.0859 | **0.0000** |
| 16 | 0.0000 | 0.0859 | 0.0859 | 0.0859 | **0.0000** |

**The write path is nearly empty, and the one positive point is not robust.**
Replication was run in both directions rather than only on the convenient side:

| point | seed 1 | seed 2 | seed 3 | verdict |
|---|---:|---:|---:|---|
| 1 fact/doc | +0.1953 | **+0.0000** | +0.1016 | **not robust** — one seed of three is exactly zero |
| 4 facts/doc | 0.0000 | 0.0000 | — | **robust zero** |

And a longer budget does not rescue it. Re-running 2 facts at **900 steps** (2.25x the
original budget), which also tests the "more slots need more steps" caveat below:

| slots | correct−swap @400 steps | correct−swap @900 steps |
|---:|---:|---:|
| 64 | +0.0078 | **+0.0000** |
| 128 | +0.0000 | +0.0000 |
| 256 | +0.0000 | +0.0000 |

More training removes the marginal +0.008 rather than growing it, and neither slot
count benefits. **So: the collapse to zero by 4 facts is robust across seeds and
budgets; the 1-fact positive is real in 2 of 3 seeds but highly variable at this
budget.** The July runs in `out_prefix_value/` reached correct 0.99 vs swap 0.05 at one
fact with a longer schedule and paired-conflict batching, so the 1-fact capability is
genuine — it is these 400-step runs that are undertrained for it, not the capability
that is absent. The load-bearing claim here is the **ceiling**, not the floor.

This single curve explains every downstream result in this report. A HotpotQA
distractor context (1697 tokens, 10 paragraphs) and a LoCoMo conversation (tens of
thousands of tokens) both contain far more than four facts, so `correct − swap ≈ 0`
downstream is exactly what this capacity ceiling predicts. It also explains why five
rescue objectives could not move the number: the failure is not the loss function
choosing the wrong thing to encode, it is **the write producing a state with
essentially no addressable capacity beyond a single binding**.

Caveats, stated after running the checks rather than before: this is a **synthetic
diagnostic**, not a benchmark. The absolute values are training-limited (the July runs
reached 0.99 at one fact with a longer schedule). The seed replication above shows the
1-fact point is variable at this budget; the 4-fact zero is not. The *collapse across
facts* is the finding, and it reproduces the July capacity result (1 fact 0.99 →
2 facts 0.48 → 4 facts 0.23) in direction, at a much smaller budget.

### Is the ceiling the slot count or the write mechanism? — the attribution

Same synthetic task, same 400-step budget, slots swept at the two fact counts where
capacity is marginal or gone:

| facts | slots | base | window | swap | correct | **correct − swap** |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 0.0078 | 0.0469 | 0.1719 | 0.1797 | **+0.0078** |
| 2 | 128 | 0.0078 | 0.0781 | 0.0781 | 0.0781 | **0.0000** |
| 2 | 256 | 0.0078 | 0.0781 | 0.0781 | 0.0781 | **0.0000** |
| 4 | 64 | 0.0000 | 0.0703 | 0.0469 | 0.0469 | **0.0000** |
| 4 | 128 | 0.0000 | 0.0781 | 0.1016 | 0.1016 | **0.0000** |
| 4 | 256 | 0.0000 | 0.0469 | 0.0391 | 0.0391 | **0.0000** |

**More slots do not restore capacity — at 2 facts they destroy the little that was
there** (+0.0078 at 64 slots → exactly 0.0000 at 128 and 256). The ceiling is therefore
**not the number of slots**; it is the single pooled-attention WRITE, which produces a
summary rather than an addressable structure. Adding slots to a pooled write just
spreads the same summary more thinly.

This measures on a task where a task adapter cannot cheat what the July runs
(`out_prefix_value/`) observed on natural QA — "capacity 64/128/256 does not help, the
attention-pool WRITE is the ceiling" — so the attribution is now clean rather than
confounded with task adaptation.

Caveat tested, not assumed: all six runs shared a 400-step budget and the 128/256-slot
models have more parameters, so the sweep was re-run at **900 steps**. Neither slot
count improved (both stay at exactly 0.0000), and P=64's marginal +0.0078 *disappeared*.
The budget was not the limiting factor.

**The next experiment is therefore no longer "test whether the write can encode
anything" — that is now answered.** It is: **replace the pooled write with an addressable one** — the slot
sweep above shows adding slots to the current pooled write is not merely insufficient
but counter-productive, so the change has to be to the write mechanism itself (explicit
key-value binding per fact, or a write objective supervised per fact), and then re-run
this same curve. Until `correct − swap` stays positive at 8+ facts here, no downstream memory
claim can succeed, and no amount of fusion-position or objective tuning will change it.

## Official evaluator manifest (§5)

| benchmark | repo | pinned SHA | status |
|---|---|---|---|
| HotpotQA | github.com/hotpotqa/hotpot | `3635853403a8735609ee997664e1528f4480762a` | cloned; gold rebuilt from HF mirror because the official curtis.ml.cmu.edu file is offline (404) — scorer itself is untouched official code |
| LoCoMo | github.com/snap-research/locomo | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | cloned; official EM/F1/ROUGE are locally runnable, LLM-judge is not |
| RULER | github.com/NVIDIA/RULER (`rulerv1-ns`) | `e8bbff677ca2c239640dc90f93310dcf32408c93` | cloned; this session's numbers come from the `simonjegou/ruler` data mirror with the official string-match scorer — marked **NON-OFFICIAL-GENERATION** until regenerated with the official generator |
