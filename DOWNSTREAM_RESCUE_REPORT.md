# Downstream audit + rescue — 12h session, 2026-08-12

node16 (8x H100 80GB) · env `/work/mingze/miniconda3/envs/deltamem` (torch 2.6.0+cu124)
backbone `/work/mingze/models/Qwen3-4B-Instruct-2507` · branch `agent/downstream-audit-2026-08-12`
Repo HEAD at start: `dc9269b` · outputs `out_downstream_audit_20260812/`

**STATUS: IN PROGRESS** — this file is updated as evidence lands. Numbers below are
copied from result JSONs, never from memory.

## Direct answers (§13) — updated live

| # | question | answer as of latest evidence |
|---|---|---|
| 1 | Bug in the pre_o implementation? | **No.** On the live model the o_proj input equals `Z + g·C` with max abs residual **0.0**; widths are Z=C=4096 into a 4096→2560 bias-free `o_proj` (GQA 32 q / 8 kv heads). `a0_preo_graph.json` |
| 2 | Is pre_o vs post_o_projected really just bf16? | **Yes as to correctness, no as to size.** In fp32 the two agree to 6.8e-3 max / 9.9e-6 mean with **top-1 disagreement 0.0000** — mathematically equivalent, differences are accumulation order. In bf16 they flip the top-1 token on **1.1%** of positions (max abs 10.2), which greedy decoding amplifies into visibly different generations (25% of 4 sampled generations differed). Calling it "bf16 noise" understates it: it is bf16-scale, but large enough to move small-n greedy F1. Both arms are internally deterministic (repeat runs bit-exact). `a1_numerics_numerics.json` |
| 3 | Is the backbone fully frozen? | **Yes, bit-identical.** SHA256 over all 398 backbone tensors is unchanged across 4 real optimizer steps (`c894f7a1…` → `c894f7a1…`); 0 backbone params ever receive a gradient; the optimizer contains exactly the 48 sidecar tensors. `--lr 2e-5` applies to an empty parameter group for `swa_steer`. `a0b_frozen_frozen.json` |
| 4 | Does the old Qasper +0.055 F1 hold on independent data? | **Partly, and larger than that on HotpotQA** — see §2. But it is NOT a memory effect (§1 F1/F2). |
| 5 | LoCoMo official | RUNNING |
| 6 | HotpotQA official EM/F1 + Joint | **Official scorer run**: base EM .4340/F1 .5659 vs ours EM .4860-.4870/F1 .6156-.6256 on internal dev-1000. Joint/Support = **0.0** (no supporting-fact head, `sp` submitted empty). Untouched-final RUNNING |
| 7 | RULER 8K/16K/32K macro | 8K done: base **0.9288** vs ours **0.9254** (13/13 tasks, at ceiling). 16K RUNNING |
| 8–11 | strongest config, significance, memory vs task adaptation | see §1 F1 and §2 — the gain is real but is **task adaptation, not memory** |

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

## 3. RULER (mirror `simonjegou/ruler`, 13/13 official tasks, 40 samples/task)

| length | arm | macro | note |
|---|---|---:|---|
| 8192 | base | 0.9288 | at ceiling: 8/13 tasks exactly 1.000 for both arms |
| 8192 | ours pre_o s0 | 0.9254 | −0.0034; qa_1 +0.025, qa_2 −0.050, cwe −0.012 |
| 16384 | base / ours | RUNNING | |

Historical runs agree (4k: base 0.9360 / ours 0.9355; 16k: base 0.9282 / ours 0.9274),
so RULER at these lengths has no headroom for this method to show anything.

## 4. LoCoMo — RUNNING

Partial (200/1540 QA, official protocol serialization, our scorer): base 0.2856,
ours 0.3231. Official `task_eval/evaluation.py` scoring and conversation-clustered CI
pending; the LLM-judge metric is **not available** (no API key) and is **not** replaced
by a substitute.

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

## Official evaluator manifest (§5)

| benchmark | repo | pinned SHA | status |
|---|---|---|---|
| HotpotQA | github.com/hotpotqa/hotpot | `3635853403a8735609ee997664e1528f4480762a` | cloned; gold rebuilt from HF mirror because the official curtis.ml.cmu.edu file is offline (404) — scorer itself is untouched official code |
| LoCoMo | github.com/snap-research/locomo | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | cloned; official EM/F1/ROUGE are locally runnable, LLM-judge is not |
| RULER | github.com/NVIDIA/RULER (`rulerv1-ns`) | `e8bbff677ca2c239640dc90f93310dcf32408c93` | cloned; this session's numbers come from the `simonjegou/ruler` data mirror with the official string-match scorer — marked **NON-OFFICIAL-GENERATION** until regenerated with the official generator |
