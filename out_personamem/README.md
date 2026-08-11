# PersonaMem-v2 experiment status — 2026-07-30 19:30

Consolidated status: official protocol, baseline reproduction, architecture
gates, formal dev arms, and how the current PersonaMem reader differs from the
Qasper memory branch.

---

## 1. Official protocol (locked)

Taken from the official Qwen-VeRL evaluation path, not the top-level
`final_answer` script:

- Full 32k persona history in the prompt, official Qwen chat template, thinking
  enabled.
- Options lowercase `(a)`–`(d)`, shuffled with the official seed (42 + original
  row index).
- Greedy, `n=1`, `max_new_tokens=2048` (`n=16` in the paper is a training
  rollout count, validation is `n=1`).
- Score the **last** `\boxed{a-d}` in the response.
- Our local scorer is strict: no `\boxed{}` means wrong. The official scorer
  falls back to remote `text-embedding-3-large`, which we do not have a key for,
  so unparsed responses are reported separately instead of guessed.

Prompt fidelity was verified against the authors' preprocessed parquet: the
first item's 191 messages, option order and gold letter match our harness
verbatim.

Backbones:

- `Qwen3-4B-Instruct-2507` (Base)
- `Qwen3-4B-PersonaMem-SFT` (public official SFT, 8.3 GB, downloaded)

---

## 2. Baseline reproduction (official 5000 MCQs, running)

| Backend | Done | Strict acc | Parse rate | Acc on parsed | Paper |
|---|---:|---:|---:|---:|---:|
| Base (Instruct-2507) | 3186/5000 | 11.0% | 35.2% | 31.3% | 30.5 |
| Public SFT | 2777/5000 | 33.3% | 100.0% | 33.3% | 35.0 |

Reading: the SFT backbone reproduces the paper number locally within ~1.7pt.
Base's paper number is only recoverable on the parsed subset (31.3% vs 30.5%),
which confirms our harness matches the official protocol and that the strict
11.0% is a formatting/fallback artifact, not a different model. Both numbers
will be reported.

Forced-choice calibration on the same 300 items (frozen, one-token, full
history): Base 33.67%, SFT 43.00%. This is a sanity channel only, not a
paper number.

Paper-reported, **not** locally reproducible (no public weights, no compatible
VeRL commit, ~160k judge API calls needed):

| Official method | MCQ |
|---|---:|
| Qwen3-4B GRPO | 53.8 |
| Qwen3-4B GRPO (MCQ-only) | 55.5 |
| Qwen3-4B + 2k agentic text memory | 55.2 |

These are cited as `paper-reported†`. **They set the real bar: beating the
public SFT (35.0) is not a paper result; beating 55.5 is.**

---

## 3. Architecture gate (4 personas, train == eval)

Purpose: check that the objective and wiring can produce persona-conditional
behaviour at all. 150 updates, 1200 label exposures, official reader prompt,
four-choice CE + identity contrast (λ=10, margin=1), frozen SFT backbone.

| Arm | correct | swap | prefix_off | Δ prefix | correct − swap |
|---|---:|---:|---:|---:|---:|
| pooled steer D84 (P=0) | 75.2 | 26.0 | — | — | +49.2 |
| hybrid, learned scalar gate | 57.1 | 29.5 | 58.1 | −1.0 | +27.6 |
| hybrid, fixed gate 1.0 | 51.4 | 28.6 | 49.5 | +1.9 | +22.8 |
| hybrid, fixed gate 1.0 + pool-drop 0.5 | 56.2 | 41.3 | 41.0 | **+15.2** | +14.9 |

Two things follow, and only these two:

1. **Branch dropout is what makes the prefix carry information.** Without it the
   optimizer routes everything through the pooled summary and the prefix is
   worth 0 to +1.9pt; with a 0.5 pooled-branch dropout the same reader gains
   +15.2pt from the prefix inside one checkpoint. In that run `prefix_off`
   equals `window` (41.0 vs 41.0), i.e. the pooled branch became the redundant
   one.
2. **The gate cannot rank pooled vs prefix.** 4 personas × 8 queries with
   train == eval is a memorisation test; one vector per layer separates four
   identities trivially. The pooled arm's 75.2 is not evidence that a single
   vector is enough at scale.

Ranking is therefore deferred entirely to the formal dev arms below.

---

## 4. Formal dev arms (the real comparison, in flight)

Identical everywhere: frozen `Qwen3-4B-PersonaMem-SFT`, 719 train personas /
16,681 queries, **80 unseen dev personas / 2,031 queries**, history ≤ 37,000
tokens (tail), official reader prompt, four-choice CE + identity contrast
(λ=10, margin=1, donor seed 7331), 3 swap derangements, 300 optimizer updates ×
64 labels = **19,200 label exposures**, K=4 read micro-batch (~5,092
microsteps), lr 1e-4 / prefix-lr 1e-3, seed 1.

K=16 and K=8 both OOM at 79.2 GB on the backward of a 37k history; K=4 runs at
~56–61 GB. K only changes micro-batching, not the label/update budget.

| GPU | Arm | Memory architecture | Trainable | Status |
|---|---|---|---:|---|
| 0 | `poolsteer` | one history-conditioned pooled vector per layer, P=0, D=84 | 16,545,792 | running, ~75/300 |
| 1 | `pool` | 64 written slots, query-independent pooled read | 16,515,072 | running, ~50/300 |
| 2 | `prefixonly` | 64 written slots, prefix-only softmax | 16,515,072 | running, ~75/300 |
| 5 | `standard` | **Qasper-native**: one softmax over [64 slots ; 256-token local window] + max-prefix bonus | 16,515,072 | queued behind SFT shard2 |
| 7 | `hybridpart_pooldrop05` | pooled vector + partitioned 64-slot query attention, fixed gate 1.0, pooled-branch dropout 0.5 | 16,545,792 | queued behind Base shard2 |

Launcher: `scripts/run_personamem_formal_arm.sh GPU ARM [SEED]` (resumes from
`*.resume.pt` automatically).

Decision rule after these five:

- winner must beat `poolsteer` on unseen dev personas,
- with `correct − swap ≥ 5pt` (target 10pt) and a persona-clustered CI clear of 0,
- and, for hybrid, `full − prefix_off > 0` inside the same checkpoint.

Only the winner is retrained on all official training personas and run on the
official 5000.

---

## 5. Qasper architecture retrospective (why `standard` is now an arm)

The Qasper checkpoint that worked was **not** "pooled steer + separate prefix
attention". It was a parallel SWA memory branch:

- WRITE: 64 learned probes attend over the frozen hidden states of the document,
  `M_l(D) = W_write · Attn(W_q P, W_k H(D), W_v H(D))`, context-only, dynamic
  (the static prefix is not added back).
- READ: **one** softmax over `[M_l(D) ; last 256 hidden states]` → `R_t`, plus an
  explicit max-prefix bonus `M_t = max_p α_{t,p} v_p`; reads = `R_t + M_t`.
- INJECT: `q,k,v += 0.1·Δ_{q,k,v}(reads)` before the frozen attention, `o +=
  0.1·Δ_o(reads)` after `o_proj`, on 12 layers (0,3,…,33). Backbone weights,
  embeddings, norms and MLPs frozen; 16,515,072 trainable params (~0.4%).
- Correct ablation there is `window_only` = mask the prefix columns but keep the
  SWA read. `method_nomem` zeroes the whole branch and degenerates to the frozen
  base, so it cannot be used to argue for the prefix.
- Training: two passes (`noctx` / `ctxmask`) — WRITE on the full document, READ
  with the document removed or masked, CE on answer tokens only, gradient flows
  back through the written prefix into the probes and `mem_q/k/v/write_proj`.

In PersonaMem this exact reader is `--read-mode standard` and had never been run
at formal scale — every PersonaMem arm so far used `pool`, `prefix_only`,
`broadcast` or the additive hybrid. That gap is now closed by the GPU5 arm.

Related, on GPU3: `out_swa_sharedv/p0_mainv_h1_fixed_g01_s1` is the Qasper
window-only reference (P=0, no prefix, shared main-V, ΔO only, 11.8M params),
followed by HotpotQA-500 and LoCoMo-10 screening from the same checkpoint.

---

## 6. Baseline coverage still owed

Local, same generator + same strict scorer:

- Base shard3 (queued on GPU3), SFT shard3 (queued on GPU4).
- Query-only (no history) SFT: shards 0–3 queued on GPUs 6, 7, 5, 3.
- BM25-RAG: wired into the shared evaluator.
- Dense-RAG: official uses `text-embedding-3-large` (no key) → run an offline
  `Qwen3-Embedding-8B` variant, labelled as a substitute, not a reproduction.
- Mem0: dependencies present locally; SFT as fact extractor, MiniLM+Chroma
  retrieval, 1–2.5 h across 4 GPUs.

Retrieval systems only ingest the test history; they never see options or gold
labels. Supervised arms all share the 19,200-label budget. The paper table will
mark each row's supervision explicitly.

Statistics: per-item predictions are kept for every run; comparisons use exact
McNemar plus 10,000-sample persona-clustered bootstrap CIs; the three swap
derangements are averaged within persona before any interval is computed.

---

## 7. Earlier pilot (8 personas, vanilla CE) — superseded

| Method | Trainable | Base | Window | Swap | Correct | Correct − swap |
|---|---:|---:|---:|---:|---:|---:|
| P0/D64 query-only steer | 12.58M | 30.3 | 66.2 | 66.2 | 66.2 | 0.0 |
| P0/D84 query-only steer, seed 1 | 16.52M | 30.3 | 65.6 | 65.6 | 65.6 | 0.0 |
| P0/D84 query-only steer, seed 2 | 16.52M | 30.3 | 76.9 | 76.9 | 76.9 | 0.0 |
| P1/D64 pool | 14.58M | 30.3 | 53.8 | 55.4 | 55.9 | +0.5 |
| P64/D64 pool | 16.52M | 30.3 | 46.7 | 53.8 | 52.3 | −1.5 |
| P64/D64 prefix-only | 16.52M | 30.3 | 30.3 | 56.9 | 56.4 | −0.5 |
| P0/D84 history-conditioned attention pool | 16.55M | 30.3 | 30.3 | 59.8 | 60.5 | +0.7 |

195 dev MCQs, one-token forced choice, 8 disjoint dev personas. Vanilla CE never
requires the same query to behave differently under correct and wrong memory, so
`correct − swap` stayed within ±1.5pt. This is why the four-choice identity
contrast objective replaced it; these rows are kept only as the negative result
that motivated it.
