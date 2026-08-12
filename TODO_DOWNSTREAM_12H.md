# 12-hour downstream audit + rescue — live TODO

START 2026-08-12 05:24:19 UTC · DEADLINE 2026-08-12 17:24:19 UTC · node16 (8x H100)
Branch `agent/downstream-audit-2026-08-12` · outputs `out_downstream_audit_20260812/`

Status legend: PENDING / RUNNING / PASS / FAIL / INVALID / BLOCKED

## Headline audit findings so far (evidence, not claims)

| # | finding | evidence |
|---|---|---|
| F1 | **The whole pre-o/post-o line has NO written memory.** `num_prefix_tokens=0`, `prefix_write=False` → the sidecar is a local sliding-window attention adapter, not a memory. `state_only`/`swap`/`zero-state` arms are undefined for these checkpoints. | `a0_preo_graph.json`, `a2_posto_isolation.json` |
| F2 | **`ours_window_only` is NOT an ablation for these checkpoints** — with P=0 there are no prefix columns to mask, so window-only output is bit-identical to full. Any past claim resting on it is void. | `a2_posto_isolation.json: window_only_equals_full_when_P0=true` |
| F3 | pre_o formula verified exactly on the live model: o_proj input == Z + g·C, residual **0.0**; widths Z=C=4096 (o_proj 4096→2560, no bias), GQA 32 q-heads / 8 kv-heads. | `a0_preo_graph.json: formula_check` |
| F4 | **Base parity is bit-exact** (n=100): wrapper with steer disabled and wrapper at gain=0 both reproduce pristine HF logits exactly. | `a2_posto_parity.json` |
| F5 | **pre_o vs post_o_projected: no implementation bug.** fp32 top-1 disagreement = **0.0000** (max_abs 6.8e-3 = accumulation order). bf16 disagreement is real but numerical: 1.1% of tokens flip top-1, max_abs 10.2 — enough to move greedy F1 on small n. Repeats are bit-exact (deterministic). | `a1_numerics_numerics.json` |

## P0 — must finish

- [x] git / env / GPU / model / dataset manifest — PASS (`env_manifest.json`)
- [x] audit WRITE/READ inputs, injection points, widths — PASS (F1, F3)
- [x] base parity — PASS bit-exact (F4)
- [x] pre_o vs post_o_projected fp32/bf16/cache/batch/repeat — PASS, no bug (F5)
- [x] state isolation for P=0 checkpoints — PASS with the caveat that there is no state (F1, F2)
- [x] frozen-backbone hash across a real optimizer step — **PASS** bit-identical, 0 backbone grads
- [x] historical result inventory + contamination manifest — **PASS** (4678 untouched Hotpot dev ids)
- [x] Hotpot dev-200 + dev-1000 screening — **PASS**: pre_o +.050/+.060 (sig), post_o +.024 (sig), all vs base_fullctx
- [ ] LoCoMo official-protocol generation: base / ours — RUNNING (GPU5)
- [x] RULER 8K — **PASS/no-signal** (base .9288 vs ours .9254, ceiling). 16K RUNNING (base .948 vs ours .963 partial), 32K RUNNING
- [ ] unified per-example prediction schema — PENDING
- [ ] Qasper regression re-run (regression only, NOT evidence) — PENDING

## P1 — first rescue round (additive memory only; subtraction is settled)

- [~] P=64 written memory, `noctx` training, post_o + fixed_add — RUNNING (GPU0, step 100/200)
- [~] P=64 written memory, `noctx` training, pre_o + fixed_add — RUNNING (GPU1, step 100/200)
- [~] P=64 `ctxmask` training — RUNNING (GPU2, step 100/200)
- [ ] delta heads `o` vs `qo` — PENDING
- [ ] gain sweep 0.03 / 0.05 / 0.10 / 0.20 — PENDING
- [ ] swap-contrastive objective (β, margin small screen) — PENDING
- [ ] matched no-memory adapter control — PENDING
- [ ] zero-shot transfer of current Qasper checkpoints — RUNNING (part of P0 screening)

## P2 — per-benchmark

HotpotQA: [x] official `hotpot_evaluate_v1.py` wiring — **matches in-repo metric to 4 decimals** · [ ] official distractor dev · [ ] hash split train/dev · [ ] 200 screen · [ ] 1000 confirm · [ ] 3 seeds · [ ] untouched final
LoCoMo: [ ] recover old 10x20 IDs · [ ] write-once/query-many verification · [ ] correct/swap/zero arms (needs P>0) · [ ] conversation-clustered CI
RULER: [ ] 13-task completeness check · [ ] 8K/16K disjoint seeds · [ ] 32K · [ ] zero/swap controls

## P3 — only with spare compute

- [ ] slots 64 vs 128 (only if capacity is shown to bind)
- [ ] layer subsets
- [ ] per-layer positive gain (softplus)
- [ ] one subtractive-from-scratch diagnostic arm (≤1 GPU)
- [ ] latency / state size / amortization
