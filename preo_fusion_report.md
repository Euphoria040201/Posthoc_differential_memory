# Pre-O prefix differential fusion — implementation, verification, experiments

Repo: `Euphoria040201/Posthoc_differential_memory`. All experiments on
lux-2-node-16 (8x H100 80GB), env `/work/mingze/miniconda3/envs/deltamem`
(torch 2.6.0+cu124), backbone `/work/mingze/models/Qwen3-4B-Instruct-2507`,
Qasper protocol identical to `dex_control_report.md` §13 (800/75 papers,
187-example val F1, greedy, 156 updates).

## 1. What the original implementation actually computed (task 1)

Read from `deltamem/core/prefix_steer.py` (commit `49db119`), not from the README:

```
reads   = _memory_read(hidden)                      # [B, L, read_dim]   (SWA/prefix sidecar)
q,k,v   = base.{q,k,v}_proj(hidden)                 # q/k/v corrections: q += g*delta_q(reads) etc.
                                                    #   (delta_heads="o" in this line => untouched)
delta_o = self.delta_o(reads)                       # C, [B, L, 2560]  = o_proj.OUT_features
attn    = attention_interface(...)                  # [B, L, 32, 128]
Z       = attn.reshape(B, L, 4096).contiguous()     # concat-head activation, o_proj.IN_features
out     = base.o_proj(Z)                            # W_O Z   [B, L, 2560]
out     = _fuse_delta_o(out, delta_o)               # Y = fuse(W_O Z, C)   <-- POST-o_proj
return out                                          # residual add happens OUTSIDE, in the
                                                    # decoder layer (hidden = residual + out)
```

So the answer to "where was it added": **after `o_proj`, before the residual
add**, with `C` living in the output space (2560) — `Y = W_O Z ± λC`. It was
never fused after the residual connection, and never before `W_O`. On
Qwen3-4B the two bases genuinely differ: `o_proj` is 4096→2560 (32 query heads
x 128; GQA's 8 KV heads do not change Z's width), so the old `C` cannot even be
shape-cast into the pre-o position — a re-trained sidecar is required, and any
"reuse the post-o checkpoint under pre-o" comparison would be silently wrong.
Loading a post-o `delta_o` (2560-wide) into a pre-o module (4096-wide) raises a
size-mismatch error (tested).

## 2. The change (tasks 2–3)

One config field, `PrefixSteerConfig.o_fusion_position` (default `post_o`):

| position | formula | C width |
|---|---|---|
| `post_o` (historical, default) | `Y = fuse(W_O Z + b, C)` | `o_proj.out_features` (2560) |
| `pre_o` | `Y = W_O fuse(Z, C) + b` | `o_proj.in_features` (4096) |
| `post_o_projected` (control) | `Y = fuse(W_O Z + b, W_O C)`, `W_O C` without bias | `o_proj.in_features` (4096) |

- q/k/v corrections untouched; residual-add position untouched; `residual`
  steer mode remains post-o only (validated).
- All six fusion modes share the single `_fuse_delta_o`; the position only
  chooses which tensor pair it sees. No duplicated fusion code.
- `post_o_projected` handles `o_proj.bias` by projecting `C` through the
  **weight only** (`F.linear(C, W_O.weight)`), so the bias enters exactly once
  and `W_O(Z ± gC) + b == (W_O Z + b) ± g(W_O C)` holds as an identity for the
  linear fusions. (Qwen3's `o_proj` has no bias; the code is correct either way.)
- Old checkpoints/configs (which predate the field) deserialize to `post_o`
  everywhere: dataclass default, `dex_stage1_fusion.py` (`saved.get(...,
  "post_o")`), and the audited `_LEGACY_DEFAULTS` sync guard in
  `deltamem/eval/steer_checkpoint.py`.
- Logs and result JSONs record the position (`fusion=fixed@pre_o` in stdout;
  `o_fusion_position` + `o_fusion_position_ckpt` + `lambda_report` +
  `final_fusion_norms` in the stage-1 payload).
- No old result file is overwritten: the post-o table stays at
  `out_dex_fusion/stage1_*`; new runs write `s1preo_*` / `s1ctlproj_*`.

Files changed:

| file | change |
|---|---|
| `deltamem/core/prefix_steer.py` | `O_FUSION_POSITIONS`, config field + validation, position-dependent `delta_o` width, forward-path fusion placement, `_record_fusion_norms` + `set_collect_fusion_norms`/`collect_fusion_norms` |
| `deltamem/eval/steer_checkpoint.py` | audited legacy default `o_fusion_position: "post_o"` |
| `scripts/dex_train_qasper.py` | `--steer-o-fusion-position` |
| `scripts/dex_stage1_fusion.py` | `--o-fusion-position` (ckpt default; only `pre_o` ↔ `post_o_projected` may swap), λ report, norm diagnostics in payload |
| `scripts/run_preo_fusion_matrix.sh` | resumable 3-seed training + 24-job stage-1 matrix |
| `tests/test_o_fusion_position.py` | 24 new tests |

Key forward diff (`prefix_steer.py`):

```python
Z = attn_output.reshape(*input_shape, -1).contiguous()
if fuse_active and pos == "pre_o":
    Z = self._fuse_delta_o(Z, delta_o)          # Y = W_O fuse(Z, C) + b
out = base.o_proj(Z)
if fuse_active and pos == "post_o":
    out = self._fuse_delta_o(out, delta_o)      # Y = fuse(W_O Z + b, C)
elif fuse_active and pos == "post_o_projected":
    out = self._fuse_delta_o(out, F.linear(delta_o, base.o_proj.weight))
```

The fusion sits in the wrapper AFTER the `attention_interface` dispatch, so
eager/SDPA/FlashAttention share the same fusion placement by construction
(covered by `test_attention_backends_share_the_fusion_path`).

## 3. Unit tests (task 4)

`tests/test_o_fusion_position.py`, run with the repo suite:

```
/work/mingze/miniconda3/envs/deltamem/bin/python -m pytest tests/ -q
225 passed   (201 pre-existing + 24 new; also 225 passed on node16)
```

What the new tests pin down (on a NON-square GQA tiny backbone, o_proj 64→32,
4 query heads / 2 KV heads, so a wrong-basis fusion cannot hide):

- **Position**: against tensors captured from the real forward, `pre_o` output
  == `W_O(Z + gC)` and `post_o` output == `W_O Z + gC`, with non-identity W_O;
  the two positions provably disagree on the same weights.
- **Equivalence boundaries**: identity `W_O` collapses the positions;
  `post_o_projected` == `pre_o` (fp32, 1e-5) for fixed_add / fixed_sub /
  learned_diff; gain=0 and λ=0 reproduce the frozen backbone **bit-exactly**
  at every position.
- **Gradients**: under `pre_o` + `learned_diff`, both `delta_o.weight` and
  `fusion_lambda` receive finite, non-zero grads through the frozen `W_O`.
- **Shapes/dtypes**: B=2, L=9, fp32 + bf16 (+ fp16 on CUDA), all positions.
- **GQA widths**: `delta_o` out-width follows `o_proj.in_features` (64) for
  pre_o vs `out_features` (32) for post_o — authoritative from `o_proj`, never
  derived from `n_kv`.
- **Checkpoint safety**: post-o state loading into a pre-o module raises;
  config dicts without the field deserialize to `post_o`.

## 4. Smoke run (node16, GPU0)

4-step pre-o sidecar training → checkpoint → stage-1 reload:

```
[smoke_preo_train_s0]  FINAL val_loss=1.5384 F1=0.2136 (n=8); saved 48 steer tensors
[smoke_s1_fixadd_preo] FINAL arm=fixed_add@pre_o F1=0.2136 val_loss=1.5384
```

identical numbers across save/reload — exact round trip. The
`post_o_projected` control on the same checkpoint gave val_loss 1.5540 / F1
0.2036 (n=8): the norm diagnostics agree to ~3 decimals (math identical) while
outputs differ slightly — that is the bf16 implementation-path effect the
control arm exists to measure (single fused matmul vs two matmuls + add,
divergence amplified by greedy decoding). Fusion scale at this init:
‖C‖/‖Z‖ ≈ 3.2%, ‖W_O C‖/‖W_O Z‖ ≈ 5.9%, cos(Z, C) ≈ 0.006.

## 5. Qasper comparison (task 5) — RUNNING

Protocol: 3 seeds (0/1/2), sidecar re-trained under `pre_o` with the exact
post-o recipe (156 updates, steer-lr 5e-4 constant, gain 0.1, `delta_heads=o`,
`main_v`, layers 0,3,…,33), then stage-1 arms on each frozen checkpoint.
Commands: `scripts/run_preo_fusion_matrix.sh`; logs
`out_dex_fusion/{preo_swa_steer,s1preo,s1ctlproj}_*.log` on node16.

Post-o reference rows are the EXISTING `stage1_*` results (not rerun, not
overwritten; environment cross-checked via the base row, which reproduced
0.2444 **exactly**, all three seeds).

### Results (Qasper val F1, 187 examples, greedy, 3 seeds)

| arm | post_o (existing) | pre_o (new) | post_o_projected (control) |
|---|---|---|---|
| base | 0.2444 | 0.2444 ± 0.0000 | — |
| fixed_add (g=0.1) | 0.2935 ± 0.0046 | **0.2993 ± 0.0090** | 0.2964 ± 0.0082 |
| fixed_sub (g=0.1) | 0.2297 ± 0.0022 | **0.2155 ± 0.0027** | 0.2175 ± 0.0039 |
| learned_diff (λ init +0.1) | 0.2888 ± 0.0076 | 0.2889 ± 0.0056 | 0.2896 ± 0.0040 |
| variance_diff (closed-form λ*) | 0.2415 ± 0.0004 | 0.2435 ± 0.0010 | — |

Per-seed pre_o F1: add .2927/.3095/.2957, sub .2161/.2178/.2125, learned
.2887/.2946/.2835, var .2445/.2426/.2435. Pre-o sidecar training finals
(.2927/.3095/.2957) equal the stage-1 fixed_add rows exactly — checkpoint
round-trips are exact. No NaN, no gradient explosion, no correction
domination in any run (grad-norm logs in `preo_swa_steer_s*.log`).

Paired per-example tests (pre_o − post_o, same seed, n=187):

| arm | s0 | s1 | s2 |
|---|---|---|---|
| fixed_add | +0.0005 (t=+0.07) | +0.0110 (t=+1.27) | +0.0061 (t=+0.80) |
| fixed_sub | −0.0114 (t=−2.41) | −0.0140 (t=−2.73) | −0.0173 (t=−2.82) |
| learned_diff | +0.0032 (t=+0.35) | −0.0028 (t=−0.31) | +0.0002 (t=+0.02) |

λ (learned_diff, init +0.1, 156 steps @ lr 1e-2): **crossed zero in every run
at both positions** and converged negative — pre_o finals −0.141/−0.132/−0.112
(11–12 of 12 layers negative), post_o_projected −0.134/−0.128/−0.118, matching
the historical post_o −0.144. `out − λC` with λ<0 IS addition: given a
subtractive parameterisation, the optimizer recovers the additive solution.

variance_diff's closed-form coefficient at pre_o: λ* ≈ 0.0007 (raw −0.0096,
clamped at 0; cov ≈ 5e-5, var ≈ 0.026) — the control's residual carries
essentially no component of Y at this basis either, so the arm degenerates to
≈ base (0.2435 vs 0.2444), as it did post-o (0.2415).

Norm/cosine diagnostics of the TRAINED pre_o sidecars (`final_fusion_norms`,
val[0], mean over the 12 steered layers, seed 0 shown; other seeds equal to
±0.02): ‖Z‖ = 18.0, ‖W_O Z‖ = 20.0, ‖C‖ = 10.8, ‖W_O C‖ = 22.1;
‖C‖/‖Z‖ = 0.55 but ‖W_O C‖/‖W_O Z‖ = **1.12**; cos(Z, C) = −0.009,
cos(W_O Z, W_O C) = −0.055. (Effective added branch is g=0.1 of that.)

Control-arm agreement (post_o_projected − pre_o, same checkpoints, pooled
n=561): fixed_add −0.0030 (t=−1.57), fixed_sub +0.0020 (t=+1.59),
learned_diff +0.0006 (t=+0.25) — the mathematically-identical paths agree
within bf16 noise, so the pre-vs-post differences above are attributable to
the mathematical position, not the implementation path.

## 6. Diagnosis (task 6)

**H1 — basis mismatch: not supported.** If post-o fusion had suffered from C
bypassing W_O's local basis, moving the fusion pre-W_O should have helped.
For the additive arm the positional gain is +0.006 mean, ≤ 1 seed-std, t ≤ 1.3
per seed; learned_diff is dead even. Nothing that was "wrong" post-o got fixed
by the position.

**H2 — W_O filtering: refuted in direction.** W_O does not attenuate the
correction — it preferentially AMPLIFIES it: the correction/base norm ratio
doubles through W_O (0.55 → 1.12) while the trained C stays near-orthogonal to
Z (cos ≈ −0.01). The projection neither filters noise directions out of C nor
stabilises the ratio; the trained sidecar simply occupies directions W_O
passes with above-average gain.

**H3 — the sign, not the position, is the issue: confirmed.** The subtraction
is the only arm the position changes significantly, and it moves the WRONG way
(pre_o_sub 0.2155 < post_o_sub 0.2297 < base 0.2444, t ≈ −2.6 on all seeds).
Learned λ crosses zero and lands negative at every position and every seed.
Both facts say the same thing: a sidecar trained under ADDITIVE fusion learns
C that carries answer-relevant signal, not a nuisance component — subtracting
it destroys information wherever you subtract it, and slightly faster pre-W_O
because W_O amplifies C relative to Z (H2's finding). The premise that C could
act as a control variable is what fails, upstream of any placement question.

## 7. Final answers (deliverable 9)

1. **Where was the original fusion?** After `o_proj`, before the residual add:
   `Y = W_O Z ± λC` with C in the 2560-d output space (§1).
2. **Is the new implementation really pre-o_proj?** Yes: C is built 4096-wide
   in the concat-head space and fused into Z before `base.o_proj`, verified by
   unit tests against tensors captured from the real forward (§3), exact
   checkpoint round-trips (§4), and the post_o_projected identity (§5).
3. **Does pre_o_sub improve?** No — it is significantly worse than post_o_sub
   (Δ ≈ −0.014, t ≈ −2.6, 3/3 seeds), and both are below base.
4. **Position, W_O projection, or sign?** Sign/content. Position: no
   significant effect on add/learned (H1 rejected). W_O projection: amplifies
   rather than filters C (H2 refuted in direction). Everything significant is
   carried by the sign interacting with what C learned to be (H3).
5. **Is this line worth continuing?** Not as a fusion-position question — that
   is now answered negatively with controls. The open question the data
   points to is upstream: C only becomes a meaningful *control variable* if it
   is trained AS one (frozen-sign subtractive training from scratch, or an
   explicit nuisance-prediction objective), rather than re-fusing a sidecar
   that additive training has already shaped into a signal carrier. The
   variance_diff λ* ≈ 0 result says today's C explains none of Y's residual
   variance in either basis — position was never the missing piece.

## Run commands

```bash
# everything (resumable; skips completed jobs):
bash scripts/run_preo_fusion_matrix.sh                     # seeds "0 1 2", 8 GPUs
# single arms:
python scripts/dex_train_qasper.py --variant swa_steer --steer-o-fusion-position pre_o ...
python scripts/dex_stage1_fusion.py --steer-ckpt out_dex_fusion/preo_swa_steer_s0_steer.pt \
    --arm fixed_sub --seed 0 --tag s1preo_fixed_sub_s0
python scripts/dex_stage1_fusion.py --steer-ckpt ... --arm fixed_add \
    --o-fusion-position post_o_projected --tag s1ctlproj_fixed_add_s0
# tests:
python -m pytest tests/ -q          # 225 passed (node12 and node16)
```

Logs/artifacts: result JSONs in `out_dex_fusion/` (committed);
`*_steer.pt` checkpoints and `.log` files on node16 under
`/work/mingze/Posthoc_differential_memory/out_dex_fusion/`; the historical
post-o table under `/work/mingze/delta-mem/out_dex_fusion/stage1_*` (both
nodes, untouched).

## Theoretical boundary

Even at pre-o, our correction is `C = delta_o(prefix_reads)` — a separate
context-aggregation path — while original DEX builds its correction from the
attention output itself. Everything here is therefore **DEX-inspired pre-O
prefix differential fusion**; no claim of equivalence with DEX (arXiv:2505.16333)
is made or licensed by these results.
