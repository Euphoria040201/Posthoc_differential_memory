# DEX control study — results

## Per-seed results (Qasper val F1, 187 examples, greedy)

| Variant | Trainable Params | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Std |
|---|---|---|---|---|---|---|---|
| base | 0 | 0.2444 | — | — | — | — | 0.2444 | 0.0000 |
| dex_minus | 566,820,900 | 0.2921 | 0.2933 | 0.2945 | 0.2945 | 0.2942 | 0.2937 | 0.0010 |
| dex_plus | 566,820,900 | 0.2948 | 0.2986 | 0.2951 | 0.2860 | 0.2917 | 0.2932 | 0.0047 |
| residual_adapter | 566,820,864 | 0.2897 | 0.2814 | 0.2792 | — | — | 0.2834 | 0.0055 |
| attn_only | 566,231,040 | 0.2980 | 0.2997 | 0.2872 | — | — | 0.2950 | 0.0068 |
| adapter_only | 589,860 | 0.2443 | 0.2460 | 0.2433 | — | — | 0.2445 | 0.0014 |

## Per-seed results (EM and final validation CE)

| Variant | EM s0 | EM s1 | EM s2 | EM s3 | EM s4 | EM mean | val CE s0 | val CE s1 | val CE s2 | val CE s3 | val CE s4 | CE mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.0000 | — | — | — | — | 0.0000 | 2.1065 | — | — | — | — | 2.1065 |
| dex_minus | 0.0000 | 0.0053 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 0.7157 | 0.7558 | 0.6890 | 0.6340 | 0.7126 | 0.7014 |
| dex_plus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6934 | 0.7540 | 0.6465 | 0.6082 | 0.6927 | 0.6790 |
| residual_adapter | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | 0.7551 | 0.7082 | 0.7506 | — | — | 0.7379 |
| attn_only | 0.0000 | 0.0053 | 0.0000 | — | — | 0.0018 | 0.7370 | 0.7662 | 0.6605 | — | — | 0.7212 |
| adapter_only | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | 2.0989 | 2.1004 | 2.1063 | — | — | 2.1018 |

## Comparisons

| Comparison | Mean Difference (F1) | 95% CI | p-value | Effect Size | Level |
|---|---:|---|---:|---:|---|
| A: dex_minus − dex_plus | +0.0005 | [-0.0033, +0.0049] | 0.8499 | dz=+0.090 | seed-level (n=5) |
| A: dex_minus − dex_plus | +0.0005 | [-0.0059, +0.0070] | 0.8828 | dz=+0.011 | example-level (n=187) |
| B: dex_minus − residual_adapter | +0.0099 | [+0.0024, +0.0153] | 0.1250 | dz=+1.476 | seed-level (n=3) |
| B: dex_minus − residual_adapter | +0.0103 | [-0.0003, +0.0208] | 0.0597 | dz=+0.139 | example-level (n=187) |
| C: dex_minus − attn_only | -0.0017 | [-0.0064, +0.0073] | 0.7459 | dz=-0.215 | seed-level (n=3) |
| C: dex_minus − attn_only | -0.0012 | [-0.0097, +0.0074] | 0.7876 | dz=-0.020 | example-level (n=187) |
| D: dex_minus − adapter_only | +0.0488 | [+0.0473, +0.0512] | 0.0006 | dz=+22.980 | seed-level (n=3) |
| D: dex_minus − adapter_only | +0.0492 | [+0.0242, +0.0748] | 0.0002 | dz=+0.275 | example-level (n=187) |

