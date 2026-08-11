# DEX control study — results

## Per-seed results (Qasper val F1, 187 examples, greedy)

| Variant | Trainable Params | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean | Std |
|---|---|---|---|---|---|---|---|
| base | 0 | 0.2444 | — | — | — | — | 0.2444 | 0.0000 |
| dex_minus | 566,820,900 | 0.2902 | 0.2933 | 0.2907 | 0.2879 | 0.2930 | 0.2910 | 0.0022 |
| dex_plus | 566,820,900 | 0.2959 | 0.3014 | 0.2936 | 0.2889 | 0.2950 | 0.2950 | 0.0045 |
| residual_adapter | 566,820,864 | 0.2897 | 0.2814 | 0.2792 | 0.2928 | 0.2709 | 0.2828 | 0.0087 |
| attn_only | 566,231,040 | 0.2980 | 0.2997 | 0.2872 | 0.2946 | 0.2872 | 0.2933 | 0.0059 |
| adapter_only | 589,860 | 0.2416 | 0.2423 | 0.2457 | — | — | 0.2432 | 0.0022 |

## Per-seed results (EM and final validation CE)

| Variant | EM s0 | EM s1 | EM s2 | EM s3 | EM s4 | EM mean | val CE s0 | val CE s1 | val CE s2 | val CE s3 | val CE s4 | CE mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.0000 | — | — | — | — | 0.0000 | 2.1065 | — | — | — | — | 2.1065 |
| dex_minus | 0.0000 | 0.0053 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 0.7218 | 0.7464 | 0.6800 | 0.6672 | 0.7217 | 0.7074 |
| dex_plus | 0.0000 | 0.0053 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 0.7260 | 0.7688 | 0.6367 | 0.5819 | 0.6805 | 0.6788 |
| residual_adapter | 0.0000 | 0.0000 | 0.0000 | 0.0053 | 0.0000 | 0.0011 | 0.7551 | 0.7082 | 0.7506 | 0.7319 | 0.7542 | 0.7400 |
| attn_only | 0.0000 | 0.0053 | 0.0000 | 0.0053 | 0.0000 | 0.0021 | 0.7370 | 0.7662 | 0.6605 | 0.5792 | 0.7340 | 0.6954 |
| adapter_only | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0000 | 2.1038 | 2.0992 | 2.1009 | — | — | 2.1013 |

## Comparisons

| Comparison | Mean Difference (F1) | 95% CI | p-value | Effect Size | Level |
|---|---:|---|---:|---:|---|
| A: dex_minus − dex_plus | -0.0039 | [-0.0064, -0.0018] | 0.0389 | dz=-1.354 | seed-level (n=5) |
| A: dex_minus − dex_plus | -0.0040 | [-0.0098, +0.0016] | 0.1679 | dz=-0.101 | example-level (n=187) |
| B: dex_minus − residual_adapter | +0.0082 | [+0.0005, +0.0159] | 0.1577 | dz=+0.776 | seed-level (n=5) |
| B: dex_minus − residual_adapter | +0.0082 | [-0.0035, +0.0194] | 0.1655 | dz=+0.102 | example-level (n=187) |
| C: dex_minus − attn_only | -0.0023 | [-0.0071, +0.0028] | 0.4654 | dz=-0.360 | seed-level (n=5) |
| C: dex_minus − attn_only | -0.0023 | [-0.0100, +0.0050] | 0.5417 | dz=-0.045 | example-level (n=187) |
| D: dex_minus − adapter_only | +0.0482 | [+0.0450, +0.0510] | 0.0013 | dz=+15.961 | seed-level (n=3) |
| D: dex_minus − adapter_only | +0.0478 | [+0.0218, +0.0737] | 0.0005 | dz=+0.258 | example-level (n=187) |

