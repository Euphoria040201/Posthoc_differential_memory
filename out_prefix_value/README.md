# Prefix value: episodic binding results

## Claim tested

With the backbone and steer weights held fixed, a memory written from the
correct episode should outperform both:

- `window`: the identical model with prefix columns removed from the read;
- `swap`: the identical model reading a prefix written from another episode.

The task randomly reassigns cities to names on every episode. The document is
written once and then removed before answering. Consequently, the question
alone cannot identify the current answer; a question-only task adapter is
bounded near the city prior.

Model: frozen Qwen3-4B-Instruct-2507, 64 prefix slots, 12 steered layers
(`0,3,...,33`). All reported accuracies below use a strict leading-city
boundary: repetition/punctuation after the city is allowed, but glued junk
such as `LagosHuman` is not.

## Main results (512 fresh random episodes)

| Setting | Correct prefix | Window-only | Swapped prefix | Correct-window | Correct-swap |
|---|---:|---:|---:|---:|---:|
| Current pool architecture, gain 1.0 | 1.000 | 0.070 | 0.055 | +0.930 | +0.945 |
| Prefix-only, plain CE (no pair, no hinge) | 0.994 | 0.008 | 0.059 | +0.986 | +0.936 |
| Prefix-only, seed 2 | 0.998 | 0.002 | 0.055 | +0.996 | +0.943 |
| Prefix-only, seed 3 | 0.965 | 0.006 | 0.055 | +0.959 | +0.910 |
| Prefix-only, seed 4 | 1.000 | 0.008 | 0.074 | +0.992 | +0.926 |

For the current pool architecture, correct vs window has 476 favorable and
zero unfavorable discordant pairs (exact McNemar p=1.0e-143); correct vs swap
has 484/0 (p=4.0e-146).

The plain-CE prefix-only run is important: neither paired-conflict batching nor
the detached window hinge is required for the result. The random reassignment
task itself removes the steer-only shortcut. Pairing and the hinge mainly make
optimization faster and more stable.

## Capacity curriculum

Starting from the solved one-fact checkpoint:

| Facts per written document | Correct prefix | Window-only | Swapped prefix |
|---:|---:|---:|---:|
| 1 | 0.994 | 0.008 | 0.059 |
| 2, current pool | 0.494 | 0.090 | 0.051 |
| 2, prefix-only | 0.475 | 0.006 | 0.057 |
| 4, prefix-only | 0.232 | 0.004 | 0.070 |

The prefix advantage remains highly significant at two and four facts, but
absolute retrieval accuracy declines with the number of bindings. Multi-slot
capacity is the next bottleneck.

## Generalization boundary

- Held-out entity names (all evaluation first names absent during training):
  correct 0.859, window 0.010, swap 0.062 on 512 episodes.
- Held-out entity names *and* held-out city values: correct 0.000 after the
  training loss reached approximately zero.

Thus the learned circuit generalizes new bindings and new keys over a known
value vocabulary. It is not yet an open-vocabulary copy mechanism.

## Interpretation

These experiments establish a clean positive regime for prefix memory:
write-once episodic selection/binding after the source document is discarded.
They do not overturn the Qasper/HotpotQA finding that most natural-QA gains
came from the steer/window adapter. The proposed track should therefore be
described as episodic associative memory, personalization/state slots, or
write-then-discard memory—not as general long-document QA compression yet.

The full per-example predictions and arguments are in the corresponding
`eval512_*.json` files. The training script is
`investigation/episodic_kv_test.py`.
