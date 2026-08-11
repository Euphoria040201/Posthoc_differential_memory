"""Global Prefix Memory + Sliding Window Attention.

A lightweight, self-contained alternative to the delta-rule memory in
``deltamem.core.delta_impl``.  Instead of a compressed online state read as a
low-rank correction, we prepend a small number of *trainable prefix memory
tokens* to the sequence and force the normal tokens to use causal
sliding-window attention (SWA).  Long-range information must therefore flow
through the prefix slots, which are allowed to read the whole context
(non-causal / global read).

Sequence layout::

    [PREFIX P (learned)] [CONTEXT C] [QUERY Q] [ANSWER Y]

Attention rules (see ``build_prefix_attention_mask``):

* Prefix rows  : attend to all P + all C (+ Q if ``prefix_sees_query``); never Y.
* Context rows : causal SWA over previous normal tokens only.
* Query rows   : causal SWA (+ all P if ``normal_attends_prefix``).
* Answer rows  : causal SWA (+ all P if ``normal_attends_prefix``).

Variant A == ``normal_attends_prefix=False`` (prefix influences indirectly).
Variant B == ``normal_attends_prefix=True``  (prefix is a readable memory).

Everything here is gated behind CLI flags in the training / eval scripts and
does not touch the delta-mem machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Segment ids used throughout.  Order matters: normal tokens are >= CTX and the
# sequence is always laid out PREFIX < CTX < QRY < ANS.
SEG_PREFIX = 0
SEG_CTX = 1
SEG_QRY = 2
SEG_ANS = 3
SEG_NAMES = {SEG_PREFIX: "P", SEG_CTX: "C", SEG_QRY: "Q", SEG_ANS: "Y"}


@dataclass
class GlobalPrefixConfig:
    num_prefix_tokens: int = 32
    sliding_window_size: int = 1024
    normal_attends_prefix: bool = True
    prefix_sees_query: bool = True
    freeze_backbone: bool = True


def build_prefix_attention_mask(
    seg: torch.Tensor,
    valid: torch.Tensor,
    *,
    sliding_window_size: int,
    normal_attends_prefix: bool,
    prefix_sees_query: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct a 4D additive attention mask ``[B, 1, N, N]``.

    ``seg``   : ``[B, N]`` int tensor with values in {SEG_PREFIX..SEG_ANS}.
    ``valid`` : ``[B, N]`` bool tensor, False for padding key positions.

    Returns an additive mask (``0`` = attend, large negative = masked) suitable
    for ``sdpa`` / ``eager`` attention (broadcast over heads).
    """
    device = seg.device
    B, N = seg.shape
    idx = torch.arange(N, device=device)
    qi = idx.view(1, N, 1)  # query position
    kj = idx.view(1, 1, N)  # key position

    sq = seg.view(B, N, 1)  # query segment
    sk = seg.view(B, 1, N)  # key segment

    is_pref_q = sq == SEG_PREFIX
    is_pref_k = sk == SEG_PREFIX
    is_norm_q = sq >= SEG_CTX
    is_norm_k = sk >= SEG_CTX

    # Normal tokens: causal sliding window over normal tokens only.
    causal = kj <= qi
    window = (qi - kj) < sliding_window_size
    swa = is_norm_q & is_norm_k & causal & window

    # Prefix rows: global read of P + C (+ Q), never Y.  Non-causal, no window.
    pref_rows = is_pref_q & (
        is_pref_k
        | (sk == SEG_CTX)
        | ((sk == SEG_QRY) if prefix_sees_query else torch.zeros_like(is_pref_k))
    )

    allowed = swa | pref_rows

    # Variant B: normal rows may additionally read all prefix slots.
    # NOTE: this used to be restricted to (SEG_QRY | SEG_ANS) rows, which silently made the
    # prefix UNREADABLE whenever a caller labelled the whole prompt SEG_CTX -- which is what
    # every eval script does. Training labelled CTX/QRY/ANS properly, so the prefix was
    # readable at train time and dead at eval time (measured: 20.5% -> 0.00% of the readout).
    # `normal_attends_prefix` now means what its name says: ANY normal row may read prefix.
    if normal_attends_prefix:
        norm_reads_pref = is_pref_k & is_norm_q
        allowed = allowed | norm_reads_pref

    # Respect padding on the key axis.
    allowed = allowed & valid.view(B, 1, N)

    min_val = torch.finfo(dtype).min
    additive = torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )
    return additive.unsqueeze(1)  # [B, 1, N, N]


class GlobalPrefixMemoryModel(nn.Module):
    """Wrap a frozen HF causal LM with trainable global prefix memory tokens."""

    def __init__(self, base_model: nn.Module, config: GlobalPrefixConfig) -> None:
        super().__init__()
        self.base = base_model
        self.cfg = config
        text_cfg = base_model.config.get_text_config() if hasattr(base_model.config, "get_text_config") else base_model.config
        hidden = text_cfg.hidden_size
        self.num_prefix_tokens = config.num_prefix_tokens

        # Trainable prefix embeddings, initialized from a sample of the real
        # input embedding matrix so they start in-distribution.
        embed = self.base.get_input_embeddings()
        with torch.no_grad():
            weight = embed.weight
            vocab = weight.shape[0]
            pick = torch.randint(0, vocab, (config.num_prefix_tokens,), device=weight.device)
            init = weight[pick].clone().float()
        self.prefix = nn.Parameter(init)  # [P, H] kept in fp32 for stable training

        if config.freeze_backbone:
            # Freeze the backbone but keep any LoRA/adapter params trainable.
            for n, p in self.base.named_parameters():
                if "lora_" not in n:
                    p.requires_grad_(False)
        self._zero_prefix = False  # ablation toggle

    # -- helpers -----------------------------------------------------------
    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def set_zero_prefix(self, enabled: bool) -> None:
        """Ablation: zero the prefix embeddings at inference time."""
        self._zero_prefix = enabled

    def _prefix_embeds(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        p = self.prefix.to(dtype=dtype, device=device)
        if self._zero_prefix:
            p = torch.zeros_like(p)
        return p.unsqueeze(0).expand(batch_size, -1, -1)

    def build_inputs(self, input_ids, seg_text, valid_text, include_prefix=True):
        """Prepend prefix (unless disabled); return (embeds, seg_full, valid_full, pos, P)."""
        B, L = input_ids.shape
        device = input_ids.device
        embed = self.base.get_input_embeddings()
        text_embeds = embed(input_ids)
        dtype = text_embeds.dtype
        P = self.num_prefix_tokens if include_prefix else 0
        if P > 0:
            pref = self._prefix_embeds(B, dtype, device)
            inputs_embeds = torch.cat([pref, text_embeds], dim=1)
            pref_seg = torch.full((B, P), SEG_PREFIX, dtype=seg_text.dtype, device=device)
            seg_full = torch.cat([pref_seg, seg_text], dim=1)
            pref_valid = torch.ones((B, P), dtype=torch.bool, device=device)
            valid_full = torch.cat([pref_valid, valid_text.bool()], dim=1)
        else:
            inputs_embeds, seg_full, valid_full = text_embeds, seg_text, valid_text.bool()
        N = inputs_embeds.shape[1]
        position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        return inputs_embeds, seg_full, valid_full, position_ids, P

    def forward(self, input_ids, seg_text, valid_text, labels_text=None,
                include_prefix=True, sliding_window_size=None):
        inputs_embeds, seg_full, valid_full, position_ids, P = self.build_inputs(
            input_ids, seg_text, valid_text, include_prefix=include_prefix
        )
        win = self.cfg.sliding_window_size if sliding_window_size is None else sliding_window_size
        mask = build_prefix_attention_mask(
            seg_full,
            valid_full,
            sliding_window_size=win,
            normal_attends_prefix=self.cfg.normal_attends_prefix,
            prefix_sees_query=self.cfg.prefix_sees_query,
            dtype=inputs_embeds.dtype,
        )
        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = out.logits  # [B, N, V]

        loss = None
        if labels_text is not None:
            B, L = input_ids.shape
            full_labels = torch.full(
                (B, P + L), -100, dtype=labels_text.dtype, device=labels_text.device
            )
            full_labels[:, P:] = labels_text
            shift_logits = logits[:, :-1, :]
            shift_labels = full_labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate_answer(self, input_ids, seg_text, valid_text, max_new_tokens=16, eos_token_id=None,
                        include_prefix=True, sliding_window_size=None):
        """Greedy decode by re-running a full forward each step (toy-scale).

        ``input_ids`` should end after the query (no answer tokens).  Newly
        generated tokens are appended and labeled as answer tokens so they use
        the same SWA + prefix-read rules.
        """
        self.eval()
        device = input_ids.device
        cur_ids = input_ids.clone()
        cur_seg = seg_text.clone()
        cur_valid = valid_text.clone()
        generated = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                cur_ids, cur_seg, cur_valid,
                include_prefix=include_prefix, sliding_window_size=sliding_window_size,
            )
            next_logits = logits[:, -1, :]
            next_tok = next_logits.argmax(dim=-1, keepdim=True)  # [B,1]
            generated.append(next_tok)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            cur_seg = torch.cat(
                [cur_seg, torch.full_like(next_tok, SEG_ANS)], dim=1
            )
            cur_valid = torch.cat(
                [cur_valid, torch.ones_like(next_tok, dtype=cur_valid.dtype)], dim=1
            )
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
        return torch.cat(generated, dim=1) if generated else cur_ids[:, :0]
