from __future__ import annotations

from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Attention,
    apply_rotary_pos_emb as smollm3_apply_rotary_pos_emb,
    eager_attention_forward as smollm3_eager_attention_forward,
)

HAS_SMOLLM3 = True

try:
    # Qwen3-MoE (e.g. Qwen3-30B-A3B) uses Qwen3MoeAttention, which is structurally
    # identical to dense Qwen3Attention (MoE only affects the MLP). In transformers
    # >=5.x it is no longer a subclass of Qwen3Attention, so register it explicitly.
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeAttention,
    )

    HAS_QWEN3_MOE = True
except Exception:  # pragma: no cover - optional backbone
    Qwen3MoeAttention = None  # type: ignore[assignment]
    HAS_QWEN3_MOE = False


def ensure_attention_compat_views(module):
    return module
