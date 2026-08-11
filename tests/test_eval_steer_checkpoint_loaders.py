from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from deltamem.core.prefix_steer import PrefixSteerConfig
from deltamem.eval.steer_checkpoint import (
    has_persistent_history_writer,
    restore_prefix_steer_config,
)
from scripts import eval_ours_hotpotqa as hotpot
from scripts import eval_ours_locomo as locomo


_DELTA_O = "model.layers.0.self_attn.delta_o.weight"
_MEM_V = "model.layers.0.self_attn.mem_v.weight"


class _FakeModel:
    def __init__(
        self,
        *,
        steer_names: tuple[str, ...] = (_DELTA_O,),
        unexpected: tuple[str, ...] = (),
    ) -> None:
        self._parameters = {
            name: torch.nn.Parameter(torch.zeros(1)) for name in steer_names
        }
        self._unexpected = list(unexpected)
        self.loaded_state = None

    def named_parameters(self):
        return iter(self._parameters.items())

    def load_state_dict(self, state, strict=False):
        assert strict is False
        self.loaded_state = state
        return [], list(self._unexpected)


def _shared_v_config() -> PrefixSteerConfig:
    return PrefixSteerConfig(
        num_prefix_tokens=0,
        sliding_window_size=256,
        mem_num_heads=1,
        mem_head_dim=128,
        steer_mode="deltamem",
        normal_attends_prefix=True,
        prefix_sees_query=False,
        prefix_write=False,
        pool_reads=False,
        memory_value_source="main_v",
        delta_heads="o",
        output_fusion="rms_match",
        output_fusion_eps=2e-6,
        output_fusion_scale_max=7.0,
    )


@pytest.mark.parametrize("module", [hotpot, locomo])
def test_eval_loaders_restore_shared_main_v_and_output_fusion(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _shared_v_config()
    checkpoint = tmp_path / f"{module.__name__.split('.')[-1]}.pt"
    state = {_DELTA_O: torch.ones(1)}
    torch.save({"cfg": asdict(config), "state": state}, checkpoint)
    model = _FakeModel()
    monkeypatch.setattr(module, "attach_prefix_steer", lambda *_: None)
    monkeypatch.setattr(module, "freeze_backbone_keep_steer", lambda *_: None)

    restored = module.load_ours(model, checkpoint)

    assert restored.memory_value_source == "main_v"
    assert restored.output_fusion == "rms_match"
    assert restored.output_fusion_eps == 2e-6
    assert restored.output_fusion_scale_max == 7.0
    assert restored.delta_heads == "o"
    assert model.loaded_state == state


def test_legacy_checkpoint_fields_use_historical_not_dataclass_defaults() -> None:
    legacy = asdict(
        PrefixSteerConfig(
            prefix_init_std=0.02,
            prefix_write=False,
            pool_reads=False,
            pool_gate_max=1.0,
        )
    )
    for field in (
        "prefix_init_dist",
        "prefix_write_layout",
        "prefix_write_overlap_tokens",
        "pool_gate_bias",
        "memory_value_source",
        "output_fusion",
        "output_fusion_eps",
        "output_fusion_scale_max",
        "history_pool_mode",
        "hybrid_read_mode",
        "hybrid_prefix_gate_mode",
        "hybrid_prefix_gate_init",
    ):
        legacy.pop(field)

    restored = restore_prefix_steer_config(legacy)

    assert restored.prefix_init_std == 0.02
    assert restored.prefix_write is False
    assert restored.pool_reads is False
    assert restored.pool_gate_max == 1.0
    assert restored.memory_value_source == "trainable"
    assert restored.output_fusion == "fixed"
    assert restored.output_fusion_eps == 1e-6
    assert restored.output_fusion_scale_max == 10.0
    assert restored.prefix_write_layout == "global"
    assert restored.history_pool_mode == "none"


def test_checkpoint_cfg_unknown_to_current_core_fails_loudly() -> None:
    raw = asdict(PrefixSteerConfig())
    raw["future_silent_architecture_switch"] = True
    with pytest.raises(ValueError, match="unknown to this evaluator/core"):
        restore_prefix_steer_config(raw)


@pytest.mark.parametrize(
    ("state", "unexpected", "message"),
    [
        ({}, (), "would stay at random init"),
        (
            {_DELTA_O: torch.ones(1), _MEM_V: torch.ones(1)},
            (_MEM_V,),
            "were NOT loaded",
        ),
    ],
)
def test_locomo_loader_rejects_missing_or_dropped_steer_tensors(
    state,
    unexpected,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"cfg": asdict(_shared_v_config()), "state": state}, checkpoint)
    model = _FakeModel(unexpected=unexpected)
    monkeypatch.setattr(locomo, "attach_prefix_steer", lambda *_: None)
    monkeypatch.setattr(locomo, "freeze_backbone_keep_steer", lambda *_: None)

    with pytest.raises(RuntimeError, match=message):
        locomo.load_ours(model, checkpoint)


def test_hotpot_skips_ctx_write_only_when_checkpoint_has_no_writer() -> None:
    no_writer = _shared_v_config()
    assert has_persistent_history_writer(no_writer) is False
    assert (
        hotpot.should_run_context_write(
            no_writer, condition="ours", write_pass="ctx"
        )
        is False
    )

    pooled_writer = PrefixSteerConfig(
        num_prefix_tokens=0,
        prefix_write=False,
        pool_reads=False,
        history_pool_mode="attn",
    )
    assert has_persistent_history_writer(pooled_writer) is True
    assert hotpot.should_run_context_write(
        pooled_writer, condition="ours", write_pass="ctx"
    )
    assert not hotpot.should_run_context_write(
        pooled_writer, condition="base", write_pass="ctx"
    )
