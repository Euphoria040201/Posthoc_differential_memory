"""Regression tests for experiment-invalidating failures found in the September audit."""
from dataclasses import asdict

import pytest
import torch

from deltamem.core.global_prefix import SEG_CTX, SEG_QRY, SEG_ANS
from deltamem.core.prefix_steer import (iter_steer_modules, set_fusion_calibrating,
                                      set_steer_segments, steer_state_dict)
from deltamem.eval.steer_checkpoint import load_steer_state_strict
from deltamem.eval.dex_checkpoint import dex_state_dict, load_dex_state_strict, restore_dex_config
from deltamem.core.dex import DexConfig, attach_dex, set_trainable
from scripts import dex_nuisance_train as nuisance
from scripts import qasper_prefix_steer as qa
from tests.test_dex import tiny_model
from tests.test_fusion_differential import build, wake_delta_o


def test_nuisance_ignores_unaligned_context_and_answer(monkeypatch):
    y = torch.tensor([[[0.], [3.], [100.]], [[10.], [3.], [-100.]]])
    c = torch.zeros_like(y, requires_grad=True)
    monkeypatch.setattr(nuisance, "collect_fusion_tensors", lambda _: [(0, y, c)])
    ln, li, _ = nuisance.nuisance_losses(None, .1, torch.tensor([1, 1]))
    assert ln.item() == li.item() == 0
    (ln + li).backward()
    assert torch.count_nonzero(c.grad) == 0


def test_nuisance_targets_actual_scaled_correction(monkeypatch):
    y = torch.tensor([[[8.], [1.], [99.]], [[-8.], [3.], [-99.]]])
    c = torch.tensor([[[88.], [-10.], [999.]], [[88.], [10.], [999.]]], requires_grad=True)
    monkeypatch.setattr(nuisance, "collect_fusion_tensors", lambda _: [(0, y, c)])
    ln, li, _ = nuisance.nuisance_losses(None, .1, torch.tensor([1, 1]))
    assert ln.item() == li.item() == 0


def test_query_end_is_before_teacher_forced_answer():
    seg = torch.tensor([[SEG_CTX, SEG_QRY, SEG_QRY, SEG_ANS],
                        [SEG_QRY, SEG_QRY, SEG_ANS, SEG_ANS]])
    assert nuisance.aligned_query_positions(seg, torch.ones_like(seg)).tolist() == [2, 1]
    with pytest.raises(ValueError, match="query token"):
        nuisance.aligned_query_positions(torch.full_like(seg, SEG_ANS), torch.ones_like(seg))


def calibrated_model():
    model = build("variance_diff").eval()
    wake_delta_o(model)
    ids = torch.randint(0, 64, (1, 9))
    set_fusion_calibrating(model, True)
    set_steer_segments(model, torch.full_like(ids, SEG_CTX), torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    set_fusion_calibrating(model, False)
    return model


def logits(model, ids):
    set_steer_segments(model, torch.full_like(ids, SEG_CTX), torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def test_variance_diff_prefix_causality_and_batch_independence():
    model = calibrated_model()
    ids = torch.randint(0, 64, (1, 6))
    before = {n: b.clone() for n, b in model.named_buffers() if ".fusion_" in n}
    one = logits(model, ids)
    extended = logits(model, torch.cat([ids, torch.randint(0, 64, (1, 3))], dim=1))
    batched = logits(model, torch.cat([ids, torch.randint(0, 64, (1, 6))], dim=0))
    torch.testing.assert_close(one, extended[:, :6], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(one, batched[:1], atol=1e-6, rtol=1e-5)
    for name, buf in model.named_buffers():
        if name in before:
            torch.testing.assert_close(buf, before[name], rtol=0, atol=0)


def test_variance_checkpoint_round_trip_includes_calibration(tmp_path):
    model = calibrated_model()
    ids = torch.randint(0, 64, (1, 7))
    before = logits(model, ids)
    state = steer_state_dict(model)
    path = tmp_path / "steer.pt"
    torch.save(state, path)
    restored = build("variance_diff").eval()
    load_steer_state_strict(restored, torch.load(path, weights_only=True))
    torch.testing.assert_close(before, logits(restored, ids), rtol=0, atol=0)
    broken = {k: v for k, v in state.items() if not k.endswith("fusion_coefficient")}
    with pytest.raises(RuntimeError, match="NOT in the ckpt"):
        load_steer_state_strict(restored, broken)


def test_calibration_fp32_buffers_survive_bf16_model_reload():
    model = calibrated_model()
    state = steer_state_dict(model)
    restored = build("variance_diff").to(torch.bfloat16).eval()
    load_steer_state_strict(restored, state)
    buffers = dict(restored.named_buffers())
    for name, value in state.items():
        if ".fusion_" in name and value.is_floating_point():
            assert buffers[name].dtype == value.dtype == torch.float32
            torch.testing.assert_close(buffers[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["dex_minus", "attn_only", "adapter_only"])
def test_dex_training_checkpoint_round_trip_and_incomplete_rejection(variant, tmp_path):
    cfg = DexConfig(variant=variant, head_selection="all", allow_no_anneal=True).resolve()
    model = tiny_model()
    attach_dex(model, cfg)
    set_trainable(model, cfg)
    ids = torch.randint(0, 64, (1, 8))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    state = dex_state_dict(model, cfg)
    path = tmp_path / "model.pt"
    torch.save({"state": state, "config": asdict(cfg)}, path)
    saved = torch.load(path, weights_only=True)
    restored_cfg = restore_dex_config(saved["config"])
    restored = tiny_model()
    attach_dex(restored, restored_cfg)
    load_dex_state_strict(restored, saved["state"], restored_cfg)
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(input_ids=ids, use_cache=False).logits,
                                   rtol=0, atol=0)
    broken = dict(state)
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError, match="incomplete DEX"):
        load_dex_state_strict(restored, broken, restored_cfg)


class CharTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kw):
        result = {"input_ids": list(map(ord, text))}
        if kw.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids if i > 1)


def episode(answer="secret", evidence="EVIDENCE"):
    return {"paper_id": "p1", "chunks": ["EVIDENCE", "x" * 60, "TAIL SECRET"],
            "queries": [{"question": "what?", "answer": answer,
                         "gold": [0], "evidence": [evidence], "question_id": 1}]}


def examples(monkeypatch, ep, **kwargs):
    monkeypatch.setattr(qa, "build_fulldoc_episodes", lambda *a, **k: [ep])
    return qa.build_examples("train", 1, CharTokenizer(), 256,
                             kwargs.get("context", 1000), kwargs.get("answer_budget", 24))


def test_targets_include_eos_and_reject_truncated_answers(monkeypatch):
    ex = examples(monkeypatch, episode())[0]
    assert ex["labels"][-1] == CharTokenizer.eos_token_id
    assert ex["qa_labels"][-1] == CharTokenizer.eos_token_id
    assert ex["answer"] == "secret"
    with pytest.raises(ValueError, match="no eligible"):
        examples(monkeypatch, episode(answer="z" * 24))


def test_evidence_must_survive_total_context_truncation(monkeypatch):
    with pytest.raises(ValueError, match="no eligible"):
        examples(monkeypatch, episode(evidence="TAIL SECRET"), context=30)


def test_partial_evidence_is_not_enough(monkeypatch):
    with pytest.raises(ValueError, match="no eligible"):
        examples(monkeypatch, episode(evidence="EVIDENCE HAS BEEN TRUNCATED"))


def test_generation_stops_on_supervised_eos(monkeypatch):
    from types import SimpleNamespace
    calls = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append(input_ids.shape[1])
            scores = torch.full((1, input_ids.shape[1], 128), -100.)
            scores[0, -1, ord("N") if len(calls) == 1 else 1] = 100
            return SimpleNamespace(logits=scores)
    monkeypatch.setattr(qa, "set_steer_segments", lambda *a: None)
    ex = {"prompt_ids": [7, 8], "prompt_seg": [SEG_QRY, SEG_QRY]}
    assert qa.generate(Model(), CharTokenizer(), ex, "cpu", 24, 1) == "N"
    assert len(calls) == 2 and ex["last_generation"]["stopped_on_eos"]


def test_add_sub_parameter_counts_are_identical():
    add, sub = build("fixed_add"), build("fixed_sub")
    from deltamem.core.prefix_steer import freeze_backbone_keep_steer
    freeze_backbone_keep_steer(add)
    freeze_backbone_keep_steer(sub)
    a = {n: p.numel() for n, p in add.named_parameters() if p.requires_grad}
    b = {n: p.numel() for n, p in sub.named_parameters() if p.requires_grad}
    assert a == b and sum(a.values()) > 0


def test_cli_rejects_two_different_fusion_coefficients(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["dex_nuisance_train.py", "--tag", "invalid",
                                     "--fusion-lambda", "0.7", "--steer-gain", "0.1"])
    with pytest.raises(SystemExit) as error:
        nuisance.main()
    assert error.value.code == 2


def test_experiment_plan_keeps_all_sidecar_budgets_identical():
    import argparse
    from scripts.run_differential_v2 import build_jobs
    args = argparse.Namespace(output_root="/tmp/plan", phase="main", seeds="", steps=0,
        eval_examples=None, robust_examples=None, model_path="model", data_manifest="data.json",
        dtype="bfloat16", device="cuda", attn_impl="sdpa", max_new_tokens=24,
        arms="", grad_accum=8, group_k=2, steer_lr=5e-4, steer_layers="0,3",
        steer_mem_head_dim=128, steer_window=256, eval_group_k=4)
    jobs = build_jobs(args)
    assert len(jobs) == 16
    flags = ("--steps", "--group-k", "--grad-accum", "--steer-lr", "--steer-layers", "--fusion-lambda")
    settings = []
    for job in jobs:
        if job["arm"] == "base":
            continue
        command = job["commands"][0]
        settings.append(tuple(command[command.index(flag) + 1] for flag in flags))
    assert len(set(settings)) == 1
