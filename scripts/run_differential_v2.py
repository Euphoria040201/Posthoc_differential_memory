#!/usr/bin/env python
"""Plan/run/resume the audit-v2 experiment with one independent job per selected GPU."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = {
    "add": ("fixed_add", "none", 0., 0.),
    "sub": ("fixed_sub", "none", 0., 0.),
    "add_aug": ("fixed_add", "shuffle", 0., 0.),
    "sub_aug": ("fixed_sub", "shuffle", 0., 0.),
    "sub_nui": ("fixed_sub", "shuffle", 1., 1.),
}


def build_jobs(args):
    root = Path(args.output_root).resolve()
    output = root / args.phase
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else ([0] if args.phase == "pilot" else [0, 1, 2])
    steps = args.steps or (32 if args.phase == "pilot" else 156)
    eval_n = args.eval_examples if args.eval_examples is not None else (64 if args.phase == "pilot" else 0)
    robust_n = args.robust_examples if args.robust_examples is not None else (32 if args.phase == "pilot" else 0)
    common = ["--model-path", args.model_path, "--data-manifest", str(Path(args.data_manifest).resolve()),
              "--dtype", args.dtype, "--device", args.device, "--attn-impl", args.attn_impl,
              "--max-new-tokens", str(args.max_new_tokens)]
    jobs = []
    selections = args.arms.split(",") if args.arms else ["base", *ARMS]
    for arm in selections:
        if arm not in ("base", "attn_only", *ARMS):
            raise ValueError(f"unknown arm: {arm}")
        for seed in ([0] if arm == "base" else seeds):
            tag = f"{arm}_s{seed}"
            tail = ["--seed", str(seed), "--output-dir", str(output), "--tag", tag,
                    "--eval-examples", str(eval_n), "--val-loss-examples", "32"]
            if arm in ("base", "attn_only"):
                train = [sys.executable, str(REPO / "scripts/dex_train_qasper.py"), *common, *tail,
                         "--variant", arm, "--steps", str(steps), "--lr", "2e-5",
                         "--grad-accum", str(args.grad_accum * args.group_k), "--batch-size", "1",
                         "--save-attn", "true", "--grad-checkpointing", "true"]
                checkpoint = output / f"{tag}_model.pt"
            else:
                fusion, augmentation, beta, gamma = ARMS[arm]
                train = [sys.executable, str(REPO / "scripts/dex_nuisance_train.py"), *common, *tail,
                         "--steps", str(steps), "--group-k", str(args.group_k),
                         "--grad-accum", str(args.grad_accum), "--steer-lr", str(args.steer_lr),
                         "--fusion-lambda", "0.1", "--output-fusion", fusion,
                         "--context-augmentation", augmentation, "--beta", str(beta), "--gamma", str(gamma),
                         "--o-fusion-position", "pre_o", "--steer-layers", args.steer_layers,
                         "--steer-mem-head-dim", str(args.steer_mem_head_dim),
                         "--steer-window", str(args.steer_window), "--grad-checkpointing", "true"]
                checkpoint = output / f"{tag}_steer.pt"
            robust = [sys.executable, str(REPO / "scripts/eval_differential_checkpoint.py"), *common,
                      "--checkpoint", str(checkpoint), "--split", "validation",
                      "--group-k", str(args.eval_group_k), "--max-examples", str(robust_n),
                      "--output", str(output / f"{tag}_robust.json")]
            jobs.append({"tag": tag, "arm": arm, "seed": seed, "commands": [train, robust],
                         "checkpoint": str(checkpoint), "output_dir": str(output),
                         "expected": [str(output / f"{tag}.json"), str(checkpoint),
                                      str(output / f"{tag}_robust.json")]})
    return jobs


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def job_signature(job, manifest_sha, source_sha):
    return hashlib.sha256(json.dumps([job["commands"], manifest_sha, source_sha], sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    digest = hashlib.sha256()
    for root in (REPO / "deltamem", REPO / "scripts"):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_job(job, gpu, args, manifest_sha, source_sha):
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / f"{job['tag']}.status.json"
    signature = job_signature(job, manifest_sha, source_sha)
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("signature") != signature:
            raise ValueError(f"{job['tag']}: configuration/source changed; choose a fresh output-root")
        if previous.get("status") == "COMPLETE" and all(Path(p).exists() for p in job["expected"]):
            print(f"[skip] {job['tag']} verified completed job", flush=True)
            return
    state = {"tag": job["tag"], "status": "RUNNING", "signature": signature,
             "source_sha256": source_sha, "manifest_sha256": manifest_sha,
             "gpu": gpu, "started": time.time(), "commands": job["commands"]}
    write_json(status_path, state)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        for index, command in enumerate(job["commands"]):
            stage_path = output / f"{job['tag']}.stage{index}.json"
            if stage_path.exists() and json.loads(stage_path.read_text()).get("signature") == signature:
                stage_outputs = job["expected"][:2] if index == 0 else job["expected"][2:]
                if all(Path(p).exists() for p in stage_outputs):
                    continue
            print(f"[start] {job['tag']} stage={index} device={gpu}", flush=True)
            with (output / f"{job['tag']}.stage{index}.log").open("w") as log:
                completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"{job['tag']} stage {index} exited {completed.returncode}; see its log")
            write_json(stage_path, {"signature": signature, "completed": time.time()})
        if not all(Path(p).exists() for p in job["expected"]):
            raise RuntimeError("a command succeeded without writing all expected artifacts")
        state["status"] = "COMPLETE"
    except Exception as exc:
        state.update(status="FAILED", error=str(exc))
        raise
    finally:
        state["finished"] = time.time()
        write_json(status_path, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "run", "status"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-manifest", required=True)
    ap.add_argument("--output-root", default="runs/differential_v2")
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--gpus", default="0", help="explicitly allocated GPUs; one job per GPU")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--arms", default="", help="default base + five matched sidecars; attn_only is optional")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--group-k", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steer-lr", type=float, default=5e-4)
    ap.add_argument("--steer-layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--steer-mem-head-dim", type=int, default=128)
    ap.add_argument("--steer-window", type=int, default=256)
    ap.add_argument("--eval-examples", type=int, default=None)
    ap.add_argument("--robust-examples", type=int, default=None)
    ap.add_argument("--eval-group-k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()
    jobs = build_jobs(args)
    if args.action == "plan":
        print(json.dumps(jobs, indent=2))
        return
    if args.action == "status":
        for job in jobs:
            path = Path(job["output_dir"]) / f"{job['tag']}.status.json"
            state = json.loads(path.read_text()) if path.exists() else {"status": "PENDING"}
            print(job["tag"], state["status"], state.get("error", ""))
        return
    sys.path.insert(0, str(REPO))
    from deltamem.kv_binding.qa_protocol import read_manifest
    manifest = read_manifest(args.data_manifest)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU is available: no full-model job was launched. Run this command on the allocated GPU host.")
    devices = args.gpus.split(",") if args.device == "cuda" else ["cpu"]
    if len(devices) != len(set(devices)) or not all(devices):
        ap.error("gpus must be a non-empty list without duplicates")
    available = queue.Queue()
    for device in devices:
        available.put(device)
    source_sha = source_fingerprint()
    def worker(job):
        device = available.get()
        try:
            run_job(job, device, args, manifest["sha256"], source_sha)
        finally:
            available.put(device)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
