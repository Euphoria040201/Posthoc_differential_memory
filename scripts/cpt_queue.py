#!/usr/bin/env python
"""GPU queue runner for the CPT matrix.

Keeps every listed GPU busy until the job list is empty, appends one ledger row
per state change, and retries a failed job once (a transient OOM caused by
another tenant grabbing the card should not silently cost an arm).

    python scripts/cpt_queue.py --jobs jobs.json --gpus 0,1,2,3 [--dry-run]

jobs.json: [{"tag": "...", "args": ["--arm", "lowrank", ...]}, ...]
Jobs are dispatched in list order, so put the highest-priority controls first:
a missing control is worse than a missing variant.
"""
from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/work/mingze/miniconda3/envs/deltamem/bin/python"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    COLS = ("tag", "host", "gpu", "pid", "start_utc", "end_utc", "secs", "rc",
            "status", "artifact", "log", "cmd")

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.write_text("\t".join(self.COLS) + "\n")

    def row(self, **kw):
        with self.path.open("a") as f:
            f.write("\t".join(str(kw.get(c, "")) for c in self.COLS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    outdir = repo / "out_cpt_20260817"
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    led = Ledger(outdir / "cpt_job_ledger.tsv")

    jobs = json.loads(Path(args.jobs).read_text())
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pending = [dict(j, tries=0) for j in jobs]
    running: dict[str, dict] = {}
    host = socket.gethostname()
    done, failed = [], []

    print(f"[queue] {len(pending)} jobs over GPUs {gpus} on {host}", flush=True)
    if args.dry_run:
        for j in pending:
            print(f"  {j['tag']}: {' '.join(j['args'])}")
        return

    while pending or running:
        for g in list(gpus):
            if g in running or not pending:
                continue
            j = pending.pop(0)
            tag = j["tag"]
            log = logdir / f"{tag}.log"
            cmd = ([PY, str(repo / "scripts" / "cpt_train.py"), "--tag", tag,
                    "--data-dir", str(outdir), "--out-dir", str(outdir)] + j["args"])
            env_prefix = f"CUDA_VISIBLE_DEVICES={g}"
            with log.open("w") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                     cwd=str(repo),
                                     env={**__import__("os").environ,
                                          "CUDA_VISIBLE_DEVICES": g})
            running[g] = {"job": j, "proc": p, "start": time.time(),
                          "start_utc": now(), "log": log, "tag": tag,
                          "cmd": env_prefix + " " + " ".join(shlex.quote(c) for c in cmd)}
            led.row(tag=tag, host=host, gpu=g, pid=p.pid, start_utc=now(),
                    status="RUNNING", artifact=str(outdir / f"{tag}.json"),
                    log=str(log), cmd=running[g]["cmd"])
            print(f"[queue] start {tag} on gpu{g} pid={p.pid} "
                  f"({len(pending)} pending)", flush=True)

        time.sleep(args.poll)

        for g, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is None:
                continue
            secs = int(time.time() - r["start"])
            tag = r["tag"]
            nll = ""
            try:
                for line in r["log"].read_text().splitlines()[::-1]:
                    if "FINAL nll=" in line:
                        nll = line.split("FINAL nll=")[1].split()[0]
                        break
            except Exception:
                pass
            if rc == 0:
                status = f"DONE(rc=0) nll={nll}" if nll else "DONE(rc=0)"
                done.append(tag)
            else:
                status = f"FAILED(rc={rc})"
                j = r["job"]
                if j["tries"] < args.retries:
                    j["tries"] += 1
                    pending.insert(0, j)          # highest priority: retry first
                    status += f" -> retry {j['tries']}/{args.retries}"
                else:
                    failed.append(tag)
            led.row(tag=tag, host=host, gpu=g, pid=r["proc"].pid,
                    start_utc=r["start_utc"], end_utc=now(), secs=secs, rc=rc,
                    status=status, artifact=str(outdir / f"{tag}.json"),
                    log=str(r["log"]), cmd=r["cmd"])
            print(f"[queue] {status} {tag} on gpu{g} in {secs}s", flush=True)
            del running[g]

    print(f"[queue] complete: {len(done)} done, {len(failed)} failed {failed}", flush=True)


if __name__ == "__main__":
    main()
