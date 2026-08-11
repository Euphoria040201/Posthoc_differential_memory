"""Run benchmark_compare gpqa_diamond loading from a local CSV (gated-dataset bypass)."""
import deltamem.eval.benchmark_compare as bc
from datasets import load_dataset

CSV = "/storage/backup/mike/15/work_mike/datasets/Idavidrein/gpqa/gpqa_diamond.csv"

def _load_gpqa_csv(*, cache_dir, max_samples, seed, local_files_only):
    ds = load_dataset("csv", data_files=CSV, split="train", cache_dir=str(cache_dir))
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    return [dict(r) for r in ds]

bc.load_gpqa_diamond = _load_gpqa_csv

if __name__ == "__main__":
    bc.main()
