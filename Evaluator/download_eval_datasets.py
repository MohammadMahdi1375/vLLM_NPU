#!/usr/bin/env python3

import os
import time
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Make sure we are NOT accidentally forcing offline mode.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

DATASETS = [
    ("gsm8k", "openai/gsm8k", "main", "test"),
    ("math500", "HuggingFaceH4/MATH-500", "default", "test"),
    ("humaneval", "openai/openai_humaneval", "openai_humaneval", "test"),
    ("mbpp", "google-research-datasets/mbpp", "sanitized", "test"),
    ("aime25", "MathArena/aime_2025", "default", "train"),
    ("mtbench", "HuggingFaceH4/mt_bench_prompts", "default", "train"),
    ("alpaca", "tatsu-lab/alpaca", "default", "train"),
]


def download_dataset(name, repo, config, split):
    print("\n" + "=" * 80)
    print(f"Downloading: {name}")
    print(f"repo   : {repo}")
    print(f"config : {config}")
    print(f"split  : {split}")
    print("=" * 80)

    last_error = None

    for attempt in range(1, 6):
        try:
            ds = load_dataset(
                repo,
                config,
                split=split,
            )

            print(f"[OK] {name}")
            print(f"     rows       : {len(ds)}")
            print(f"     fingerprint: {getattr(ds, '_fingerprint', None)}")
            print(f"     cache files:")
            for item in getattr(ds, "cache_files", []):
                print(f"       {item.get('filename')}")

            return

        except Exception as exc:
            last_error = exc
            print(f"[ERROR] attempt {attempt}/5: {exc}")

            if attempt < 5:
                wait = min(30, 5 * attempt)
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {name} after 5 attempts"
    ) from last_error


for args in DATASETS:
    download_dataset(*args)


# ------------------------------------------------------------------
# LiveCodeBench
#
# evaluator.py uses release_v5, which means:
#   test.jsonl
#   test2.jsonl
#   test3.jsonl
#   test4.jsonl
#   test5.jsonl
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("Downloading LiveCodeBench release_v5 files")
print("=" * 80)

for filename in [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
]:
    last_error = None

    for attempt in range(1, 6):
        try:
            path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=filename,
                repo_type="dataset",
            )

            print(f"[OK] {filename}")
            print(f"     {path}")
            break

        except Exception as exc:
            last_error = exc
            print(f"[ERROR] {filename}, attempt {attempt}/5: {exc}")

            if attempt < 5:
                wait = min(30, 5 * attempt)
                print(f"Retrying in {wait}s...")
                time.sleep(wait)

    else:
        raise RuntimeError(
            f"Failed downloading LiveCodeBench {filename}"
        ) from last_error


print("\n")
print("=" * 80)
print("ALL EVALUATION DATASETS DOWNLOADED SUCCESSFULLY")
print("=" * 80)
