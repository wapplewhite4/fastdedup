#!/usr/bin/env python3
"""
Fuzzy dedup baseline using datatrove MinHashDedup.
Config: 128 hashes (16 buckets x 8 hashes), 5-grams, threshold via bucket config.
"""
import sys, time, os

input_path = sys.argv[1]
output_dir = sys.argv[2]
field = "text"

print(f"Input     : {input_path}", flush=True)
print(f"Output dir: {output_dir}", flush=True)
print(f"Field     : {field}", flush=True)

os.makedirs(output_dir, exist_ok=True)
MINHASH_BASE_PATH = os.path.join(output_dir, "minhash_tmp")

input_dir = os.path.dirname(os.path.abspath(input_path))
input_filename = os.path.basename(input_path)

from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupFilter
from datatrove.pipeline.writers import ParquetWriter

config = MinhashConfig(
    n_grams=5,
    num_buckets=16,
    hashes_per_bucket=8,
)
print(f"Config    : {config.num_buckets} buckets x {config.hashes_per_bucket} hashes, {config.n_grams}-grams", flush=True)

t0 = time.perf_counter()

print("Stage 1/3: computing MinHash signatures...", flush=True)
t1 = time.perf_counter()
LocalPipelineExecutor(
    pipeline=[
        ParquetReader(input_dir, text_key=field, glob_pattern=input_filename),
        MinhashDedupSignature(
            output_folder=os.path.join(MINHASH_BASE_PATH, "sigs"),
            config=config,
        ),
    ],
    tasks=1,
).run()
print(f"  Done in {time.perf_counter() - t1:.2f}s", flush=True)

print("Stage 2/3: bucket clustering...", flush=True)
t2 = time.perf_counter()
LocalPipelineExecutor(
    pipeline=[
        MinhashDedupBuckets(
            input_folder=os.path.join(MINHASH_BASE_PATH, "sigs"),
            output_folder=os.path.join(MINHASH_BASE_PATH, "buckets"),
            config=config,
        ),
    ],
    tasks=1,
).run()
print(f"  Done in {time.perf_counter() - t2:.2f}s", flush=True)

print("Stage 3/3: filtering duplicates...", flush=True)
t3 = time.perf_counter()
LocalPipelineExecutor(
    pipeline=[
        ParquetReader(input_dir, text_key=field, glob_pattern=input_filename),
        MinhashDedupFilter(input_folder=os.path.join(MINHASH_BASE_PATH, "buckets")),
        ParquetWriter(output_dir),
    ],
    tasks=1,
).run()
print(f"  Done in {time.perf_counter() - t3:.2f}s", flush=True)

total = time.perf_counter() - t0
print(f"Total time: {total:.2f}s", flush=True)
