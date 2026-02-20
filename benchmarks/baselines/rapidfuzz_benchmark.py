#!/usr/bin/env python3

"""
RapidFuzz Parquet Benchmark Script

Purpose:
    - Load a Parquet file
    - Perform fuzzy matching within blocked groups
    - Display CLI progress bar
    - Measure runtime and throughput

Usage:
    python rapidfuzz_benchmark.py wikipedia.parquet title
"""

import sys
import time
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm
from collections import defaultdict


def block_records(values):
    """
    Simple blocking strategy:
    Group strings by first lowercase character.

    This avoids full O(n^2) comparison.
    Replace with better blocking if needed.
    """
    blocks = defaultdict(list)
    for idx, val in enumerate(values):
        if isinstance(val, str) and val:
            key = val[0].lower()
            blocks[key].append((idx, val))
    return blocks


def fuzzy_compare_block(block, threshold=90):
    """
    Compare all records within a block.
    Returns list of matching index pairs above threshold.
    """
    matches = []
    size = len(block)

    for i in range(size):
        idx1, val1 = block[i]
        for j in range(i + 1, size):
            idx2, val2 = block[j]

            score = fuzz.ratio(val1, val2)
            if score >= threshold:
                matches.append((idx1, idx2, score))

    return matches


def main():
    if len(sys.argv) != 3:
        print("Usage: python rapidfuzz_benchmark.py <parquet_file> <column_name>")
        sys.exit(1)

    parquet_file = sys.argv[1]
    column_name = sys.argv[2]

    print("Loading parquet file...")
    df = pd.read_parquet(parquet_file)

    if column_name not in df.columns:
        print(f"Column '{column_name}' not found.")
        sys.exit(1)

    values = df[column_name].dropna().astype(str).tolist()
    total_records = len(values)

    print(f"Loaded {total_records} records.")

    print("Blocking records...")
    blocks = block_records(values)

    print(f"Created {len(blocks)} blocks.")

    total_matches = 0
    start_time = time.time()

    print("Starting fuzzy matching...")

    for block in tqdm(blocks.values(), desc="Processing blocks"):
        matches = fuzzy_compare_block(block)
        total_matches += len(matches)

    elapsed = time.time() - start_time

    print("\n=== Benchmark Results ===")
    print(f"Records processed: {total_records}")
    print(f"Matches found: {total_matches}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Records/sec: {total_records / elapsed:.2f}")


if __name__ == "__main__":
    main()
