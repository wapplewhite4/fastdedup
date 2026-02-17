#!/usr/bin/env python3
"""
Baseline: Polars deduplication
Faster alternative to pandas (uses Rust under the hood)
"""
import polars as pl
import hashlib
import time
import sys

def exact_dedup_polars(input_file, output_file, text_field='text'):
    """Exact deduplication using polars (lazy evaluation)"""
    print(f"Reading {input_file}...")
    start = time.time()

    # Read with lazy evaluation
    df = pl.scan_parquet(input_file)

    # Add hash column and deduplicate
    df_clean = (
        df
        .with_columns([
            pl.col(text_field)
            .map_elements(lambda x: hashlib.sha256(str(x).encode()).hexdigest(), return_dtype=pl.Utf8)
            .alias('hash')
        ])
        .unique(subset=['hash'], keep='first')
        .drop('hash')
    )

    # Collect (execute the lazy query)
    result = df_clean.collect()

    total_time = time.time() - start

    # Write output
    result.write_parquet(output_file)

    return {
        'final_count': len(result),
        'total_time': total_time
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python polars_dedup.py <input.parquet> <output.parquet>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("=" * 60)
    print("Polars Exact Deduplication Baseline")
    print("=" * 60)

    start = time.time()
    stats = exact_dedup_polars(input_file, output_file)
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print(f"Completed in {total_time:.2f}s")
    print(f"Final count: {stats['final_count']:,}")
    print("=" * 60)
