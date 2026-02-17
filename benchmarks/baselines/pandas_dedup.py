#!/usr/bin/env python3
"""
Baseline: Pandas deduplication
Most common approach developers use
"""
import pandas as pd
import hashlib
import time
import sys
from pathlib import Path

def exact_dedup_pandas(input_file, output_file, text_field='text'):
    """Exact deduplication using pandas"""
    print(f"Reading {input_file}...")
    start_read = time.time()

    # Read entire file into memory
    df = pd.read_parquet(input_file)
    read_time = time.time() - start_read

    total_records = len(df)
    print(f"Read {total_records:,} records in {read_time:.2f}s")

    # Hash the text field
    print("Computing hashes...")
    start_hash = time.time()
    df['hash'] = df[text_field].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    hash_time = time.time() - start_hash
    print(f"Hashed in {hash_time:.2f}s")

    # Drop duplicates
    print("Removing duplicates...")
    start_dedup = time.time()
    df_clean = df.drop_duplicates(subset=['hash'], keep='first')
    dedup_time = time.time() - start_dedup

    duplicates = total_records - len(df_clean)
    print(f"Found {duplicates:,} duplicates ({duplicates/total_records*100:.1f}%)")
    print(f"Deduplication took {dedup_time:.2f}s")

    # Write output
    print("Writing output...")
    start_write = time.time()
    df_clean.drop(columns=['hash']).to_parquet(output_file, index=False)
    write_time = time.time() - start_write
    print(f"Wrote {len(df_clean):,} records in {write_time:.2f}s")

    return {
        'total_records': total_records,
        'duplicates': duplicates,
        'final_count': len(df_clean),
        'read_time': read_time,
        'hash_time': hash_time,
        'dedup_time': dedup_time,
        'write_time': write_time,
        'total_time': read_time + hash_time + dedup_time + write_time
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python pandas_dedup.py <input.parquet> <output.parquet>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("=" * 60)
    print("Pandas Exact Deduplication Baseline")
    print("=" * 60)

    start = time.time()
    stats = exact_dedup_pandas(input_file, output_file)
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total records:     {stats['total_records']:,}")
    print(f"  Duplicates found:  {stats['duplicates']:,} ({stats['duplicates']/stats['total_records']*100:.1f}%)")
    print(f"  Final count:       {stats['final_count']:,}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Throughput:        {stats['total_records']/total_time:,.0f} records/sec")
    print("=" * 60)
