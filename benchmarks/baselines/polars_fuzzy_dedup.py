#!/usr/bin/env python3
"""
Baseline: Polars fuzzy deduplication using MinHash + LSH
Faster alternative to pandas (but still Python MinHash)
"""
import polars as pl
import time
import sys
import re
from datasketch import MinHash, MinHashLSH

def normalize_text(text):
    """Normalize text for comparison"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text"""
    tokens = text.split()
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode('utf8'))
    return m

def fuzzy_dedup_polars(input_file, output_file, text_field='text', threshold=0.8, num_perm=128):
    """Fuzzy deduplication using polars + datasketch"""
    print(f"Reading {input_file}...")
    start = time.time()

    # Read file
    df = pl.read_parquet(input_file)
    total_records = len(df)
    print(f"Read {total_records:,} records")

    # Normalize text
    print("Normalizing text...")
    df = df.with_columns([
        pl.col(text_field)
        .map_elements(normalize_text, return_dtype=pl.Utf8)
        .alias('normalized')
    ])

    # Create MinHash signatures
    print(f"Computing MinHash signatures (num_perm={num_perm})...")
    df = df.with_columns([
        pl.col('normalized')
        .map_elements(lambda x: create_minhash(x, num_perm), return_dtype=pl.Object)
        .alias('minhash')
    ])

    # Build LSH index
    print(f"Building LSH index (threshold={threshold})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    duplicates_found = 0
    keep_indices = []

    # Convert to Python for LSH processing
    data = df.to_dicts()

    for idx, row in enumerate(data):
        minhash = row['minhash']
        # Query for similar items
        result = lsh.query(minhash)

        if len(result) == 0:
            # No duplicates found, keep this record
            lsh.insert(str(idx), minhash)
            keep_indices.append(idx)
        else:
            # Duplicate found, skip this record
            duplicates_found += 1

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1:,} records, {duplicates_found:,} duplicates found...")

    print(f"Found {duplicates_found:,} duplicates ({duplicates_found/total_records*100:.1f}%)")

    # Filter to keep only non-duplicates
    print("Filtering dataset...")
    df_clean = df.select([col for col in df.columns if col not in ['normalized', 'minhash']])
    df_clean = df_clean[keep_indices]

    # Write output
    print("Writing output...")
    df_clean.write_parquet(output_file)

    total_time = time.time() - start

    return {
        'total_records': total_records,
        'duplicates': duplicates_found,
        'final_count': len(df_clean),
        'total_time': total_time
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python polars_fuzzy_dedup.py <input.parquet> <output.parquet> [threshold] [num_perm]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    num_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    print("=" * 60)
    print("Polars Fuzzy Deduplication Baseline (MinHash + LSH)")
    print("=" * 60)

    start = time.time()
    stats = fuzzy_dedup_polars(input_file, output_file, threshold=threshold, num_perm=num_perm)
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total records:     {stats['total_records']:,}")
    print(f"  Duplicates found:  {stats['duplicates']:,} ({stats['duplicates']/stats['total_records']*100:.1f}%)")
    print(f"  Final count:       {stats['final_count']:,}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Throughput:        {stats['total_records']/total_time:,.0f} records/sec")
    print("=" * 60)
