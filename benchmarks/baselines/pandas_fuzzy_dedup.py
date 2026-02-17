#!/usr/bin/env python3
"""
Baseline: Pandas fuzzy deduplication using MinHash + LSH
Most common approach for fuzzy dedup in Python
"""
import pandas as pd
import time
import sys
import re
from datasketch import MinHash, MinHashLSH


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _print_progress(done: int, total: int, elapsed: float, duplicates: int,
                    bar_width: int = 30) -> None:
    frac = done / total if total > 0 else 0
    filled = int(bar_width * frac)
    bar = '#' * filled + '.' * (bar_width - filled)
    rate = done / elapsed if elapsed > 0 else 0
    eta_str = _fmt_duration((total - done) / rate) if rate > 0 and done < total else '0:00'
    sys.stdout.write(
        f"\r  [{bar}] {frac * 100:5.1f}% | "
        f"{done:,}/{total:,} | "
        f"{_fmt_duration(elapsed)} elapsed | "
        f"ETA {eta_str} | "
        f"{rate:,.0f} rec/s | "
        f"{duplicates:,} dupes"
    )
    sys.stdout.flush()

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

def fuzzy_dedup_pandas(input_file, output_file, text_field='text', threshold=0.8, num_perm=128):
    """Fuzzy deduplication using pandas + datasketch"""
    print(f"Reading {input_file}...")
    start_read = time.time()

    # Read entire file into memory
    df = pd.read_parquet(input_file)
    read_time = time.time() - start_read

    total_records = len(df)
    print(f"Read {total_records:,} records in {read_time:.2f}s")

    # Normalize text
    print("Normalizing text...")
    start_normalize = time.time()
    df['normalized'] = df[text_field].apply(normalize_text)
    normalize_time = time.time() - start_normalize
    print(f"Normalized in {normalize_time:.2f}s")

    # Create MinHash signatures
    print(f"Computing MinHash signatures (num_perm={num_perm})...")
    start_minhash = time.time()
    df['minhash'] = df['normalized'].apply(lambda x: create_minhash(x, num_perm))
    minhash_time = time.time() - start_minhash
    print(f"MinHash computed in {minhash_time:.2f}s")

    # Build LSH index
    print(f"Building LSH index (threshold={threshold})...")
    start_lsh = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    duplicates_found = 0
    keep_indices = []

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        minhash = row['minhash']
        result = lsh.query(minhash)

        if len(result) == 0:
            lsh.insert(str(idx), minhash)
            keep_indices.append(idx)
        else:
            duplicates_found += 1

        if i % 500 == 0 or i == total_records:
            _print_progress(i, total_records, time.time() - start_lsh, duplicates_found)

    print()  # newline after progress bar
    lsh_time = time.time() - start_lsh
    print(f"LSH deduplication took {lsh_time:.2f}s")
    print(f"Found {duplicates_found:,} duplicates ({duplicates_found/total_records*100:.1f}%)")

    # Filter to keep only non-duplicates
    print("Filtering dataset...")
    start_filter = time.time()
    df_clean = df.loc[keep_indices].drop(columns=['normalized', 'minhash'])
    filter_time = time.time() - start_filter
    print(f"Filtered in {filter_time:.2f}s")

    # Write output
    print("Writing output...")
    start_write = time.time()
    df_clean.to_parquet(output_file, index=False)
    write_time = time.time() - start_write
    print(f"Wrote {len(df_clean):,} records in {write_time:.2f}s")

    return {
        'total_records': total_records,
        'duplicates': duplicates_found,
        'final_count': len(df_clean),
        'read_time': read_time,
        'normalize_time': normalize_time,
        'minhash_time': minhash_time,
        'lsh_time': lsh_time,
        'filter_time': filter_time,
        'write_time': write_time,
        'total_time': read_time + normalize_time + minhash_time + lsh_time + filter_time + write_time
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python pandas_fuzzy_dedup.py <input.parquet> <output.parquet> [threshold] [num_perm]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    num_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    print("=" * 60)
    print("Pandas Fuzzy Deduplication Baseline (MinHash + LSH)")
    print("=" * 60)

    start = time.time()
    stats = fuzzy_dedup_pandas(input_file, output_file, threshold=threshold, num_perm=num_perm)
    total_time = time.time() - start

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total records:     {stats['total_records']:,}")
    print(f"  Duplicates found:  {stats['duplicates']:,} ({stats['duplicates']/stats['total_records']*100:.1f}%)")
    print(f"  Final count:       {stats['final_count']:,}")
    print(f"  Read time:         {stats['read_time']:.2f}s")
    print(f"  Normalize time:    {stats['normalize_time']:.2f}s")
    print(f"  MinHash time:      {stats['minhash_time']:.2f}s")
    print(f"  LSH time:          {stats['lsh_time']:.2f}s")
    print(f"  Filter time:       {stats['filter_time']:.2f}s")
    print(f"  Write time:        {stats['write_time']:.2f}s")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Throughput:        {stats['total_records']/total_time:,.0f} records/sec")
    print("=" * 60)
