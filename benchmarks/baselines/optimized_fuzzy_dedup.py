#!/usr/bin/env python3
"""
Optimized Python fuzzy deduplication baseline.

Written as a fair comparison against the Rust implementation.
Fixes every major performance issue found in the naive baselines:

NAIVE BASELINE PROBLEMS:
  - iterrows(): converts every row to a Series object (~50-100x slower than
    direct array access)
  - No parallelism: single-threaded despite MinHash being embarrassingly parallel
  - Word unigrams: different algorithm than Rust (char 3-grams)
  - Pandas overhead: unnecessary DataFrame conversions in the hot loop

THIS IMPLEMENTATION:
  1. Parallel MinHash via multiprocessing.Pool (bypasses Python GIL; each
     worker process computes MinHash independently)
  2. Char 3-grams (matches Rust's default shingle method exactly)
  3. PyArrow-native I/O: reads batches directly as Arrow RecordBatches with
     no pandas conversion
  4. Compile regex once (not per-call)
  5. Pre-compute all signatures in parallel, then serial LSH pass
     (LSH query/insert is inherently sequential — order matters)
  6. Return only numpy hashvalues from workers (avoids pickling full
     MinHash objects across processes)

Usage:
  python optimized_fuzzy_dedup.py <input.parquet> <output.parquet> [threshold] [num_perm] [workers]
"""
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import itertools
import numpy as np
import time
import sys
import re
import os
from multiprocessing import Pool, cpu_count
from datasketch import MinHash, MinHashLSH


# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as m:ss or h:mm:ss."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _print_progress(done: int, total: int, elapsed: float, duplicates: int,
                    bar_width: int = 30) -> None:
    """
    Overwrite the current line with a compact progress bar.

      [######................]  35% | 3,500/10,000 | 0:23 elapsed | ETA 0:42 | 152 rec/s | 120 dupes
    """
    frac = done / total if total > 0 else 0
    filled = int(bar_width * frac)
    bar = '#' * filled + '.' * (bar_width - filled)

    rate = done / elapsed if elapsed > 0 else 0
    eta_str = _fmt_duration((total - done) / rate) if rate > 0 and done < total else '0:00'

    line = (
        f"\r  [{bar}] {frac * 100:5.1f}% | "
        f"{done:,}/{total:,} | "
        f"{_fmt_duration(elapsed)} elapsed | "
        f"ETA {eta_str} | "
        f"{rate:,.0f} rec/s | "
        f"{duplicates:,} dupes"
    )
    sys.stdout.write(line)
    sys.stdout.flush()

# Compile regexes once at module level (shared across all calls in a process)
_PUNCT_RE = re.compile(r'[^a-z0-9\s]')
_SPACE_RE = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def _compute_char3_minhash(args: tuple) -> np.ndarray:
    """
    Worker function: compute MinHash from character 3-grams.

    Returns a numpy array of hashvalues (not the MinHash object) so that
    pickling back to the main process is cheap (~1KB per record at num_perm=128).

    We deduplicate shingles with set() before hashing: MinHash estimates the
    Jaccard of *sets*, so duplicate shingles add zero information but cost a
    full SHA-1 + 128-permutation update each. Wikipedia articles typically have
    40-70% unique 3-grams, so this cuts hashing work by 30-60%.
    """
    text, num_perm = args
    normalized = normalize_text(text if text else '')
    m = MinHash(num_perm=num_perm)
    n = len(normalized)
    if n >= 3:
        # set() deduplication: identical 3-grams are no-ops for min-hash
        # (min(h, h) == h) so skipping them is always correct and much faster.
        for s in set(normalized[i:i + 3] for i in range(n - 2)):
            m.update(s.encode('utf8'))
    elif n > 0:
        # Text shorter than shingle size: hash the whole string
        m.update(normalized.encode('utf8'))
    return m.hashvalues.copy()


def optimized_fuzzy_dedup(
    input_file: str,
    output_file: str,
    text_field: str = 'text',
    threshold: float = 0.8,
    num_perm: int = 128,
    batch_size: int = 4000,
    num_workers: int = None,
) -> dict:
    """
    Fuzzy deduplication with parallel MinHash + serial LSH.

    The parallelizable work (text normalization + MinHash computation) runs
    across all CPU cores. The inherently sequential work (LSH query/insert,
    where each result depends on all prior inserts) runs in the main process.
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"  Workers:    {num_workers} (of {cpu_count()} logical CPUs)")
    print(f"  Shingles:   char 3-grams (matches Rust implementation)")
    print(f"  Threshold:  {threshold}")
    print(f"  num_perm:   {num_perm}")
    print(f"  Batch size: {batch_size:,}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    total_records = 0
    duplicates = 0
    record_id = 0
    output_batches = []

    start = time.time()
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows  # known upfront from Parquet metadata
    print(f"  Total rows: {total_rows:,}")

    _spinner = itertools.cycle('|/-\\')

    # Reuse a single MinHash object for LSH queries/inserts in the serial phase.
    # datasketch.MinHashLSH.insert() converts band slices to bytes (a copy) so
    # it does NOT hold a reference to the numpy array — reuse is safe.
    _query_minhash = MinHash(num_perm=num_perm)

    with Pool(processes=num_workers) as pool:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            n = len(batch)

            # Extract text column as a Python list directly from Arrow
            # (no pandas conversion, no Series overhead)
            text_col = batch.column(text_field)
            texts = text_col.to_pylist()

            # --- PARALLEL PHASE ---
            # map_async returns immediately so we can spin while workers run.
            # This keeps the terminal alive during the slow first-batch startup
            # (worker process spawning + module imports can take 10-30 seconds).
            worker_args = [(t if t is not None else '', num_perm) for t in texts]
            # chunksize: send tasks in groups to reduce IPC round-trips.
            # Default (None) recomputes this every call; being explicit avoids
            # the overhead of repeated len() + division on the iterable.
            chunksize = max(1, len(worker_args) // (num_workers * 4))
            async_result = pool.map_async(_compute_char3_minhash, worker_args,
                                          chunksize=chunksize)

            while not async_result.ready():
                elapsed = time.time() - start
                rate = total_records / elapsed if elapsed > 0 and total_records > 0 else 0
                eta_str = (
                    _fmt_duration((total_rows - total_records) / rate)
                    if rate > 0 else '?:??'
                )
                sys.stdout.write(
                    f"\r  {next(_spinner)} Hashing...  "
                    f"{total_records:,}/{total_rows:,} | "
                    f"{_fmt_duration(elapsed)} elapsed | "
                    f"ETA {eta_str} | "
                    f"{rate:,.0f} rec/s | "
                    f"{duplicates:,} dupes"
                )
                sys.stdout.flush()
                async_result.wait(timeout=0.12)

            all_hashvalues = async_result.get()
            total_records += n

            # --- SERIAL PHASE ---
            # LSH query/insert must be sequential: the decision for record N
            # depends on all records 0..N-1 already being in the index.
            # Reuse _query_minhash (created once before the batch loop) to
            # avoid 4,000 MinHash allocations per batch.
            keep_mask = []
            for hashvalues in all_hashvalues:
                _query_minhash.hashvalues = hashvalues

                candidates = lsh.query(_query_minhash)
                if candidates:
                    keep_mask.append(False)
                    duplicates += 1
                else:
                    lsh.insert(str(record_id), _query_minhash)
                    keep_mask.append(True)
                record_id += 1

            # Filter the Arrow RecordBatch without converting to pandas
            mask = pa.chunked_array([pa.array(keep_mask)])
            filtered = batch.filter(mask)
            if len(filtered) > 0:
                output_batches.append(filtered)

            # Full progress bar after each completed batch
            _print_progress(total_records, total_rows, time.time() - start, duplicates)

    # Move to next line after the progress bar
    print()

    # Write output as Parquet
    write_start = time.time()
    if output_batches:
        schema = output_batches[0].schema
        with pq.ParquetWriter(output_file, schema) as writer:
            for b in output_batches:
                writer.write_batch(b)
    write_time = time.time() - write_start

    total_time = time.time() - start

    return {
        'total_records': total_records,
        'duplicates': duplicates,
        'final_count': total_records - duplicates,
        'total_time': total_time,
        'write_time': write_time,
        'throughput': total_records / total_time if total_time > 0 else 0,
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python optimized_fuzzy_dedup.py "
              "<input.parquet> <output.parquet> [threshold] [num_perm] [workers]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    num_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else None

    print("=" * 60)
    print("Optimized Python Fuzzy Deduplication (MinHash + LSH)")
    print("=" * 60)

    start = time.time()
    stats = optimized_fuzzy_dedup(
        input_file, output_file,
        threshold=threshold,
        num_perm=num_perm,
        num_workers=workers,
    )
    total_time = time.time() - start

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total records:    {stats['total_records']:,}")
    print(f"  Duplicates found: {stats['duplicates']:,} "
          f"({stats['duplicates'] / stats['total_records'] * 100:.1f}%)")
    print(f"  Final dataset:    {stats['final_count']:,} "
          f"({stats['final_count'] / stats['total_records'] * 100:.1f}%)")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Throughput:       {stats['throughput']:,.0f} records/sec")
    print("=" * 60)
