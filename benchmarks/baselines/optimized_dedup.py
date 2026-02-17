#!/usr/bin/env python3
"""
optimized_dedup.py
==================
Fastest possible Python deduplication using:
  - Exact dedup:  xxhash (fastest non-cryptographic hash available in Python)
  - Fuzzy dedup:  MinHash + LSH via datasketch with multiprocessing to bypass the GIL
  - I/O:          PyArrow for zero-copy Parquet reads, polars for fast JSONL
  - Parallelism:  multiprocessing.Pool for CPU-bound MinHash computation
  - Progress:     rich library for a live terminal progress bar

Install dependencies:
    pip install xxhash datasketch pyarrow polars rich psutil

Usage:
    # Exact dedup
    python optimized_dedup.py exact --input data.parquet --output clean.parquet --field text

    # Fuzzy dedup
    python optimized_dedup.py fuzzy --input data.parquet --output clean.parquet \
                                    --field text --threshold 0.8 --num-perm 128

    # Both passes (exact first, then fuzzy on the result)
    python optimized_dedup.py both --input data.parquet --output clean.parquet --field text
"""

import argparse
import math
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ── Third-party imports (all must be installed) ──────────────────────────────
try:
    import xxhash                          # fastest hash in Python
    import pyarrow as pa                   # zero-copy columnar I/O
    import pyarrow.parquet as pq
    from datasketch import MinHash, MinHashLSH  # MinHash + LSH index
    from rich.console import Console       # pretty terminal output
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn, TaskProgressColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    import psutil                          # memory reporting
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install xxhash datasketch pyarrow polars rich psutil")
    sys.exit(1)

# ── Global console (used throughout) ─────────────────────────────────────────
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Number of worker processes for MinHash computation.
# Leave one core free so the main process stays responsive.
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# Chunk size sent to each worker process.
# Larger = less IPC overhead; smaller = finer progress granularity.
CHUNK_SIZE = 500

# Character n-gram size for MinHash shingling.
# 3-grams strike a good balance between speed and accuracy.
NGRAM_SIZE = 3


# ─────────────────────────────────────────────────────────────────────────────
#  I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(path: Path) -> str:
    """Return 'parquet' or 'jsonl' based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in (".jsonl", ".json", ".gz"):
        return "jsonl"
    raise ValueError(f"Unsupported file format: {suffix}")


def read_dataset(path: Path, field: str) -> Tuple[pa.Table, List[str]]:
    """
    Read a Parquet or JSONL file and return:
      - the full PyArrow Table (for writing back later)
      - a plain Python list of strings for the target field

    PyArrow's memory-mapped Parquet reader avoids copying data
    into Python objects until we explicitly access a column.
    """
    fmt = detect_format(path)

    if fmt == "parquet":
        # Use memory-mapping so the OS handles paging - avoids OOM on large files
        table = pq.read_table(str(path), memory_map=True)
    else:
        # Polars is the fastest JSONL reader available in Python
        import polars as pl
        if str(path).endswith(".gz"):
            df = pl.read_ndjson(path)          # polars handles gzip natively
        else:
            df = pl.read_ndjson(path)
        table = df.to_arrow()                  # convert to Arrow for uniform handling

    # Extract the text column as a Python list once (avoids repeated Arrow overhead)
    if field not in table.schema.names:
        raise KeyError(f"Field '{field}' not found. Available: {table.schema.names}")

    texts: List[str] = table.column(field).to_pylist()
    return table, texts


def write_dataset(table: pa.Table, mask: List[bool], path: Path) -> int:
    """
    Write rows where mask[i] is True to *path*.
    Uses PyArrow's boolean filter - zero Python-level row iteration.
    Returns the number of rows written.
    """
    # Convert mask to an Arrow BooleanArray for vectorised filtering
    bool_array = pa.array(mask, type=pa.bool_())
    filtered = table.filter(bool_array)

    fmt = detect_format(path)
    if fmt == "parquet":
        # snappy compression is fast and gives good ratio
        pq.write_table(filtered, str(path), compression="snappy")
    else:
        import polars as pl
        pl.from_arrow(filtered).write_ndjson(str(path))

    return len(filtered)


# ─────────────────────────────────────────────────────────────────────────────
#  EXACT DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def exact_dedup(
    table: pa.Table,
    texts: List[str],
    progress: Progress,
) -> List[bool]:
    """
    Mark duplicates using xxhash (128-bit) for speed.

    xxhash is ~4x faster than SHA-256 and has negligible collision
    probability at dataset scales. We use the 128-bit variant to be safe.

    Returns a boolean mask: True = keep, False = duplicate.
    """
    total = len(texts)
    seen: set = set()             # stores integer digests (ints hash faster in Python sets)
    mask: List[bool] = [False] * total
    duplicates = 0

    # Create a rich progress task
    task = progress.add_task(
        "[cyan]Exact dedup (xxhash-128)…",
        total=total,
    )

    for i, text in enumerate(texts):
        # xxhash intdigest returns an int directly - faster than hex string storage
        digest = xxhash.xxh128_intdigest(str(text).encode("utf-8", errors="replace"))

        if digest not in seen:
            seen.add(digest)
            mask[i] = True        # unique - keep
        else:
            duplicates += 1       # duplicate - discard

        # Update progress every 5000 records to avoid Rich overhead per-row
        if i % 5000 == 0:
            progress.update(task, completed=i)

    progress.update(task, completed=total)
    return mask, duplicates


# ─────────────────────────────────────────────────────────────────────────────
#  MINHASH WORKER (runs in a subprocess - bypasses the GIL)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_minhash_batch(args: Tuple) -> List[Tuple[int, List[int]]]:
    """
    Worker function executed in a subprocess pool.

    Receives a batch of (index, text) pairs and returns
    (index, minhash_hashvalues) pairs.

    Running in separate processes bypasses Python's GIL entirely,
    giving true CPU parallelism for the expensive MinHash computation.

    Note: datasketch MinHash objects are not picklable, so we return
    the raw hashvalues list and reconstruct the MinHash on the main process.
    """
    # Unpack args (multiprocessing requires a single argument)
    batch, num_perm, ngram_size = args

    results = []
    for idx, text in batch:
        # Build character n-gram set for shingling
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)   # normalise punctuation

        # Use a set comprehension - faster than a loop + set.add()
        shingles = {
            text[j: j + ngram_size]
            for j in range(max(1, len(text) - ngram_size + 1))
        }

        # Create MinHash and update with each shingle
        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode("utf-8"))   # encode once per shingle

        # Return raw hash values (a numpy array) - picklable across processes
        results.append((idx, m.hashvalues.tolist()))

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FUZZY DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_dedup(
    table: pa.Table,
    texts: List[str],
    threshold: float,
    num_perm: int,
    progress: Progress,
) -> Tuple[List[bool], int]:
    """
    Fuzzy deduplication using MinHash + LSH.

    Pipeline:
      1. Compute MinHash signatures in parallel (multiprocessing pool)
      2. Insert into LSH index sequentially (LSH is not thread-safe)
      3. Query LSH before inserting - if a match exists, mark as duplicate

    The parallelism in step 1 is what makes this fast: all CPU cores
    compute signatures simultaneously, bypassing the GIL.
    """
    total = len(texts)

    # ── Step 1: Parallel MinHash computation ─────────────────────────────────
    console.print(
        f"  [dim]Spawning {NUM_WORKERS} worker processes "
        f"(chunk size {CHUNK_SIZE})…[/dim]"
    )

    # Build list of (index, text) pairs for all records
    indexed_texts = list(enumerate(texts))

    # Split into chunks for the worker pool
    chunks = [
        (indexed_texts[i: i + CHUNK_SIZE], num_perm, NGRAM_SIZE)
        for i in range(0, total, CHUNK_SIZE)
    ]

    # Pre-allocate result storage (indexed by original row position)
    hashvalues_by_idx: dict = {}

    # Progress task for the parallel hashing phase
    hash_task = progress.add_task(
        "[magenta]Computing MinHash signatures (parallel)…",
        total=total,
    )

    # imap_unordered streams results back as workers finish - lower latency
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for batch_results in pool.imap_unordered(_compute_minhash_batch, chunks):
            for idx, hashvals in batch_results:
                hashvalues_by_idx[idx] = hashvals
            # Advance progress by the number of records in this batch
            progress.advance(hash_task, CHUNK_SIZE)

    progress.update(hash_task, completed=total)

    # ── Step 2 & 3: LSH insertion + duplicate detection ──────────────────────
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mask: List[bool] = [False] * total
    duplicates = 0

    lsh_task = progress.add_task(
        "[yellow]Building LSH index + querying…",
        total=total,
    )

    for i in range(total):
        # Reconstruct the MinHash object from the stored hash values
        m = MinHash(num_perm=num_perm, hashvalues=hashvalues_by_idx[i])

        # Query the index *before* inserting (like a seen-set check)
        matches = lsh.query(m)

        if matches:
            # At least one similar record already in the index → duplicate
            duplicates += 1
        else:
            # No match found → keep this record and add to index
            lsh.insert(str(i), m)   # key must be a string for datasketch
            mask[i] = True

        # Update progress every 2000 records
        if i % 2000 == 0:
            progress.update(lsh_task, completed=i)

    progress.update(lsh_task, completed=total)
    return mask, duplicates


# ─────────────────────────────────────────────────────────────────────────────
#  STATS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    mode: str,
    input_path: Path,
    output_path: Path,
    total: int,
    duplicates: int,
    elapsed: float,
    peak_mb: float,
) -> None:
    """Print a Rich-formatted summary table to the terminal."""
    kept = total - duplicates
    dup_pct = duplicates / total * 100 if total else 0
    throughput = total / elapsed if elapsed else 0

    # Build a Rich table for clean alignment
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    table.add_row("Mode",             mode)
    table.add_row("Input",            str(input_path))
    table.add_row("Output",           str(output_path))
    table.add_row("Total records",    f"{total:,}")
    table.add_row("Duplicates found", f"{duplicates:,}  ({dup_pct:.2f}%)")
    table.add_row("Final dataset",    f"{kept:,}")
    table.add_row("Elapsed time",     f"{elapsed:.2f}s")
    table.add_row("Throughput",       f"{throughput:,.0f} records/sec")
    table.add_row("Peak memory",      f"{peak_mb:.0f} MB")

    console.print(Panel(table, title="[bold green]Deduplication Complete[/bold green]",
                        border_style="green"))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_progress() -> Progress:
    """
    Build a Rich Progress bar with multiple columns.
    Displayed as a context manager in the calling function.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,   # 10 Hz refresh - smooth without CPU waste
    )


def run(args: argparse.Namespace) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)
    field       = args.field
    threshold   = getattr(args, "threshold", 0.8)
    num_perm    = getattr(args, "num_perm", 128)
    mode        = args.mode

    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Optimized Python Deduplication Baseline")
    console.print(f"  Mode      : [bold]{mode}[/bold]")
    console.print(f"  Input     : {input_path}  ({input_path.stat().st_size / 1e6:.1f} MB)")
    console.print(f"  Field     : {field}")
    if mode in ("fuzzy", "both"):
        console.print(f"  Threshold : {threshold}")
        console.print(f"  num_perm  : {num_perm}")
        console.print(f"  Workers   : {NUM_WORKERS}")
    console.print()

    # Track peak memory via psutil for a fair comparison with Rust
    process    = psutil.Process(os.getpid())
    start_mem  = process.memory_info().rss / 1e6    # MB
    wall_start = time.perf_counter()

    # ── Read dataset ──────────────────────────────────────────────────────────
    console.print("[bold]Reading dataset…[/bold]")
    t0 = time.perf_counter()
    table, texts = read_dataset(input_path, field)
    read_time = time.perf_counter() - t0
    console.print(f"  [green]✓[/green] {len(texts):,} records read in {read_time:.2f}s\n")

    total_duplicates = 0

    with build_progress() as progress:
        # ── Exact pass ───────────────────────────────────────────────────────
        if mode in ("exact", "both"):
            mask, dups = exact_dedup(table, texts, progress)
            total_duplicates += dups

            if mode == "both":
                # For the fuzzy pass, only pass texts that survived exact dedup
                # Rebuild table and texts from the mask before the fuzzy pass
                bool_array = pa.array(mask, type=pa.bool_())
                table = table.filter(bool_array)
                texts = table.column(field).to_pylist()
                console.print(
                    f"\n  [green]✓[/green] Exact pass complete: "
                    f"{dups:,} duplicates removed, "
                    f"{len(texts):,} records remain\n"
                )

        # ── Fuzzy pass ───────────────────────────────────────────────────────
        if mode in ("fuzzy", "both"):
            mask, dups = fuzzy_dedup(table, texts, threshold, num_perm, progress)
            total_duplicates += dups

    # ── Write output ──────────────────────────────────────────────────────────
    console.print("\n[bold]Writing output…[/bold]")
    t0 = time.perf_counter()
    written = write_dataset(table, mask, output_path)
    write_time = time.perf_counter() - t0
    console.print(f"  [green]✓[/green] {written:,} records written in {write_time:.2f}s\n")

    # ── Final stats ───────────────────────────────────────────────────────────
    elapsed  = time.perf_counter() - wall_start
    peak_mem = (process.memory_info().rss / 1e6) - start_mem

    print_summary(
        mode        = mode,
        input_path  = input_path,
        output_path = output_path,
        total       = len(texts) + total_duplicates,   # original total
        duplicates  = total_duplicates,
        elapsed     = elapsed,
        peak_mb     = max(peak_mem, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fastest possible Python deduplication baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["exact", "fuzzy", "both"],
        help="Deduplication mode",
    )
    parser.add_argument("--input",     required=True,  help="Input file (.parquet or .jsonl)")
    parser.add_argument("--output",    required=True,  help="Output file")
    parser.add_argument("--field",     default="text", help="Text field to deduplicate on")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for fuzzy dedup")
    parser.add_argument("--num-perm",  type=int,   default=128,
                        help="Number of MinHash permutations (more = more accurate, slower)")
    return parser


if __name__ == "__main__":
    # Required for multiprocessing on macOS (spawn start method)
    multiprocessing.set_start_method("spawn", force=True)
    args = build_parser().parse_args()
    run(args)
