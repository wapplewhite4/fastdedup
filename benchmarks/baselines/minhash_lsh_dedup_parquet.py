#!/usr/bin/env python3
"""
MinHash + LSH dedup baseline for a Parquet file (Python).

This script is designed to be apples-to-apples with a Rust pipeline that:
- extracts a single named field
- shingles it (character trigrams by default)
- computes a MinHash signature (128 permutations)
- uses LSH banding (16 bands x 8 rows)
- verifies candidate pairs using MinHash signature Jaccard similarity
- reports only summary stats (total, unique, duplicates removed)

Important notes:
- We verify duplicates using MinHash signature Jaccard (same notion as comparing hash-value agreement),
  NOT true Jaccard over shingles. This matches your Rust description.
- We process records in-order and treat a record as a duplicate if it matches ANY earlier kept record.
  This mimics typical “dedup keeps first occurrence” behavior.

Usage:
  python minhash_lsh_dedup_parquet.py --parquet wikipedia.parquet --field title

Common tweaks:
  --threshold 0.8
  --shingle-size 3
  --word-shingles   (if you ever switch your Rust tool to word shingles)
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def normalize_text(s: str) -> str:
    """
    Normalize text before shingling.

    Your Rust tool may or may not normalize. If it doesn't, you should remove
    or reduce normalization here for a fair comparison.

    Current normalization:
      - strip outer whitespace
      - casefold (stronger than lower() for Unicode)
      - collapse internal whitespace runs to single spaces
    """
    s = s.strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s


def char_ngrams(s: str, n: int) -> Iterable[str]:
    """
    Generate character n-grams (shingles) from a string.

    Example: n=3, "abcd" -> "abc", "bcd"
    """
    if len(s) < n:
        return []
    return (s[i : i + n] for i in range(len(s) - n + 1))


def word_ngrams(s: str, n: int) -> Iterable[str]:
    """
    Generate word n-grams from whitespace-tokenized text.

    If your Rust tool uses character trigrams by default, you likely do NOT want this.
    """
    toks = s.split()
    if len(toks) < n:
        return []
    return (" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def build_minhash(
    text: str,
    num_perm: int,
    shingle_size: int,
    word_shingles: bool,
) -> Optional[MinHash]:
    """
    Build a MinHash signature for the given text.

    Returns None if there are no shingles (e.g., empty/too-short input).
    """
    if not text:
        return None

    m = MinHash(num_perm=num_perm)

    shingles = word_ngrams(text, shingle_size) if word_shingles else char_ngrams(text, shingle_size)

    count = 0
    for sh in shingles:
        # MinHash expects bytes
        m.update(sh.encode("utf-8", errors="ignore"))
        count += 1

    if count == 0:
        return None

    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to input parquet file")
    ap.add_argument("--field", required=True, help="Column name to dedup on (single field)")
    ap.add_argument("--threshold", type=float, default=0.8, help="MinHash signature Jaccard threshold")
    ap.add_argument("--num-perm", type=int, default=128, help="Number of MinHash permutations (hash functions)")
    ap.add_argument("--bands", type=int, default=16, help="LSH bands (b)")
    ap.add_argument("--rows", type=int, default=8, help="LSH rows per band (r). b*r should equal num_perm.")
    ap.add_argument("--shingle-size", type=int, default=3, help="Shingle size (3 = character trigrams)")
    ap.add_argument("--word-shingles", action="store_true", help="Use word shingles instead of character shingles")
    ap.add_argument("--no-normalize", action="store_true", help="Disable normalization to match raw behavior")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional: limit number of rows read (debug)")
    args = ap.parse_args()

    if args.bands * args.rows != args.num_perm:
        raise SystemExit(
            f"Invalid LSH params: bands*rows must equal num_perm "
            f"({args.bands}*{args.rows} != {args.num_perm})."
        )

    # Read only the one column we care about.
    df = pd.read_parquet(args.parquet, columns=[args.field])
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    values = df[args.field].dropna().astype(str).tolist()
    total = len(values)

    # LSH index; params=(b,r) forces 16x8 banding like your Rust tool.
    lsh = MinHashLSH(num_perm=args.num_perm, params=(args.bands, args.rows))

    # Store signatures for verification (so we can do exact signature-jaccard like Rust).
    signatures: List[Optional[MinHash]] = [None] * total

    # Keep flags: whether record is considered unique (kept) or a duplicate (dropped).
    kept = [False] * total

    # 1) Build MinHash signatures (progress bar)
    for i in tqdm(range(total), desc="Building MinHash", unit="rec"):
        text = values[i]
        if not args.no_normalize:
            text = normalize_text(text)

        signatures[i] = build_minhash(
            text=text,
            num_perm=args.num_perm,
            shingle_size=args.shingle_size,
            word_shingles=args.word_shingles,
        )

    # 2) Dedup pass: query LSH for candidates, verify with signature Jaccard, then insert if kept
    duplicates_removed = 0
    unique_count = 0

    for i in tqdm(range(total), desc="LSH dedup pass", unit="rec"):
        sig = signatures[i]

        # If no signature (empty/too short), treat as unique (or choose to drop—your call)
        if sig is None:
            kept[i] = True
            unique_count += 1
            continue

        # Query for candidate matches among previously inserted (kept) records
        candidates = lsh.query(sig)

        is_dup = False
        for cand_key in candidates:
            j = int(cand_key)
            cand_sig = signatures[j]
            if cand_sig is None:
                continue

            # This is MinHash signature Jaccard (agreement fraction), matching your Rust description.
            sim = sig.jaccard(cand_sig)
            if sim >= args.threshold:
                is_dup = True
                break

        if is_dup:
            duplicates_removed += 1
            kept[i] = False
            continue

        # Keep: insert into LSH for future comparisons
        lsh.insert(str(i), sig)
        kept[i] = True
        unique_count += 1

    print("\n=== Summary ===")
    print(f"Total records (non-null {args.field}): {total}")
    print(f"Unique kept: {unique_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Threshold: {args.threshold}")
    print(f"MinHash num_perm: {args.num_perm}")
    print(f"LSH bands x rows: {args.bands} x {args.rows}")
    print(f"Shingle mode: {'word' if args.word_shingles else 'char'}")
    print(f"Shingle size: {args.shingle_size}")
    print(f"Normalization: {'off' if args.no_normalize else 'on'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
