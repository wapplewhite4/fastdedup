#!/bin/bash

INPUT_FILE="test_data/wikipedia_sample.parquet"
OUTPUT_DIR="benchmarks/output_fuzzy"
THRESHOLD=0.8
NUM_PERM=128

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Fuzzy Deduplication Benchmark"
echo "=========================================="
echo "Parameters:"
echo "  Threshold:  $THRESHOLD"
echo "  Num Perm:   $NUM_PERM"
echo ""

# Check file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Get file info
FILE_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
echo "Input file: $INPUT_FILE ($FILE_SIZE)"
echo ""

echo "=========================================="
echo "1. Pandas Fuzzy Baseline (MinHash + LSH)"
echo "=========================================="
time python3 benchmarks/baselines/pandas_fuzzy_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/pandas_fuzzy_output.parquet" \
    $THRESHOLD \
    $NUM_PERM
echo ""

echo "=========================================="
echo "2. Polars Fuzzy Baseline (MinHash + LSH)"
echo "=========================================="
time python3 benchmarks/baselines/polars_fuzzy_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/polars_fuzzy_output.parquet" \
    $THRESHOLD \
    $NUM_PERM
echo ""

echo "=========================================="
echo "3. Streaming Python Fuzzy Baseline"
echo "=========================================="
time python3 benchmarks/baselines/streaming_fuzzy_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/streaming_fuzzy_output.parquet" \
    $THRESHOLD \
    $NUM_PERM
echo ""

echo "=========================================="
echo "4. Rust Dataset-Dedup Fuzzy (YOUR TOOL)"
echo "=========================================="
time cargo run --release --package dataset-dedup-cli -- fuzzy-dedup \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR/rust_fuzzy_output.parquet" \
    --field text \
    --threshold $THRESHOLD \
    --num-perm $NUM_PERM
echo ""

echo "=========================================="
echo "Results Comparison"
echo "=========================================="
echo "Output file sizes:"
ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null | awk '{print $9, $5}' || echo "No output files found"
echo ""

echo "Verification (record counts):"
python3 << 'PYTHON'
import pandas as pd
import glob

files = sorted(glob.glob("benchmarks/output_fuzzy/*.parquet"))
if files:
    for file in files:
        try:
            df = pd.read_parquet(file)
            print(f"{file}: {len(df):,} records")
        except Exception as e:
            print(f"{file}: ERROR - {e}")
else:
    print("No output files found")
PYTHON

echo ""
echo "=========================================="
echo "Performance Summary"
echo "=========================================="
echo "Expected results:"
echo "  Pandas:    Very slow (100-300s)"
echo "  Polars:    Slow (80-200s)"
echo "  Streaming: Slow (90-250s)"
echo "  Rust:      Fast (20-60s) - 5-10x faster!"
echo "=========================================="
