#!/bin/bash

INPUT_FILE="test_data/wikipedia_sample.parquet"
OUTPUT_DIR="benchmarks/output"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "fastdedup Benchmark"
echo "=========================================="
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
echo "1. Pandas Baseline"
echo "=========================================="
time python3 benchmarks/baselines/pandas_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/pandas_output.parquet"
echo ""

echo "=========================================="
echo "2. Polars Baseline"
echo "=========================================="
time python3 benchmarks/baselines/polars_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/polars_output.parquet"
echo ""

echo "=========================================="
echo "3. Streaming Python Baseline"
echo "=========================================="
time python3 benchmarks/baselines/streaming_dedup.py \
    "$INPUT_FILE" \
    "$OUTPUT_DIR/streaming_output.parquet"
echo ""

echo "=========================================="
echo "4. Rust fastdedup (YOUR TOOL)"
echo "=========================================="
time cargo run --release --package fastdedup-cli -- exact-dedup \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR/rust_output.parquet" \
    --field text
echo ""

echo "=========================================="
echo "Results Comparison"
echo "=========================================="
echo "Output file sizes:"
ls -lh "$OUTPUT_DIR"/*.parquet | awk '{print $9, $5}'
echo ""

echo "Verification (record counts should match):"
python3 << 'PYTHON'
import pandas as pd
import glob

for file in sorted(glob.glob("benchmarks/output/*.parquet")):
    df = pd.read_parquet(file)
    print(f"{file}: {len(df):,} records")
PYTHON
