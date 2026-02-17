#!/bin/bash

INPUT="test_data/wikipedia_sample.parquet"
OUTPUT_DIR="benchmarks/output"

mkdir -p $OUTPUT_DIR

echo "Running benchmark comparison with hyperfine..."
echo ""

hyperfine \
    --warmup 1 \
    --export-markdown benchmarks/results.md \
    --export-json benchmarks/results.json \
    "python3 benchmarks/baselines/pandas_dedup.py $INPUT $OUTPUT_DIR/pandas_out.parquet" \
    "python3 benchmarks/baselines/polars_dedup.py $INPUT $OUTPUT_DIR/polars_out.parquet" \
    "python3 benchmarks/baselines/streaming_dedup.py $INPUT $OUTPUT_DIR/streaming_out.parquet" \
    "cargo run --release --package dataset-dedup-cli -- exact-dedup -i $INPUT -o $OUTPUT_DIR/rust_out.parquet -f text"

echo ""
echo "Results saved to:"
echo "  - benchmarks/results.md"
echo "  - benchmarks/results.json"
