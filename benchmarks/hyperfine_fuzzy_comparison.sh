#!/bin/bash

INPUT="test_data/wikipedia_sample.parquet"
OUTPUT_DIR="benchmarks/output_fuzzy"
THRESHOLD=0.8
NUM_PERM=128

mkdir -p $OUTPUT_DIR

echo "Running fuzzy dedup benchmark comparison with hyperfine..."
echo "Parameters: threshold=$THRESHOLD, num_perm=$NUM_PERM"
echo ""

hyperfine \
    --warmup 0 \
    --runs 1 \
    --export-markdown benchmarks/results_fuzzy.md \
    --export-json benchmarks/results_fuzzy.json \
    "python3 benchmarks/baselines/pandas_fuzzy_dedup.py $INPUT $OUTPUT_DIR/pandas_fuzzy_out.parquet $THRESHOLD $NUM_PERM" \
    "python3 benchmarks/baselines/polars_fuzzy_dedup.py $INPUT $OUTPUT_DIR/polars_fuzzy_out.parquet $THRESHOLD $NUM_PERM" \
    "python3 benchmarks/baselines/streaming_fuzzy_dedup.py $INPUT $OUTPUT_DIR/streaming_fuzzy_out.parquet $THRESHOLD $NUM_PERM" \
    "cargo run --release --package dataset-dedup-cli -- fuzzy-dedup -i $INPUT -o $OUTPUT_DIR/rust_fuzzy_out.parquet -f text --threshold $THRESHOLD --num-perm $NUM_PERM"

echo ""
echo "Results saved to:"
echo "  - benchmarks/results_fuzzy.md"
echo "  - benchmarks/results_fuzzy.json"
