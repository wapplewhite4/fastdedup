# Deduplication Benchmarks

Comparing Rust dataset-dedup against Python baselines.

## Setup

### Install Python Dependencies

```bash
# Option 1: Global install
pip3 install -r benchmarks/requirements.txt

# Option 2: Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r benchmarks/requirements.txt
```

### Install Hyperfine (Optional, for cleaner benchmarks)

```bash
# macOS
brew install hyperfine

# Linux
cargo install hyperfine
# or
apt install hyperfine  # Debian/Ubuntu
```

## Running Benchmarks

### Option 1: Full Comparison Suite

Runs all baselines and verifies results:

```bash
./benchmarks/run_comparison.sh
```

### Option 2: Hyperfine (Cleaner Output)

Requires hyperfine installed:

```bash
./benchmarks/hyperfine_comparison.sh
```

This creates:
- `benchmarks/results.md` - Markdown table
- `benchmarks/results.json` - JSON results

### Option 3: Individual Tests

```bash
# Pandas baseline
python3 benchmarks/baselines/pandas_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output/pandas_out.parquet

# Polars baseline
python3 benchmarks/baselines/polars_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output/polars_out.parquet

# Streaming baseline
python3 benchmarks/baselines/streaming_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output/streaming_out.parquet

# Rust tool (exact dedup)
cargo run --release --package dataset-dedup-cli -- exact-dedup \
    --input test_data/wikipedia_sample.parquet \
    --output benchmarks/output/rust_out.parquet \
    --field text
```

## Visualizing Results

After running hyperfine benchmarks:

```bash
pip install matplotlib
python3 benchmarks/visualize_results.py
```

Creates `benchmarks/comparison_chart.png` with a bar chart.

## Expected Results

**Note:** These are for exact deduplication only (SHA256 hashing).

| Tool | Expected Time | Throughput | Notes |
|------|--------------|------------|-------|
| **Pandas** | 45-90s | ~2K-3K rec/s | In-memory, single-threaded |
| **Polars** | 15-30s | ~5K-10K rec/s | Lazy eval, Rust internals |
| **Streaming** | 30-60s | ~3K-5K rec/s | Memory-efficient |
| **Rust (yours)** | **3-10s** | **15K-50K rec/s** | **10-20x faster** |

**For fuzzy deduplication (MinHash + LSH):**
- Python implementations would be 5-10x slower
- Your Rust tool maintains similar performance (581 rec/s on 157K dataset)

## File Structure

```
benchmarks/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_comparison.sh                  # Full benchmark suite
├── hyperfine_comparison.sh            # Hyperfine benchmark
├── visualize_results.py               # Create charts
├── baselines/
│   ├── pandas_dedup.py                # Pandas baseline
│   ├── polars_dedup.py                # Polars baseline
│   └── streaming_dedup.py             # Streaming baseline
└── output/                            # Output files (generated)
    ├── pandas_output.parquet
    ├── polars_output.parquet
    ├── streaming_output.parquet
    └── rust_output.parquet
```

## Troubleshooting

**Import errors:**
```bash
pip3 install pandas polars pyarrow
```

**Permission denied:**
```bash
chmod +x benchmarks/*.sh
```

**Hyperfine not found:**
```bash
brew install hyperfine  # macOS
cargo install hyperfine # Cross-platform
```
