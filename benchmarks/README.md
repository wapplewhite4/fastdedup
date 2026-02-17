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

### Exact Deduplication Benchmarks

#### Option 1: Full Comparison Suite

Runs all baselines and verifies results:

```bash
./benchmarks/run_comparison.sh
```

#### Option 2: Hyperfine (Cleaner Output)

Requires hyperfine installed:

```bash
./benchmarks/hyperfine_comparison.sh
```

This creates:
- `benchmarks/results.md` - Markdown table
- `benchmarks/results.json` - JSON results

#### Option 3: Individual Tests

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

### Fuzzy Deduplication Benchmarks (MinHash + LSH)

#### Option 1: Full Fuzzy Comparison

```bash
./benchmarks/run_fuzzy_comparison.sh
```

#### Option 2: Hyperfine Fuzzy Benchmark

```bash
./benchmarks/hyperfine_fuzzy_comparison.sh
```

#### Option 3: Individual Fuzzy Tests

```bash
# Pandas fuzzy dedup
python3 benchmarks/baselines/pandas_fuzzy_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output_fuzzy/pandas_out.parquet \
    0.8 128

# Polars fuzzy dedup
python3 benchmarks/baselines/polars_fuzzy_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output_fuzzy/polars_out.parquet \
    0.8 128

# Streaming fuzzy dedup
python3 benchmarks/baselines/streaming_fuzzy_dedup.py \
    test_data/wikipedia_sample.parquet \
    benchmarks/output_fuzzy/streaming_out.parquet \
    0.8 128

# Rust fuzzy dedup
cargo run --release --package dataset-dedup-cli -- fuzzy-dedup \
    --input test_data/wikipedia_sample.parquet \
    --output benchmarks/output_fuzzy/rust_out.parquet \
    --field text \
    --threshold 0.8 \
    --num-perm 128
```

## Visualizing Results

After running hyperfine benchmarks:

```bash
pip install matplotlib
python3 benchmarks/visualize_results.py
```

Creates `benchmarks/comparison_chart.png` with a bar chart.

## Expected Results

### Exact Deduplication (SHA256 hashing)

| Tool | Expected Time | Throughput | Notes |
|------|--------------|------------|-------|
| **Pandas** | 10-20s | ~10K-15K rec/s | In-memory, single-threaded |
| **Polars** | 7-10s | ~15K-20K rec/s | Lazy eval, Rust internals |
| **Streaming** | 9-12s | ~13K-17K rec/s | Memory-efficient |
| **Rust (yours)** | **3-5s** | **30K-50K rec/s** | **3-5x faster** |

### Fuzzy Deduplication (MinHash + LSH)

| Tool | Expected Time | Throughput | Notes |
|------|--------------|------------|-------|
| **Pandas** | 100-300s | ~500-1.5K rec/s | Python MinHash + LSH |
| **Polars** | 80-200s | ~800-2K rec/s | Still Python MinHash |
| **Streaming** | 90-250s | ~600-1.7K rec/s | Memory-efficient |
| **Rust (yours)** | **20-60s** | **2.5K-8K rec/s** | **5-10x faster** |

**Why fuzzy is slower:**
- MinHash computation is CPU-intensive (128 permutations)
- LSH index queries are O(log n) per record
- Python MinHash is pure Python, much slower than Rust

## File Structure

```
benchmarks/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_comparison.sh                  # Exact dedup benchmark suite
├── run_fuzzy_comparison.sh            # Fuzzy dedup benchmark suite
├── hyperfine_comparison.sh            # Hyperfine exact benchmark
├── hyperfine_fuzzy_comparison.sh      # Hyperfine fuzzy benchmark
├── visualize_results.py               # Create charts
├── baselines/
│   ├── pandas_dedup.py                # Pandas exact dedup
│   ├── polars_dedup.py                # Polars exact dedup
│   ├── streaming_dedup.py             # Streaming exact dedup
│   ├── pandas_fuzzy_dedup.py          # Pandas fuzzy dedup (MinHash)
│   ├── polars_fuzzy_dedup.py          # Polars fuzzy dedup (MinHash)
│   └── streaming_fuzzy_dedup.py       # Streaming fuzzy dedup (MinHash)
├── output/                            # Exact dedup output (generated)
│   ├── pandas_output.parquet
│   ├── polars_output.parquet
│   ├── streaming_output.parquet
│   └── rust_output.parquet
└── output_fuzzy/                      # Fuzzy dedup output (generated)
    ├── pandas_fuzzy_output.parquet
    ├── polars_fuzzy_output.parquet
    ├── streaming_fuzzy_output.parquet
    └── rust_fuzzy_output.parquet
```

## Troubleshooting

**Import errors:**
```bash
pip3 install pandas polars pyarrow datasketch
# Or from requirements.txt
pip3 install -r benchmarks/requirements.txt
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
