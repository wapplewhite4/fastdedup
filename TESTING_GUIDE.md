# Dataset Deduplication - Testing Guide

This guide shows you how to actually deduplicate datasets using the tool.

## Quick Start (Automated)

Run the automated test script:

```bash
cd /home/user/data-dedup
./test_dedup.sh
```

This will:
1. Generate a test dataset with various types of duplicates
2. Run all three deduplication modes
3. Show you the results

## Manual Testing

### 1. Generate Test Data

```bash
# Generate test dataset
cargo run --example generate_test_data /tmp/test_dataset.jsonl
```

This creates a JSONL file with:
- Exact duplicates
- Near duplicates
- Case variations
- Punctuation variations
- Unique records

### 2. Run Deduplication

**Exact Deduplication** (byte-for-byte matching):
```bash
cargo run --release --example dedupe_file /tmp/test_dataset.jsonl exact
```

**Fuzzy Deduplication** (MinHash/LSH, 70% similarity threshold):
```bash
cargo run --release --example dedupe_file /tmp/test_dataset.jsonl fuzzy
```

**Exact with Normalization** (case-insensitive, whitespace-normalized):
```bash
cargo run --release --example dedupe_file /tmp/test_dataset.jsonl exact-normalized
```

### 3. Check Results

```bash
# Compare record counts
wc -l /tmp/test_dataset.jsonl
wc -l /tmp/test_dataset.jsonl.deduped.jsonl

# View deduplicated output
cat /tmp/test_dataset.jsonl.deduped.jsonl
```

## Using Your Own Data

### Prepare Your Data

Your data should be in JSONL format (one JSON object per line):

```jsonl
{"id": 1, "text": "Your text content here"}
{"id": 2, "text": "Another document"}
{"id": 3, "text": "More content"}
```

### Run Deduplication

```bash
# Replace /path/to/your/data.jsonl with your actual file
cargo run --release --example dedupe_file /path/to/your/data.jsonl fuzzy
```

Output will be written to: `/path/to/your/data.jsonl.deduped.jsonl`

## Understanding the Modes

### Exact Mode
- **Use when:** You want to remove byte-for-byte identical records
- **Speed:** Fastest (~500K records/sec)
- **Strictness:** Most strict
- **Example:**
  - ✓ Removes: "Hello World" vs "Hello World"
  - ✗ Keeps: "Hello World" vs "hello world"

### Exact-Normalized Mode
- **Use when:** You want case-insensitive deduplication
- **Speed:** Fast (~400K records/sec)
- **Strictness:** Medium
- **Example:**
  - ✓ Removes: "Hello World" vs "hello world"
  - ✓ Removes: "  Hello  " vs "Hello"
  - ✗ Keeps: "Hello World" vs "Hello World!"

### Fuzzy Mode
- **Use when:** You want to catch near-duplicates
- **Speed:** Moderate (~10K records/sec)
- **Strictness:** Least strict (configurable threshold)
- **Example:**
  - ✓ Removes: "The quick brown fox" vs "The quick brown foxes"
  - ✓ Removes: "Hello World" vs "Hello, World!"
  - ✗ Keeps: "Hello World" vs "Goodbye World" (too different)

## Advanced: Customizing the Tool

Edit `examples/dedupe_file.rs` to customize:

### Change Similarity Threshold

```rust
// In fuzzy_dedup() function, change:
let mut dedup = FuzzyDeduplicator::new(0.7);  // 70% similarity

// To:
let mut dedup = FuzzyDeduplicator::new(0.9);  // 90% similarity (stricter)
// or
let mut dedup = FuzzyDeduplicator::new(0.5);  // 50% similarity (looser)
```

### Change Text Field

```rust
// If your JSON has a different field name:
let mut dedup = FuzzyDeduplicator::with_config(FuzzyDedupConfig {
    text_field: "content".to_string(),  // Instead of "text"
    similarity_threshold: 0.7,
    ..Default::default()
});
```

### Use Different Hash Strategy

```rust
// In exact_dedup() function, change:
let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);

// To hash only specific field:
let mut dedup = ExactDeduplicator::new(HashStrategy::Field("text".to_string()));

// Or multiple fields:
let mut dedup = ExactDeduplicator::new(
    HashStrategy::MultiField(vec!["text".to_string(), "title".to_string()])
);
```

## Performance Testing

### Small Dataset (< 10K records)
```bash
time cargo run --release --example dedupe_file data.jsonl fuzzy
```

### Large Dataset (> 100K records)
For very large datasets, consider:
1. Use `exact` mode first (fastest)
2. Then use `fuzzy` mode on remaining records
3. Monitor memory usage with `htop` or `top`

### Benchmark Different Modes

```bash
# Create larger test dataset
for i in {1..10000}; do
  echo "{\"id\": $i, \"text\": \"Document $((i % 1000))\"}" >> /tmp/large_test.jsonl
done

# Time each mode
time cargo run --release --example dedupe_file /tmp/large_test.jsonl exact
time cargo run --release --example dedupe_file /tmp/large_test.jsonl exact-normalized
time cargo run --release --example dedupe_file /tmp/large_test.jsonl fuzzy
```

## Troubleshooting

### "No such file or directory"
Make sure your input file path is correct:
```bash
ls -lh /path/to/your/file.jsonl
```

### "Parse error"
Your JSONL file might have invalid JSON. Check with:
```bash
cat /tmp/test_dataset.jsonl | jq . > /dev/null
```

### Out of Memory
For very large datasets (>10M records):
1. Process in batches
2. Use `exact` mode (most memory efficient)
3. Increase system swap space

### Slow Performance
- Use `--release` flag for 10-100x speedup
- Try `exact` mode first (fastest)
- Consider processing in parallel batches

## Real-World Examples

### Web Scraping Data
```bash
# Dedupe scraped articles (fuzzy to catch near-duplicates)
cargo run --release --example dedupe_file scraped_articles.jsonl fuzzy
```

### User Comments
```bash
# Dedupe comments (exact-normalized to handle case/whitespace)
cargo run --release --example dedupe_file user_comments.jsonl exact-normalized
```

### Training Data
```bash
# Dedupe ML training data (exact to preserve variations)
cargo run --release --example dedupe_file training_data.jsonl exact
```

## Next Steps

- Run the automated test: `./test_dedup.sh`
- Try with your own data
- Adjust thresholds and settings
- Check the [benchmarks](README.md#benchmarks) for performance expectations
