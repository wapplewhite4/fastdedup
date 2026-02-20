# fastdedup

High-performance Rust CLI for deduplicating AI training datasets on a single machine.
Handles exact and fuzzy (near-duplicate) detection on datasets with tens of millions of
records, reading and writing JSONL, gzip-compressed JSONL, and Apache Parquet.

**[Blog post: Deduplicating 15 Million Records in 3 Minutes with Rust](https://YOUR_BLOG_URL_HERE)**

## Benchmarks

Tested on [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
— 14.8 million records, 29 GB on disk — on a Hetzner CCX43 (16 vCPU AMD EPYC Milan,
64 GB RAM).

### Exact deduplication

|  | **fastdedup** | DuckDB + SHA-256 |
|--|--|--|
| Wall clock | **2m 55s** | 7m 55s |
| Peak RAM | **688 MB** | 21.9 GB |
| CPU cores used | 1 | 4+ |
| Records/sec | ~85,000 | ~31,000 |
| Duplicates found | 51,392 | 51,392 ✓ |

2.7× faster, 32× less RAM, on a single core.

### Fuzzy deduplication (MinHash + LSH)

Baseline: [datatrove](https://github.com/huggingface/datatrove), the reference
implementation used to produce FineWeb.

|  | **fastdedup** | datatrove |
|--|--|--|
| Wall clock | **36m 44s** | 3h 50m+ (incomplete) |
| Peak RAM | 23 GB | 1.1 GB |
| CPU cores used | ~5.5 | 1 |
| Completed | **Yes** | No |
| Duplicates found | 105,044 | — |

datatrove did not finish in under 4 hours. fastdedup completed the full run including
output writing.

> **RAM trade-off:** fastdedup keeps the LSH index in memory (23 GB peak on this
> dataset). datatrove streams intermediate data to disk (1.1 GB RAM) at the cost of
> heavy inter-stage I/O. On a machine with adequate RAM the in-memory approach is
> significantly faster. See [System requirements](#system-requirements) below.

Full methodology and raw results: `benchmarks/README.md`.

## Table of contents

- [System requirements](#system-requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Library usage](#library-usage)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## System requirements

- **Rust 1.70+** to build from source
- **Exact dedup:** RAM scales with unique record count (~12 bytes/unique record).
  The 29 GB FineWeb benchmark peaked at 688 MB.
- **Fuzzy dedup:** RAM scales with total record count. The LSH index requires
  roughly 1.5–2 KB per record. Rough estimates:

  | Dataset size | Estimated peak RAM |
  |---|---|
  | 1M records | ~2 GB |
  | 5M records | ~8 GB |
  | 15M records | ~23 GB |
  | 50M records | ~70 GB |

  If RAM is constrained, reducing `--num-hashes` (e.g. 64 instead of 128) roughly
  halves index memory at a small accuracy trade-off.

## Installation

Requires Rust 1.70+.

```bash
git clone <repository-url>
cd fastdedup

# Option A: install the binary into ~/.cargo/bin (then it's on your PATH)
cargo install --path crates/cli

# Option B: build only, then run from the repo
cargo build --release
./target/release/fastdedup --help
```

The binary is named `fastdedup`. After `cargo install` you can call it
directly; with Option B prefix every command with `./target/release/`.

## Quick start

```bash
# Inspect a dataset
fastdedup inspect data.jsonl -n 5

# Count records
fastdedup count data.parquet

# Exact dedup on the "text" field
fastdedup exact-dedup -i data.jsonl -o deduped.jsonl --field text

# Fuzzy dedup at 85% similarity
fastdedup fuzzy-dedup -i data.jsonl -o deduped.jsonl --threshold 0.85 --field text

# Full pipeline from a config file
fastdedup pipeline -i data.jsonl -o clean.jsonl --config pipeline.yaml

# Interactive terminal UI
fastdedup tui
```

## Commands

### Global flags

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable DEBUG-level logging |
| `--json` | Output statistics as JSON |

### `exact-dedup`

Remove exact duplicates using content hashing.

```
fastdedup exact-dedup [OPTIONS] -i <INPUT> -o <OUTPUT>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | required | Input file (JSONL, JSONL.gz, or Parquet) |
| `-o, --output` | required | Output file (JSONL or Parquet, auto-detected by extension) |
| `-f, --field` | full record | Field to hash. Omit to hash the entire JSON record |
| `-n, --normalize` | off | Lowercase + trim before hashing (`--field` required) |
| `--dry-run` | off | Print statistics without writing output |
| `--stats-only` | off | Print statistics only |

**How it works.** Each record is hashed with `ahash`. A Bloom filter (1% false-positive
rate) provides a fast negative check; positives are confirmed against an in-memory
`AHashSet`. Four hash strategies are available via the library API:

| Strategy | Hashes |
|----------|--------|
| `FullContent` | Entire JSON record |
| `Field(name)` | Single field value |
| `Normalized(name)` | Field value after lowercase + trim |
| `MultiField(names)` | Concatenation of multiple fields |

### `fuzzy-dedup`

Remove near-duplicate records using MinHash + LSH.

```
fastdedup fuzzy-dedup [OPTIONS] -i <INPUT> -o <OUTPUT>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | required | Input file |
| `-o, --output` | required | Output file |
| `-t, --threshold` | `0.8` | Jaccard similarity threshold (0.0 -- 1.0) |
| `-F, --field` | `text` | JSON field to compare |
| `--num-hashes` | `128` | Number of MinHash permutations |
| `--shingle-size` | `3` | n-gram size (characters or words) |
| `--word-shingles` | `false` | Use word n-grams instead of character n-grams |
| `--bands` | `16` | Number of LSH bands |
| `--rows-per-band` | `8` | Rows per LSH band (`bands * rows_per_band` must equal `num_hashes`) |
| `--dry-run` | off | Print statistics without writing output |
| `--stats-only` | off | Print statistics only |

**Output files.** The output format is determined by the file extension you supply:

- `.jsonl` / `.json` — unique records written as JSON Lines (one record per line)
- `.parquet` — unique records written as a valid Apache Parquet file (schema inferred
  from the first batch of records)

In addition, a companion audit file is always written alongside the clean output:

- `<stem>.removed.jsonl` — one JSON object per removed duplicate (always JSONL regardless
  of the clean-output format)

Example for `deduped.parquet`:

- `deduped.parquet` -- unique (kept) records in Parquet format
- `deduped.removed.jsonl` -- one JSON object per removed duplicate:

Example for `deduped.jsonl`:

- `deduped.jsonl` -- unique (kept) records
- `deduped.removed.jsonl` -- one JSON object per removed duplicate:

```json
{
  "row_id": 42,
  "duplicate_of_row_id": 7,
  "field": "text",
  "value": "the removed record's text",
  "matched_value": "the kept record's text",
  "similarity": 0.92,
  "threshold": 0.8
}
```

### `filter`

Apply quality and language filters.

```
fastdedup filter -i <INPUT> -o <OUTPUT> [--config filters.yaml]
```

### `pipeline`

Run a full dedup + filter pipeline from a YAML/TOML config.

```
fastdedup pipeline -i <INPUT> -o <OUTPUT> --config pipeline.yaml [--dry-run]
```

### `inspect`

Print the first N records from a dataset.

```
fastdedup inspect <FILE> [-n 10]
```

### `count`

Count records in a dataset.

```
fastdedup count <FILE>
```

### `completions`

Generate shell completions.

```bash
fastdedup completions bash > ~/.local/share/bash-completion/completions/fastdedup
fastdedup completions zsh  > ~/.zsh/completions/_fastdedup
fastdedup completions fish > ~/.config/fish/completions/fastdedup.fish
```

### `tui`

Launch an interactive terminal UI for guided deduplication.

## Algorithms

### Exact deduplication

Records are hashed with [ahash](https://github.com/tkaitchuck/ahash) (a fast,
non-cryptographic hash). The deduplicator maintains:

1. **Bloom filter** -- probabilistic set membership with ~1% false-positive
   rate. Records that fail the Bloom check are guaranteed unique and skip the
   hash-set lookup entirely.
2. **AHashSet** -- definitive set of seen hashes. Only consulted when the Bloom
   filter returns a positive.

This two-layer design keeps the average lookup at ~1 hash + 1 bit-probe for
unique records.

For large datasets a **tiered hash storage** is available as a library API
(`fastdedup_core::hash_storage::TieredHashStorage`): an in-memory hot cache
backed by an on-disk sled database. This is not currently exposed as a CLI flag.

### Fuzzy deduplication (MinHash + LSH)

Fuzzy dedup detects records whose text is *similar but not identical*. The
pipeline for each record is:

```
extract field -> normalize text -> compute MinHash signature -> query LSH index -> verify candidates
```

#### 1. Text normalization

Text is preprocessed before hashing. Three presets are available:

| Preset | Lowercase | Remove punctuation | Collapse whitespace | Unicode NFKD |
|--------|-----------|-------------------|---------------------|-------------|
| **Aggressive** | yes | yes | yes | yes |
| **Balanced** (default) | yes | yes | yes | no |
| **Conservative** | yes | no | yes | no |

#### 2. Shingling

Text is split into overlapping n-grams (*shingles*). Two modes:

- **Character n-grams** (default, `--word-shingles false`): e.g. `"the"`,
  `"he "`, `"e q"` for shingle size 3. Matches Python
  [datasketch](https://ekzhu.com/datasketch/) behaviour.
- **Word n-grams** (`--word-shingles true`): e.g. `"the quick"`,
  `"quick brown"` for shingle size 2. More discriminative for natural language
  (fewer false positives from common character sequences like "the", "ing").

#### 3. MinHash signatures

Each shingle set is compressed into a fixed-size signature of `num_hashes`
(default 128) minimum hash values. The signature preserves the *Jaccard
similarity* between any two documents:

```
J(A, B) = |A ∩ B| / |A ∪ B| ≈ (# matching signature positions) / num_hashes
```

Hash functions use the form `h(x) = (a * x + b) mod p` with a large prime
`p = 2^31 - 1` and deterministic coefficients (seeded LCG).

#### 4. Locality Sensitive Hashing (LSH)

The 128-element signature is divided into **bands** of consecutive rows. Two
documents become *candidates* if they match in all rows of at least one band.

With `b` bands of `r` rows each, the probability that two documents with true
Jaccard similarity `s` become candidates is:

```
P(candidate) = 1 - (1 - s^r)^b
```

Default configuration: **16 bands x 8 rows = 128 hashes**.

| True similarity | P(candidate) | Behavior |
|----------------|-------------|-----------|
| 0.2 | ~0.0004% | Almost never flagged |
| 0.5 | ~0.5% | Rarely flagged |
| 0.7 | ~18% | Sometimes flagged |
| 0.8 | ~66% | Usually flagged |
| 0.9 | ~97% | Almost always flagged |
| 1.0 | 100% | Always flagged |

Compared to a 32 x 4 configuration, 16 x 8 reduces false positives by ~1000x at
low similarity while keeping ~95% true-positive rate at `s = 0.8`.

#### 5. Candidate verification

Each candidate pair returned by LSH is verified by computing the exact MinHash
Jaccard estimate. Only pairs meeting the `--threshold` are marked as duplicates.

#### LSH index performance optimizations

The LSH index is tuned for datasets with millions of records:

1. **Pre-hashed u64 band keys.** Band signatures (`rows_per_band` hash values)
   are reduced to a single `u64` via ahash before HashMap lookup. This
   eliminates a `Vec<u64>` allocation per lookup and makes key comparison O(1)
   instead of O(rows_per_band).

2. **ahash-backed HashMaps.** All band tables use ahash (`RandomState`) instead
   of the default SipHash hasher. Since keys are not adversarially controlled,
   this is ~30% faster.

3. **Vec-backed signature storage.** Signatures are stored in a
   `Vec<Option<MinHashSignature>>` indexed by document ID rather than a
   `HashMap`. Since IDs are sequential 0..N this gives O(1) access with better
   cache locality.

4. **Capped candidate verification.** Queries return at most 200 candidates,
   bounding worst-case verification cost. Most true duplicates appear early
   (inserted close in time), so recall loss is negligible.

5. **Capacity-hint pre-allocation.** When the record count is known in advance
   (e.g. from Parquet metadata), `LSHIndex::with_capacity()` pre-allocates all
   internal structures to avoid rehashing during ingestion.

6. **Periodic band-bucket compaction.** IDs that were removed as duplicates
   still occupy band buckets, inflating candidate lists. The `compact()` method
   prunes stale entries; `query()` also inline-filters stale IDs.

**Algorithmic complexity per record (after optimizations):**

| Step | Cost |
|------|------|
| Band key computation | 16 u64 hashes |
| Band lookups | 16 ahash HashMap lookups |
| Candidate verification | min(candidates, 200) x 128 comparisons |
| Signature storage | O(1) Vec index |

### Quality filtering

Quality scores are computed across multiple metrics. A record must pass all
enabled checks:

| Metric | Default | Description |
|--------|---------|-------------|
| `min_length` | 50 | Minimum character count |
| `max_length` | 100,000 | Maximum character count |
| `min_word_count` | 10 | Minimum word count |
| `max_word_count` | 10,000 | Maximum word count |
| `max_repetition_ratio` | 0.3 | Maximum fraction of repeated 3-4 word n-grams |
| `min_unique_words_ratio` | 0.3 | Minimum vocabulary diversity |
| `max_url_ratio` | 0.1 | Maximum fraction of URL characters |
| `max_special_char_ratio` | 0.3 | Maximum non-alphanumeric ratio |
| `min_avg_word_length` | 2.5 | Minimum average word length |
| `reject_html` | true | Reject records containing HTML tags |
| `filter_profanity` | false | Enable profanity filter |

Three presets are available: **default**, **strict** (tighter thresholds), and
**lenient** (more permissive).

### Language detection

Language detection uses [whatlang](https://github.com/grstrainern/whatlang) and supports 20+ languages via ISO 639-3 codes:

`eng`, `spa`, `fra`, `deu`, `por`, `rus`, `jpn`, `zho`, `ara`, `hin`, `ita`,
`nld`, `pol`, `tur`, `vie`, `kor`, `swe`, `dan`, `fin`, `nor`, ...

Configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| `allowed_languages` | `["eng"]` | ISO 639-3 codes to accept |
| `confidence_threshold` | 0.5 | Minimum detection confidence (0.0 -- 1.0) |
| `min_text_length` | 50 | Skip detection for texts shorter than this |

Code-heavy text is detected via keyword heuristics (`function`, `class`,
`import`, `return`, etc.) and can be accepted with the `"code"` pseudo-language.

## Configuration

Pipeline and filter commands accept YAML or TOML config files. Example
(`examples/config.yaml`):

```yaml
input:
  path: "raw_dataset.jsonl.gz"
  format: jsonl

output:
  path: "clean_dataset.jsonl"
  format: jsonl
  compression: none          # none | gzip | zstd

deduplication:
  exact:
    field: "text"            # null = full record
    normalize: true

  fuzzy:
    threshold: 0.85
    field: "text"

filters:
  language:
    allowed_languages: ["eng", "code"]
    confidence_threshold: 0.7
    min_text_length: 50

  quality:
    min_length: 100
    max_length: 10000
    min_word_count: 20
    max_word_count: 2000
    max_repetition_ratio: 0.3
    min_unique_words_ratio: 0.3
    max_url_ratio: 0.1
    max_special_char_ratio: 0.3
    reject_html: true
    filter_profanity: false
    min_avg_word_length: 2.5
```

Additional example configs in `examples/`:

| File | Use case |
|------|----------|
| `config.yaml` | Balanced English-language pipeline |
| `config.toml` | Same pipeline in TOML format |
| `config-multilingual.yaml` | Top-10 languages with lenient quality |
| `config-strict.yaml` | Aggressive quality filtering |
| `filters-only.yaml` | Language + quality filters, no dedup |

## Supported formats

### Input

| Extension | Format | Notes |
|-----------|--------|-------|
| `.jsonl`, `.json` | JSON Lines | Streaming, line-by-line |
| `.jsonl.gz`, `.json.gz` | Gzip JSONL | Auto-decompressed |
| `.parquet` | Apache Parquet | Batch reading, column projection |

### Output (`fuzzy-dedup`)

| Extension | Format | Notes |
|-----------|--------|-------|
| `.jsonl`, `.json` | JSON Lines | One record per line |
| `.parquet` | Apache Parquet | Schema inferred from first batch |

Format is auto-detected from the file extension.

## Library usage

The workspace crates can be used as libraries independently.

### Reading datasets

```rust
use fastdedup_formats::{open_dataset, DatasetReader};

let mut reader = open_dataset("data.parquet")?;
for result in reader.by_ref() {
    let record = result?;
    println!("{}", record.data);
}
println!("Processed {} records", reader.records_processed());
```

### Exact dedup

```rust
use fastdedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};

let mut dedup = ExactDeduplicator::new(HashStrategy::Normalized("text".into()));
// dedup.is_duplicate(&record.data) returns true for duplicates
```

### Fuzzy dedup

```rust
use fastdedup_core::fuzzy_dedup::{FuzzyDeduplicator, FuzzyDedupConfig};

let config = FuzzyDedupConfig {
    similarity_threshold: 0.85,
    num_hashes: 128,
    shingle_size: 2,
    word_shingles: true,
    num_bands: 16,
    rows_per_band: 8,
    text_field: "text".into(),
};
let mut dedup = FuzzyDeduplicator::with_config(config);

// Single-pass: query + insert in one call
match dedup.process_record(id, &record) {
    Some(duplicate_ids) => { /* record is a duplicate */ }
    None                => { /* record is unique, added to index */ }
}
```

### Tiered hash storage (for billion-scale datasets)

```rust
use fastdedup_core::hash_storage::{TieredHashStorage, TieredStorageConfig};

let config = TieredStorageConfig {
    max_hot_size: 10_000_000,  // 10M hashes in memory (~160 MB)
    db_path: "./cold_storage".into(),
    sync_on_write: false,
};
let mut storage = TieredHashStorage::with_config(config)?;
// storage.contains(hash)? / storage.insert(hash)?
```

### Text normalization

```rust
use fastdedup_filters::text_preprocessing::TextNormalizer;

let norm = TextNormalizer::aggressive();
assert_eq!(norm.normalize("  Hello, WORLD!!!  "), "hello world");
```

## Reproducing benchmarks

```bash
# Exact dedup vs DuckDB
./benchmarks/run_comparison.sh

# Fuzzy dedup vs datatrove
./benchmarks/run_fuzzy_comparison.sh

# Rust micro-benchmarks
cargo bench --package fastdedup-core
cargo bench --package fastdedup-filters
```

See `benchmarks/README.md` for full setup instructions. Benchmark results and
methodology are summarised [at the top of this README](#benchmarks) and in the
accompanying [blog post](https://YOUR_BLOG_URL_HERE).

## Project structure

```
fastdedup/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── core/                     # Deduplication engine
│   │   └── src/
│   │       ├── minhash.rs        # MinHash signatures + LSH index
│   │       ├── fuzzy_dedup.rs    # Fuzzy dedup pipeline
│   │       ├── exact_dedup.rs    # Exact dedup with Bloom filter
│   │       ├── hash_storage.rs   # Tiered hot/cold hash storage
│   │       ├── pipeline.rs       # Parallel processing pipeline
│   │       ├── memory.rs         # Memory tracking + limits
│   │       ├── hash.rs           # Hashing utilities
│   │       ├── dedup.rs          # Basic dedup tracker
│   │       └── error.rs          # Error types
│   ├── formats/                  # File format readers/writers
│   │   └── src/
│   │       ├── jsonl.rs          # JSONL streaming reader (+ gzip)
│   │       ├── parquet_reader.rs # Parquet batch reader
│   │       ├── parquet_writer.rs # Parquet streaming writer (schema inference)
│   │       ├── reader.rs         # Unified DatasetReader trait
│   │       └── record.rs         # Record data structure
│   ├── filters/                  # Text processing + quality
│   │   └── src/
│   │       ├── text_preprocessing.rs  # Text normalization
│   │       ├── quality.rs        # Quality scoring
│   │       ├── language.rs       # Language detection
│   │       └── length_filter.rs  # Length constraints
│   └── cli/                      # Command-line interface + TUI
├── examples/                     # Example config files
└── benchmarks/                   # Python baselines + comparison scripts
```

## Testing

```bash
# Full test suite
cargo test --workspace

# Single crate
cargo test -p fastdedup-core

# With debug logging
RUST_LOG=debug cargo test
```

## Contributing

Bug reports, feature requests, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).
