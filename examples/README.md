# Configuration Examples

This directory contains example configuration files for the dataset deduplication tool.

## Configuration Files

### Pipeline Configurations (Full Pipeline)

- **`config.yaml`** - Default configuration with balanced settings
- **`config.toml`** - Same as above, but in TOML format
- **`config-multilingual.yaml`** - Multilingual support with lenient quality filters
- **`config-strict.yaml`** - Strict quality filtering for high-quality datasets

### Filter-Only Configurations

- **`filters-only.yaml`** - Language and quality filters without deduplication

## Usage

### Run Full Pipeline

```bash
fastdedup pipeline \
  --input raw_dataset.jsonl \
  --output clean_dataset.jsonl \
  --config examples/config.yaml
```

### Run Exact Deduplication

```bash
fastdedup exact-dedup \
  --input raw_dataset.jsonl \
  --output deduped.jsonl \
  --field text \
  --normalize
```

### Run Fuzzy Deduplication

```bash
fastdedup fuzzy-dedup \
  --input raw_dataset.jsonl \
  --output deduped.jsonl \
  --threshold 0.85 \
  --field text
```

### Apply Quality Filters

```bash
fastdedup filter \
  --input dataset.jsonl \
  --output filtered.jsonl \
  --config examples/filters-only.yaml
```

## Configuration Options

### Input/Output

```yaml
input:
  path: "input.jsonl"
  format: jsonl          # Options: jsonl, json, parquet, csv

output:
  path: "output.jsonl"
  format: jsonl
  compression: gzip      # Options: none, gzip, zstd
```

### Deduplication

#### Exact Deduplication

```yaml
deduplication:
  exact:
    field: "text"        # Field to hash (null = full record)
    normalize: true      # Normalize before hashing
```

#### Fuzzy Deduplication

```yaml
deduplication:
  fuzzy:
    threshold: 0.85      # Similarity threshold (0.0-1.0)
    field: "text"        # Field to compare
```

### Filters

#### Language Filter

```yaml
filters:
  language:
    allowed_languages:
      - "eng"            # English
      - "spa"            # Spanish
      - "fra"            # French
    confidence_threshold: 0.7
    min_text_length: 50
```

**Supported Languages:**
- `eng` (English)
- `spa` (Spanish)
- `fra` (French)
- `deu` (German)
- `rus` (Russian)
- `por` (Portuguese)
- `ita` (Italian)
- `jpn` (Japanese)
- `zho` (Chinese)
- `ara` (Arabic)
- And 20+ more

#### Quality Filter

```yaml
filters:
  quality:
    # Length constraints
    min_length: 100
    max_length: 10000
    min_word_count: 20
    max_word_count: 2000

    # Content quality
    max_repetition_ratio: 0.3       # Max repeated n-grams
    min_unique_words_ratio: 0.3     # Min vocabulary diversity
    max_url_ratio: 0.1              # Max URL density
    max_special_char_ratio: 0.3     # Max non-alphanumeric chars
    min_avg_word_length: 2.5        # Min average word length

    # Optional filters
    reject_html: true               # Reject HTML content
    filter_profanity: false         # Enable profanity filter
```

## Creating Custom Configurations

1. Copy an example config:
   ```bash
   cp examples/config.yaml my-config.yaml
   ```

2. Edit settings to match your needs

3. Run with your config:
   ```bash
   fastdedup pipeline \
     --input mydata.jsonl \
     --output clean.jsonl \
     --config my-config.yaml
   ```

## Presets

### Default (Balanced)
- English only
- Moderate quality filters
- Both exact and fuzzy dedup
- Good for general datasets

### Multilingual
- Top 10 languages
- Lenient quality filters
- Lower fuzzy threshold
- Good for diverse datasets

### Strict
- English only
- Aggressive quality filtering
- High fuzzy threshold
- Good for high-quality training data

## Tips

- **Start with `--dry-run`** to see statistics without writing output
- **Use `--stats-only`** to analyze your dataset first
- **Enable `--json`** for programmatic output
- **Test on a sample** before processing full dataset
- **Adjust thresholds** based on `--dry-run` results

## Shell Completions

Generate completions for your shell:

```bash
# Bash
fastdedup completions bash > ~/.local/share/bash-completion/completions/fastdedup

# Zsh
fastdedup completions zsh > ~/.zsh/completions/_fastdedup

# Fish
fastdedup completions fish > ~/.config/fish/completions/fastdedup.fish
```

## Examples

See `../crates/cli/examples/` for code examples:
- `dedupe_file.rs` - Deduplication example
- `generate_test_data.rs` - Test data generator
- `quality_demo.rs` - Quality filters demo
