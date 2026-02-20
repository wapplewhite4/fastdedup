//! fastdedup CLI
//!
//! High-performance tool for deduplicating and cleaning AI training datasets

mod config;
mod disk_kv;
mod progress;
mod resource_monitor;
mod tui;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use fastdedup_formats::ParquetWriter;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use config::PipelineConfig;
use progress::ProgressReporter;

#[derive(Parser)]
#[command(name = "fastdedup")]
#[command(version, about = "High-performance dataset deduplication and cleaning", long_about = None)]
#[command(author = "fastdedup")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Output statistics in JSON format
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Remove exact duplicates using content hashing
    ExactDedup {
        /// Input file (JSONL or Parquet)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Field to use for hashing (defaults to full record)
        #[arg(short, long)]
        field: Option<String>,

        /// Normalize text before hashing
        #[arg(short, long)]
        normalize: bool,

        /// Show statistics without writing output
        #[arg(long)]
        dry_run: bool,

        /// Only show statistics, don't deduplicate
        #[arg(long)]
        stats_only: bool,
    },

    /// Remove fuzzy duplicates using MinHash + LSH
    FuzzyDedup {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Similarity threshold (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        threshold: f64,

        /// Text field to compare
        #[arg(short = 'F', long, default_value = "text")]
        field: String,

        /// Number of MinHash hash functions
        #[arg(long, default_value = "128")]
        num_hashes: usize,

        /// Shingle size for MinHash (character n-grams by default, words with --word-shingles)
        #[arg(long, default_value = "3")]
        shingle_size: usize,

        /// Use word n-grams instead of character n-grams
        ///
        /// Word bigrams are faster but less sensitive: misses near-duplicates
        /// that share phrasing but differ in word choice. Use char n-grams
        /// (the default) for accuracy matching Python datasketch results.
        #[arg(long, default_value = "false")]
        word_shingles: bool,

        /// LSH number of bands (default 16; more bands = higher recall, more false positives)
        #[arg(long)]
        bands: Option<usize>,

        /// LSH rows per band (default 8; more rows = fewer false positives, lower recall)
        #[arg(long)]
        rows_per_band: Option<usize>,

        /// Show statistics without writing output
        #[arg(long)]
        dry_run: bool,

        /// Only show statistics, don't deduplicate
        #[arg(long)]
        stats_only: bool,
    },

    /// Apply quality filters (language, quality metrics)
    Filter {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Config file with filter settings (YAML or TOML)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Show statistics without writing output
        #[arg(long)]
        dry_run: bool,
    },

    /// Run full pipeline (dedup + filter)
    Pipeline {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Pipeline config file (YAML or TOML)
        #[arg(short, long)]
        config: PathBuf,

        /// Show statistics without writing output
        #[arg(long)]
        dry_run: bool,
    },

    /// Inspect a dataset file
    Inspect {
        /// Path to the dataset file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Number of records to show
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },

    /// Count records in a dataset
    Count {
        /// Path to the dataset file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Generate shell completion scripts
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Launch the interactive terminal UI
    Tui,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .with_ansi(!cli.json) // Disable colors if JSON output
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::ExactDedup {
            input,
            output,
            field,
            normalize,
            dry_run,
            stats_only,
        } => {
            exact_dedup(input, output, field, normalize, dry_run, stats_only, cli.json).await?;
        }
        Commands::FuzzyDedup {
            input,
            output,
            threshold,
            field,
            num_hashes,
            shingle_size,
            word_shingles,
            bands,
            rows_per_band,
            dry_run,
            stats_only,
        } => {
            fuzzy_dedup(input, output, threshold, field, num_hashes, shingle_size, word_shingles, bands, rows_per_band, dry_run, stats_only, cli.json).await?;
        }
        Commands::Filter {
            input,
            output,
            config,
            dry_run,
        } => {
            apply_filters(input, output, config, dry_run, cli.json).await?;
        }
        Commands::Pipeline {
            input,
            output,
            config,
            dry_run,
        } => {
            run_pipeline(input, output, config, dry_run, cli.json).await?;
        }
        Commands::Inspect { input, limit } => {
            inspect_dataset(input, limit).await?;
        }
        Commands::Count { input } => {
            count_dataset(input).await?;
        }
        Commands::Completions { shell } => {
            generate_completions(shell);
        }
        Commands::Tui => {
            tui::run_tui()?;
        }
    }

    Ok(())
}

async fn exact_dedup(
    input: PathBuf,
    output: PathBuf,
    field: Option<String>,
    normalize: bool,
    dry_run: bool,
    stats_only: bool,
    json_output: bool,
) -> Result<()> {
    use fastdedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
    use fastdedup_formats::open_dataset;

    info!("Starting exact deduplication");
    info!("  Input: {:?}", input);
    if !stats_only {
        info!("  Output: {:?}", output);
    }
    info!("  Field: {:?}", field);
    info!("  Normalize: {}", normalize);

    let mut reader = open_dataset(&input)?;

    let strategy = match field {
        Some(f) => HashStrategy::Field(f),
        None => HashStrategy::FullContent,
    };

    let mut deduplicator = ExactDeduplicator::new(strategy);

    // Note: Normalization should be integrated into the deduplicator itself
    // For now, we just use the normalize flag to indicate intent
    let _normalize_flag = normalize;

    let mut total = 0;
    let mut unique = 0;
    let mut duplicates = 0;

    // Use record-based progress if available (Parquet), otherwise bytes
    let progress = if let Some(total_records) = reader.total_records() {
        ProgressReporter::new_record_based(total_records)
    } else {
        let total_bytes = reader.total_bytes().unwrap_or(0);
        ProgressReporter::new(total_bytes)
    };

    while let Some(result) = reader.next() {
        let record = result?;
        total += 1;

        // Pass the record directly to the deduplicator
        // It handles field extraction based on the HashStrategy
        if !deduplicator.is_duplicate(&record.data) {
            unique += 1;
        } else {
            duplicates += 1;
        }

        if total % 1000 == 0 {
            progress.update(reader.bytes_processed(), total, duplicates, 0);
        }
    }

    progress.finish();

    let stats = deduplicator.stats();

    // Print report
    if json_output {
        let report = serde_json::json!({
            "input": input.to_string_lossy().to_string(),
            "output": if stats_only { serde_json::Value::Null } else { serde_json::Value::String(output.to_string_lossy().to_string()) },
            "total_records": total,
            "unique_records": unique,
            "duplicates_removed": duplicates,
            "deduplication_rate": (duplicates as f64 / total as f64) * 100.0,
            "bloom_filter_effectiveness": stats.bloom_effectiveness(),
            "dry_run": dry_run,
            "stats_only": stats_only,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        progress::print_summary_report(
            &input,
            if stats_only { None } else { Some(&output) },
            total,
            unique,
            duplicates,
            0,
        );
    }

    Ok(())
}

/// Derive the companion "removed records" path from the clean output path.
///
/// Examples:
///   output.jsonl   → output.removed.jsonl
///   output.parquet → output.removed.jsonl
fn removed_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{}.removed.jsonl", stem))
}

async fn fuzzy_dedup(
    input: PathBuf,
    output: PathBuf,
    threshold: f64,
    field: String,
    num_hashes: usize,
    shingle_size: usize,
    word_shingles: bool,
    bands_arg: Option<usize>,
    rows_per_band_arg: Option<usize>,
    dry_run: bool,
    stats_only: bool,
    json_output: bool,
) -> Result<()> {
    use fastdedup_core::fuzzy_dedup::{FuzzyDeduplicator, FuzzyDedupConfig};
    use fastdedup_formats::open_dataset;

    // Resolve LSH band parameters.
    // Default 16 bands × 8 rows = 128 hashes.
    // Compared to the naive 32×4: FP rate drops ~1000x (from ~5% → 0.0004% at s=0.2)
    // while TP rate at s=0.8 only drops from ~100% to ~95%.
    let rows_per_band = rows_per_band_arg.unwrap_or(8);
    let num_bands = bands_arg.unwrap_or_else(|| num_hashes / rows_per_band);

    if num_hashes % rows_per_band != 0 {
        eprintln!("Warning: num_hashes ({}) is not divisible by rows_per_band ({})",
                  num_hashes, rows_per_band);
        eprintln!("LSH will use {} bands × {} rows = {} hashes",
                  num_bands, rows_per_band, num_bands * rows_per_band);
    }

    info!("Starting fuzzy deduplication");
    info!("  Input: {:?}", input);
    if !stats_only {
        info!("  Output: {:?}", output);
    }
    info!("  Threshold: {}", threshold);
    info!("  Field: {}", field);
    info!("  MinHash functions: {}", num_hashes);
    info!("  Shingle: size={} mode={}", shingle_size, if word_shingles { "word-ngrams" } else { "char-ngrams" });
    info!("  LSH: {} bands × {} rows per band", num_bands, rows_per_band);

    let mut reader = open_dataset(&input)?;

    // Create config with the specified field and parameters
    let config = FuzzyDedupConfig {
        similarity_threshold: threshold,
        text_field: field.clone(),
        num_hashes,
        shingle_size,
        word_shingles,
        num_bands,
        rows_per_band,
        ..Default::default()
    };
    let mut deduplicator = FuzzyDeduplicator::with_config(config);

    let mut total = 0;
    let mut unique = 0;
    let mut duplicates = 0;

    // Maps row_id → extracted field text for every kept record so we can
    // populate `matched_value` in the removed log without re-reading the file.
    // Uses disk-backed storage to bound memory at scale.
    let mut field_values = disk_kv::DiskBackedStringMap::new(500_000)?;

    // Open output writers unless this is a dry-run or stats-only pass.
    //
    // clean_writer         → JSONL BufWriter (used when output extension is not .parquet)
    // clean_parquet_writer → ParquetWriter   (used when output extension is .parquet)
    // removed_writer       → auto-derived <stem>.removed.jsonl; one JSON object per
    //                        duplicate relationship with row ids, field values, and
    //                        similarity score for easy downstream inspection.
    let write_output = !dry_run && !stats_only;
    let removed_output = removed_path(&output);

    let output_is_parquet = output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "parquet")
        .unwrap_or(false);

    let mut clean_writer: Option<BufWriter<File>> = None;
    let mut clean_parquet_writer: Option<ParquetWriter> = None;
    let mut removed_writer: Option<BufWriter<File>> = None;

    if write_output {
        if output_is_parquet {
            clean_parquet_writer = Some(ParquetWriter::open(&output)?);
        } else {
            clean_writer = Some(BufWriter::new(File::create(&output)?));
        }
        removed_writer = Some(BufWriter::new(File::create(&removed_output)?));
        info!("  Clean output:   {:?}", output);
        info!("  Removed output: {:?}", removed_output);
    }

    // Use record-based progress if available (Parquet), otherwise bytes
    let progress = if let Some(total_records) = reader.total_records() {
        ProgressReporter::new_record_based(total_records)
    } else {
        let total_bytes = reader.total_bytes().unwrap_or(0);
        ProgressReporter::new(total_bytes)
    };

    // Process in batches: compute MinHash signatures in parallel (rayon),
    // then do LSH query/insert serially (LSH is not thread-safe).
    const BATCH_SIZE: usize = 2000;

    loop {
        // Collect a batch of records
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            match reader.next() {
                Some(Ok(record)) => batch.push(record),
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }

        // Parallel phase: compute MinHash signatures for all records in batch
        let signatures: Vec<_> = batch
            .par_iter()
            .map(|record| deduplicator.prepare_signature(&record.data))
            .collect();

        // Serial phase: LSH query + insert (order matters for dedup correctness)
        for (record, sig_opt) in batch.into_iter().zip(signatures) {
            // Capture the 0-based row id before incrementing so it matches
            // pandas df.iloc[row_id] on the original file.
            let row_id = total;
            let dups = match sig_opt {
                Some(sig) => deduplicator.process_prepared(row_id, sig),
                None => None, // missing/empty text field — treat as unique
            };
            total += 1;

            // Extract the compared field value once; used for both the
            // field_values cache (kept records) and the removed log.
            let field_text = record
                .data
                .get(&field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            match dups {
                None => {
                    unique += 1;
                    if write_output {
                        field_values.insert(row_id, field_text)?;
                    }
                    if let Some(ref mut w) = clean_writer {
                        writeln!(w, "{}", serde_json::to_string(&record.data)?)?;
                    } else if let Some(ref mut w) = clean_parquet_writer {
                        w.write_record(&record)?;
                    }
                }
                Some(ref dup_matches) => {
                    duplicates += 1;
                    if let Some(ref mut w) = removed_writer {
                        // One log line per duplicate relationship.
                        for &(dup_id, sim) in dup_matches {
                            let matched_value = field_values
                                .get(&dup_id)?
                                .unwrap_or_default();
                            let entry = serde_json::json!({
                                "row_id": row_id,
                                "duplicate_of_row_id": dup_id,
                                "field": field,
                                "value": field_text,
                                "matched_value": matched_value,
                                "similarity": (sim * 10_000.0).round() / 10_000.0,
                                "threshold": threshold,
                            });
                            writeln!(w, "{}", serde_json::to_string(&entry)?)?;
                        }
                    }
                }
            }
        }

        progress.update(reader.bytes_processed(), total, duplicates, 0);
    }

    // Flush / close writers before printing the summary so all data is on disk.
    if let Some(ref mut w) = clean_writer {
        w.flush()?;
    }
    // close() flushes remaining buffered records and writes the parquet footer —
    // without this the file is corrupt.
    if let Some(w) = clean_parquet_writer {
        w.close()?;
    }
    if let Some(ref mut w) = removed_writer {
        w.flush()?;
    }

    progress.finish();

    let stats = deduplicator.stats();

    if json_output {
        let report = serde_json::json!({
            "input": input.to_string_lossy().to_string(),
            "output": if write_output { serde_json::Value::String(output.to_string_lossy().to_string()) } else { serde_json::Value::Null },
            "removed_output": if write_output { serde_json::Value::String(removed_output.to_string_lossy().to_string()) } else { serde_json::Value::Null },
            "total_records": total,
            "unique_records": unique,
            "duplicates_removed": duplicates,
            "deduplication_rate": (duplicates as f64 / total as f64) * 100.0,
            "lsh_candidates_checked": stats.lsh_candidates_checked,
            "verified_duplicates": stats.verified_duplicates,
            "lsh_precision": stats.lsh_precision(),
            "dry_run": dry_run,
            "stats_only": stats_only,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        progress::print_summary_report(
            &input,
            if write_output { Some(&output) } else { None },
            total,
            unique,
            duplicates,
            0,
        );
        if write_output {
            println!("  Removed records: {:?} ({} records with __duplicate_of annotation)",
                     removed_output, duplicates);
        }
    }

    Ok(())
}

async fn apply_filters(
    input: PathBuf,
    output: PathBuf,
    config: Option<PathBuf>,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    use fastdedup_filters::language::{LanguageFilter, LanguageFilterConfig};
    use fastdedup_filters::quality::{QualityScorer, QualityConfig};
    use fastdedup_formats::open_dataset;

    info!("Applying quality filters");
    info!("  Input: {:?}", input);
    info!("  Output: {:?}", output);
    info!("  Config: {:?}", config);

    let mut reader = open_dataset(&input)?;

    // Load config or use defaults
    let (lang_filter, quality_scorer) = if let Some(config_path) = config {
        let config = config::load_filter_config(&config_path)?;
        (
            LanguageFilter::new(config.language)?,
            QualityScorer::new(config.quality),
        )
    } else {
        (
            LanguageFilter::new(LanguageFilterConfig::english_only())?,
            QualityScorer::new(QualityConfig::default()),
        )
    };

    let mut total = 0;
    let mut filtered = 0;
    let mut passed = 0;

    // Use record-based progress if available (Parquet), otherwise bytes
    let progress = if let Some(total_records) = reader.total_records() {
        ProgressReporter::new_record_based(total_records)
    } else {
        let total_bytes = reader.total_bytes().unwrap_or(0);
        ProgressReporter::new(total_bytes)
    };

    while let Some(result) = reader.next() {
        let record = result?;
        total += 1;

        let text = serde_json::to_string(&record.data)?;

        let lang_ok = lang_filter.is_accepted(&text);
        let quality_score = quality_scorer.score(&text);

        if lang_ok && quality_score.passes {
            passed += 1;
        } else {
            filtered += 1;
        }

        if total % 1000 == 0 {
            progress.update(reader.bytes_processed(), total, 0, filtered);
        }
    }

    progress.finish();

    if json_output {
        let report = serde_json::json!({
            "input": input.to_string_lossy().to_string(),
            "output": output.to_string_lossy().to_string(),
            "total_records": total,
            "passed": passed,
            "filtered": filtered,
            "filter_rate": (filtered as f64 / total as f64) * 100.0,
            "dry_run": dry_run,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        progress::print_summary_report(&input, Some(&output), total, passed, 0, filtered);
    }

    Ok(())
}

async fn run_pipeline(
    input: PathBuf,
    output: PathBuf,
    config_path: PathBuf,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    info!("Running deduplication pipeline");
    info!("  Input: {:?}", input);
    info!("  Output: {:?}", output);
    info!("  Config: {:?}", config_path);

    let config = PipelineConfig::load(&config_path)?;

    // TODO: Implement full pipeline with config
    // For now, just validate config and print it
    info!("Pipeline config loaded successfully");

    if json_output {
        println!("{}", serde_json::to_string_pretty(&config)?);
    } else {
        println!("Pipeline configuration:");
        println!("{:#?}", config);
    }

    Ok(())
}

async fn inspect_dataset(input: PathBuf, limit: usize) -> Result<()> {
    use fastdedup_formats::open_dataset;

    info!("Inspecting dataset: {:?}", input);

    let mut reader = open_dataset(&input)?;
    let mut count = 0;

    while let Some(result) = reader.next() {
        let record = result?;
        println!(
            "Record #{}: {}",
            record.source_line,
            serde_json::to_string_pretty(&record.data)?
        );

        count += 1;
        if count >= limit {
            break;
        }
    }

    info!(
        "Processed {} records ({} bytes)",
        reader.records_processed(),
        reader.bytes_processed()
    );

    Ok(())
}

async fn count_dataset(input: PathBuf) -> Result<()> {
    use fastdedup_formats::open_dataset;
    use indicatif::{ProgressBar, ProgressStyle};

    info!("Counting records in: {:?}", input);

    let mut reader = open_dataset(&input)?;

    let pb = if let Some(total) = reader.total_bytes() {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut count = 0;
    while let Some(result) = reader.next() {
        let _record = result?;
        count += 1;

        if let Some(ref pb) = pb {
            pb.set_position(reader.bytes_processed());
        }

        if count % 10000 == 0 {
            info!("Processed {} records...", count);
        }
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    println!("Total records: {}", count);
    info!("Processed {} bytes", reader.bytes_processed());

    Ok(())
}

fn generate_completions(shell: Shell) {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
}
