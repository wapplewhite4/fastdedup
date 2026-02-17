//! Dataset Deduplication CLI
//!
//! High-performance tool for deduplicating and cleaning AI training datasets

mod config;
mod progress;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use config::PipelineConfig;
use progress::ProgressReporter;

#[derive(Parser)]
#[command(name = "dataset-dedup")]
#[command(version, about = "High-performance dataset deduplication and cleaning", long_about = None)]
#[command(author = "Dataset Dedup Team")]
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

        /// Number of MinHash functions (lower = faster, less accurate)
        #[arg(long, default_value = "128")]
        num_hashes: usize,

        /// Shingle size for MinHash (character n-grams)
        #[arg(long, default_value = "3")]
        shingle_size: usize,

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
            dry_run,
            stats_only,
        } => {
            fuzzy_dedup(input, output, threshold, field, num_hashes, shingle_size, dry_run, stats_only, cli.json).await?;
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
    use dataset_dedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
    use dataset_dedup_formats::open_dataset;

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

    // Create progress reporter
    let total_bytes = reader.total_bytes().unwrap_or(0);
    let progress = ProgressReporter::new(total_bytes);

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

async fn fuzzy_dedup(
    input: PathBuf,
    output: PathBuf,
    threshold: f64,
    field: String,
    num_hashes: usize,
    shingle_size: usize,
    dry_run: bool,
    stats_only: bool,
    json_output: bool,
) -> Result<()> {
    use dataset_dedup_core::fuzzy_dedup::{FuzzyDeduplicator, FuzzyDedupConfig};
    use dataset_dedup_formats::open_dataset;

    info!("Starting fuzzy deduplication");
    info!("  Input: {:?}", input);
    if !stats_only {
        info!("  Output: {:?}", output);
    }
    info!("  Threshold: {}", threshold);
    info!("  Field: {}", field);
    info!("  MinHash functions: {}", num_hashes);
    info!("  Shingle size: {}", shingle_size);

    let mut reader = open_dataset(&input)?;

    // Calculate LSH parameters based on num_hashes
    // Formula: num_bands * rows_per_band = num_hashes
    // We keep rows_per_band = 4 for good balance
    let rows_per_band = 4;
    let num_bands = num_hashes / rows_per_band;

    if num_hashes % rows_per_band != 0 {
        eprintln!("Warning: num_hashes ({}) is not divisible by rows_per_band ({})",
                  num_hashes, rows_per_band);
        eprintln!("LSH will use {} bands × {} rows = {} hashes",
                  num_bands, rows_per_band, num_bands * rows_per_band);
    }

    info!("  LSH bands: {}, rows per band: {}", num_bands, rows_per_band);

    // Create config with the specified field and parameters
    let config = FuzzyDedupConfig {
        similarity_threshold: threshold,
        text_field: field.clone(),
        num_hashes,
        shingle_size,
        num_bands,
        rows_per_band,
    };
    let mut deduplicator = FuzzyDeduplicator::with_config(config);

    let mut total = 0;
    let mut unique = 0;
    let mut duplicates = 0;

    let total_bytes = reader.total_bytes().unwrap_or(0);
    let progress = ProgressReporter::new(total_bytes);

    while let Some(result) = reader.next() {
        let record = result?;
        total += 1;

        let dups = deduplicator.find_duplicates(&record.data);

        if dups.is_empty() {
            deduplicator.add_record(total, &record.data);
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

    if json_output {
        let report = serde_json::json!({
            "input": input.to_string_lossy().to_string(),
            "output": if stats_only { serde_json::Value::Null } else { serde_json::Value::String(output.to_string_lossy().to_string()) },
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
            if stats_only { None } else { Some(&output) },
            total,
            unique,
            duplicates,
            0,
        );
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
    use dataset_dedup_filters::language::{LanguageFilter, LanguageFilterConfig};
    use dataset_dedup_filters::quality::{QualityScorer, QualityConfig};
    use dataset_dedup_formats::open_dataset;

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

    let total_bytes = reader.total_bytes().unwrap_or(0);
    let progress = ProgressReporter::new(total_bytes);

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
    use dataset_dedup_formats::open_dataset;

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
    use dataset_dedup_formats::open_dataset;
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
