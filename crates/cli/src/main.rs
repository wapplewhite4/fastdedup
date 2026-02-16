//! Dataset Deduplication CLI
//!
//! High-performance tool for deduplicating and cleaning AI training datasets

use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset_dedup_core::hash::hash_string;
use dataset_dedup_formats::open_dataset;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "dataset-dedup")]
#[command(about = "High-performance dataset deduplication tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
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

    /// Deduplicate a dataset
    Dedup {
        /// Input dataset file
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output file path
        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,

        /// Field to use for deduplication
        #[arg(short, long, default_value = "text")]
        field: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Inspect { input, limit } => {
            inspect_dataset(input, limit)?;
        }
        Commands::Count { input } => {
            count_dataset(input)?;
        }
        Commands::Dedup {
            input,
            output,
            field,
        } => {
            deduplicate_dataset(input, output, field)?;
        }
    }

    Ok(())
}

fn inspect_dataset(input: PathBuf, limit: usize) -> Result<()> {
    info!("Inspecting dataset: {:?}", input);

    let mut reader = open_dataset(&input)?;
    let mut count = 0;

    for result in reader.by_ref() {
        let record = result?;
        println!("Record #{}: {}", record.source_line, record.data);

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

fn count_dataset(input: PathBuf) -> Result<()> {
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
    loop {
        match reader.next() {
            Some(result) => {
                let _record = result?;
                count += 1;

                if let Some(ref pb) = pb {
                    pb.set_position(reader.bytes_processed());
                }

                if count % 10000 == 0 {
                    info!("Processed {} records...", count);
                }
            }
            None => break,
        }
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    println!("Total records: {}", count);
    info!("Processed {} bytes", reader.bytes_processed());

    Ok(())
}

fn deduplicate_dataset(input: PathBuf, output: PathBuf, field: String) -> Result<()> {
    info!("Deduplicating dataset: {:?} -> {:?}", input, output);
    info!("Using field: {}", field);

    let mut reader = open_dataset(&input)?;
    let mut seen_hashes = std::collections::HashSet::new();
    let mut unique_count = 0;
    let mut duplicate_count = 0;

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

    // For now, just count unique vs duplicates
    // In a full implementation, we'd write to the output file
    loop {
        match reader.next() {
            Some(result) => {
                let record = result?;

                if let Some(text) = record.data.get(&field).and_then(|v| v.as_str()) {
                    let hash = hash_string(text);

                    if seen_hashes.insert(hash) {
                        unique_count += 1;
                    } else {
                        duplicate_count += 1;
                    }
                }

                if let Some(ref pb) = pb {
                    pb.set_position(reader.bytes_processed());
                }

                if (unique_count + duplicate_count) % 10000 == 0 {
                    info!(
                        "Processed {} records ({} unique, {} duplicates)...",
                        unique_count + duplicate_count,
                        unique_count,
                        duplicate_count
                    );
                }
            }
            None => break,
        }
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    println!("Deduplication complete!");
    println!("  Total records: {}", unique_count + duplicate_count);
    println!("  Unique: {}", unique_count);
    println!("  Duplicates: {}", duplicate_count);
    println!(
        "  Deduplication rate: {:.2}%",
        (duplicate_count as f64 / (unique_count + duplicate_count) as f64) * 100.0
    );

    Ok(())
}
