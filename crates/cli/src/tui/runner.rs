//! Dedup execution for the TUI.
//!
//! Each function runs synchronously in a `std::thread` and sends
//! `ProgressMsg` updates over an `mpsc` channel.  The TUI polls
//! the channel every 50 ms without blocking the event loop.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::time::Instant;

use rayon::prelude::*;

use dataset_dedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
use dataset_dedup_core::fuzzy_dedup::{FuzzyDeduplicator, FuzzyDedupConfig};
use dataset_dedup_formats::open_dataset;

use super::{ProgressMsg, RunResults};

const BATCH_SIZE: usize = 2000;
const REPORT_EVERY: u64 = 10_000;

pub fn run_fuzzy(
    tx: Sender<ProgressMsg>,
    input: PathBuf,
    output: PathBuf,
    field: String,
    threshold: f64,
    num_hashes: usize,
    shingle_size: usize,
    word_shingles: bool,
    bands: Option<usize>,
    rows_per_band: Option<usize>,
) {
    let start = Instant::now();

    let result = run_fuzzy_inner(
        &tx,
        input,
        output,
        field,
        threshold,
        num_hashes,
        shingle_size,
        word_shingles,
        bands,
        rows_per_band,
        start,
    );

    match result {
        Ok(r) => {
            let _ = tx.send(ProgressMsg::Done(r));
        }
        Err(e) => {
            let _ = tx.send(ProgressMsg::Error(e.to_string()));
        }
    }
}

fn run_fuzzy_inner(
    tx: &Sender<ProgressMsg>,
    input: PathBuf,
    output: PathBuf,
    field: String,
    threshold: f64,
    num_hashes: usize,
    shingle_size: usize,
    word_shingles: bool,
    bands_arg: Option<usize>,
    rows_per_band_arg: Option<usize>,
    start: Instant,
) -> anyhow::Result<RunResults> {
    let rows_per_band = rows_per_band_arg.unwrap_or(8);
    let num_bands = bands_arg.unwrap_or(num_hashes / rows_per_band);
    let removed_output = removed_path(&output);

    let mut reader = open_dataset(&input)?;
    let total_records = reader.total_records().map(|r| r as u64);

    let config = FuzzyDedupConfig {
        similarity_threshold: threshold,
        text_field: field,
        num_hashes,
        shingle_size,
        word_shingles,
        num_bands,
        rows_per_band,
    };
    let mut deduplicator = FuzzyDeduplicator::with_config(config);

    let mut clean_writer = BufWriter::new(File::create(&output)?);
    let mut removed_writer = BufWriter::new(File::create(&removed_output)?);

    let mut total: usize = 0;
    let mut unique: usize = 0;
    let mut duplicates: usize = 0;

    loop {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            match reader.next() {
                Some(Ok(r)) => batch.push(r),
                Some(Err(e)) => return Err(e.into()),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }

        let signatures: Vec<_> = batch
            .par_iter()
            .map(|r| deduplicator.prepare_signature(&r.data))
            .collect();

        for (record, sig_opt) in batch.into_iter().zip(signatures) {
            let dups = sig_opt.and_then(|sig| deduplicator.process_prepared(total, sig));
            total += 1;

            match dups {
                None => {
                    unique += 1;
                    writeln!(clean_writer, "{}", serde_json::to_string(&record.data)?)?;
                }
                Some(ref dup_matches) => {
                    duplicates += 1;
                    let mut annotated = record.data.clone();
                    if let serde_json::Value::Object(ref mut map) = annotated {
                        let ids: Vec<_> = dup_matches
                            .iter()
                            .map(|(id, _)| serde_json::json!(id))
                            .collect();
                        let scores: Vec<_> = dup_matches
                            .iter()
                            .map(|(_, sim)| {
                                let rounded = (sim * 10_000.0).round() / 10_000.0;
                                serde_json::json!(rounded)
                            })
                            .collect();
                        map.insert(
                            "__duplicate_of".to_string(),
                            serde_json::Value::Array(ids),
                        );
                        map.insert(
                            "__similarity_scores".to_string(),
                            serde_json::Value::Array(scores),
                        );
                    }
                    writeln!(removed_writer, "{}", serde_json::to_string(&annotated)?)?;
                }
            }

            if total as u64 % REPORT_EVERY == 0 {
                let _ = tx.send(ProgressMsg::Update {
                    processed: total as u64,
                    duplicates: duplicates as u64,
                    total: total_records,
                });
            }
        }
    }

    clean_writer.flush()?;
    removed_writer.flush()?;

    let stats = deduplicator.stats();

    Ok(RunResults {
        total: total as u64,
        unique: unique as u64,
        duplicates: duplicates as u64,
        elapsed: start.elapsed(),
        lsh_precision: Some(stats.lsh_precision()),
        output_path: output.to_string_lossy().into_owned(),
        removed_path: Some(removed_output.to_string_lossy().into_owned()),
    })
}

pub fn run_exact(
    tx: Sender<ProgressMsg>,
    input: PathBuf,
    output: PathBuf,
    field: String,
    _normalize: bool,
) {
    let start = Instant::now();

    let result = run_exact_inner(&tx, input, output, field, start);
    match result {
        Ok(r) => {
            let _ = tx.send(ProgressMsg::Done(r));
        }
        Err(e) => {
            let _ = tx.send(ProgressMsg::Error(e.to_string()));
        }
    }
}

fn run_exact_inner(
    tx: &Sender<ProgressMsg>,
    input: PathBuf,
    output: PathBuf,
    field: String,
    start: Instant,
) -> anyhow::Result<RunResults> {
    let mut reader = open_dataset(&input)?;
    let total_records = reader.total_records().map(|r| r as u64);

    let strategy = if field.is_empty() {
        HashStrategy::FullContent
    } else {
        HashStrategy::Field(field)
    };
    let mut deduplicator = ExactDeduplicator::new(strategy);
    let mut writer = BufWriter::new(File::create(&output)?);

    let mut total: u64 = 0;
    let mut unique: u64 = 0;
    let mut duplicates: u64 = 0;

    while let Some(result) = reader.next() {
        let record = result?;
        total += 1;

        if !deduplicator.is_duplicate(&record.data) {
            unique += 1;
            writeln!(writer, "{}", serde_json::to_string(&record.data)?)?;
        } else {
            duplicates += 1;
        }

        if total % REPORT_EVERY == 0 {
            let _ = tx.send(ProgressMsg::Update {
                processed: total,
                duplicates,
                total: total_records,
            });
        }
    }

    writer.flush()?;

    Ok(RunResults {
        total,
        unique,
        duplicates,
        elapsed: start.elapsed(),
        lsh_precision: None,
        output_path: output.to_string_lossy().into_owned(),
        removed_path: None,
    })
}

fn removed_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let parent = output.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.removed.jsonl", stem))
}
