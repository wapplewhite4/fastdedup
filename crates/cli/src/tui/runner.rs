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

    let field_name = field.clone();
    let config = FuzzyDedupConfig {
        similarity_threshold: threshold,
        text_field: field,
        num_hashes,
        shingle_size,
        word_shingles,
        num_bands,
        rows_per_band,
        ..Default::default()
    };
    let mut deduplicator = FuzzyDeduplicator::with_config(config);

    let mut clean_writer = BufWriter::new(File::create(&output)?);
    let mut removed_writer = BufWriter::new(File::create(&removed_output)?);

    // Maps row_id → field text for kept records to populate `matched_value`.
    // Uses disk-backed storage to bound memory at scale.
    let mut field_values = crate::disk_kv::DiskBackedStringMap::new(2_000_000)?;

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
            let row_id = total;
            let dups = sig_opt.and_then(|sig| deduplicator.process_prepared(row_id, sig));
            total += 1;

            let field_text = record
                .data
                .get(&field_name)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            match dups {
                None => {
                    unique += 1;
                    field_values.insert(row_id, field_text)?;
                    writeln!(clean_writer, "{}", serde_json::to_string(&record.data)?)?;
                }
                Some(ref dup_matches) => {
                    duplicates += 1;
                    for &(dup_id, sim) in dup_matches {
                        let matched_value = field_values
                            .get(&dup_id)?
                            .unwrap_or_default();
                        let entry = serde_json::json!({
                            "row_id": row_id,
                            "duplicate_of_row_id": dup_id,
                            "field": field_name,
                            "value": field_text,
                            "matched_value": matched_value,
                            "similarity": (sim * 10_000.0).round() / 10_000.0,
                            "threshold": threshold,
                        });
                        writeln!(removed_writer, "{}", serde_json::to_string(&entry)?)?;
                    }
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
