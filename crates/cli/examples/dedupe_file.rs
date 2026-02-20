use fastdedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
use fastdedup_core::fuzzy_dedup::FuzzyDeduplicator;
use fastdedup_filters::text_preprocessing::TextNormalizer;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 fastdedup\n");

    // Configuration
    let input_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/test_dataset.jsonl".to_string());
    let output_file = format!("{}.deduped.jsonl", input_file);
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "fuzzy".to_string());

    println!("Input: {}", input_file);
    println!("Output: {}", output_file);
    println!("Mode: {}\n", mode);

    // Read input file
    let file = File::open(&input_file)?;
    let reader = BufReader::new(file);
    let mut records: Vec<Value> = Vec::new();

    println!("📖 Reading records...");
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("⚠️  Line {}: Parse error: {}", line_num + 1, e),
        }
    }
    println!("✓ Read {} records\n", records.len());

    // Deduplicate based on mode
    let unique_records = match mode.as_str() {
        "exact" => exact_dedup(&records),
        "fuzzy" => fuzzy_dedup(&records),
        "exact-normalized" => exact_dedup_normalized(&records),
        _ => {
            eprintln!("Unknown mode: {}. Use 'exact', 'fuzzy', or 'exact-normalized'", mode);
            return Ok(());
        }
    };

    // Write output
    println!("\n💾 Writing deduplicated records...");
    let out_file = File::create(&output_file)?;
    let mut writer = BufWriter::new(out_file);

    for record in &unique_records {
        writeln!(writer, "{}", serde_json::to_string(record)?)?;
    }
    writer.flush()?;

    // Summary
    println!("✓ Wrote {} unique records to {}", unique_records.len(), output_file);
    println!("\n📊 Summary:");
    println!("  Total records: {}", records.len());
    println!("  Unique records: {}", unique_records.len());
    println!("  Duplicates removed: {}", records.len() - unique_records.len());
    println!("  Deduplication rate: {:.2}%",
             (records.len() - unique_records.len()) as f64 / records.len() as f64 * 100.0);

    Ok(())
}

fn exact_dedup(records: &[Value]) -> Vec<Value> {
    println!("🔨 Running exact deduplication (full content hash)...");
    let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);
    let mut unique = Vec::new();

    for record in records {
        if !dedup.is_duplicate(record) {
            unique.push(record.clone());
        }
    }

    let stats = dedup.stats();
    println!("  Bloom filter effectiveness: {:.2}%", stats.bloom_effectiveness());

    unique
}

fn exact_dedup_normalized(records: &[Value]) -> Vec<Value> {
    println!("🔨 Running exact deduplication (normalized text field)...");
    let mut dedup = ExactDeduplicator::new(HashStrategy::Normalized("text".to_string()));
    let mut unique = Vec::new();

    for record in records {
        if !dedup.is_duplicate(record) {
            unique.push(record.clone());
        }
    }

    let stats = dedup.stats();
    println!("  Bloom filter effectiveness: {:.2}%", stats.bloom_effectiveness());

    unique
}

fn fuzzy_dedup(records: &[Value]) -> Vec<Value> {
    println!("🌐 Running fuzzy deduplication (MinHash + LSH, threshold=0.7)...");
    let mut dedup = FuzzyDeduplicator::new(0.7);
    let mut unique = Vec::new();
    let mut duplicate_clusters = Vec::new();

    for (id, record) in records.iter().enumerate() {
        let duplicates = dedup.find_duplicates(record);

        if duplicates.is_empty() {
            // No duplicates found - add to index and output
            dedup.add_record(id, record);
            unique.push(record.clone());
        } else {
            // Found duplicates - track the cluster
            if !duplicate_clusters.iter().any(|cluster: &Vec<usize>| cluster.contains(&id)) {
                let mut cluster = vec![id];
                cluster.extend(&duplicates);
                duplicate_clusters.push(cluster);
            }
        }
    }

    let stats = dedup.stats();
    println!("  LSH candidates checked: {}", stats.lsh_candidates_checked);
    println!("  Verified duplicates: {}", stats.verified_duplicates);
    println!("  LSH precision: {:.2}%", stats.lsh_precision());

    if !duplicate_clusters.is_empty() {
        println!("\n  📋 Duplicate clusters found:");
        for (i, cluster) in duplicate_clusters.iter().enumerate().take(5) {
            println!("    Cluster {}: {:?}", i + 1, cluster);
        }
        if duplicate_clusters.len() > 5 {
            println!("    ... and {} more clusters", duplicate_clusters.len() - 5);
        }
    }

    unique
}
