use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/test_dataset.jsonl".to_string());

    let mut writer = BufWriter::new(File::create(&output_file)?);

    println!("🔧 Generating test dataset: {}\n", output_file);

    // Exact duplicates
    println!("Adding exact duplicates...");
    for _ in 0..3 {
        writeln!(writer, "{}", json!({"id": 1, "text": "The quick brown fox jumps over the lazy dog"}))?;
    }

    // Near duplicates (should be caught by fuzzy dedup)
    println!("Adding near duplicates...");
    writeln!(writer, "{}", json!({"id": 2, "text": "The quick brown fox jumps over the lazy dog"}))?;
    writeln!(writer, "{}", json!({"id": 3, "text": "The quick brown fox jumps over a lazy dog"}))?;
    writeln!(writer, "{}", json!({"id": 4, "text": "The quick brown foxes jump over the lazy dog"}))?;

    // Case variations (should be caught by normalized exact dedup)
    println!("Adding case variations...");
    writeln!(writer, "{}", json!({"id": 5, "text": "Hello World"}))?;
    writeln!(writer, "{}", json!({"id": 6, "text": "hello world"}))?;
    writeln!(writer, "{}", json!({"id": 7, "text": "HELLO WORLD"}))?;
    writeln!(writer, "{}", json!({"id": 8, "text": "  Hello   World  "}))?;

    // Punctuation variations
    println!("Adding punctuation variations...");
    writeln!(writer, "{}", json!({"id": 9, "text": "Hello, World!"}))?;
    writeln!(writer, "{}", json!({"id": 10, "text": "Hello World!!!"}))?;
    writeln!(writer, "{}", json!({"id": 11, "text": "Hello... World?"}))?;

    // Similar content
    println!("Adding similar content...");
    writeln!(writer, "{}", json!({"id": 12, "text": "Python is a great programming language"}))?;
    writeln!(writer, "{}", json!({"id": 13, "text": "Python is an excellent programming language"}))?;
    writeln!(writer, "{}", json!({"id": 14, "text": "Python is a wonderful programming language"}))?;

    // Completely unique
    println!("Adding unique records...");
    writeln!(writer, "{}", json!({"id": 15, "text": "Rust provides memory safety without garbage collection"}))?;
    writeln!(writer, "{}", json!({"id": 16, "text": "Machine learning is transforming industries"}))?;
    writeln!(writer, "{}", json!({"id": 17, "text": "Climate change requires immediate action"}))?;
    writeln!(writer, "{}", json!({"id": 18, "text": "Artificial intelligence and ethics"}))?;
    writeln!(writer, "{}", json!({"id": 19, "text": "Quantum computing breakthrough announced"}))?;
    writeln!(writer, "{}", json!({"id": 20, "text": "Space exploration reaches new milestone"}))?;

    // More exact duplicates scattered throughout
    writeln!(writer, "{}", json!({"id": 21, "text": "Duplicate test record number one"}))?;
    writeln!(writer, "{}", json!({"id": 22, "text": "Duplicate test record number one"}))?;
    writeln!(writer, "{}", json!({"id": 23, "text": "Another unique record here"}))?;
    writeln!(writer, "{}", json!({"id": 24, "text": "Duplicate test record number one"}))?;

    // Near duplicates with typos
    writeln!(writer, "{}", json!({"id": 25, "text": "The cat sat on the mat"}))?;
    writeln!(writer, "{}", json!({"id": 26, "text": "The cat sat on the mat."}))?;
    writeln!(writer, "{}", json!({"id": 27, "text": "The cat sits on the mat"}))?;
    writeln!(writer, "{}", json!({"id": 28, "text": "A cat sat on the mat"}))?;

    writer.flush()?;

    println!("\n✓ Generated test dataset with 28 records");
    println!("  - Exact duplicates: ~10");
    println!("  - Near duplicates: ~10");
    println!("  - Case variations: ~4");
    println!("  - Unique records: ~10");
    println!("\nRun deduplication with:");
    println!("  cargo run --example dedupe_file {} exact", output_file);
    println!("  cargo run --example dedupe_file {} fuzzy", output_file);
    println!("  cargo run --example dedupe_file {} exact-normalized", output_file);

    Ok(())
}
