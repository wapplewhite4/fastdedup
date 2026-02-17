//! Performance benchmarks for deduplication algorithms
//!
//! Run with: cargo bench -p dataset-dedup-core

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dataset_dedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
use dataset_dedup_core::fuzzy_dedup::FuzzyDeduplicator;
use dataset_dedup_core::minhash::MinHasher;
use dataset_dedup_core::pipeline::PipelineBuilder;
use serde_json::json;

/// Generate test data with configurable duplicate ratio
fn generate_test_data(size: usize, duplicate_ratio: f64) -> Vec<serde_json::Value> {
    let mut data = Vec::with_capacity(size);
    let unique_count = (size as f64 * (1.0 - duplicate_ratio)) as usize;

    // Generate unique records
    for i in 0..unique_count {
        data.push(json!({
            "id": i,
            "text": format!("This is unique text number {} with some content", i),
            "value": i * 2,
        }));
    }

    // Generate duplicates by repeating some records
    while data.len() < size {
        let idx = data.len() % unique_count;
        data.push(data[idx].clone());
    }

    data
}

fn bench_exact_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_dedup");

    for size in [1_000, 10_000, 100_000] {
        for dup_ratio in [0.1, 0.3, 0.5] {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("n={}_dup={}", size, dup_ratio)),
                &(size, dup_ratio),
                |b, &(size, dup_ratio)| {
                    let data = generate_test_data(size, dup_ratio);
                    b.iter(|| {
                        let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);
                        for record in &data {
                            black_box(dedup.is_duplicate(record));
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash");

    for size in [100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let data: Vec<String> = (0..size)
                    .map(|i| format!("This is test document {} with content", i))
                    .collect();

                b.iter(|| {
                    let hasher = MinHasher::new(128, 4);
                    for text in &data {
                        black_box(hasher.compute(text));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_pipeline_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_parallel");

    for size in [1_000, 10_000, 50_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let data = generate_test_data(size, 0.3);
                b.iter(|| {
                    let pipeline = PipelineBuilder::new()
                        .chunk_size(1_000)
                        .exact_dedup(HashStrategy::FullContent)
                        .build();

                    black_box(pipeline.process_batch(data.clone()));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_exact_dedup, bench_minhash, bench_pipeline_parallel);
criterion_main!(benches);
