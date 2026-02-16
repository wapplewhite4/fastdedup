use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dataset_dedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
use dataset_dedup_core::hash_storage::TieredHashStorage;
use dataset_dedup_core::fuzzy_dedup::FuzzyDeduplicator;
use dataset_dedup_core::minhash::{MinHasher, LSHIndex};
use serde_json::json;

fn bench_exact_dedup_full_content(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_dedup_full_content");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("10k_unique", |b| {
        b.iter(|| {
            let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::FullContent, 10_000);
            for i in 0..10_000 {
                let record = json!({"id": i, "text": format!("text_{}", i)});
                black_box(dedup.is_duplicate(&record));
            }
        });
    });

    group.bench_function("10k_50pct_dup", |b| {
        b.iter(|| {
            let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::FullContent, 10_000);
            for i in 0..10_000 {
                let record = json!({"id": i % 5000, "text": format!("text_{}", i % 5000)});
                black_box(dedup.is_duplicate(&record));
            }
        });
    });

    group.finish();
}

fn bench_exact_dedup_field_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_dedup_field");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("field_10k_unique", |b| {
        b.iter(|| {
            let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::Field("text".to_string()), 10_000);
            for i in 0..10_000 {
                let record = json!({"id": i, "text": format!("text_{}", i)});
                black_box(dedup.is_duplicate(&record));
            }
        });
    });

    group.finish();
}

fn bench_exact_dedup_normalized(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_dedup_normalized");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("normalized_10k", |b| {
        b.iter(|| {
            let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::Normalized("text".to_string()), 10_000);
            for i in 0..10_000 {
                let text = if i % 2 == 0 {
                    format!("  TEXT_{}  ", i / 2)
                } else {
                    format!("text_{}", i / 2)
                };
                let record = json!({"text": text});
                black_box(dedup.is_duplicate(&record));
            }
        });
    });

    group.finish();
}

fn bench_hash_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_storage");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("tiered_insert_10k", |b| {
        b.iter(|| {
            let mut storage = TieredHashStorage::temporary().unwrap();
            for i in 0..10_000u64 {
                black_box(storage.insert(i).unwrap());
            }
            storage.clear().unwrap();
        });
    });

    group.bench_function("tiered_lookup_10k", |b| {
        let mut storage = TieredHashStorage::temporary().unwrap();
        for i in 0..10_000u64 {
            storage.insert(i).unwrap();
        }

        b.iter(|| {
            for i in 0..10_000u64 {
                black_box(storage.contains(i).unwrap());
            }
        });

        storage.clear().unwrap();
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Benchmark 100k records
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("100k_records", |b| {
        b.iter(|| {
            let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::Field("text".to_string()), 100_000);
            for i in 0..100_000 {
                let record = json!({"text": format!("doc_{}", i % 50_000)}); // 50% dup rate
                black_box(dedup.is_duplicate(&record));
            }
        });
    });

    group.finish();
}

fn bench_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash");
    group.throughput(Throughput::Elements(1_000));

    let texts: Vec<String> = (0..1_000)
        .map(|i| format!("Document {} with some sample text content", i))
        .collect();

    group.bench_function("compute_signatures_1k", |b| {
        let hasher = MinHasher::new(128, 3);
        b.iter(|| {
            for text in &texts {
                black_box(hasher.compute_signature(text));
            }
        });
    });

    group.finish();
}

fn bench_lsh_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("lsh_index");

    // Prepare signatures
    let hasher = MinHasher::new(128, 3);
    let signatures: Vec<_> = (0..1_000)
        .map(|i| hasher.compute_signature(&format!("Document {} content", i)))
        .collect();

    group.throughput(Throughput::Elements(1_000));
    group.bench_function("insert_1k", |b| {
        b.iter(|| {
            let mut index = LSHIndex::new(32, 4);
            for (i, sig) in signatures.iter().enumerate() {
                index.insert(i, sig.clone());
            }
        });
    });

    // Setup index for query benchmark
    let mut index = LSHIndex::new(32, 4);
    for (i, sig) in signatures.iter().enumerate() {
        index.insert(i, sig.clone());
    }

    group.bench_function("query_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(index.query(&signatures[i], 0.7));
            }
        });
    });

    group.finish();
}

fn bench_fuzzy_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuzzy_dedup");

    let records: Vec<_> = (0..1_000)
        .map(|i| json!({"text": format!("Document {} with sample content", i % 500)}))
        .collect();

    group.throughput(Throughput::Elements(1_000));
    group.bench_function("process_1k_records", |b| {
        b.iter(|| {
            let mut dedup = FuzzyDeduplicator::new(0.7);
            for (i, record) in records.iter().enumerate() {
                dedup.add_record(i, record);
                black_box(dedup.find_duplicates(record));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_exact_dedup_full_content,
    bench_exact_dedup_field_hash,
    bench_exact_dedup_normalized,
    bench_hash_storage,
    bench_throughput,
    bench_minhash,
    bench_lsh_index,
    bench_fuzzy_dedup
);
criterion_main!(benches);
