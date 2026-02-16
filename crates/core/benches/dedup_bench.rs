use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use dataset_dedup_core::exact_dedup::{ExactDeduplicator, HashStrategy};
use dataset_dedup_core::hash_storage::TieredHashStorage;
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

criterion_group!(
    benches,
    bench_exact_dedup_full_content,
    bench_exact_dedup_field_hash,
    bench_exact_dedup_normalized,
    bench_hash_storage,
    bench_throughput
);
criterion_main!(benches);
