use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fastdedup_filters::text_preprocessing::TextNormalizer;

fn bench_text_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_normalization");

    let sample_texts = vec![
        "The quick brown fox jumps over the lazy dog",
        "Hello, World! How are you doing today?",
        "This is a sample text with UPPERCASE and lowercase letters.",
        "Special characters: $100.50, 50% off!!!",
        "   Multiple   spaces   and   tabs\t\there   ",
        "café résumé naïve", // Accented characters
    ];

    // Benchmark aggressive normalization
    group.throughput(Throughput::Elements(sample_texts.len() as u64));
    group.bench_function("aggressive", |b| {
        let normalizer = TextNormalizer::aggressive();
        b.iter(|| {
            for text in &sample_texts {
                black_box(normalizer.normalize(text));
            }
        });
    });

    // Benchmark conservative normalization
    group.bench_function("conservative", |b| {
        let normalizer = TextNormalizer::conservative();
        b.iter(|| {
            for text in &sample_texts {
                black_box(normalizer.normalize(text));
            }
        });
    });

    // Benchmark balanced normalization
    group.bench_function("balanced", |b| {
        let normalizer = TextNormalizer::balanced();
        b.iter(|| {
            for text in &sample_texts {
                black_box(normalizer.normalize(text));
            }
        });
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Generate 1000 sample documents
    let documents: Vec<String> = (0..1000)
        .map(|i| {
            format!(
                "Document {} contains some sample text with various characters and numbers like {}",
                i,
                i * 100
            )
        })
        .collect();

    group.throughput(Throughput::Elements(documents.len() as u64));
    group.bench_function("1000_docs_balanced", |b| {
        let normalizer = TextNormalizer::balanced();
        b.iter(|| {
            for doc in &documents {
                black_box(normalizer.normalize(doc));
            }
        });
    });

    group.finish();
}

fn bench_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_reuse");

    let texts = vec![
        "Sample text 1",
        "Another sample text",
        "Yet another text",
    ];

    group.bench_function("with_reuse", |b| {
        let normalizer = TextNormalizer::balanced();
        let mut buffer = String::new();
        b.iter(|| {
            for text in &texts {
                normalizer.normalize_into(text, &mut buffer);
                black_box(&buffer);
            }
        });
    });

    group.bench_function("without_reuse", |b| {
        let normalizer = TextNormalizer::balanced();
        b.iter(|| {
            for text in &texts {
                black_box(normalizer.normalize(text));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_text_normalization, bench_throughput, bench_reuse);
criterion_main!(benches);
