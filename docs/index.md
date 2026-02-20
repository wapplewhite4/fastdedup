---
layout: post
title: "Deduplicating 15 Million Records in 3 Minutes with Rust"
---

Deduplicating 15 Million Records in 3 Minutes with Rust
Dataset deduplication is one of those problems that sounds simple until you're staring at a 30GB Parquet file and a Python script that's been running for four hours.
If you've worked on LLM dataset preparation, you know the pain. The standard tools — datatrove, text-dedup, custom pandas scripts — are slow, memory-hungry, or both. They were built for distributed clusters, not the single machine most ML engineers actually have.
I built fastdedup, a Rust CLI for exactly this use case: fast, memory-efficient deduplication on a single machine. This post covers the benchmarks I ran against standard Python baselines on the FineWeb sample-10BT dataset.
 
The Dataset
All benchmarks ran on FineWeb sample-10BT, a well-known public subset of the full FineWeb dataset used as a standard reference in the ML community.
•	14,868,862 records
•	29GB on disk (merged Parquet)
•	Field: text
Hardware: Hetzner CCX43 — 16 dedicated AMD EPYC Milan vCPUs, 64GB RAM, Ubuntu 24.04. All runs are single-machine, no distributed infrastructure.
 
Exact Deduplication
Exact dedup removes records with identical content. fastdedup uses a Bloom filter + AHash set pipeline to process records in a single streaming pass without loading the full dataset into memory.
Results
	fastdedup	DuckDB + SHA-256
Wall clock	2:55	7:55
Peak RAM	688 MB	21.9 GB
CPU cores used	1	4+
Records/sec	~85,000	~31,000
Duplicates removed	51,392	51,392
The duplicate counts match exactly, confirming correctness. fastdedup ran 2.7x faster using a single core and 32x less RAM than DuckDB using four cores.
A few things worth noting:
The 688MB peak RAM on a 29GB dataset is the standout number. The Bloom filter pre-screens candidates before the hash set, keeping memory usage flat regardless of dataset size. DuckDB peaked at 21.9GB because it had to materialize hash aggregations in memory — even with spill-to-disk configured it was operating at the edge of available RAM.
The current exact-dedup implementation is intentionally single-threaded. The deduplication state (AHashSet + Bloom filter) requires sequential access and the bottleneck is I/O rather than CPU. At ~85,000 records/sec the throughput is high enough that parallelism would yield diminishing returns for most dataset sizes.
 
Fuzzy Deduplication
Fuzzy dedup removes near-duplicate records using MinHash + LSH. This is significantly more expensive than exact dedup and is where Python tooling struggles most.
fastdedup uses character 3-grams + 128 MinHash signatures with 16 bands of 8 rows, matching standard FineWeb pipeline parameters. The comparison baseline is datatrove — the reference implementation used to produce FineWeb itself, making it the most credible possible comparison.
Results
	fastdedup	datatrove
Wall clock	36:44	3h50m+ (stage 1 only, terminated)
Peak RAM	23 GB	1.1 GB
CPU cores used	~5.5	1
Completed	Yes	No
Duplicates removed	105,044 (0.7%)	—
datatrove did not complete. After 3 hours and 50 minutes, stage 1 (MinHash signature computation) was still running and we terminated it. Stages 2 (bucket clustering) and 3 (filtering) had not started.
Why is datatrove so slow?
Profiling the traceback revealed the bottleneck: datatrove runs a full spaCy NLP pipeline on every document before computing shingles — tokenization, vocab lookup, lexeme creation. This is orders of magnitude more expensive than the character n-gram shingling fastdedup uses. It's doing linguistic analysis where simple character slicing suffices for deduplication purposes.
datatrove is also designed for distributed execution across hundreds of workers. Running it with tasks=1 on a single machine is not its intended use case — the FineWeb team ran it across a large CPU cluster. This benchmark represents how a typical ML engineer would actually run it locally.
RAM trade-off
The RAM difference is a real trade-off, not a clear win. datatrove streams intermediate data to disk (keeping RAM at 1.1GB) at the cost of heavy I/O between stages. fastdedup holds the LSH index in memory (23GB peak) for significantly faster processing. On a machine with sufficient RAM, the in-memory approach wins decisively on wall clock time.
23GB is well within the capacity of a standard cloud instance (this benchmark ran on a €0.15/hr Hetzner CCX43). If RAM is constrained, configuring fewer hashes or a lower band count reduces memory usage at a slight accuracy trade-off.
 
Reproducing These Benchmarks
All benchmark scripts, methodology, and raw results are available in the repository. The setup is straightforward:
# Install
cargo install fastdedup

# Exact dedup
fastdedup exact-dedup \
  -i ./dataset.parquet \
  -o ./deduped.parquet \
  --field text --normalize

# Fuzzy dedup
fastdedup fuzzy-dedup \
  -i ./dataset.parquet \
  -o ./deduped.parquet \
  --field text --threshold 0.8 \
  --num-hashes 128 --shingle-size 3
Hardware: Hetzner CCX43 (16 vCPU, 64GB RAM). Dataset: FineWeb sample-10BT, merged into a single Parquet file using DuckDB.
 
When to Use This
fastdedup is a good fit if you're:
•	Preparing training datasets on a single machine or modest cloud instance
•	Running deduplication as part of a pipeline where speed matters
•	Working with datasets in the 1M–100M record range
It's not the right tool if you need distributed processing across a cluster (use datatrove), or if you're working at trillion-token scale where no single-machine tool is appropriate.
 
What's Next
The tool is under active development. 
Feedback, issues, and contributions welcome on GitHub: [link]
 
Benchmarks run February 2026 on Hetzner CCX43 (16 vCPU AMD EPYC Milan, 64GB RAM). All results reproducible using scripts in the /benchmarks directory of the repository.

