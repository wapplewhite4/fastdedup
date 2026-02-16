//! High-performance parallel processing pipeline
//!
//! Orchestrates the full deduplication and filtering pipeline with
//! parallel processing using Rayon for maximum throughput.

use crate::exact_dedup::{ExactDeduplicator, HashStrategy};
use crate::fuzzy_dedup::FuzzyDeduplicator;
use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_records: usize,
    pub exact_duplicates: usize,
    pub fuzzy_duplicates: usize,
    pub filtered_records: usize,
    pub unique_records: usize,
    pub bytes_processed: u64,
}

impl PipelineStats {
    pub fn deduplication_rate(&self) -> f64 {
        let total_dups = self.exact_duplicates + self.fuzzy_duplicates;
        if self.total_records > 0 {
            (total_dups as f64 / self.total_records as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn filter_rate(&self) -> f64 {
        if self.total_records > 0 {
            (self.filtered_records as f64 / self.total_records as f64) * 100.0
        } else {
            0.0
        }
    }

    pub fn retention_rate(&self) -> f64 {
        if self.total_records > 0 {
            (self.unique_records as f64 / self.total_records as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable exact deduplication
    pub exact_dedup: Option<ExactDedupConfig>,
    /// Enable fuzzy deduplication
    pub fuzzy_dedup: Option<FuzzyDedupConfig>,
    /// Number of threads (None = auto-detect)
    pub num_threads: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            exact_dedup: Some(ExactDedupConfig {
                strategy: HashStrategy::FullContent,
            }),
            fuzzy_dedup: None,
            num_threads: None,
        }
    }
}

/// Exact deduplication configuration
#[derive(Debug, Clone)]
pub struct ExactDedupConfig {
    pub strategy: HashStrategy,
}

/// Fuzzy deduplication configuration
#[derive(Debug, Clone)]
pub struct FuzzyDedupConfig {
    pub threshold: f64,
}

/// High-performance deduplication pipeline
pub struct Pipeline {
    config: PipelineConfig,
    exact_dedup: Option<Arc<Mutex<ExactDeduplicator>>>,
    fuzzy_dedup: Option<Arc<Mutex<FuzzyDeduplicator>>>,
    stats: Arc<Mutex<PipelineStats>>,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        // Configure rayon thread pool
        if let Some(num_threads) = config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .ok();
        }

        let exact_dedup = config.exact_dedup.as_ref().map(|cfg| {
            Arc::new(Mutex::new(ExactDeduplicator::new(cfg.strategy.clone())))
        });

        let fuzzy_dedup = config.fuzzy_dedup.as_ref().map(|cfg| {
            Arc::new(Mutex::new(FuzzyDeduplicator::new(cfg.threshold)))
        });

        Self {
            config,
            exact_dedup,
            fuzzy_dedup,
            stats: Arc::new(Mutex::new(PipelineStats::default())),
        }
    }

    /// Process a single record through the pipeline
    pub fn process_record(&self, record: &Value) -> Result<bool> {
        // Check exact duplicates
        if let Some(ref dedup) = self.exact_dedup {
            let mut dedup = dedup.lock().unwrap();
            if dedup.is_duplicate(record) {
                let mut stats = self.stats.lock().unwrap();
                stats.exact_duplicates += 1;
                return Ok(false);
            }
        }

        // Check fuzzy duplicates
        if let Some(ref dedup) = self.fuzzy_dedup {
            let mut dedup = dedup.lock().unwrap();
            let duplicates = dedup.find_duplicates(record);
            if !duplicates.is_empty() {
                let mut stats = self.stats.lock().unwrap();
                stats.fuzzy_duplicates += 1;
                return Ok(false);
            }
            // Add to index for future comparisons
            let record_id = {
                let stats = self.stats.lock().unwrap();
                stats.total_records
            };
            dedup.add_record(record_id, record);
        }

        Ok(true)
    }

    /// Process a chunk of records in parallel
    pub fn process_chunk(&self, chunk: Vec<Value>) -> Vec<Value> {
        // Update total count
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_records += chunk.len();
        }

        // Process records in parallel
        let results: Vec<(Value, bool)> = chunk
            .into_par_iter()
            .map(|record| {
                let keep = self.process_record(&record).unwrap_or(false);
                (record, keep)
            })
            .collect();

        // Filter and collect results
        let unique: Vec<Value> = results
            .into_iter()
            .filter_map(|(record, keep)| if keep { Some(record) } else { None })
            .collect();

        {
            let mut stats = self.stats.lock().unwrap();
            stats.unique_records += unique.len();
        }

        unique
    }

    /// Process records in batches for better performance
    pub fn process_batch(&self, records: Vec<Value>) -> Vec<Value> {
        // Split into chunks
        let chunks: Vec<Vec<Value>> = records
            .chunks(self.config.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Process chunks sequentially to maintain ordering
        // (parallel processing happens within each chunk)
        let mut results = Vec::new();
        for chunk in chunks {
            let mut processed = self.process_chunk(chunk);
            results.append(&mut processed);
        }

        results
    }

    /// Get current statistics
    pub fn stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = PipelineStats::default();
    }
}

/// Builder for pipeline configuration
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    pub fn exact_dedup(mut self, strategy: HashStrategy) -> Self {
        self.config.exact_dedup = Some(ExactDedupConfig { strategy });
        self
    }

    pub fn fuzzy_dedup(mut self, threshold: f64) -> Self {
        self.config.fuzzy_dedup = Some(FuzzyDedupConfig { threshold });
        self
    }

    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    pub fn build(self) -> Pipeline {
        Pipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_pipeline_exact_dedup() {
        let pipeline = PipelineBuilder::new()
            .exact_dedup(HashStrategy::FullContent)
            .build();

        let records = vec![
            json!({"text": "hello world"}),
            json!({"text": "hello world"}), // duplicate
            json!({"text": "goodbye world"}),
        ];

        let results = pipeline.process_batch(records);
        assert_eq!(results.len(), 2);

        let stats = pipeline.stats();
        assert_eq!(stats.total_records, 3);
        assert_eq!(stats.exact_duplicates, 1);
        assert_eq!(stats.unique_records, 2);
    }

    #[test]
    fn test_pipeline_stats() {
        let pipeline = PipelineBuilder::new()
            .exact_dedup(HashStrategy::FullContent)
            .build();

        let records = vec![
            json!({"text": "a"}),
            json!({"text": "a"}),
            json!({"text": "b"}),
            json!({"text": "c"}),
        ];

        pipeline.process_batch(records);
        let stats = pipeline.stats();

        assert_eq!(stats.total_records, 4);
        assert_eq!(stats.exact_duplicates, 1);
        assert_eq!(stats.unique_records, 3);
        assert_eq!(stats.deduplication_rate(), 25.0);
        assert_eq!(stats.retention_rate(), 75.0);
    }

    #[test]
    fn test_pipeline_chunking() {
        let pipeline = PipelineBuilder::new()
            .chunk_size(2)
            .exact_dedup(HashStrategy::FullContent)
            .build();

        let records = vec![
            json!({"id": 1}),
            json!({"id": 2}),
            json!({"id": 3}),
            json!({"id": 4}),
            json!({"id": 5}),
        ];

        let results = pipeline.process_batch(records);
        assert_eq!(results.len(), 5);
    }
}
