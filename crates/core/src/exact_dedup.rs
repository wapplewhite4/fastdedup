//! Exact deduplication using content hashing
//!
//! Provides high-performance exact duplicate detection using various
//! hash strategies and optimizations like bloom filters.

use ahash::{AHashSet, AHasher};
use bloomfilter::Bloom;
use serde_json::Value;
use std::hash::{Hash, Hasher};
use tracing::{debug, info};

/// Statistics for deduplication operations
#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    /// Total number of records seen
    pub total_seen: usize,
    /// Number of duplicates found
    pub duplicates_found: usize,
    /// Number of unique records
    pub unique_count: usize,
    /// Number of bloom filter hits (potential duplicates)
    pub bloom_hits: usize,
    /// Number of bloom filter misses (definitely unique)
    pub bloom_misses: usize,
}

impl DedupStats {
    /// Get the deduplication rate as a percentage
    pub fn dedup_rate(&self) -> f64 {
        if self.total_seen == 0 {
            0.0
        } else {
            (self.duplicates_found as f64 / self.total_seen as f64) * 100.0
        }
    }

    /// Get the bloom filter effectiveness (true negatives)
    pub fn bloom_effectiveness(&self) -> f64 {
        let total_checks = self.bloom_hits + self.bloom_misses;
        if total_checks == 0 {
            0.0
        } else {
            (self.bloom_misses as f64 / total_checks as f64) * 100.0
        }
    }
}

/// Strategy for hashing records
#[derive(Debug, Clone)]
pub enum HashStrategy {
    /// Hash the entire JSON content
    FullContent,
    /// Hash only a specific field
    Field(String),
    /// Hash a field with normalization (lowercase, trim)
    Normalized(String),
    /// Hash multiple fields
    MultiField(Vec<String>),
}

impl HashStrategy {
    /// Compute hash for a record based on the strategy
    pub fn compute_hash(&self, value: &Value) -> Option<u64> {
        match self {
            HashStrategy::FullContent => {
                let content = value.to_string();
                Some(Self::hash_string(&content))
            }
            HashStrategy::Field(field) => value
                .get(field)
                .and_then(|v| v.as_str())
                .map(|s| Self::hash_string(s)),
            HashStrategy::Normalized(field) => value
                .get(field)
                .and_then(|v| v.as_str())
                .map(|s| {
                    let normalized = s.trim().to_lowercase();
                    Self::hash_string(&normalized)
                }),
            HashStrategy::MultiField(fields) => {
                let mut hasher = AHasher::default();
                for field in fields {
                    if let Some(v) = value.get(field) {
                        v.to_string().hash(&mut hasher);
                    }
                }
                Some(hasher.finish())
            }
        }
    }

    /// Hash a string using ahash
    fn hash_string(s: &str) -> u64 {
        let mut hasher = AHasher::default();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// Exact deduplicator with bloom filter optimization
pub struct ExactDeduplicator {
    /// Set of seen hashes
    seen_hashes: AHashSet<u64>,
    /// Bloom filter for quick negative lookups
    bloom: Bloom<u64>,
    /// Hash strategy
    hash_strategy: HashStrategy,
    /// Statistics
    stats: DedupStats,
}

impl ExactDeduplicator {
    /// Create a new exact deduplicator with default settings
    pub fn new(hash_strategy: HashStrategy) -> Self {
        Self::with_capacity(hash_strategy, 10_000_000) // Default 10M capacity
    }

    /// Create a new exact deduplicator with specified capacity
    ///
    /// The capacity is used to initialize both the HashSet and Bloom filter.
    /// Bloom filter is configured for ~1% false positive rate.
    pub fn with_capacity(hash_strategy: HashStrategy, capacity: usize) -> Self {
        info!(
            "Creating ExactDeduplicator with capacity {} and strategy {:?}",
            capacity, hash_strategy
        );

        // Bloom filter with 1% false positive rate
        let bloom = Bloom::new_for_fp_rate(capacity, 0.01);

        Self {
            seen_hashes: AHashSet::with_capacity(capacity),
            bloom,
            hash_strategy,
            stats: DedupStats::default(),
        }
    }

    /// Check if a record is a duplicate
    ///
    /// Returns `true` if this is a duplicate, `false` if unique.
    /// Uses bloom filter for fast negative lookups before checking HashSet.
    pub fn is_duplicate(&mut self, value: &Value) -> bool {
        self.stats.total_seen += 1;

        // Compute hash based on strategy
        let hash = match self.hash_strategy.compute_hash(value) {
            Some(h) => h,
            None => {
                debug!("Could not compute hash for record, treating as unique");
                self.stats.unique_count += 1;
                return false;
            }
        };

        // First check bloom filter
        if !self.bloom.check(&hash) {
            // Definitely not seen before (bloom filter negative)
            self.bloom.set(&hash);
            self.seen_hashes.insert(hash);
            self.stats.bloom_misses += 1;
            self.stats.unique_count += 1;
            return false;
        }

        // Bloom filter positive - might be duplicate
        self.stats.bloom_hits += 1;

        // Check actual HashSet
        if self.seen_hashes.contains(&hash) {
            // Definitely a duplicate
            self.stats.duplicates_found += 1;
            true
        } else {
            // False positive from bloom filter
            self.seen_hashes.insert(hash);
            self.stats.unique_count += 1;
            false
        }
    }

    /// Check if a hash is a duplicate (for pre-computed hashes)
    pub fn is_duplicate_hash(&mut self, hash: u64) -> bool {
        self.stats.total_seen += 1;

        // First check bloom filter
        if !self.bloom.check(&hash) {
            // Definitely not seen before
            self.bloom.set(&hash);
            self.seen_hashes.insert(hash);
            self.stats.bloom_misses += 1;
            self.stats.unique_count += 1;
            return false;
        }

        // Bloom filter positive
        self.stats.bloom_hits += 1;

        if self.seen_hashes.contains(&hash) {
            self.stats.duplicates_found += 1;
            true
        } else {
            self.seen_hashes.insert(hash);
            self.stats.unique_count += 1;
            false
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &DedupStats {
        &self.stats
    }

    /// Get the number of unique hashes stored
    pub fn unique_count(&self) -> usize {
        self.seen_hashes.len()
    }

    /// Clear all seen hashes and reset statistics
    pub fn clear(&mut self) {
        self.seen_hashes.clear();
        self.bloom.clear();
        self.stats = DedupStats::default();
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        // HashSet: ~24 bytes overhead + capacity * (8 bytes per u64 + ~8 bytes overhead)
        let hashset_size = 24 + self.seen_hashes.capacity() * 16;

        // Bloom filter bitmap size
        let bloom_size = (self.bloom.number_of_bits() / 8) as usize;

        hashset_size + bloom_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_full_content_hash() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);

        let record1 = json!({"text": "hello", "id": 1});
        let record2 = json!({"text": "hello", "id": 1});
        let record3 = json!({"text": "world", "id": 2});

        assert!(!dedup.is_duplicate(&record1));
        assert!(dedup.is_duplicate(&record2));
        assert!(!dedup.is_duplicate(&record3));

        assert_eq!(dedup.stats().unique_count, 2);
        assert_eq!(dedup.stats().duplicates_found, 1);
        assert_eq!(dedup.stats().total_seen, 3);
    }

    #[test]
    fn test_field_hash() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::Field("text".to_string()));

        let record1 = json!({"text": "hello", "id": 1});
        let record2 = json!({"text": "hello", "id": 2}); // Different ID, same text
        let record3 = json!({"text": "world", "id": 3});

        assert!(!dedup.is_duplicate(&record1));
        assert!(dedup.is_duplicate(&record2)); // Duplicate because text is same
        assert!(!dedup.is_duplicate(&record3));

        assert_eq!(dedup.stats().unique_count, 2);
        assert_eq!(dedup.stats().duplicates_found, 1);
    }

    #[test]
    fn test_normalized_hash() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::Normalized("text".to_string()));

        let record1 = json!({"text": "  Hello World  "});
        let record2 = json!({"text": "hello world"});
        let record3 = json!({"text": "HELLO WORLD"});
        let record4 = json!({"text": "different"});

        assert!(!dedup.is_duplicate(&record1));
        assert!(dedup.is_duplicate(&record2)); // Same after normalization
        assert!(dedup.is_duplicate(&record3)); // Same after normalization
        assert!(!dedup.is_duplicate(&record4));

        assert_eq!(dedup.stats().unique_count, 2);
        assert_eq!(dedup.stats().duplicates_found, 2);
    }

    #[test]
    fn test_multi_field_hash() {
        let mut dedup =
            ExactDeduplicator::new(HashStrategy::MultiField(vec!["text".to_string(), "id".to_string()]));

        let record1 = json!({"text": "hello", "id": 1});
        let record2 = json!({"text": "hello", "id": 1}); // Exact duplicate
        let record3 = json!({"text": "hello", "id": 2}); // Different ID
        let record4 = json!({"text": "world", "id": 1}); // Different text

        assert!(!dedup.is_duplicate(&record1));
        assert!(dedup.is_duplicate(&record2));
        assert!(!dedup.is_duplicate(&record3));
        assert!(!dedup.is_duplicate(&record4));

        assert_eq!(dedup.stats().unique_count, 3);
        assert_eq!(dedup.stats().duplicates_found, 1);
    }

    #[test]
    fn test_missing_field() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::Field("nonexistent".to_string()));

        let record = json!({"text": "hello"});

        // Should return false (not duplicate) when field is missing
        assert!(!dedup.is_duplicate(&record));
        assert_eq!(dedup.stats().unique_count, 1);
    }

    #[test]
    fn test_bloom_filter_effectiveness() {
        let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::FullContent, 1000);

        // Add 100 unique records
        for i in 0..100 {
            let record = json!({"id": i});
            dedup.is_duplicate(&record);
        }

        let stats = dedup.stats();
        assert_eq!(stats.unique_count, 100);
        assert_eq!(stats.duplicates_found, 0);

        // Bloom filter should have caught most as new (misses)
        assert!(stats.bloom_misses > 90); // Should be close to 100
    }

    #[test]
    fn test_memory_usage() {
        let dedup = ExactDeduplicator::with_capacity(HashStrategy::FullContent, 10_000);
        let mem = dedup.memory_usage();

        // Should be reasonable for 10k capacity
        assert!(mem > 0);
        assert!(mem < 10_000_000); // Less than 10MB for 10k capacity
    }

    #[test]
    fn test_clear() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);

        let record = json!({"text": "hello"});
        dedup.is_duplicate(&record);

        assert_eq!(dedup.unique_count(), 1);
        assert_eq!(dedup.stats().total_seen, 1);

        dedup.clear();

        assert_eq!(dedup.unique_count(), 0);
        assert_eq!(dedup.stats().total_seen, 0);
        assert_eq!(dedup.stats().duplicates_found, 0);
    }

    #[test]
    fn test_dedup_rate_calculation() {
        let mut dedup = ExactDeduplicator::new(HashStrategy::FullContent);

        let record1 = json!({"id": 1});
        let record2 = json!({"id": 1});
        let record3 = json!({"id": 2});

        dedup.is_duplicate(&record1); // unique
        dedup.is_duplicate(&record2); // duplicate
        dedup.is_duplicate(&record3); // unique

        let stats = dedup.stats();
        let rate = stats.dedup_rate();
        assert!((rate - 33.333333333333336).abs() < 0.0001); // 1/3 = 33.33%
    }

    #[test]
    fn test_large_scale() {
        let mut dedup = ExactDeduplicator::with_capacity(HashStrategy::FullContent, 100_000);

        // Add 10k unique records
        for i in 0..10_000 {
            let record = json!({"id": i});
            assert!(!dedup.is_duplicate(&record));
        }

        // Add 10k duplicates
        for i in 0..10_000 {
            let record = json!({"id": i});
            assert!(dedup.is_duplicate(&record));
        }

        assert_eq!(dedup.stats().unique_count, 10_000);
        assert_eq!(dedup.stats().duplicates_found, 10_000);
        assert_eq!(dedup.stats().total_seen, 20_000);
        assert_eq!(dedup.stats().dedup_rate(), 50.0);
    }
}
