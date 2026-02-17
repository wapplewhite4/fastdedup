//! Fuzzy deduplication pipeline combining MinHash, LSH, and text normalization
//!
//! Provides a production-ready fuzzy deduplicator for detecting near-duplicate
//! documents using MinHash signatures and LSH indexing.

use crate::minhash::{LSHIndex, MinHasher, MinHashSignature};
use dataset_dedup_filters::text_preprocessing::TextNormalizer;
use serde_json::Value;
use tracing::{debug, info, warn};

/// Statistics for fuzzy deduplication
#[derive(Debug, Clone, Default)]
pub struct FuzzyDedupStats {
    /// Total records processed
    pub total_processed: usize,
    /// Number of records with duplicates found
    pub records_with_duplicates: usize,
    /// Total number of duplicate relationships found
    pub total_duplicates_found: usize,
    /// Number of LSH candidates checked
    pub lsh_candidates_checked: usize,
    /// Number of verified duplicates (above threshold)
    pub verified_duplicates: usize,
}

impl FuzzyDedupStats {
    /// Get the duplicate rate as a percentage
    pub fn duplicate_rate(&self) -> f64 {
        if self.total_processed == 0 {
            0.0
        } else {
            (self.records_with_duplicates as f64 / self.total_processed as f64) * 100.0
        }
    }

    /// Get the LSH precision (verified / candidates)
    pub fn lsh_precision(&self) -> f64 {
        if self.lsh_candidates_checked == 0 {
            0.0
        } else {
            (self.verified_duplicates as f64 / self.lsh_candidates_checked as f64) * 100.0
        }
    }
}

/// Configuration for fuzzy deduplication
#[derive(Debug, Clone)]
pub struct FuzzyDedupConfig {
    /// Similarity threshold (0.0 to 1.0)
    pub similarity_threshold: f64,
    /// Number of MinHash functions
    pub num_hashes: usize,
    /// Shingle size for MinHash (words if word_shingles=true, else characters)
    pub shingle_size: usize,
    /// Use word-level shingles (recommended for natural language text)
    ///
    /// Word bigrams are far more discriminative than char trigrams for NLP text.
    /// Char trigrams cause massive LSH false positives due to common English
    /// sequences ("the", " to", "ing"), leading to O(n²) slowdown on large datasets.
    pub word_shingles: bool,
    /// Number of LSH bands
    pub num_bands: usize,
    /// Rows per LSH band
    pub rows_per_band: usize,
    /// Text field to use for deduplication
    pub text_field: String,
}

impl Default for FuzzyDedupConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            num_hashes: 128,
            shingle_size: 3, // character trigrams (matches Python datasketch default)
            word_shingles: false,
            // 16 bands × 8 rows = 128 hashes
            // vs naive 32×4: FP rate drops from ~5% → 0.0004% at s=0.2
            // TP rate at s=0.8: ~95% (vs ~100% for 32×4) — acceptable tradeoff
            num_bands: 16,
            rows_per_band: 8,
            text_field: "text".to_string(),
        }
    }
}

/// Fuzzy deduplicator combining MinHash, LSH, and text normalization
pub struct FuzzyDeduplicator {
    /// MinHash hasher
    minhash: MinHasher,
    /// LSH index for fast similarity search
    lsh_index: LSHIndex,
    /// Text normalizer
    normalizer: TextNormalizer,
    /// Configuration
    config: FuzzyDedupConfig,
    /// Statistics
    stats: FuzzyDedupStats,
}

impl FuzzyDeduplicator {
    /// Create a new fuzzy deduplicator with default configuration
    pub fn new(threshold: f64) -> Self {
        let config = FuzzyDedupConfig {
            similarity_threshold: threshold,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new fuzzy deduplicator with custom configuration
    pub fn with_config(config: FuzzyDedupConfig) -> Self {
        info!(
            "Creating FuzzyDeduplicator with threshold {}, {} hash functions",
            config.similarity_threshold, config.num_hashes
        );

        let minhash = MinHasher::new_with_mode(config.num_hashes, config.shingle_size, config.word_shingles);
        let lsh_index = LSHIndex::new(config.num_bands, config.rows_per_band);
        let normalizer = TextNormalizer::balanced();

        Self {
            minhash,
            lsh_index,
            normalizer,
            config,
            stats: FuzzyDedupStats::default(),
        }
    }

    /// Create with custom text normalizer
    pub fn with_normalizer(threshold: f64, normalizer: TextNormalizer) -> Self {
        let mut dedup = Self::new(threshold);
        dedup.normalizer = normalizer;
        dedup
    }

    /// Extract text from a record
    fn extract_text(&self, record: &Value) -> Option<String> {
        record
            .get(&self.config.text_field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Process a record and add it if it's not a duplicate
    ///
    /// This is more efficient than calling find_duplicates + add_record
    /// because it only computes the MinHash signature once.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this record
    /// * `record` - The record to process
    ///
    /// # Returns
    /// * `Some(duplicates)` - If duplicates found, returns their IDs
    /// * `None` - If no duplicates (record was added to index)
    pub fn process_record(&mut self, id: usize, record: &Value) -> Option<Vec<usize>> {
        self.stats.total_processed += 1;

        // 1. Extract text field
        let text = match self.extract_text(record) {
            Some(t) => t,
            None => {
                debug!("Record missing text field '{}'", self.config.text_field);
                return None;
            }
        };

        // 2. Normalize text
        let normalized_text = self.normalizer.normalize(&text);

        if normalized_text.is_empty() {
            debug!("Record normalized to empty text");
            return None;
        }

        // 3. Compute MinHash signature ONCE
        let signature = self.minhash.compute_signature(&normalized_text);

        // 4. Query LSH index for candidates
        let candidates = self.lsh_index.query(&signature, self.config.similarity_threshold);

        if candidates.is_empty() {
            // No duplicates found - add to index
            self.lsh_index.insert(id, signature);
            debug!("Added record {} to fuzzy dedup index", id);
            return None;
        }

        // 5. Verify with Jaccard similarity
        let mut duplicates = Vec::new();

        for &candidate_id in &candidates {
            self.stats.lsh_candidates_checked += 1;

            if let Some(candidate_sig) = self.lsh_index.get_signature(candidate_id) {
                let similarity = signature.jaccard_similarity(candidate_sig);

                if similarity >= self.config.similarity_threshold {
                    duplicates.push(candidate_id);
                    self.stats.verified_duplicates += 1;
                }
            }
        }

        if !duplicates.is_empty() {
            self.stats.records_with_duplicates += 1;
            self.stats.total_duplicates_found += duplicates.len();
            duplicates.sort_unstable();
            Some(duplicates)
        } else {
            // LSH gave false positives - add to index
            self.lsh_index.insert(id, signature);
            debug!("Added record {} to fuzzy dedup index (no verified duplicates)", id);
            None
        }
    }

    /// Find duplicate IDs for a given record
    ///
    /// Returns a list of document IDs that are similar above the threshold.
    /// The list is sorted by document ID.
    pub fn find_duplicates(&mut self, record: &Value) -> Vec<usize> {
        self.stats.total_processed += 1;

        // 1. Extract text field
        let text = match self.extract_text(record) {
            Some(t) => t,
            None => {
                debug!("Record missing text field '{}'", self.config.text_field);
                return Vec::new();
            }
        };

        // 2. Normalize text
        let normalized_text = self.normalizer.normalize(&text);

        if normalized_text.is_empty() {
            debug!("Record normalized to empty text");
            return Vec::new();
        }

        // 3. Compute MinHash signature
        let signature = self.minhash.compute_signature(&normalized_text);

        // 4. Query LSH index for candidates
        let candidates = self.lsh_index.query(&signature, self.config.similarity_threshold);

        if candidates.is_empty() {
            return Vec::new();
        }

        // 5. Verify with Jaccard similarity
        let mut duplicates = Vec::new();

        for &candidate_id in &candidates {
            self.stats.lsh_candidates_checked += 1;

            if let Some(candidate_sig) = self.lsh_index.get_signature(candidate_id) {
                let similarity = signature.jaccard_similarity(candidate_sig);

                if similarity >= self.config.similarity_threshold {
                    duplicates.push(candidate_id);
                    self.stats.verified_duplicates += 1;
                }
            }
        }

        if !duplicates.is_empty() {
            self.stats.records_with_duplicates += 1;
            self.stats.total_duplicates_found += duplicates.len();
        }

        duplicates.sort_unstable();
        duplicates
    }

    /// Add a record to the index
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this record
    /// * `record` - The record to add
    pub fn add_record(&mut self, id: usize, record: &Value) {
        // Extract and normalize text
        let text = match self.extract_text(record) {
            Some(t) => t,
            None => {
                warn!(
                    "Cannot add record {}: missing text field '{}'",
                    id, self.config.text_field
                );
                return;
            }
        };

        let normalized_text = self.normalizer.normalize(&text);

        if normalized_text.is_empty() {
            debug!("Record {} normalized to empty text, skipping", id);
            return;
        }

        // Compute signature and add to index
        let signature = self.minhash.compute_signature(&normalized_text);
        self.lsh_index.insert(id, signature);

        debug!("Added record {} to fuzzy dedup index", id);
    }

    /// Prepare a record by computing its MinHash signature (can run in parallel).
    ///
    /// Returns None if the record is missing the text field or normalizes to empty.
    pub fn prepare_signature(&self, record: &Value) -> Option<MinHashSignature> {
        let text = self.extract_text(record)?;
        let normalized = self.normalizer.normalize(&text);
        if normalized.is_empty() {
            return None;
        }
        Some(self.minhash.compute_signature(&normalized))
    }

    /// Process a pre-computed signature against the LSH index (must run serially).
    ///
    /// Returns `Some(matches)` if duplicates found, where each element is
    /// `(matched_id, minhash_similarity)`.  Returns `None` (and inserts into
    /// the index) when no verified duplicate is found.
    pub fn process_prepared(&mut self, id: usize, signature: MinHashSignature) -> Option<Vec<(usize, f64)>> {
        self.stats.total_processed += 1;

        let candidates = self.lsh_index.query(&signature, self.config.similarity_threshold);

        if candidates.is_empty() {
            self.lsh_index.insert(id, signature);
            return None;
        }

        let mut duplicates: Vec<(usize, f64)> = Vec::new();
        for &candidate_id in &candidates {
            self.stats.lsh_candidates_checked += 1;
            if let Some(candidate_sig) = self.lsh_index.get_signature(candidate_id) {
                let similarity = signature.jaccard_similarity(candidate_sig);
                if similarity >= self.config.similarity_threshold {
                    duplicates.push((candidate_id, similarity));
                    self.stats.verified_duplicates += 1;
                }
            }
        }

        if !duplicates.is_empty() {
            self.stats.records_with_duplicates += 1;
            self.stats.total_duplicates_found += duplicates.len();
            duplicates.sort_unstable_by_key(|&(id, _)| id);
            Some(duplicates)
        } else {
            self.lsh_index.insert(id, signature);
            None
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &FuzzyDedupStats {
        &self.stats
    }

    /// Get the number of indexed documents
    pub fn index_size(&self) -> usize {
        self.lsh_index.len()
    }

    /// Clear the index and reset statistics
    pub fn clear(&mut self) {
        self.lsh_index.clear();
        self.stats = FuzzyDedupStats::default();
    }

    /// Process a batch of records and return duplicate clusters
    ///
    /// Returns a Vec of clusters, where each cluster is a Vec of record IDs
    /// that are similar to each other.
    pub fn find_all_duplicates(&mut self, records: &[Value]) -> Vec<Vec<usize>> {
        let mut visited = vec![false; records.len()];
        let mut clusters = Vec::new();

        for (i, record) in records.iter().enumerate() {
            if visited[i] {
                continue;
            }

            // Add this record to index
            self.add_record(i, record);

            // Find duplicates
            let duplicates = self.find_duplicates(record);

            if !duplicates.is_empty() {
                // Create cluster with current record and its duplicates
                let mut cluster = vec![i];
                cluster.extend(duplicates.iter().filter(|&&id| !visited[id]));

                // Mark all in cluster as visited
                for &id in &cluster {
                    if id < visited.len() {
                        visited[id] = true;
                    }
                }

                clusters.push(cluster);
            } else {
                visited[i] = true;
            }
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_wikipedia_title_no_false_positive() {
        // Regression: Wikipedia title-only dataset, threshold=0.9.
        // "Buru_thrush"    → normalised "buru_thrush"    (9 char-trigrams)
        // "Buru_White-eye" → normalised "buru_white eye" (12 char-trigrams)
        // 3 shared trigrams → true Jaccard = 3/18 = 0.166.
        // MinHash estimate ≈ 0.22.  Must NOT be flagged as duplicate.
        let config = FuzzyDedupConfig {
            similarity_threshold: 0.9,
            num_hashes: 128,
            shingle_size: 3,
            word_shingles: false,
            num_bands: 16,
            rows_per_band: 8,
            text_field: "text".to_string(),
        };
        let mut dedup = FuzzyDeduplicator::with_config(config);

        let r1 = json!({"text": "Buru_thrush"});
        let r2 = json!({"text": "Buru_White-eye"});

        let sig1 = dedup.prepare_signature(&r1).expect("sig1");
        let sig2 = dedup.prepare_signature(&r2).expect("sig2");
        let minhash_sim = sig1.jaccard_similarity(&sig2);
        // Confirm the estimate is close to the true value (0.166 ± noise).
        assert!(
            minhash_sim < 0.5,
            "MinHash Jaccard unexpectedly high: {:.4} (true ≈ 0.166)",
            minhash_sim
        );

        dedup.add_record(1, &r1);
        let dups = dedup.find_duplicates(&r2);
        assert!(
            dups.is_empty(),
            "FALSE POSITIVE at threshold=0.9: MinHash sim={:.4}",
            minhash_sim
        );
    }

    #[test]
    fn test_exact_duplicates() {
        let mut dedup = FuzzyDeduplicator::new(0.9);

        let record1 = json!({"text": "The quick brown fox jumps over the lazy dog"});
        let record2 = json!({"text": "The quick brown fox jumps over the lazy dog"});

        dedup.add_record(0, &record1);

        let duplicates = dedup.find_duplicates(&record2);
        assert!(duplicates.contains(&0));
    }

    #[test]
    fn test_near_duplicates() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        // Longer texts with only 1 word changed → ~80% word-bigram Jaccard
        let record1 = json!({"text": "the cat sat on the mat near the window and the light was bright outside"});
        let record2 = json!({"text": "the cat sat on the mat near the window and the light was dim outside"});

        dedup.add_record(0, &record1);

        let duplicates = dedup.find_duplicates(&record2);
        assert!(duplicates.contains(&0), "Should find near-duplicate");
    }

    #[test]
    fn test_different_texts() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record1 = json!({"text": "The quick brown fox"});
        let record2 = json!({"text": "Completely different text"});

        dedup.add_record(0, &record1);

        let duplicates = dedup.find_duplicates(&record2);
        assert!(duplicates.is_empty());
    }

    #[test]
    fn test_missing_text_field() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record = json!({"id": 123, "other": "data"});

        let duplicates = dedup.find_duplicates(&record);
        assert!(duplicates.is_empty());
    }

    #[test]
    fn test_custom_text_field() {
        let mut config = FuzzyDedupConfig::default();
        config.text_field = "content".to_string();

        let mut dedup = FuzzyDeduplicator::with_config(config);

        let record1 = json!({"content": "Hello world"});
        let record2 = json!({"content": "Hello world"});

        dedup.add_record(0, &record1);

        let duplicates = dedup.find_duplicates(&record2);
        assert!(duplicates.contains(&0));
    }

    #[test]
    fn test_normalization() {
        let mut dedup = FuzzyDeduplicator::new(0.9);

        let record1 = json!({"text": "Hello, WORLD!!!"});
        let record2 = json!({"text": "hello world"});

        dedup.add_record(0, &record1);

        // Should find as duplicate due to normalization
        let duplicates = dedup.find_duplicates(&record2);
        assert!(duplicates.contains(&0));
    }

    #[test]
    fn test_multiple_duplicates() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let base = json!({"text": "The quick brown fox"});
        let dup1 = json!({"text": "The quick brown fox"});
        let dup2 = json!({"text": "The quick brown foxes"});
        let different = json!({"text": "Completely different"});

        dedup.add_record(0, &base);
        dedup.add_record(1, &dup1);
        dedup.add_record(2, &dup2);
        dedup.add_record(3, &different);

        let duplicates = dedup.find_duplicates(&base);
        assert!(duplicates.contains(&0));
        assert!(duplicates.contains(&1));
        // May or may not contain 2 depending on threshold
    }

    #[test]
    fn test_statistics() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record1 = json!({"text": "Hello world"});
        let record2 = json!({"text": "Hello world"});

        dedup.add_record(0, &record1);
        dedup.find_duplicates(&record2);

        let stats = dedup.stats();
        assert_eq!(stats.total_processed, 1);
        assert!(stats.lsh_candidates_checked > 0);
    }

    #[test]
    fn test_clear() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record = json!({"text": "Hello world"});
        dedup.add_record(0, &record);

        assert_eq!(dedup.index_size(), 1);

        dedup.clear();

        assert_eq!(dedup.index_size(), 0);
        assert_eq!(dedup.stats().total_processed, 0);
    }

    #[test]
    fn test_empty_text() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record = json!({"text": ""});
        dedup.add_record(0, &record);

        // Should not be added
        assert_eq!(dedup.index_size(), 0);
    }

    #[test]
    fn test_whitespace_only() {
        let mut dedup = FuzzyDeduplicator::new(0.7);

        let record = json!({"text": "   \t\n   "});
        let duplicates = dedup.find_duplicates(&record);

        assert!(duplicates.is_empty());
    }

    #[test]
    fn test_find_all_duplicates() {
        let mut dedup = FuzzyDeduplicator::new(0.8);

        let records = vec![
            json!({"text": "The quick brown fox"}),
            json!({"text": "The quick brown fox"}), // Duplicate of 0
            json!({"text": "Hello world"}),
            json!({"text": "Hello world"}),         // Duplicate of 2
            json!({"text": "Different text"}),
        ];

        let clusters = dedup.find_all_duplicates(&records);

        // Should have 2 clusters (0,1) and (2,3)
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn test_threshold_sensitivity() {
        // 11-word texts with 1 word different → ~67% word-bigram Jaccard
        let record1 = json!({"text": "the quick brown fox jumps over the lazy dog near the river"});
        let record2 = json!({"text": "the quick brown cat jumps over the lazy dog near the river"});

        // High threshold (0.95) - should not match (Jaccard ≈ 0.67)
        let mut dedup_high = FuzzyDeduplicator::new(0.95);
        dedup_high.add_record(0, &record1);
        let dups_high = dedup_high.find_duplicates(&record2);
        assert!(dups_high.is_empty());

        // Low threshold (0.5) - should match (Jaccard ≈ 0.67 > 0.5)
        let mut dedup_low = FuzzyDeduplicator::new(0.5);
        dedup_low.add_record(0, &record1);
        let dups_low = dedup_low.find_duplicates(&record2);
        assert!(dups_low.contains(&0));
    }
}
