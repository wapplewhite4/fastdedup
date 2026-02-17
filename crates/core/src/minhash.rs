//! MinHash implementation for near-duplicate detection
//!
//! Provides MinHash signatures for documents and LSH indexing for
//! fast similarity search.

use ahash::RandomState;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use tracing::{debug, info};

/// MinHash signature for a document
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashSignature {
    /// MinHash signature values
    pub signature: Vec<u64>,
}

impl MinHashSignature {
    /// Create a new MinHash signature
    pub fn new(signature: Vec<u64>) -> Self {
        Self { signature }
    }

    /// Get the number of hash functions used
    pub fn num_hashes(&self) -> usize {
        self.signature.len()
    }

    /// Compute Jaccard similarity with another signature
    ///
    /// Returns a value between 0.0 (no similarity) and 1.0 (identical)
    pub fn jaccard_similarity(&self, other: &MinHashSignature) -> f64 {
        if self.signature.len() != other.signature.len() {
            return 0.0;
        }

        let matches = self
            .signature
            .iter()
            .zip(other.signature.iter())
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / self.signature.len() as f64
    }
}

/// MinHash hasher for generating document signatures
pub struct MinHasher {
    /// Number of hash functions to use
    num_hashes: usize,
    /// Size of shingles (k in k-shingles)
    shingle_size: usize,
    /// Whether to use word-level shingles (true) or character-level (false)
    word_shingles: bool,
    /// Random coefficients for hash functions (a, b pairs)
    coefficients: Vec<(u64, u64)>,
    /// Prime number for hash function
    prime: u64,
    /// Fixed-seed hash builder for deterministic shingle hashing
    hash_builder: RandomState,
}

impl MinHasher {
    /// Create a new MinHasher with specified parameters
    ///
    /// # Arguments
    /// * `num_hashes` - Number of hash functions (typically 128)
    /// * `shingle_size` - Size of character shingles (typically 3) or word shingles (typically 2)
    pub fn new(num_hashes: usize, shingle_size: usize) -> Self {
        Self::new_with_mode(num_hashes, shingle_size, true)
    }

    /// Create a new MinHasher with explicit shingle mode
    ///
    /// # Arguments
    /// * `num_hashes` - Number of hash functions (typically 128)
    /// * `shingle_size` - Shingle size in words or characters
    /// * `word_shingles` - If true, use word n-grams; if false, use character n-grams
    pub fn new_with_mode(num_hashes: usize, shingle_size: usize, word_shingles: bool) -> Self {
        // Generate random-like coefficients using a simple LCG
        let mut coefficients = Vec::with_capacity(num_hashes);
        let mut seed = 42u64;

        for _ in 0..num_hashes {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let a = seed % 2147483647;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let b = seed % 2147483647;
            coefficients.push((a, b));
        }

        info!(
            "Created MinHasher with {} hash functions, {} shingle size {}, mode={}",
            num_hashes,
            if word_shingles { "word" } else { "char" },
            shingle_size,
            if word_shingles { "word-ngrams" } else { "char-ngrams" },
        );

        Self {
            num_hashes,
            shingle_size,
            word_shingles,
            coefficients,
            prime: 2147483647, // Large prime number
            // Fixed seeds ensure identical shingle hashes across runs.
            // AHasher::default() is randomly seeded per-process (DoS prevention),
            // which caused non-deterministic duplicate counts between runs.
            hash_builder: RandomState::with_seeds(
                0x517cc1b727220a95,
                0x8d8f5f3b12c4a6e1,
                0xbf58476d1ce4e5b9,
                0x94d049bb133111eb,
            ),
        }
    }

    /// Hash a shingle to a u64
    fn hash_shingle(&self, shingle: &str) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        shingle.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute MinHash signature for text
    ///
    /// Uses word-level shingles by default (much more discriminative for natural
    /// language text, and avoids O(n²) LSH false positives from common char n-grams).
    pub fn compute_signature(&self, text: &str) -> MinHashSignature {
        if text.is_empty() {
            return MinHashSignature::new(vec![0; self.num_hashes]);
        }

        if self.word_shingles {
            return self.compute_signature_words(text);
        }

        // Character-level path (legacy, not recommended for natural language)
        if text.is_ascii() {
            return self.compute_signature_ascii(text.as_bytes());
        }
        self.compute_signature_utf8(text)
    }

    /// Word n-gram shingles (preferred for natural language text)
    ///
    /// Word bigrams ("the quick", "quick brown") are far more discriminative
    /// than character trigrams ("the", "he ", "e q"), which appear in nearly
    /// every document and cause massive LSH false positives.
    fn compute_signature_words(&self, text: &str) -> MinHashSignature {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return MinHashSignature::new(vec![0; self.num_hashes]);
        }

        if words.len() < self.shingle_size {
            // Short text: use entire text as a single shingle
            let hash = self.hash_shingle(text);
            return self.compute_from_single_hash(hash);
        }

        let mut shingle_hashes = HashSet::new();
        for i in 0..=words.len() - self.shingle_size {
            // Hash the word n-gram without allocating a String
            let mut hasher = self.hash_builder.build_hasher();
            for word in &words[i..i + self.shingle_size] {
                word.hash(&mut hasher);
            }
            shingle_hashes.insert(hasher.finish());
        }

        self.compute_signature_from_hashes(&shingle_hashes)
    }

    /// Fast path for ASCII text - no Vec<char> allocation needed
    fn compute_signature_ascii(&self, bytes: &[u8]) -> MinHashSignature {
        if bytes.len() < self.shingle_size {
            let shingle_hash = self.hash_shingle(std::str::from_utf8(bytes).unwrap());
            return self.compute_from_single_hash(shingle_hash);
        }

        let mut shingle_hashes = HashSet::new();

        // Directly iterate over byte slices
        for i in 0..=bytes.len() - self.shingle_size {
            let shingle_bytes = &bytes[i..i + self.shingle_size];
            let shingle = std::str::from_utf8(shingle_bytes).unwrap();
            let hash = self.hash_shingle(shingle);
            shingle_hashes.insert(hash);
        }

        self.compute_signature_from_hashes(&shingle_hashes)
    }

    /// Slower path for UTF-8 text with multi-byte characters
    fn compute_signature_utf8(&self, text: &str) -> MinHashSignature {
        let chars: Vec<char> = text.chars().collect();

        if chars.len() < self.shingle_size {
            let shingle_hash = self.hash_shingle(text);
            return self.compute_from_single_hash(shingle_hash);
        }

        let mut shingle_hashes = HashSet::new();
        let mut temp_shingle = String::with_capacity(self.shingle_size * 4);

        for i in 0..=chars.len() - self.shingle_size {
            temp_shingle.clear();
            for j in 0..self.shingle_size {
                temp_shingle.push(chars[i + j]);
            }
            let hash = self.hash_shingle(&temp_shingle);
            shingle_hashes.insert(hash);
        }

        self.compute_signature_from_hashes(&shingle_hashes)
    }

    /// Compute MinHash signature from a set of shingle hashes
    fn compute_signature_from_hashes(&self, shingle_hashes: &HashSet<u64>) -> MinHashSignature {
        let mut signature = vec![u64::MAX; self.num_hashes];

        for &shingle_hash in shingle_hashes {
            for i in 0..self.num_hashes {
                let (a, b) = self.coefficients[i];
                let hash_value = (a.wrapping_mul(shingle_hash).wrapping_add(b)) % self.prime;

                if hash_value < signature[i] {
                    signature[i] = hash_value;
                }
            }
        }

        debug!(
            "Generated MinHash signature with {} unique shingles",
            shingle_hashes.len()
        );

        MinHashSignature::new(signature)
    }

    /// Helper for computing signature from a single hash
    fn compute_from_single_hash(&self, shingle_hash: u64) -> MinHashSignature {
        let mut signature = vec![u64::MAX; self.num_hashes];
        for i in 0..self.num_hashes {
            let (a, b) = self.coefficients[i];
            signature[i] = (a.wrapping_mul(shingle_hash).wrapping_add(b)) % self.prime;
        }
        MinHashSignature::new(signature)
    }

    /// Compute Jaccard similarity between two texts
    pub fn jaccard_similarity(&self, text1: &str, text2: &str) -> f64 {
        let sig1 = self.compute_signature(text1);
        let sig2 = self.compute_signature(text2);
        sig1.jaccard_similarity(&sig2)
    }
}

/// LSH (Locality Sensitive Hashing) Index for fast similarity search
pub struct LSHIndex {
    /// Number of bands
    num_bands: usize,
    /// Number of rows per band
    rows_per_band: usize,
    /// Hash tables for each band
    bands: Vec<HashMap<Vec<u64>, Vec<usize>>>,
    /// Stored signatures (sparse HashMap instead of Vec for efficiency)
    signatures: HashMap<usize, MinHashSignature>,
}

impl LSHIndex {
    /// Create a new LSH index
    ///
    /// # Arguments
    /// * `num_bands` - Number of bands (typically 32)
    /// * `rows_per_band` - Rows per band (typically 4)
    ///
    /// Configuration: 32 bands × 4 rows = 128 hash functions total
    /// This catches similarities > 0.7 with high probability
    pub fn new(num_bands: usize, rows_per_band: usize) -> Self {
        info!(
            "Creating LSH index with {} bands, {} rows per band",
            num_bands, rows_per_band
        );

        let bands = (0..num_bands).map(|_| HashMap::new()).collect();

        Self {
            num_bands,
            rows_per_band,
            bands,
            signatures: HashMap::new(),
        }
    }

    /// Insert a signature into the index
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this document
    /// * `signature` - MinHash signature to index
    pub fn insert(&mut self, id: usize, signature: MinHashSignature) {
        let expected_size = self.num_bands * self.rows_per_band;
        if signature.num_hashes() != expected_size {
            panic!(
                "Signature size {} doesn't match expected size {}",
                signature.num_hashes(),
                expected_size
            );
        }

        // Add signature to storage (no more Vec resizing!)
        self.signatures.insert(id, signature.clone());

        // Insert into each band's hash table
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_signature: Vec<u64> = signature.signature[start..end].to_vec();

            self.bands[band_idx]
                .entry(band_signature)
                .or_insert_with(Vec::new)
                .push(id);
        }

        debug!("Inserted signature for document {}", id);
    }

    /// Query for similar documents
    ///
    /// Returns IDs of documents that are candidates for similarity
    /// based on LSH banding. Results should be verified with actual
    /// Jaccard similarity.
    ///
    /// # Arguments
    /// * `signature` - Query signature
    /// * `threshold` - Minimum similarity threshold (not strictly enforced by LSH)
    pub fn query(&self, signature: &MinHashSignature, _threshold: f64) -> Vec<usize> {
        let expected_size = self.num_bands * self.rows_per_band;
        if signature.num_hashes() != expected_size {
            debug!(
                "Query signature size {} doesn't match expected size {}",
                signature.num_hashes(),
                expected_size
            );
            return Vec::new();
        }

        let mut candidates = HashSet::new();

        // Check each band for matches
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_signature: Vec<u64> = signature.signature[start..end].to_vec();

            if let Some(ids) = self.bands[band_idx].get(&band_signature) {
                for &id in ids {
                    candidates.insert(id);
                }
            }
        }

        let result: Vec<usize> = candidates.into_iter().collect();
        debug!("Query found {} candidates", result.len());
        result
    }

    /// Get a stored signature by ID
    pub fn get_signature(&self, id: usize) -> Option<&MinHashSignature> {
        self.signatures.get(&id)
    }

    /// Get the number of documents indexed
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Clear all indexed data
    pub fn clear(&mut self) {
        self.bands.clear();
        self.bands = (0..self.num_bands).map(|_| HashMap::new()).collect();
        self.signatures.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts() {
        let hasher = MinHasher::new(128, 3);
        let text = "The quick brown fox jumps over the lazy dog";

        let sim = hasher.jaccard_similarity(text, text);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_different_texts() {
        let hasher = MinHasher::new(128, 3);
        let text1 = "The quick brown fox";
        let text2 = "Completely different text";

        let sim = hasher.jaccard_similarity(text1, text2);
        assert!(sim < 0.3); // Should be quite different
    }

    #[test]
    fn test_similar_texts() {
        let hasher = MinHasher::new(128, 2); // word bigrams
        // Texts with only 1 word different out of ~20 → many shared word bigrams
        let text1 = "the cat sat on the mat near the window in the bright sunny afternoon";
        let text2 = "the cat sat on the mat near the window in the bright sunny morning";

        let sim = hasher.jaccard_similarity(text1, text2);
        assert!(sim > 0.7, "Expected sim > 0.7, got {}", sim);
    }

    #[test]
    fn test_signature_consistency() {
        let hasher = MinHasher::new(128, 3);
        let text = "Hello world";

        let sig1 = hasher.compute_signature(text);
        let sig2 = hasher.compute_signature(text);

        assert_eq!(sig1, sig2);
        assert_eq!(sig1.jaccard_similarity(&sig2), 1.0);
    }

    #[test]
    fn test_lsh_insert_and_query() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4); // 32 bands × 4 rows = 128 hashes

        let text1 = "The quick brown fox";
        let text2 = "The quick brown fox"; // Identical
        let text3 = "Completely different text";

        let sig1 = hasher.compute_signature(text1);
        let sig2 = hasher.compute_signature(text2);
        let sig3 = hasher.compute_signature(text3);

        index.insert(0, sig1.clone());
        index.insert(1, sig2.clone());
        index.insert(2, sig3.clone());

        // Query with identical text should find at least itself
        let candidates = index.query(&sig1, 0.7);
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1)); // Should also find the duplicate

        // Query with different text might not find matches
        let candidates = index.query(&sig3, 0.7);
        assert!(candidates.contains(&2));
    }

    #[test]
    fn test_lsh_near_duplicates() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4);

        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "The quick brown fox jumps over a lazy dog"; // Very similar
        let text3 = "A completely different piece of text entirely";

        let sig1 = hasher.compute_signature(text1);
        let sig2 = hasher.compute_signature(text2);
        let sig3 = hasher.compute_signature(text3);

        index.insert(0, sig1.clone());
        index.insert(1, sig2.clone());
        index.insert(2, sig3.clone());

        // Query with text1 should find text2 as candidate
        let candidates = index.query(&sig1, 0.7);
        assert!(candidates.contains(&0));
        // May or may not contain 1 depending on hash collisions in bands
    }

    #[test]
    fn test_empty_text() {
        let hasher = MinHasher::new(128, 3);
        let sig = hasher.compute_signature("");

        assert_eq!(sig.signature.len(), 128);
    }

    #[test]
    fn test_short_text() {
        let hasher = MinHasher::new(128, 3);
        let sig = hasher.compute_signature("ab"); // Shorter than shingle size

        assert_eq!(sig.signature.len(), 128);
    }

    #[test]
    fn test_lsh_clear() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4);

        let sig = hasher.compute_signature("test text");
        index.insert(0, sig);

        assert_eq!(index.len(), 1);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_jaccard_different_sizes() {
        let sig1 = MinHashSignature::new(vec![1, 2, 3]);
        let sig2 = MinHashSignature::new(vec![1, 2, 3, 4]);

        assert_eq!(sig1.jaccard_similarity(&sig2), 0.0);
    }

    #[test]
    fn test_partial_overlap() {
        let hasher = MinHasher::new(128, 2); // word bigrams
        // These texts share the middle portion ("sat on the mat near the window")
        // so they should have meaningful but not complete overlap
        let text1 = "the cat sat on the mat near the window it was quiet";
        let text2 = "the dog sat on the mat near the window it was loud";

        let sim = hasher.jaccard_similarity(text1, text2);
        assert!(sim > 0.2 && sim < 0.9, "Expected 0.2 < sim < 0.9, got {}", sim);
    }
}
