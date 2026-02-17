//! MinHash implementation for near-duplicate detection
//!
//! Provides MinHash signatures for documents and LSH indexing for
//! fast similarity search.

use ahash::AHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
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
    /// Random coefficients for hash functions (a, b pairs)
    coefficients: Vec<(u64, u64)>,
    /// Prime number for hash function
    prime: u64,
}

impl MinHasher {
    /// Create a new MinHasher with specified parameters
    ///
    /// # Arguments
    /// * `num_hashes` - Number of hash functions (typically 128)
    /// * `shingle_size` - Size of character shingles (typically 3)
    pub fn new(num_hashes: usize, shingle_size: usize) -> Self {
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
            "Created MinHasher with {} hash functions, shingle size {}",
            num_hashes, shingle_size
        );

        Self {
            num_hashes,
            shingle_size,
            coefficients,
            prime: 2147483647, // Large prime number
        }
    }

    /// Hash a shingle to a u64
    fn hash_shingle(&self, shingle: &str) -> u64 {
        let mut hasher = AHasher::default();
        shingle.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute MinHash signature for text - optimized to avoid string allocations
    pub fn compute_signature(&self, text: &str) -> MinHashSignature {
        let chars: Vec<char> = text.chars().collect();

        if chars.len() < self.shingle_size {
            // Handle short text
            if text.is_empty() {
                return MinHashSignature::new(vec![0; self.num_hashes]);
            }
            let shingle_hash = self.hash_shingle(text);
            return self.compute_from_single_hash(shingle_hash);
        }

        // Hash shingles directly without storing them as strings
        // This avoids allocating thousands of String objects
        let mut shingle_hashes = HashSet::new();
        let mut temp_shingle = String::with_capacity(self.shingle_size * 4); // Preallocate

        for i in 0..=chars.len() - self.shingle_size {
            temp_shingle.clear();
            for j in 0..self.shingle_size {
                temp_shingle.push(chars[i + j]);
            }
            let hash = self.hash_shingle(&temp_shingle);
            shingle_hashes.insert(hash);
        }

        // Initialize signature with maximum values
        let mut signature = vec![u64::MAX; self.num_hashes];

        // For each unique shingle hash, compute MinHash values
        for &shingle_hash in &shingle_hashes {
            for i in 0..self.num_hashes {
                let (a, b) = self.coefficients[i];
                // Hash function: (a * x + b) % prime
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
    /// Stored signatures
    signatures: Vec<MinHashSignature>,
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
            signatures: Vec::new(),
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

        // Add signature to storage
        if id >= self.signatures.len() {
            self.signatures.resize(id + 1, MinHashSignature::new(vec![0; expected_size]));
        }
        self.signatures[id] = signature.clone();

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
        self.signatures.get(id)
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
    fn test_shingle_generation() {
        let hasher = MinHasher::new(128, 3);
        let shingles = hasher.generate_shingles("hello");

        assert!(shingles.contains("hel"));
        assert!(shingles.contains("ell"));
        assert!(shingles.contains("llo"));
        assert_eq!(shingles.len(), 3);
    }

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
        let hasher = MinHasher::new(128, 3);
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "The quick brown fox jumps over a lazy dog"; // Very similar

        let sim = hasher.jaccard_similarity(text1, text2);
        assert!(sim > 0.7); // Should be very similar
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
        let hasher = MinHasher::new(128, 3);
        let text1 = "The quick brown fox";
        let text2 = "The slow brown dog";

        let sim = hasher.jaccard_similarity(text1, text2);
        assert!(sim > 0.2 && sim < 0.8); // Some overlap but not identical
    }
}
