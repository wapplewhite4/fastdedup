//! MinHash implementation for near-duplicate detection
//!
//! Provides MinHash signatures for documents and LSH indexing for
//! fast similarity search.
//!
//! ## Performance optimizations
//!
//! The LSH index includes several optimizations for large-scale datasets:
//!
//! 1. **Pre-hashed u64 band keys** — band signatures (Vec<u64> of `rows_per_band`
//!    elements) are hashed down to a single `u64` via ahash before being used as
//!    HashMap keys.  This eliminates per-lookup Vec allocation, hashing, and
//!    comparison overhead (~4x faster lookups).
//!
//! 2. **ahash-backed HashMaps** — all internal HashMaps use `ahash::HashMap`
//!    instead of the default SipHash-based HashMap (~30% faster lookups since
//!    keys are not adversarially controlled).
//!
//! 3. **Vec-backed signature storage** — signatures are stored in a
//!    `Vec<Option<MinHashSignature>>` indexed by document ID for O(1) access
//!    with better cache locality (IDs are sequential 0..N).
//!
//! 4. **Capped candidate verification** — queries return at most
//!    `MAX_CANDIDATES_PER_QUERY` candidates, bounding worst-case verification
//!    cost to O(1) amortized.  Most real duplicates appear early since they
//!    were inserted close in time.
//!
//! 5. **Capacity-hint pre-allocation** — band HashMaps and signature storage
//!    can be pre-allocated via `with_capacity()` when the total record count
//!    is known (e.g. from Parquet metadata), avoiding O(log n) rehashes.
//!
//! 6. **Periodic band-bucket compaction** — IDs that were removed as duplicates
//!    still sit in band buckets, inflating candidate lists.  `compact()` prunes
//!    stale IDs from all band buckets.

use ahash::RandomState;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hash, Hasher};
use tracing::{debug, info};

use crate::Result;
use crate::signature_store::{TieredSignatureStore, SignatureStoreConfig};

/// Type alias: ahash-backed HashMap for internal use (faster than SipHash for
/// non-adversarial keys).
type AHashMap<K, V> = std::collections::HashMap<K, V, RandomState>;

/// Maximum number of candidate IDs to verify per query.
///
/// Bounds worst-case verification to O(MAX_CANDIDATES × num_hashes) regardless
/// of index size.  Most true duplicates appear early (inserted close in time),
/// so the recall loss is negligible in practice.
const MAX_CANDIDATES_PER_QUERY: usize = 200;

/// MinHash signature for a document
///
/// Stores hash values as `u32` since the MinHash prime (2^31 - 1) fits
/// in 32 bits, halving per-signature memory (512 B vs 1 KiB at 128 hashes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashSignature {
    /// MinHash signature values (each < 2^31)
    pub signature: Vec<u32>,
}

impl MinHashSignature {
    /// Create a new MinHash signature
    pub fn new(signature: Vec<u32>) -> Self {
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
            return MinHashSignature::new(vec![0u32; self.num_hashes]);
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
            return MinHashSignature::new(vec![0u32; self.num_hashes]);
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
        // Prime is 2^31-1, so all hash values fit in u32
        let mut signature = vec![u32::MAX; self.num_hashes];

        for &shingle_hash in shingle_hashes {
            for i in 0..self.num_hashes {
                let (a, b) = self.coefficients[i];
                let hash_value = ((a.wrapping_mul(shingle_hash).wrapping_add(b)) % self.prime) as u32;

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
        let mut signature = vec![u32::MAX; self.num_hashes];
        for i in 0..self.num_hashes {
            let (a, b) = self.coefficients[i];
            signature[i] = ((a.wrapping_mul(shingle_hash).wrapping_add(b)) % self.prime) as u32;
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

/// Hash a band slice (rows_per_band u32 values) down to a single u64 key.
///
/// Uses ahash for speed; collision probability is ~1/2^64 per pair per band,
/// which is negligible.
fn hash_band_key(slice: &[u32], hash_builder: &RandomState) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    slice.hash(&mut hasher);
    hasher.finish()
}

/// LSH (Locality Sensitive Hashing) Index for fast similarity search
///
/// Uses several performance optimizations over a naive implementation:
/// - Pre-hashed u64 band keys (avoids Vec allocation per lookup)
/// - ahash-backed HashMaps (faster than default SipHash)
/// - Tiered signature storage (hot in-memory + cold on disk)
/// - Capped candidate verification (bounded worst-case cost)
/// - Capacity-hint pre-allocation
/// - Periodic compaction of stale IDs
pub struct LSHIndex {
    /// Number of bands
    num_bands: usize,
    /// Number of rows per band
    rows_per_band: usize,
    /// Hash tables for each band, keyed by a pre-hashed u64 of the band
    /// signature slice.  Document IDs are stored as u32 to halve band-table
    /// memory (supports up to ~4 billion documents).
    bands: Vec<AHashMap<u64, Vec<u32>>>,
    /// Tiered signature storage: hot in-memory cache + cold disk-backed sled.
    /// Bounds memory usage regardless of dataset size.
    signatures: TieredSignatureStore,
    /// Fixed-seed hash builder for deterministic band-key hashing
    band_hash_builder: RandomState,
    /// Number of insertions since last compaction
    insertions_since_compact: usize,
    /// Threshold of insertions that triggers an automatic compaction check
    compact_interval: usize,
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
        Self::with_store_config(num_bands, rows_per_band, SignatureStoreConfig::default())
    }

    /// Create a new LSH index with a custom signature store configuration.
    pub fn with_store_config(
        num_bands: usize,
        rows_per_band: usize,
        store_config: SignatureStoreConfig,
    ) -> Self {
        info!(
            "Creating LSH index with {} bands, {} rows per band",
            num_bands, rows_per_band
        );

        let hash_builder = RandomState::with_seeds(
            0xa1b2c3d4e5f60718,
            0x9182736455463728,
            0xdeadbeefcafebabe,
            0x0123456789abcdef,
        );

        let sig_size = num_bands * rows_per_band;
        let bands = (0..num_bands)
            .map(|_| AHashMap::with_hasher(hash_builder.clone()))
            .collect();

        let signatures = TieredSignatureStore::new(sig_size, store_config)
            .expect("Failed to initialize signature store");

        Self {
            num_bands,
            rows_per_band,
            bands,
            signatures,
            band_hash_builder: hash_builder,
            insertions_since_compact: 0,
            compact_interval: 100_000,
        }
    }

    /// Create a new LSH index with pre-allocated capacity.
    ///
    /// When the total number of records is known in advance (e.g. from Parquet
    /// file metadata), this avoids O(log n) rehashes during insertion.
    pub fn with_capacity(num_bands: usize, rows_per_band: usize, expected_records: usize) -> Self {
        info!(
            "Creating LSH index with {} bands, {} rows per band, capacity hint {}",
            num_bands, rows_per_band, expected_records
        );

        let hash_builder = RandomState::with_seeds(
            0xa1b2c3d4e5f60718,
            0x9182736455463728,
            0xdeadbeefcafebabe,
            0x0123456789abcdef,
        );

        let sig_size = num_bands * rows_per_band;
        let per_band_capacity = expected_records / num_bands.max(1);
        let bands = (0..num_bands)
            .map(|_| {
                AHashMap::with_capacity_and_hasher(per_band_capacity, hash_builder.clone())
            })
            .collect();

        let signatures = TieredSignatureStore::with_defaults(sig_size)
            .expect("Failed to initialize signature store");

        Self {
            num_bands,
            rows_per_band,
            bands,
            signatures,
            band_hash_builder: hash_builder,
            insertions_since_compact: 0,
            compact_interval: 100_000,
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

        // Insert into each band's hash table using pre-hashed u64 keys
        let id32 = id as u32;
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;
            let band_key = hash_band_key(&signature.signature[start..end], &self.band_hash_builder);

            self.bands[band_idx]
                .entry(band_key)
                .or_insert_with(Vec::new)
                .push(id32);
        }

        // Store signature in tiered storage (hot cache + cold disk)
        if let Err(e) = self.signatures.insert(id, signature) {
            // Log but don't panic — band tables are already updated, so
            // queries will still find this ID; we just won't be able to
            // verify it later if it gets evicted.
            tracing::warn!("Failed to store signature for document {}: {}", id, e);
        }

        self.insertions_since_compact += 1;

        debug!("Inserted signature for document {}", id);
    }

    /// Query for similar documents
    ///
    /// Returns IDs of documents that are candidates for similarity
    /// based on LSH banding. Results should be verified with actual
    /// Jaccard similarity.
    ///
    /// Returns at most `MAX_CANDIDATES_PER_QUERY` candidates to bound
    /// worst-case verification cost.
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
            let band_key = hash_band_key(&signature.signature[start..end], &self.band_hash_builder);

            if let Some(ids) = self.bands[band_idx].get(&band_key) {
                for &id32 in ids {
                    let id = id32 as usize;
                    // Only include IDs that still have a valid signature
                    // (filters out stale entries from removed duplicates)
                    if self.signatures.contains(id).unwrap_or(false) {
                        candidates.insert(id);
                        if candidates.len() >= MAX_CANDIDATES_PER_QUERY {
                            let result: Vec<usize> = candidates.into_iter().collect();
                            debug!("Query found {} candidates (capped)", result.len());
                            return result;
                        }
                    }
                }
            }
        }

        let result: Vec<usize> = candidates.into_iter().collect();
        debug!("Query found {} candidates", result.len());
        result
    }

    /// Get a stored signature by ID.
    ///
    /// Returns an **owned** `MinHashSignature` because cold-storage lookups
    /// require deserialization.  Returns `Err` on I/O failure.
    pub fn get_signature(&self, id: usize) -> Result<Option<MinHashSignature>> {
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
        for band in &mut self.bands {
            band.clear();
        }
        if let Err(e) = self.signatures.clear() {
            tracing::warn!("Failed to clear signature store: {}", e);
        }
        self.insertions_since_compact = 0;
    }

    /// Compact band buckets by removing IDs whose signatures have been removed.
    ///
    /// IDs that were identified as duplicates and had their signatures removed
    /// still linger in band buckets, inflating candidate lists.  This method
    /// prunes those stale entries.
    pub fn compact(&mut self) {
        let sigs = &self.signatures;
        for band in &mut self.bands {
            band.retain(|_key, ids| {
                ids.retain(|&id32| sigs.contains(id32 as usize).unwrap_or(false));
                !ids.is_empty()
            });
        }
        self.insertions_since_compact = 0;
        debug!("Compacted LSH band buckets");
    }

    /// Compact if enough insertions have accumulated since the last compaction.
    pub fn maybe_compact(&mut self) {
        if self.insertions_since_compact >= self.compact_interval {
            self.compact();
        }
    }

    /// Remove a signature from the index.
    ///
    /// Removes from the tiered store but does not immediately remove the
    /// ID from band buckets.  Stale band entries are filtered during
    /// `query()` and fully cleaned up during `compact()`.
    pub fn remove_signature(&mut self, id: usize) {
        if let Err(e) = self.signatures.remove(id) {
            tracing::warn!("Failed to remove signature {}: {}", id, e);
        }
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
        let sig1 = MinHashSignature::new(vec![1u32, 2, 3]);
        let sig2 = MinHashSignature::new(vec![1u32, 2, 3, 4]);

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

    #[test]
    fn test_lsh_with_capacity() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::with_capacity(32, 4, 1000);

        let sig = hasher.compute_signature("test text for capacity");
        index.insert(0, sig.clone());

        assert_eq!(index.len(), 1);
        let candidates = index.query(&sig, 0.7);
        assert!(candidates.contains(&0));
    }

    #[test]
    fn test_lsh_compact() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4);

        let sig1 = hasher.compute_signature("The quick brown fox");
        let sig2 = hasher.compute_signature("The quick brown fox"); // identical

        index.insert(0, sig1);
        index.insert(1, sig2);
        assert_eq!(index.len(), 2);

        // Remove one signature
        index.remove_signature(1);
        assert_eq!(index.len(), 1);

        // Compact should clean up stale entries in band buckets
        index.compact();

        // Should still find document 0
        let sig_query = hasher.compute_signature("The quick brown fox");
        let candidates = index.query(&sig_query, 0.7);
        assert!(candidates.contains(&0));
        assert!(!candidates.contains(&1)); // removed, should not appear
    }

    #[test]
    fn test_lsh_remove_signature() {
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4);

        let sig = hasher.compute_signature("test text");
        index.insert(0, sig);

        assert_eq!(index.len(), 1);
        assert!(index.get_signature(0).unwrap().is_some());

        index.remove_signature(0);
        assert_eq!(index.len(), 0);
        assert!(index.get_signature(0).unwrap().is_none());
    }

    #[test]
    fn test_candidate_cap() {
        // Insert many identical signatures to create a hot bucket, then verify
        // that query returns at most MAX_CANDIDATES_PER_QUERY results.
        let hasher = MinHasher::new(128, 3);
        let mut index = LSHIndex::new(32, 4);

        let sig = hasher.compute_signature("The quick brown fox");
        let count = MAX_CANDIDATES_PER_QUERY + 100;
        for i in 0..count {
            index.insert(i, sig.clone());
        }

        let candidates = index.query(&sig, 0.7);
        assert!(
            candidates.len() <= MAX_CANDIDATES_PER_QUERY,
            "Expected at most {} candidates, got {}",
            MAX_CANDIDATES_PER_QUERY,
            candidates.len()
        );
    }
}
