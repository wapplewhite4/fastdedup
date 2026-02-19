//! Flat in-memory signature storage indexed by document ID.
//!
//! Stores MinHash signatures as a contiguous `Vec<u32>` buffer with O(1) access
//! by document ID.  A parallel `Vec<bool>` tracks which slots are occupied.
//!
//! Memory at 5.5M records (128 u32 hashes per signature):
//!   data  = 5.5M × 128 × 4 = ~2.8 GB
//!   valid = 5.5M × 1        = ~5.5 MB
//!   Total ≈ 2.8 GB  (vs ~5.8 GB with the original HashMap<usize, Vec<u64>>)

use crate::minhash::MinHashSignature;
use tracing::info;

/// Flat in-memory signature store.
///
/// Signatures are laid out contiguously: signature for document `id` occupies
/// `data[id * sig_size .. (id+1) * sig_size]`.  Gaps (IDs that were never
/// inserted, or were removed) are tracked via a `valid` bitmap.
pub struct SignatureStore {
    /// Flat buffer: signature `i` at `data[i * sig_size .. (i+1) * sig_size]`
    data: Vec<u32>,
    /// Whether each slot contains a valid signature
    valid: Vec<bool>,
    /// Number of u32 values per signature (e.g. 128)
    sig_size: usize,
    /// Number of valid signatures currently stored
    count: usize,
}

impl SignatureStore {
    /// Create a new empty signature store.
    ///
    /// `sig_size` is the number of u32 values per signature (e.g. 128).
    pub fn new(sig_size: usize) -> Self {
        info!("SignatureStore: sig_size={}, in-memory flat Vec", sig_size);
        Self {
            data: Vec::new(),
            valid: Vec::new(),
            sig_size,
            count: 0,
        }
    }

    /// Create a new signature store with pre-allocated capacity.
    pub fn with_capacity(sig_size: usize, capacity: usize) -> Self {
        info!(
            "SignatureStore: sig_size={}, capacity={}, in-memory flat Vec",
            sig_size, capacity,
        );
        Self {
            data: vec![0u32; capacity * sig_size],
            valid: vec![false; capacity],
            sig_size,
            count: 0,
        }
    }

    /// Ensure the store can hold a signature at index `id`.
    fn ensure_capacity(&mut self, id: usize) {
        let needed = id + 1;
        if needed > self.valid.len() {
            self.valid.resize(needed, false);
            self.data.resize(needed * self.sig_size, 0);
        }
    }

    /// Insert a signature for the given document ID.
    pub fn insert(&mut self, id: usize, signature: &MinHashSignature) {
        self.ensure_capacity(id);
        let start = id * self.sig_size;
        self.data[start..start + self.sig_size].copy_from_slice(&signature.signature);
        if !self.valid[id] {
            self.count += 1;
        }
        self.valid[id] = true;
    }

    /// Retrieve a signature by document ID.
    ///
    /// Returns `None` if the ID was never inserted or was removed.
    pub fn get(&self, id: usize) -> Option<MinHashSignature> {
        if id >= self.valid.len() || !self.valid[id] {
            return None;
        }
        let start = id * self.sig_size;
        let slice = &self.data[start..start + self.sig_size];
        Some(MinHashSignature::new(slice.to_vec()))
    }

    /// Check whether a signature exists for the given ID.
    pub fn contains(&self, id: usize) -> bool {
        id < self.valid.len() && self.valid[id]
    }

    /// Remove a signature by document ID.
    pub fn remove(&mut self, id: usize) {
        if id < self.valid.len() && self.valid[id] {
            self.valid[id] = false;
            self.count -= 1;
        }
    }

    /// Number of valid signatures stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all signatures.
    pub fn clear(&mut self) {
        self.data.clear();
        self.valid.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sig(size: usize, seed: u32) -> MinHashSignature {
        let sig: Vec<u32> = (0..size).map(|i| seed.wrapping_add(i as u32)).collect();
        MinHashSignature::new(sig)
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = SignatureStore::new(128);
        let sig = make_sig(128, 42);

        store.insert(0, &sig);
        assert_eq!(store.len(), 1);

        let got = store.get(0).unwrap();
        assert_eq!(got, sig);
    }

    #[test]
    fn test_contains() {
        let mut store = SignatureStore::new(128);
        assert!(!store.contains(0));

        store.insert(0, &make_sig(128, 1));
        assert!(store.contains(0));
        assert!(!store.contains(1));
    }

    #[test]
    fn test_remove() {
        let mut store = SignatureStore::new(128);
        store.insert(0, &make_sig(128, 1));
        assert_eq!(store.len(), 1);

        store.remove(0);
        assert_eq!(store.len(), 0);
        assert!(!store.contains(0));
        assert!(store.get(0).is_none());
    }

    #[test]
    fn test_clear() {
        let mut store = SignatureStore::new(128);
        for i in 0..10 {
            store.insert(i, &make_sig(128, i as u32));
        }
        assert_eq!(store.len(), 10);

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_sparse_ids() {
        let mut store = SignatureStore::new(128);

        // Insert at non-contiguous IDs
        store.insert(0, &make_sig(128, 0));
        store.insert(100, &make_sig(128, 100));
        store.insert(1000, &make_sig(128, 1000));

        assert_eq!(store.len(), 3);
        assert!(store.contains(0));
        assert!(!store.contains(50));
        assert!(store.contains(100));
        assert!(store.contains(1000));

        assert_eq!(store.get(0).unwrap(), make_sig(128, 0));
        assert_eq!(store.get(100).unwrap(), make_sig(128, 100));
        assert_eq!(store.get(1000).unwrap(), make_sig(128, 1000));
    }

    #[test]
    fn test_update_existing() {
        let mut store = SignatureStore::new(128);
        let sig1 = make_sig(128, 1);
        let sig2 = make_sig(128, 2);

        store.insert(0, &sig1);
        store.insert(0, &sig2);

        assert_eq!(store.len(), 1); // count should not double
        assert_eq!(store.get(0).unwrap(), sig2);
    }

    #[test]
    fn test_with_capacity() {
        let mut store = SignatureStore::with_capacity(128, 1000);
        let sig = make_sig(128, 42);

        store.insert(0, &sig);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(0).unwrap(), sig);

        // Insert beyond initial capacity
        store.insert(2000, &make_sig(128, 2000));
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_get_out_of_range() {
        let store = SignatureStore::new(128);
        assert!(store.get(999).is_none());
        assert!(!store.contains(999));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut store = SignatureStore::new(128);
        // Should not panic
        store.remove(999);
        assert_eq!(store.len(), 0);
    }
}
