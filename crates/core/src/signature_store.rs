//! Tiered signature storage with disk backing for MinHash signatures
//!
//! Implements a two-tier storage system to bound memory usage at scale:
//! - Hot cache: In-memory HashMap for recent signatures (fast O(1) access)
//! - Cold storage: Disk-backed sled database for older signatures
//!
//! At 15M records with 128-hash u32 signatures (~512 B each), keeping all
//! signatures in memory requires ~7.5 GB.  With a 2M-entry hot cache
//! (~1 GB) and the rest on disk, peak RAM drops to a manageable level.

use crate::{Error, Result};
use crate::minhash::MinHashSignature;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Configuration for tiered signature storage
#[derive(Debug, Clone)]
pub struct SignatureStoreConfig {
    /// Maximum number of signatures in hot cache before eviction
    pub max_hot: usize,
    /// Path for cold storage database.  If `None`, a temporary directory
    /// is created and cleaned up on `Drop`.
    pub db_path: Option<PathBuf>,
}

impl Default for SignatureStoreConfig {
    fn default() -> Self {
        Self {
            max_hot: 2_000_000, // ~1 GB for 128-hash u32 signatures
            db_path: None,
        }
    }
}

/// Two-tier signature storage: hot in-memory HashMap + cold sled database.
pub struct TieredSignatureStore {
    /// Hot cache keyed by document ID
    hot: HashMap<usize, MinHashSignature>,
    /// Cold storage (disk-backed)
    cold: sled::Db,
    /// Maximum hot-cache entries before eviction
    max_hot: usize,
    /// FIFO insertion order for eviction
    insertion_order: VecDeque<usize>,
    /// Total number of signatures stored (hot + cold)
    count: usize,
    /// Number of hash values per signature (for deserialization)
    sig_size: usize,
    /// If we created a temp directory, store its path for cleanup on Drop
    temp_dir: Option<PathBuf>,
}

impl TieredSignatureStore {
    /// Create a new tiered signature store.
    ///
    /// `sig_size` is the number of u64 values per signature (e.g. 128).
    pub fn new(sig_size: usize, config: SignatureStoreConfig) -> Result<Self> {
        // Limit sled's page cache to 64 MB (default is 1 GB) to avoid
        // wasting RAM — most reads are served from the hot HashMap cache.
        const SLED_CACHE_BYTES: u64 = 64 * 1024 * 1024;

        let (cold, temp_dir) = match config.db_path {
            Some(ref p) => {
                let db = sled::Config::new()
                    .path(p)
                    .cache_capacity(SLED_CACHE_BYTES)
                    .open()
                    .map_err(|e| {
                        Error::ProcessingError(format!(
                            "Failed to open signature cold storage at {:?}: {}",
                            p, e
                        ))
                    })?;
                (db, None)
            }
            None => {
                let dir = Self::make_temp_path();
                let db = sled::Config::new()
                    .path(&dir)
                    .cache_capacity(SLED_CACHE_BYTES)
                    .open()
                    .map_err(|e| {
                        Error::ProcessingError(format!(
                            "Failed to open temporary signature storage at {:?}: {}",
                            dir, e
                        ))
                    })?;
                (db, Some(dir))
            }
        };

        info!(
            "TieredSignatureStore: sig_size={}, max_hot={}, path={:?}",
            sig_size,
            config.max_hot,
            config.db_path.as_deref().unwrap_or(Path::new("<temp>")),
        );

        Ok(Self {
            hot: HashMap::with_capacity(config.max_hot.min(1_000_000)),
            cold,
            max_hot: config.max_hot,
            insertion_order: VecDeque::with_capacity(config.max_hot.min(1_000_000)),
            count: 0,
            sig_size,
            temp_dir,
        })
    }

    /// Create a store using all defaults (2M hot cache, temp directory).
    pub fn with_defaults(sig_size: usize) -> Result<Self> {
        Self::new(sig_size, SignatureStoreConfig::default())
    }

    /// Insert a signature.
    pub fn insert(&mut self, id: usize, signature: MinHashSignature) -> Result<()> {
        // If already present, just update in place
        if self.hot.contains_key(&id) {
            self.hot.insert(id, signature);
            return Ok(());
        }
        if self.cold_contains(id)? {
            self.cold_put(id, &signature)?;
            return Ok(());
        }

        // New entry — maybe evict first
        if self.hot.len() >= self.max_hot {
            self.evict_to_cold()?;
        }

        self.hot.insert(id, signature);
        self.insertion_order.push_back(id);
        self.count += 1;
        Ok(())
    }

    /// Retrieve a signature by document ID.
    ///
    /// Returns an **owned** `MinHashSignature` because cold-storage
    /// lookups require deserialization.
    pub fn get(&self, id: usize) -> Result<Option<MinHashSignature>> {
        // Check hot cache first
        if let Some(sig) = self.hot.get(&id) {
            return Ok(Some(sig.clone()));
        }
        // Fall through to cold
        self.cold_get(id)
    }

    /// Check whether a signature exists (hot or cold) without fetching it.
    pub fn contains(&self, id: usize) -> Result<bool> {
        if self.hot.contains_key(&id) {
            return Ok(true);
        }
        self.cold_contains(id)
    }

    /// Remove a signature.
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if self.hot.remove(&id).is_some() {
            self.count = self.count.saturating_sub(1);
            return Ok(());
        }
        let key = id_to_key(id);
        if self.cold.remove(&key).map_err(cold_err)?.is_some() {
            self.count = self.count.saturating_sub(1);
        }
        Ok(())
    }

    /// Number of signatures stored (hot + cold).
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all signatures.
    pub fn clear(&mut self) -> Result<()> {
        self.hot.clear();
        self.insertion_order.clear();
        self.cold.clear().map_err(cold_err)?;
        self.count = 0;
        Ok(())
    }

    // ---- internal helpers ----

    /// Evict the oldest 10% of the hot cache to cold storage.
    fn evict_to_cold(&mut self) -> Result<()> {
        let evict_count = (self.max_hot / 10).max(1);
        debug!("Evicting {} signatures from hot to cold storage", evict_count);

        let mut batch = sled::Batch::default();
        let mut evicted = 0;

        while evicted < evict_count {
            let id = match self.insertion_order.pop_front() {
                Some(id) => id,
                None => break,
            };
            if let Some(sig) = self.hot.remove(&id) {
                let key = id_to_key(id);
                let value = sig_to_bytes(&sig);
                batch.insert(&key, value);
                evicted += 1;
            }
            // If the id was already removed from hot (e.g. via `remove()`),
            // just skip it.
        }

        self.cold.apply_batch(batch).map_err(cold_err)?;
        debug!("Evicted {} signatures. hot={}, total={}", evicted, self.hot.len(), self.count);
        Ok(())
    }

    fn cold_get(&self, id: usize) -> Result<Option<MinHashSignature>> {
        let key = id_to_key(id);
        match self.cold.get(&key).map_err(cold_err)? {
            Some(bytes) => Ok(Some(bytes_to_sig(&bytes, self.sig_size)?)),
            None => Ok(None),
        }
    }

    fn cold_contains(&self, id: usize) -> Result<bool> {
        let key = id_to_key(id);
        self.cold.contains_key(&key).map_err(cold_err)
    }

    fn cold_put(&self, id: usize, sig: &MinHashSignature) -> Result<()> {
        let key = id_to_key(id);
        let value = sig_to_bytes(sig);
        self.cold.insert(&key, value).map_err(cold_err)?;
        Ok(())
    }

    fn make_temp_path() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dedup_sig_store_{}_{}", std::process::id(), ts))
    }
}

impl Drop for TieredSignatureStore {
    fn drop(&mut self) {
        // Flush sled so it shuts down cleanly
        if let Err(e) = self.cold.flush() {
            warn!("Failed to flush signature cold storage on drop: {}", e);
        }
        // Drop the sled Db before removing the directory
        drop(std::mem::replace(&mut self.cold, sled::Config::new().temporary(true).open().unwrap()));
        if let Some(ref dir) = self.temp_dir {
            if let Err(e) = std::fs::remove_dir_all(dir) {
                debug!("Failed to remove temp signature store {:?}: {}", dir, e);
            }
        }
    }
}

// ---- serialization helpers ----

/// Encode document ID as 8-byte big-endian key (for sled ordering).
fn id_to_key(id: usize) -> [u8; 8] {
    (id as u64).to_be_bytes()
}

/// Serialize a MinHash signature as raw little-endian u32 bytes.
fn sig_to_bytes(sig: &MinHashSignature) -> Vec<u8> {
    let mut buf = Vec::with_capacity(sig.signature.len() * 4);
    for &v in &sig.signature {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Deserialize raw bytes back into a MinHash signature.
fn bytes_to_sig(bytes: &[u8], expected_len: usize) -> Result<MinHashSignature> {
    let expected_bytes = expected_len * 4;
    if bytes.len() != expected_bytes {
        return Err(Error::ProcessingError(format!(
            "Signature byte length {} doesn't match expected {} ({} × 4)",
            bytes.len(),
            expected_bytes,
            expected_len,
        )));
    }
    let mut signature = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        signature.push(u32::from_le_bytes(arr));
    }
    Ok(MinHashSignature::new(signature))
}

fn cold_err(e: sled::Error) -> Error {
    Error::ProcessingError(format!("Signature cold storage error: {}", e))
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
        let mut store = TieredSignatureStore::with_defaults(128).unwrap();
        let sig = make_sig(128, 42);

        store.insert(0, sig.clone()).unwrap();
        assert_eq!(store.len(), 1);

        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got, sig);
    }

    #[test]
    fn test_contains() {
        let mut store = TieredSignatureStore::with_defaults(128).unwrap();
        assert!(!store.contains(0).unwrap());

        store.insert(0, make_sig(128, 1)).unwrap();
        assert!(store.contains(0).unwrap());
        assert!(!store.contains(1).unwrap());
    }

    #[test]
    fn test_remove() {
        let mut store = TieredSignatureStore::with_defaults(128).unwrap();
        store.insert(0, make_sig(128, 1)).unwrap();
        assert_eq!(store.len(), 1);

        store.remove(0).unwrap();
        assert_eq!(store.len(), 0);
        assert!(!store.contains(0).unwrap());
        assert!(store.get(0).unwrap().is_none());
    }

    #[test]
    fn test_clear() {
        let mut store = TieredSignatureStore::with_defaults(128).unwrap();
        for i in 0..10 {
            store.insert(i, make_sig(128, i as u32)).unwrap();
        }
        assert_eq!(store.len(), 10);

        store.clear().unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_eviction_to_cold() {
        let config = SignatureStoreConfig {
            max_hot: 10,
            db_path: None,
        };
        let mut store = TieredSignatureStore::new(128, config).unwrap();

        // Insert more than max_hot entries
        for i in 0..25 {
            store.insert(i, make_sig(128, i as u32)).unwrap();
        }

        assert_eq!(store.len(), 25);
        // Hot cache should be less than 25 (some evicted)
        assert!(store.hot.len() <= 10);

        // All entries should still be accessible
        for i in 0..25 {
            let sig = store.get(i).unwrap();
            assert!(sig.is_some(), "Signature {} not found", i);
            assert_eq!(sig.unwrap(), make_sig(128, i as u32));
        }
    }

    #[test]
    fn test_cold_roundtrip_fidelity() {
        let config = SignatureStoreConfig {
            max_hot: 2,
            db_path: None,
        };
        let mut store = TieredSignatureStore::new(128, config).unwrap();

        let sig = make_sig(128, 0xDEADBEEF);
        store.insert(0, sig.clone()).unwrap();
        // Force eviction by inserting more
        store.insert(1, make_sig(128, 1)).unwrap();
        store.insert(2, make_sig(128, 2)).unwrap();
        store.insert(3, make_sig(128, 3)).unwrap();

        // sig 0 should be in cold storage now
        let got = store.get(0).unwrap().unwrap();
        assert_eq!(got, sig, "Cold storage roundtrip corrupted signature");
    }

    #[test]
    fn test_update_existing() {
        let mut store = TieredSignatureStore::with_defaults(128).unwrap();
        let sig1 = make_sig(128, 1);
        let sig2 = make_sig(128, 2);

        store.insert(0, sig1).unwrap();
        store.insert(0, sig2.clone()).unwrap();

        assert_eq!(store.len(), 1); // count should not double
        assert_eq!(store.get(0).unwrap().unwrap(), sig2);
    }

    #[test]
    fn test_serialization_helpers() {
        let sig = make_sig(4, 100);
        let bytes = sig_to_bytes(&sig);
        assert_eq!(bytes.len(), 16); // 4 × 4 bytes (u32)
        let restored = bytes_to_sig(&bytes, 4).unwrap();
        assert_eq!(restored, sig);
    }

    #[test]
    fn test_serialization_bad_length() {
        let result = bytes_to_sig(&[0u8; 10], 4);
        assert!(result.is_err());
    }
}
