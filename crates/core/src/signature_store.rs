//! Tiered signature storage with flat-file disk backing
//!
//! Replaces the previous sled-based cold storage with a simple append-only
//! binary file + in-memory offset index.  This avoids sled's memory-mapped
//! files which caused the OS page cache to balloon to 30-50 GB at scale.
//!
//! Memory model at 15M records (128 u32 hashes per signature):
//! - Offset index: HashMap<usize, u64> → 15M × ~16 B = ~240 MB
//! - Hot cache (500K entries): 500K × 536 B = ~268 MB
//! - Flat file on disk: 15M × 512 B = ~7.5 GB (disk only, not RAM)
//! - **Total RAM: ~500 MB** for the signature store

use crate::{Error, Result};
use crate::minhash::MinHashSignature;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use tracing::{debug, info};

/// Configuration for tiered signature storage
#[derive(Debug, Clone)]
pub struct SignatureStoreConfig {
    /// Maximum number of signatures in hot cache before eviction
    pub max_hot: usize,
    /// Path for cold storage file.  If `None`, a temporary file
    /// is created and cleaned up on `Drop`.
    pub db_path: Option<PathBuf>,
}

impl Default for SignatureStoreConfig {
    fn default() -> Self {
        Self {
            max_hot: 500_000, // ~268 MB for 128-hash u32 signatures
            db_path: None,
        }
    }
}

/// Two-tier signature storage: hot in-memory HashMap + cold flat file.
///
/// Cold storage is a simple append-only binary file.  Each signature is
/// written as `sig_size` little-endian u32 values (fixed 512 bytes at
/// 128 hashes).  An in-memory `HashMap<usize, u64>` maps document ID →
/// byte offset in the file, costing only ~16 bytes per evicted record.
pub struct TieredSignatureStore {
    /// Hot cache keyed by document ID
    hot: HashMap<usize, MinHashSignature>,
    /// Cold storage: append-only flat file
    cold_writer: BufWriter<File>,
    /// Cold storage: random-access reader (separate handle)
    cold_reader: BufReader<File>,
    /// Maps document ID → byte offset in the cold file
    cold_index: HashMap<usize, u64>,
    /// Current write position in the cold file
    cold_write_pos: u64,
    /// Maximum hot-cache entries before eviction
    max_hot: usize,
    /// FIFO insertion order for eviction
    insertion_order: VecDeque<usize>,
    /// Total number of signatures stored (hot + cold)
    count: usize,
    /// Number of hash values per signature (for deserialization)
    sig_size: usize,
    /// Size in bytes of one signature on disk
    sig_bytes: usize,
    /// If we created a temp file, store its path for cleanup on Drop
    temp_path: Option<PathBuf>,
}

impl TieredSignatureStore {
    /// Create a new tiered signature store.
    ///
    /// `sig_size` is the number of u32 values per signature (e.g. 128).
    pub fn new(sig_size: usize, config: SignatureStoreConfig) -> Result<Self> {
        let sig_bytes = sig_size * 4; // u32 = 4 bytes each

        let (file_path, is_temp) = match config.db_path {
            Some(ref p) => (p.clone(), false),
            None => (Self::make_temp_path(), true),
        };

        // Open file for writing (append) and a separate handle for reading
        let writer_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| {
                Error::ProcessingError(format!(
                    "Failed to open signature cold file {:?}: {}", file_path, e
                ))
            })?;

        let reader_file = File::open(&file_path).map_err(|e| {
            Error::ProcessingError(format!(
                "Failed to open signature cold file for reading {:?}: {}", file_path, e
            ))
        })?;

        info!(
            "TieredSignatureStore: sig_size={}, sig_bytes={}, max_hot={}, path={:?}",
            sig_size, sig_bytes, config.max_hot, file_path,
        );

        Ok(Self {
            hot: HashMap::with_capacity(config.max_hot.min(500_000)),
            cold_writer: BufWriter::with_capacity(64 * 1024, writer_file),
            cold_reader: BufReader::with_capacity(sig_bytes, reader_file),
            cold_index: HashMap::new(),
            cold_write_pos: 0,
            max_hot: config.max_hot,
            insertion_order: VecDeque::with_capacity(config.max_hot.min(500_000)),
            count: 0,
            sig_size,
            sig_bytes,
            temp_path: if is_temp { Some(file_path) } else { None },
        })
    }

    /// Create a store using all defaults (500K hot cache, temp file).
    pub fn with_defaults(sig_size: usize) -> Result<Self> {
        Self::new(sig_size, SignatureStoreConfig::default())
    }

    /// Insert a signature.
    pub fn insert(&mut self, id: usize, signature: MinHashSignature) -> Result<()> {
        // If already present in hot, just update in place
        if self.hot.contains_key(&id) {
            self.hot.insert(id, signature);
            return Ok(());
        }
        // If already in cold, overwrite by appending new copy and updating index
        if self.cold_index.contains_key(&id) {
            self.cold_append(id, &signature)?;
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
    /// lookups require deserialization from disk.
    pub fn get(&mut self, id: usize) -> Result<Option<MinHashSignature>> {
        // Check hot cache first
        if let Some(sig) = self.hot.get(&id) {
            return Ok(Some(sig.clone()));
        }
        // Fall through to cold file
        self.cold_get(id)
    }

    /// Check whether a signature exists (hot or cold) without fetching it.
    ///
    /// Cold check is O(1) — just an index lookup, no disk I/O.
    pub fn contains(&self, id: usize) -> Result<bool> {
        Ok(self.hot.contains_key(&id) || self.cold_index.contains_key(&id))
    }

    /// Remove a signature.
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if self.hot.remove(&id).is_some() {
            self.count = self.count.saturating_sub(1);
            return Ok(());
        }
        if self.cold_index.remove(&id).is_some() {
            // The bytes remain in the file (append-only) but are unreachable.
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
        self.cold_index.clear();
        // Truncate the cold file
        self.cold_writer.get_mut().set_len(0).map_err(|e| {
            Error::ProcessingError(format!("Failed to truncate cold file: {}", e))
        })?;
        self.cold_write_pos = 0;
        self.count = 0;
        Ok(())
    }

    // ---- internal helpers ----

    /// Evict the oldest 10% of the hot cache to the cold file.
    fn evict_to_cold(&mut self) -> Result<()> {
        let evict_count = (self.max_hot / 10).max(1);
        debug!("Evicting {} signatures to cold file", evict_count);

        let mut evicted = 0;
        while evicted < evict_count {
            let id = match self.insertion_order.pop_front() {
                Some(id) => id,
                None => break,
            };
            if let Some(sig) = self.hot.remove(&id) {
                self.cold_append(id, &sig)?;
                evicted += 1;
            }
        }

        self.cold_writer.flush().map_err(|e| {
            Error::ProcessingError(format!("Failed to flush cold file: {}", e))
        })?;

        debug!("Evicted {} signatures. hot={}, cold={}, total={}",
            evicted, self.hot.len(), self.cold_index.len(), self.count);
        Ok(())
    }

    /// Append a signature to the cold file and record its offset.
    fn cold_append(&mut self, id: usize, sig: &MinHashSignature) -> Result<()> {
        let offset = self.cold_write_pos;
        let bytes = sig_to_bytes(sig);
        self.cold_writer.write_all(&bytes).map_err(|e| {
            Error::ProcessingError(format!("Failed to write signature to cold file: {}", e))
        })?;
        self.cold_write_pos += bytes.len() as u64;
        self.cold_index.insert(id, offset);
        Ok(())
    }

    /// Read a signature from the cold file by document ID.
    fn cold_get(&mut self, id: usize) -> Result<Option<MinHashSignature>> {
        let offset = match self.cold_index.get(&id) {
            Some(&off) => off,
            None => return Ok(None),
        };

        // Flush writer so the reader can see all data
        self.cold_writer.flush().map_err(|e| {
            Error::ProcessingError(format!("Failed to flush before cold read: {}", e))
        })?;

        self.cold_reader.seek(SeekFrom::Start(offset)).map_err(|e| {
            Error::ProcessingError(format!(
                "Failed to seek to offset {} in cold file: {}", offset, e
            ))
        })?;

        let mut buf = vec![0u8; self.sig_bytes];
        self.cold_reader.read_exact(&mut buf).map_err(|e| {
            Error::ProcessingError(format!(
                "Failed to read signature at offset {}: {}", offset, e
            ))
        })?;

        Ok(Some(bytes_to_sig(&buf, self.sig_size)?))
    }

    fn make_temp_path() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dedup_sig_{}.bin", ts))
    }
}

impl Drop for TieredSignatureStore {
    fn drop(&mut self) {
        let _ = self.cold_writer.flush();
        if let Some(ref path) = self.temp_path {
            if let Err(e) = std::fs::remove_file(path) {
                debug!("Failed to remove temp signature file {:?}: {}", path, e);
            }
        }
    }
}

// ---- serialization helpers ----

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
