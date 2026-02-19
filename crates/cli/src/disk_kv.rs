//! Disk-backed key-value store for field values.
//!
//! Wraps a bounded in-memory HashMap with sled overflow so that
//! the `field_values` cache in fuzzy dedup doesn't grow unboundedly
//! when processing large datasets.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use anyhow::Result;

/// A bounded string-value store: recent entries live in memory,
/// older ones are spilled to sled.
pub struct DiskBackedStringMap {
    hot: HashMap<usize, String>,
    cold: sled::Db,
    insertion_order: VecDeque<usize>,
    max_hot: usize,
    temp_dir: Option<PathBuf>,
}

impl DiskBackedStringMap {
    /// Create a new store with the given in-memory capacity.
    pub fn new(max_hot: usize) -> Result<Self> {
        let dir = Self::make_temp_path();
        // Limit sled page cache to 64 MB (default 1 GB is wasteful here)
        let cold = sled::Config::new()
            .path(&dir)
            .cache_capacity(64 * 1024 * 1024)
            .open()?;
        Ok(Self {
            hot: HashMap::with_capacity(max_hot.min(500_000)),
            cold,
            insertion_order: VecDeque::with_capacity(max_hot.min(500_000)),
            max_hot,
            temp_dir: Some(dir),
        })
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, id: usize, value: String) -> Result<()> {
        if self.hot.len() >= self.max_hot {
            self.evict()?;
        }
        self.hot.insert(id, value);
        self.insertion_order.push_back(id);
        Ok(())
    }

    /// Get a value by key.  Returns an owned `String`.
    pub fn get(&self, id: &usize) -> Result<Option<String>> {
        if let Some(v) = self.hot.get(id) {
            return Ok(Some(v.clone()));
        }
        let key = (*id as u64).to_be_bytes();
        match self.cold.get(&key)? {
            Some(bytes) => {
                let s = String::from_utf8(bytes.to_vec())
                    .unwrap_or_default();
                Ok(Some(s))
            }
            None => Ok(None),
        }
    }

    fn evict(&mut self) -> Result<()> {
        let evict_count = (self.max_hot / 10).max(1);
        let mut batch = sled::Batch::default();
        let mut evicted = 0;

        while evicted < evict_count {
            let id = match self.insertion_order.pop_front() {
                Some(id) => id,
                None => break,
            };
            if let Some(value) = self.hot.remove(&id) {
                let key = (id as u64).to_be_bytes();
                batch.insert(&key, value.as_bytes());
                evicted += 1;
            }
        }

        self.cold.apply_batch(batch)?;
        Ok(())
    }

    fn make_temp_path() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dedup_field_values_{}_{}", std::process::id(), ts))
    }
}

impl Drop for DiskBackedStringMap {
    fn drop(&mut self) {
        let _ = self.cold.flush();
        // Replace with a temporary db so we can drop the original and delete
        // its directory.
        self.cold = sled::Config::new()
            .temporary(true)
            .open()
            .expect("failed to open temp sled for cleanup");
        if let Some(ref dir) = self.temp_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
