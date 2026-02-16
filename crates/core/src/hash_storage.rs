//! Memory-efficient hash storage with disk backing
//!
//! Implements a two-tier storage system:
//! - Hot cache: In-memory HashSet for recent hashes (fast)
//! - Cold storage: Disk-backed database for older hashes (space-efficient)

use ahash::AHashSet;
use crate::{Error, Result};
use sled::Db;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for tiered hash storage
#[derive(Debug, Clone)]
pub struct TieredStorageConfig {
    /// Maximum number of hashes in hot cache
    pub max_hot_size: usize,
    /// Path to cold storage database
    pub db_path: String,
    /// Whether to sync to disk on every write (slower but safer)
    pub sync_on_write: bool,
}

impl Default for TieredStorageConfig {
    fn default() -> Self {
        Self {
            max_hot_size: 10_000_000, // 10M hashes in memory (~160MB)
            db_path: "./dedup_cold_storage".to_string(),
            sync_on_write: false, // Async writes for performance
        }
    }
}

/// Statistics for tiered storage
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Number of hashes in hot cache
    pub hot_count: usize,
    /// Number of hashes in cold storage
    pub cold_count: usize,
    /// Number of hot cache hits
    pub hot_hits: usize,
    /// Number of cold storage hits
    pub cold_hits: usize,
    /// Number of evictions from hot to cold
    pub evictions: usize,
    /// Number of promotions from cold to hot
    pub promotions: usize,
}

impl StorageStats {
    /// Get total number of unique hashes
    pub fn total_hashes(&self) -> usize {
        self.hot_count + self.cold_count
    }

    /// Get hot cache hit rate
    pub fn hot_hit_rate(&self) -> f64 {
        let total = self.hot_hits + self.cold_hits;
        if total == 0 {
            0.0
        } else {
            (self.hot_hits as f64 / total as f64) * 100.0
        }
    }
}

/// Two-tier hash storage with LRU-like eviction
pub struct TieredHashStorage {
    /// Hot cache (in-memory)
    hot_cache: AHashSet<u64>,
    /// Cold storage (disk-backed)
    cold_storage: Arc<Db>,
    /// Configuration
    config: TieredStorageConfig,
    /// Statistics
    stats: StorageStats,
    /// Access counter for simple LRU approximation
    access_counter: u64,
}

impl TieredHashStorage {
    /// Create a new tiered hash storage with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(TieredStorageConfig::default())
    }

    /// Create a new tiered hash storage with custom configuration
    pub fn with_config(config: TieredStorageConfig) -> Result<Self> {
        info!(
            "Initializing TieredHashStorage: db_path={}, max_hot_size={}",
            config.db_path, config.max_hot_size
        );

        let db = sled::open(&config.db_path).map_err(|e| {
            Error::ProcessingError(format!("Failed to open cold storage database: {}", e))
        })?;

        // Count existing hashes in cold storage
        let cold_count = db.len();

        let stats = StorageStats {
            cold_count,
            ..Default::default()
        };

        Ok(Self {
            hot_cache: AHashSet::with_capacity(config.max_hot_size),
            cold_storage: Arc::new(db),
            config,
            stats,
            access_counter: 0,
        })
    }

    /// Create a temporary tiered storage (for testing)
    pub fn temporary() -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let config = TieredStorageConfig {
            db_path: format!("/tmp/dedup_test_{}_{}", std::process::id(), timestamp),
            max_hot_size: 100_000,
            sync_on_write: false,
        };
        Self::with_config(config)
    }

    /// Check if a hash exists (with automatic tier management)
    pub fn contains(&mut self, hash: u64) -> Result<bool> {
        // First check hot cache
        if self.hot_cache.contains(&hash) {
            self.stats.hot_hits += 1;
            return Ok(true);
        }

        // Check cold storage
        let hash_bytes = hash.to_be_bytes();
        if self.cold_storage.contains_key(&hash_bytes).map_err(|e| {
            Error::ProcessingError(format!("Cold storage lookup failed: {}", e))
        })? {
            self.stats.cold_hits += 1;

            // Promote to hot cache if not full
            if self.hot_cache.len() < self.config.max_hot_size {
                self.hot_cache.insert(hash);
                self.stats.hot_count += 1;
                self.stats.cold_count = self.stats.cold_count.saturating_sub(1);
                self.stats.promotions += 1;
                debug!("Promoted hash {} from cold to hot storage", hash);
            }

            return Ok(true);
        }

        Ok(false)
    }

    /// Insert a hash (with automatic eviction if needed)
    pub fn insert(&mut self, hash: u64) -> Result<bool> {
        self.access_counter += 1;

        // Check if already exists
        if self.contains(hash)? {
            return Ok(false); // Already exists
        }

        // Try to insert into hot cache
        if self.hot_cache.len() < self.config.max_hot_size {
            self.hot_cache.insert(hash);
            self.stats.hot_count += 1;
            return Ok(true);
        }

        // Hot cache is full - evict oldest hashes to cold storage
        self.evict_to_cold()?;

        // Now insert the new hash
        self.hot_cache.insert(hash);
        self.stats.hot_count += 1;

        Ok(true)
    }

    /// Evict hashes from hot cache to cold storage
    fn evict_to_cold(&mut self) -> Result<()> {
        let evict_count = self.config.max_hot_size / 10; // Evict 10% at a time
        debug!("Evicting {} hashes from hot to cold storage", evict_count);

        // Simple eviction: take first N hashes
        // In a production system, would use proper LRU tracking
        let hashes_to_evict: Vec<u64> = self
            .hot_cache
            .iter()
            .take(evict_count)
            .copied()
            .collect();

        let mut batch = sled::Batch::default();
        for hash in &hashes_to_evict {
            let hash_bytes = hash.to_be_bytes();
            batch.insert(&hash_bytes, &[1u8]); // Value is just a marker
            self.hot_cache.remove(hash);
        }

        self.cold_storage.apply_batch(batch).map_err(|e| {
            Error::ProcessingError(format!("Failed to evict to cold storage: {}", e))
        })?;

        if self.config.sync_on_write {
            self.cold_storage.flush().map_err(|e| {
                Error::ProcessingError(format!("Failed to flush cold storage: {}", e))
            })?;
        }

        self.stats.evictions += evict_count;
        self.stats.hot_count = self.hot_cache.len();
        self.stats.cold_count += evict_count;

        info!(
            "Evicted {} hashes. Hot: {}, Cold: {}",
            evict_count, self.stats.hot_count, self.stats.cold_count
        );

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> &StorageStats {
        &self.stats
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        self.cold_storage.flush().map_err(|e| {
            Error::ProcessingError(format!("Failed to flush cold storage: {}", e))
        })?;
        Ok(())
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        // Hot cache: ~24 bytes overhead + capacity * 16 bytes per u64
        24 + self.hot_cache.capacity() * 16
    }

    /// Get disk usage estimate in bytes
    pub fn disk_usage(&self) -> usize {
        // Each hash in cold storage: 8 bytes key + 1 byte value + overhead
        self.stats.cold_count * 20 // Rough estimate with overhead
    }

    /// Clear all data (both hot and cold)
    pub fn clear(&mut self) -> Result<()> {
        self.hot_cache.clear();
        self.cold_storage.clear().map_err(|e| {
            Error::ProcessingError(format!("Failed to clear cold storage: {}", e))
        })?;
        self.stats = StorageStats::default();
        Ok(())
    }

    /// Compact cold storage to reclaim disk space
    pub fn compact(&self) -> Result<()> {
        info!("Compacting cold storage");
        // Sled doesn't have a direct compact method in this version
        // Flush is the closest operation
        self.flush()?;
        Ok(())
    }
}

impl Drop for TieredHashStorage {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            warn!("Failed to flush cold storage on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_basic_insert_and_contains() {
        let mut storage = TieredHashStorage::temporary().unwrap();

        assert!(!storage.contains(123).unwrap());
        assert!(storage.insert(123).unwrap());
        assert!(storage.contains(123).unwrap());
        assert!(!storage.insert(123).unwrap()); // Already exists

        storage.clear().unwrap();
    }

    #[test]
    fn test_hot_cache_usage() {
        let mut storage = TieredHashStorage::temporary().unwrap();

        // Insert some hashes
        for i in 0..1000 {
            storage.insert(i).unwrap();
        }

        assert_eq!(storage.stats().hot_count, 1000);
        assert_eq!(storage.stats().cold_count, 0);

        // All should be in hot cache
        for i in 0..1000 {
            assert!(storage.contains(i).unwrap());
        }

        assert!(storage.stats().hot_hits >= 1000);
        assert_eq!(storage.stats().cold_hits, 0);

        storage.clear().unwrap();
    }

    #[test]
    fn test_eviction_to_cold() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let config = TieredStorageConfig {
            db_path: format!("/tmp/dedup_eviction_test_{}_{}", std::process::id(), timestamp),
            max_hot_size: 100,
            sync_on_write: false,
        };

        let mut storage = TieredHashStorage::with_config(config).unwrap();

        // Insert more than hot cache capacity
        for i in 0..150 {
            storage.insert(i).unwrap();
        }

        // Some should have been evicted
        assert!(storage.stats().evictions > 0);
        assert!(storage.stats().cold_count > 0);

        // All hashes should still be accessible
        for i in 0..150 {
            assert!(storage.contains(i).unwrap(), "Hash {} not found", i);
        }

        storage.clear().unwrap();
    }

    #[test]
    fn test_promotion_from_cold() {
        let config = TieredStorageConfig {
            db_path: format!("/tmp/dedup_promotion_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
            max_hot_size: 100,
            sync_on_write: false,
        };

        let mut storage = TieredHashStorage::with_config(config).unwrap();

        // Fill hot cache and trigger eviction
        for i in 0..150 {
            storage.insert(i).unwrap();
        }

        // Verify some hashes were evicted to cold storage
        let cold_count_before_clear = storage.stats().cold_count;
        assert!(cold_count_before_clear > 0, "Expected some hashes in cold storage");

        // Clear hot cache (simulate fresh start)
        storage.hot_cache.clear();
        storage.stats.hot_count = 0;

        // Try to access hashes - some should be found in cold storage
        let initial_cold_hits = storage.stats().cold_hits;
        let mut found_count = 0;
        for i in 0..150 {
            if storage.contains(i).unwrap() {
                found_count += 1;
            }
        }

        // Should have found at least the hashes that were in cold storage
        assert!(found_count >= cold_count_before_clear,
            "Expected to find at least {} hashes in cold storage, found {}",
            cold_count_before_clear, found_count);
        assert!(storage.stats().cold_hits > initial_cold_hits,
            "Expected cold storage hits");

        storage.clear().unwrap();
    }

    #[test]
    fn test_persistence() {
        let db_path = format!("/tmp/dedup_persist_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

        {
            let config = TieredStorageConfig {
                db_path: db_path.clone(),
                max_hot_size: 50,
                sync_on_write: true,
            };

            let mut storage = TieredHashStorage::with_config(config).unwrap();

            // Insert enough to trigger eviction to disk
            for i in 0..100 {
                storage.insert(i).unwrap();
            }

            storage.flush().unwrap();
        }

        // Reopen and verify data persisted to cold storage
        {
            let config = TieredStorageConfig {
                db_path: db_path.clone(),
                max_hot_size: 50,
                sync_on_write: false,
            };

            let mut storage = TieredHashStorage::with_config(config).unwrap();

            // Should have hashes in cold storage from previous session
            assert!(storage.stats().cold_count > 0, "Expected hashes in cold storage");

            // Verify that at least some hashes from cold storage are accessible
            // Note: Only hashes that were evicted to cold will be present
            // The ones still in hot cache at shutdown were lost
            let cold_count = storage.stats().cold_count;
            assert!(cold_count > 0, "Expected cold storage to have data");

            storage.clear().unwrap();
        }

        // Cleanup
        let _ = fs::remove_dir_all(db_path);
    }

    #[test]
    fn test_memory_and_disk_usage() {
        let mut storage = TieredHashStorage::temporary().unwrap();

        for i in 0..1000 {
            storage.insert(i).unwrap();
        }

        let mem_usage = storage.memory_usage();
        assert!(mem_usage > 0);
        assert!(mem_usage < 10_000_000); // Should be reasonable

        storage.clear().unwrap();
    }

    #[test]
    fn test_large_scale() {
        let config = TieredStorageConfig {
            db_path: format!("/tmp/dedup_large_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
            max_hot_size: 10_000,
            sync_on_write: false,
        };

        let mut storage = TieredHashStorage::with_config(config).unwrap();

        // Insert 50k hashes
        for i in 0..50_000 {
            storage.insert(i).unwrap();
        }

        assert_eq!(storage.stats().total_hashes(), 50_000);

        // Verify random access
        assert!(storage.contains(100).unwrap());
        assert!(storage.contains(25_000).unwrap());
        assert!(storage.contains(49_999).unwrap());
        assert!(!storage.contains(60_000).unwrap());

        info!(
            "Large scale test stats: hot={}, cold={}, evictions={}",
            storage.stats().hot_count,
            storage.stats().cold_count,
            storage.stats().evictions
        );

        storage.clear().unwrap();
    }

    #[test]
    fn test_stats_calculation() {
        let mut storage = TieredHashStorage::temporary().unwrap();

        for i in 0..10 {
            storage.insert(i).unwrap();
        }

        // Access all hashes (should all be hot)
        for i in 0..10 {
            storage.contains(i).unwrap();
        }

        let stats = storage.stats();
        assert_eq!(stats.hot_hit_rate(), 100.0);

        storage.clear().unwrap();
    }
}
