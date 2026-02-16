//! Deduplication algorithm implementations

use std::collections::HashSet;

/// Tracks seen hashes for deduplication
pub struct DedupTracker {
    seen_hashes: HashSet<u64>,
}

impl DedupTracker {
    /// Create a new deduplication tracker
    pub fn new() -> Self {
        Self {
            seen_hashes: HashSet::new(),
        }
    }

    /// Check if a hash has been seen before
    /// Returns true if this is a duplicate
    pub fn is_duplicate(&mut self, hash: u64) -> bool {
        !self.seen_hashes.insert(hash)
    }

    /// Get the number of unique items seen
    pub fn unique_count(&self) -> usize {
        self.seen_hashes.len()
    }

    /// Clear all tracked hashes
    pub fn clear(&mut self) {
        self.seen_hashes.clear();
    }
}

impl Default for DedupTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_tracker() {
        let mut tracker = DedupTracker::new();

        assert!(!tracker.is_duplicate(123));
        assert!(tracker.is_duplicate(123));
        assert!(!tracker.is_duplicate(456));
        assert_eq!(tracker.unique_count(), 2);
    }
}
