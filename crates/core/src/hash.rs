//! Hashing utilities for deduplication

use seahash::hash;

/// Compute a 64-bit hash of the given bytes
pub fn compute_hash(data: &[u8]) -> u64 {
    hash(data)
}

/// Compute a hash from a string
pub fn hash_string(s: &str) -> u64 {
    compute_hash(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let data = b"test data";
        let hash1 = compute_hash(data);
        let hash2 = compute_hash(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_string() {
        let s = "hello world";
        let hash1 = hash_string(s);
        let hash2 = hash_string(s);
        assert_eq!(hash1, hash2);
    }
}
