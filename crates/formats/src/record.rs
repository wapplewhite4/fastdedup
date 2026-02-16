//! Record data structure for unified dataset representation

use serde_json::Value;

/// A single record from a dataset
#[derive(Debug, Clone)]
pub struct Record {
    /// The JSON data for this record
    pub data: Value,
    /// Source line number or batch index
    pub source_line: usize,
    /// Lazily computed hash
    pub hash: Option<u64>,
}

impl Record {
    /// Create a new record
    pub fn new(data: Value, source_line: usize) -> Self {
        Self {
            data,
            source_line,
            hash: None,
        }
    }

    /// Compute and cache the hash for this record
    pub fn compute_hash(&mut self) -> u64 {
        if let Some(h) = self.hash {
            return h;
        }

        let hash = seahash::hash(self.data.to_string().as_bytes());
        self.hash = Some(hash);
        hash
    }

    /// Get the cached hash, computing it if necessary
    pub fn get_hash(&mut self) -> u64 {
        self.compute_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_record_creation() {
        let data = json!({"text": "hello", "id": 1});
        let record = Record::new(data.clone(), 0);
        assert_eq!(record.data, data);
        assert_eq!(record.source_line, 0);
        assert!(record.hash.is_none());
    }

    #[test]
    fn test_hash_computation() {
        let data = json!({"text": "hello"});
        let mut record = Record::new(data, 0);

        let hash1 = record.compute_hash();
        let hash2 = record.get_hash();

        assert_eq!(hash1, hash2);
        assert_eq!(record.hash, Some(hash1));
    }
}
