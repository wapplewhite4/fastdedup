//! Streaming JSONL (JSON Lines) reader
//!
//! Provides memory-efficient streaming reading of JSONL files with
//! automatic gzip decompression support.

use crate::{Error, Record, Result};
use flate2::read::GzDecoder;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use tracing::{debug, warn};

/// Configuration for JSONL reader
#[derive(Debug, Clone)]
pub struct JsonlConfig {
    /// Fields to extract (None = all fields)
    pub fields: Option<Vec<String>>,
    /// Buffer size for BufReader
    pub buffer_size: usize,
}

impl Default for JsonlConfig {
    fn default() -> Self {
        Self {
            fields: None,
            buffer_size: 64 * 1024, // 64KB buffer
        }
    }
}

/// Streaming JSONL reader that processes files line-by-line
pub struct JsonlReader<R: Read> {
    reader: BufReader<R>,
    config: JsonlConfig,
    line_number: usize,
    bytes_read: u64,
    total_bytes: Option<u64>,
}

impl JsonlReader<Box<dyn Read>> {
    /// Open a JSONL file, auto-detecting gzip compression
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let total_bytes = file.metadata()?.len();

        let extension = path.extension().and_then(|e| e.to_str());

        match extension {
            Some("gz") => {
                debug!("Opening gzip-compressed JSONL file: {:?}", path);
                let decoder = GzDecoder::new(file);
                let reader: Box<dyn Read> = Box::new(decoder);
                Ok(Self::new_with_config(reader, JsonlConfig::default(), None))
            }
            _ => {
                debug!("Opening plain JSONL file: {:?}", path);
                let reader: Box<dyn Read> = Box::new(file);
                Ok(Self::new_with_config(reader, JsonlConfig::default(), Some(total_bytes)))
            }
        }
    }
}

impl<R: Read> JsonlReader<R> {
    /// Create a new JSONL reader from any Read source
    pub fn new(reader: R) -> Self {
        Self::new_with_config(reader, JsonlConfig::default(), None)
    }

    /// Create a new JSONL reader with custom configuration
    pub fn new_with_config(reader: R, config: JsonlConfig, total_bytes: Option<u64>) -> Self {
        let buf_reader = BufReader::with_capacity(config.buffer_size, reader);
        Self {
            reader: buf_reader,
            config,
            line_number: 0,
            bytes_read: 0,
            total_bytes,
        }
    }

    /// Set specific fields to extract
    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.config.fields = Some(fields);
        self
    }

    /// Get the number of lines processed
    pub fn lines_processed(&self) -> usize {
        self.line_number
    }

    /// Get the number of bytes read
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_read
    }

    /// Get total file size if known
    pub fn total_bytes(&self) -> Option<u64> {
        self.total_bytes
    }

    /// Extract only specified fields from a JSON value
    fn extract_fields(&self, value: Value) -> Value {
        if let Some(ref fields) = self.config.fields {
            if let Value::Object(map) = value {
                let filtered: serde_json::Map<String, Value> = map
                    .into_iter()
                    .filter(|(k, _)| fields.contains(k))
                    .collect();
                Value::Object(filtered)
            } else {
                value
            }
        } else {
            value
        }
    }
}

impl<R: Read> Iterator for JsonlReader<R> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(n) => {
                    self.bytes_read += n as u64;
                    self.line_number += 1;

                    // Skip empty lines
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Parse JSON
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => {
                            let extracted = self.extract_fields(value);
                            let record = Record::new(extracted, self.line_number);
                            return Some(Ok(record));
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse JSON at line {}: {} - Error: {}",
                                self.line_number, trimmed, e
                            );
                            // Skip malformed lines and continue
                            continue;
                        }
                    }
                }
                Err(e) => {
                    return Some(Err(Error::Io(e)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_jsonl_reader_basic() {
        let data = r#"{"text": "hello", "id": 1}
{"text": "world", "id": 2}
{"text": "rust", "id": 3}"#;

        let reader = JsonlReader::new(data.as_bytes());
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].data["text"], "hello");
        assert_eq!(records[1].data["text"], "world");
        assert_eq!(records[2].data["text"], "rust");
    }

    #[test]
    fn test_jsonl_reader_with_empty_lines() {
        let data = r#"{"text": "hello"}

{"text": "world"}

"#;

        let reader = JsonlReader::new(data.as_bytes());
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_jsonl_reader_with_malformed_json() {
        let data = r#"{"text": "hello"}
{invalid json}
{"text": "world"}"#;

        let reader = JsonlReader::new(data.as_bytes());
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        // Should skip malformed line and continue
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data["text"], "hello");
        assert_eq!(records[1].data["text"], "world");
    }

    #[test]
    fn test_jsonl_reader_field_extraction() {
        let data = r#"{"text": "hello", "id": 1, "meta": "extra"}
{"text": "world", "id": 2, "meta": "data"}"#;

        let reader = JsonlReader::new(data.as_bytes())
            .with_fields(vec!["text".to_string(), "id".to_string()]);

        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
        assert!(records[0].data.get("text").is_some());
        assert!(records[0].data.get("id").is_some());
        assert!(records[0].data.get("meta").is_none());
    }

    #[test]
    fn test_jsonl_reader_progress_tracking() {
        let data = r#"{"text": "hello"}
{"text": "world"}"#;

        let mut reader = JsonlReader::new(data.as_bytes());

        assert_eq!(reader.lines_processed(), 0);
        assert_eq!(reader.bytes_processed(), 0);

        let _ = reader.next();
        assert_eq!(reader.lines_processed(), 1);
        assert!(reader.bytes_processed() > 0);

        let _ = reader.next();
        assert_eq!(reader.lines_processed(), 2);
    }

    #[test]
    fn test_jsonl_reader_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"{{"text": "hello"}}"#).unwrap();
        writeln!(temp_file, r#"{{"text": "world"}}"#).unwrap();
        temp_file.flush().unwrap();

        let reader = JsonlReader::open(temp_file.path()).unwrap();
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_jsonl_reader_gzip() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("jsonl.gz");

        {
            let file = File::create(&temp_path).unwrap();
            let mut encoder = GzEncoder::new(file, Compression::default());
            writeln!(encoder, r#"{{"text": "compressed"}}"#).unwrap();
            writeln!(encoder, r#"{{"text": "data"}}"#).unwrap();
            encoder.finish().unwrap();
        }

        let reader = JsonlReader::open(&temp_path).unwrap();
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data["text"], "compressed");
        assert_eq!(records[1].data["text"], "data");

        std::fs::remove_file(temp_path).unwrap();
    }
}
