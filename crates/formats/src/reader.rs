//! Unified dataset reader abstraction
//!
//! Provides a common interface for reading different dataset formats
//! with automatic format detection based on file extensions.

use crate::{jsonl::JsonlReader, parquet_reader::{ParquetReader, ParquetConfig}, Error, Record, Result};
use std::path::Path;
use tracing::info;

/// Trait for dataset readers providing unified interface
pub trait DatasetReader: Iterator<Item = Result<Record>> {
    /// Get total file size in bytes if known
    fn total_bytes(&self) -> Option<u64>;

    /// Get total number of records if known (from file metadata)
    fn total_records(&self) -> Option<u64>;

    /// Get number of bytes processed so far
    fn bytes_processed(&self) -> u64;

    /// Get the number of records processed
    fn records_processed(&self) -> usize;
}

/// JSONL dataset reader wrapper
pub struct JsonlDatasetReader {
    reader: JsonlReader<Box<dyn std::io::Read>>,
}

impl Iterator for JsonlDatasetReader {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next()
    }
}

impl DatasetReader for JsonlDatasetReader {
    fn total_bytes(&self) -> Option<u64> {
        self.reader.total_bytes()
    }

    fn total_records(&self) -> Option<u64> {
        None // JSONL doesn't have metadata for total record count
    }

    fn bytes_processed(&self) -> u64 {
        self.reader.bytes_processed()
    }

    fn records_processed(&self) -> usize {
        self.reader.lines_processed()
    }
}

/// Parquet dataset reader wrapper
pub struct ParquetDatasetReader {
    reader: ParquetReader,
}

impl Iterator for ParquetDatasetReader {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next()
    }
}

impl DatasetReader for ParquetDatasetReader {
    fn total_bytes(&self) -> Option<u64> {
        self.reader.total_bytes()
    }

    fn total_records(&self) -> Option<u64> {
        self.reader.total_records()
    }

    fn bytes_processed(&self) -> u64 {
        // For Parquet, we estimate based on records processed
        // This is approximate since we don't track actual byte position
        if let Some(total_bytes) = self.total_bytes() {
            if let Some(total_records) = self.total_records() {
                let records = self.records_processed() as u64;
                // Accurate estimation based on actual totals
                (records * total_bytes) / total_records.max(1)
            } else {
                // Fallback: rough estimation
                let records = self.records_processed() as u64;
                records * 100 // Assume ~100 bytes per record
            }
        } else {
            0
        }
    }

    fn records_processed(&self) -> usize {
        self.reader.records_processed()
    }
}

/// Factory function to open a dataset with automatic format detection
///
/// Supported formats:
/// - `.jsonl`, `.json` - JSON Lines format
/// - `.gz` - Gzip-compressed JSON Lines
/// - `.parquet` - Apache Parquet format
pub fn open_dataset<P: AsRef<Path>>(path: P) -> Result<Box<dyn DatasetReader>> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| Error::UnsupportedFormat("No file extension found".to_string()))?;

    info!("Opening dataset: {:?} (format: {})", path, extension);

    match extension {
        "jsonl" | "json" => {
            let reader = JsonlReader::open(path)?;
            Ok(Box::new(JsonlDatasetReader { reader }))
        }
        "gz" => {
            // Assume gzipped JSONL
            let reader = JsonlReader::open(path)?;
            Ok(Box::new(JsonlDatasetReader { reader }))
        }
        "parquet" => {
            let reader = ParquetReader::open(path)?;
            Ok(Box::new(ParquetDatasetReader { reader }))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "Unsupported file extension: {}",
            extension
        ))),
    }
}

/// Open a dataset with specific columns to read (if supported by format)
pub fn open_dataset_with_columns<P: AsRef<Path>>(
    path: P,
    columns: Vec<String>,
) -> Result<Box<dyn DatasetReader>> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| Error::UnsupportedFormat("No file extension found".to_string()))?;

    info!(
        "Opening dataset: {:?} (format: {}, columns: {:?})",
        path, extension, columns
    );

    match extension {
        "jsonl" | "json" => {
            let reader = JsonlReader::open(path)?.with_fields(columns);
            Ok(Box::new(JsonlDatasetReader { reader }))
        }
        "gz" => {
            let reader = JsonlReader::open(path)?.with_fields(columns);
            Ok(Box::new(JsonlDatasetReader { reader }))
        }
        "parquet" => {
            let config = ParquetConfig {
                batch_size: 4096,
                columns: Some(columns),
            };
            let reader = ParquetReader::open_with_config(path, config)?;
            Ok(Box::new(ParquetDatasetReader { reader }))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "Unsupported file extension: {}",
            extension
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_open_jsonl_dataset() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("jsonl");

        {
            let mut file = std::fs::File::create(&temp_path).unwrap();
            writeln!(file, r#"{{"text": "hello"}}"#).unwrap();
            writeln!(file, r#"{{"text": "world"}}"#).unwrap();
        }

        let mut reader = open_dataset(&temp_path).unwrap();
        let records: Vec<_> = reader.by_ref().collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data["text"], "hello");

        std::fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_open_dataset_unsupported_format() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("txt");

        let result = open_dataset(&temp_path);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::UnsupportedFormat(_))));
    }

    #[test]
    fn test_dataset_reader_progress() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("jsonl");

        {
            let mut file = std::fs::File::create(&temp_path).unwrap();
            writeln!(file, r#"{{"text": "hello"}}"#).unwrap();
            writeln!(file, r#"{{"text": "world"}}"#).unwrap();
        }

        let mut reader = open_dataset(&temp_path).unwrap();

        assert_eq!(reader.records_processed(), 0);

        let _ = reader.next();
        assert_eq!(reader.records_processed(), 1);
        assert!(reader.bytes_processed() > 0);

        std::fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_open_dataset_with_columns() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().with_extension("jsonl");

        {
            let mut file = std::fs::File::create(&temp_path).unwrap();
            writeln!(file, r#"{{"text": "hello", "id": 1, "meta": "extra"}}"#).unwrap();
            writeln!(file, r#"{{"text": "world", "id": 2, "meta": "data"}}"#).unwrap();
        }

        let reader = open_dataset_with_columns(&temp_path, vec!["text".to_string()]).unwrap();
        let records: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(records.len(), 2);
        assert!(records[0].data.get("text").is_some());
        assert!(records[0].data.get("id").is_none());
        assert!(records[0].data.get("meta").is_none());

        std::fs::remove_file(temp_path).unwrap();
    }
}
