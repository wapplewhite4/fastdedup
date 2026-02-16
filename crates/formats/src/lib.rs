//! File format readers for dataset processing
//!
//! This crate provides streaming readers for various dataset formats
//! with memory-efficient implementations optimized for large files.

pub mod error;
pub mod jsonl;
pub mod parquet_reader;
pub mod reader;
pub mod record;

pub use error::{Error, Result};
pub use reader::{open_dataset, DatasetReader};
pub use record::Record;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
