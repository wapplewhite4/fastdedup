//! Error types for the core deduplication engine

use thiserror::Error;

/// Core deduplication errors
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hash computation error: {0}")]
    HashError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Result type alias for core operations
pub type Result<T> = std::result::Result<T, Error>;
