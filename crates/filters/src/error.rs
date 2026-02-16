//! Error types for filters

use thiserror::Error;

/// Filter errors
#[derive(Error, Debug)]
pub enum Error {
    #[error("Filter error: {0}")]
    FilterError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type alias for filter operations
pub type Result<T> = std::result::Result<T, Error>;
