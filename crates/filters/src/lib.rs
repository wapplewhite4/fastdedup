//! Quality filters for dataset cleaning
//!
//! This crate provides various filters for cleaning and validating
//! dataset records based on quality metrics.

pub mod error;
pub mod length_filter;
pub mod text_preprocessing;

// Phase 5: Quality Filters
pub mod language;
pub mod quality;

pub use error::{Error, Result};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
