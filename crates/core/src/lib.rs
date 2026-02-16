//! Core deduplication logic for dataset processing
//!
//! This crate provides the fundamental data structures and algorithms
//! for high-performance dataset deduplication.

pub mod error;
pub mod hash;
pub mod dedup;
pub mod exact_dedup;
pub mod hash_storage;
pub mod minhash;
pub mod fuzzy_dedup;
pub mod pipeline;
pub mod memory;

pub use error::{Error, Result};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
