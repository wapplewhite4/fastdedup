//! Disk-backed key-value store for field values.
//!
//! Uses a bounded in-memory HashMap with flat-file overflow so that
//! the `field_values` cache in fuzzy dedup doesn't grow unboundedly.
//! No sled, no mmap — just plain file I/O to keep RSS predictable.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use anyhow::Result;

/// Entry in the cold file index: byte offset + byte length.
struct ColdEntry {
    offset: u64,
    len: u32,
}

/// A bounded string-value store: recent entries live in memory,
/// older ones are spilled to a flat file.
pub struct DiskBackedStringMap {
    hot: HashMap<usize, String>,
    cold_writer: BufWriter<File>,
    cold_reader: BufReader<File>,
    cold_index: HashMap<usize, ColdEntry>,
    cold_write_pos: u64,
    insertion_order: VecDeque<usize>,
    max_hot: usize,
    temp_path: Option<PathBuf>,
}

impl DiskBackedStringMap {
    /// Create a new store with the given in-memory capacity.
    pub fn new(max_hot: usize) -> Result<Self> {
        let path = Self::make_temp_path();

        let writer = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        let reader = File::open(&path)?;

        Ok(Self {
            hot: HashMap::with_capacity(max_hot.min(500_000)),
            cold_writer: BufWriter::with_capacity(64 * 1024, writer),
            cold_reader: BufReader::with_capacity(4096, reader),
            cold_index: HashMap::new(),
            cold_write_pos: 0,
            insertion_order: VecDeque::with_capacity(max_hot.min(500_000)),
            max_hot,
            temp_path: Some(path),
        })
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, id: usize, value: String) -> Result<()> {
        if self.hot.len() >= self.max_hot {
            self.evict()?;
        }
        self.hot.insert(id, value);
        self.insertion_order.push_back(id);
        Ok(())
    }

    /// Get a value by key.  Returns an owned `String`.
    pub fn get(&mut self, id: &usize) -> Result<Option<String>> {
        if let Some(v) = self.hot.get(id) {
            return Ok(Some(v.clone()));
        }
        // Check cold file
        let entry = match self.cold_index.get(id) {
            Some(e) => e,
            None => return Ok(None),
        };

        self.cold_writer.flush()?;
        self.cold_reader.seek(SeekFrom::Start(entry.offset))?;
        let mut buf = vec![0u8; entry.len as usize];
        self.cold_reader.read_exact(&mut buf)?;
        Ok(Some(String::from_utf8(buf).unwrap_or_default()))
    }

    fn evict(&mut self) -> Result<()> {
        let evict_count = (self.max_hot / 10).max(1);
        let mut evicted = 0;

        while evicted < evict_count {
            let id = match self.insertion_order.pop_front() {
                Some(id) => id,
                None => break,
            };
            if let Some(value) = self.hot.remove(&id) {
                let bytes = value.as_bytes();
                let offset = self.cold_write_pos;
                self.cold_writer.write_all(bytes)?;
                self.cold_write_pos += bytes.len() as u64;
                self.cold_index.insert(id, ColdEntry {
                    offset,
                    len: bytes.len() as u32,
                });
                evicted += 1;
            }
        }

        self.cold_writer.flush()?;
        Ok(())
    }

    fn make_temp_path() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dedup_fv_{}.bin", ts))
    }
}

impl Drop for DiskBackedStringMap {
    fn drop(&mut self) {
        let _ = self.cold_writer.flush();
        if let Some(ref path) = self.temp_path {
            let _ = std::fs::remove_file(path);
        }
    }
}
