//! Memory profiling and optimization
//!
//! Provides memory tracking and limits to prevent OOM errors
//! during large dataset processing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracking allocator that monitors memory usage
pub struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let size = layout.size();
            let prev = ALLOCATED.fetch_add(size, Ordering::SeqCst);
            let new = prev + size;

            // Update peak if needed
            let mut peak = PEAK_ALLOCATED.load(Ordering::SeqCst);
            while new > peak {
                match PEAK_ALLOCATED.compare_exchange(
                    peak,
                    new,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => peak = x,
                }
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

/// Get current memory usage in bytes
pub fn current_memory_usage() -> usize {
    ALLOCATED.load(Ordering::SeqCst)
}

/// Get peak memory usage in bytes
pub fn peak_memory_usage() -> usize {
    PEAK_ALLOCATED.load(Ordering::SeqCst)
}

/// Reset peak memory tracking
pub fn reset_peak_memory() {
    let current = ALLOCATED.load(Ordering::SeqCst);
    PEAK_ALLOCATED.store(current, Ordering::SeqCst);
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let bytes_f = bytes as f64;

    if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Parse human-readable byte size string (e.g., "8GB", "512MB")
pub fn parse_bytes(s: &str) -> Option<usize> {
    let s = s.trim().to_uppercase();

    if let Some(gb_pos) = s.find("GB") {
        let num: f64 = s[..gb_pos].trim().parse().ok()?;
        Some((num * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(mb_pos) = s.find("MB") {
        let num: f64 = s[..mb_pos].trim().parse().ok()?;
        Some((num * 1024.0 * 1024.0) as usize)
    } else if let Some(kb_pos) = s.find("KB") {
        let num: f64 = s[..kb_pos].trim().parse().ok()?;
        Some((num * 1024.0) as usize)
    } else {
        s.parse().ok()
    }
}

/// Memory limiter to prevent OOM errors
#[derive(Debug, Clone)]
pub struct MemoryLimiter {
    /// Maximum memory in bytes
    max_bytes: usize,
    /// Warning threshold (percentage)
    warning_threshold: f64,
}

impl MemoryLimiter {
    /// Create a new memory limiter
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            warning_threshold: 0.8, // Warn at 80%
        }
    }

    /// Create from human-readable string (e.g., "8GB")
    pub fn from_string(s: &str) -> Option<Self> {
        parse_bytes(s).map(Self::new)
    }

    /// Set warning threshold (0.0-1.0)
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Check if memory limit is exceeded
    pub fn check(&self) -> MemoryStatus {
        let current = current_memory_usage();
        let usage_ratio = current as f64 / self.max_bytes as f64;

        if current >= self.max_bytes {
            MemoryStatus::LimitExceeded {
                current,
                limit: self.max_bytes,
            }
        } else if usage_ratio >= self.warning_threshold {
            MemoryStatus::Warning {
                current,
                limit: self.max_bytes,
                usage_percent: usage_ratio * 100.0,
            }
        } else {
            MemoryStatus::Ok {
                current,
                limit: self.max_bytes,
            }
        }
    }

    /// Get current memory usage percentage
    pub fn usage_percent(&self) -> f64 {
        let current = current_memory_usage();
        (current as f64 / self.max_bytes as f64) * 100.0
    }

    /// Get available memory
    pub fn available(&self) -> usize {
        let current = current_memory_usage();
        self.max_bytes.saturating_sub(current)
    }
}

/// Memory status
#[derive(Debug, Clone)]
pub enum MemoryStatus {
    /// Memory usage is within acceptable limits
    Ok { current: usize, limit: usize },
    /// Memory usage is approaching the limit
    Warning {
        current: usize,
        limit: usize,
        usage_percent: f64,
    },
    /// Memory limit exceeded
    LimitExceeded { current: usize, limit: usize },
}

impl MemoryStatus {
    /// Check if status is OK
    pub fn is_ok(&self) -> bool {
        matches!(self, MemoryStatus::Ok { .. })
    }

    /// Check if status is warning
    pub fn is_warning(&self) -> bool {
        matches!(self, MemoryStatus::Warning { .. })
    }

    /// Check if limit exceeded
    pub fn is_exceeded(&self) -> bool {
        matches!(self, MemoryStatus::LimitExceeded { .. })
    }

    /// Get status message
    pub fn message(&self) -> String {
        match self {
            MemoryStatus::Ok { current, limit } => {
                format!(
                    "Memory usage OK: {} / {}",
                    format_bytes(*current),
                    format_bytes(*limit)
                )
            }
            MemoryStatus::Warning {
                current,
                limit,
                usage_percent,
            } => {
                format!(
                    "Memory warning: {} / {} ({:.1}% used)",
                    format_bytes(*current),
                    format_bytes(*limit),
                    usage_percent
                )
            }
            MemoryStatus::LimitExceeded { current, limit } => {
                format!(
                    "Memory limit exceeded: {} / {}",
                    format_bytes(*current),
                    format_bytes(*limit)
                )
            }
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current: usize,
    pub peak: usize,
}

impl MemoryStats {
    /// Get current memory statistics
    pub fn current() -> Self {
        Self {
            current: current_memory_usage(),
            peak: peak_memory_usage(),
        }
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Current: {}, Peak: {}",
            format_bytes(self.current),
            format_bytes(self.peak)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_parse_bytes() {
        assert_eq!(parse_bytes("8GB"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_bytes("512MB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_bytes("1024KB"), Some(1024 * 1024));
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("  8 GB  "), Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_memory_limiter() {
        let limiter = MemoryLimiter::new(1024 * 1024 * 1024); // 1GB

        // Should be OK initially
        let status = limiter.check();
        assert!(status.is_ok() || status.is_warning());

        // Test usage percentage
        let usage = limiter.usage_percent();
        assert!(usage >= 0.0 && usage <= 100.0);
    }

    #[test]
    fn test_memory_limiter_from_string() {
        let limiter = MemoryLimiter::from_string("8GB").unwrap();
        assert_eq!(limiter.max_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats::current();
        assert!(stats.current >= 0); // Current memory should be non-negative
        assert!(stats.peak >= stats.current);
    }
}
