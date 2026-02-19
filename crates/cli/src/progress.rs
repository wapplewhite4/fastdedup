//! Progress reporting and visualization for CLI

use std::path::Path;
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::resource_monitor;

/// Progress reporter with multiple progress bars
pub struct ProgressReporter {
    _multi: MultiProgress,
    main_bar: ProgressBar,
    stats_bar: ProgressBar,
    resource_bar: ProgressBar,
    _start_time: Instant,
    mode: ProgressMode,
}

/// Progress tracking mode
enum ProgressMode {
    /// Track progress by bytes processed
    Bytes,
    /// Track progress by records processed
    Records,
}

impl ProgressReporter {
    /// Create a new progress reporter tracking bytes
    pub fn new(total_bytes: u64) -> Self {
        let multi = MultiProgress::new();

        // Main progress bar for bytes processed
        let main_bar = multi.add(ProgressBar::new(total_bytes));
        main_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) {msg}")
                .unwrap()
                .progress_chars("█▓▒░-"),
        );

        // Stats bar for duplicates/filtered counts
        let stats_bar = multi.add(ProgressBar::new(0));
        stats_bar.set_style(
            ProgressStyle::default_bar()
                .template("Stats: {msg}")
                .unwrap(),
        );

        // Resource bar for memory and CPU usage
        let resource_bar = multi.add(ProgressBar::new(0));
        resource_bar.set_style(
            ProgressStyle::default_bar()
                .template("Sys:   {msg}")
                .unwrap(),
        );
        resource_bar.set_message("collecting…");
        resource_monitor::spawn_into_bar(resource_bar.clone(), Duration::from_secs(1));

        Self {
            _multi: multi,
            main_bar,
            stats_bar,
            resource_bar,
            _start_time: Instant::now(),
            mode: ProgressMode::Bytes,
        }
    }

    /// Create a new progress reporter tracking records
    pub fn new_record_based(total_records: u64) -> Self {
        let multi = MultiProgress::new();

        // Main progress bar for records processed
        let main_bar = multi.add(ProgressBar::new(total_records));
        main_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {human_pos}/{human_len} ({per_sec}) {msg}")
                .unwrap()
                .progress_chars("█▓▒░-"),
        );

        // Stats bar for duplicates/filtered counts
        let stats_bar = multi.add(ProgressBar::new(0));
        stats_bar.set_style(
            ProgressStyle::default_bar()
                .template("Stats: {msg}")
                .unwrap(),
        );

        // Resource bar for memory and CPU usage
        let resource_bar = multi.add(ProgressBar::new(0));
        resource_bar.set_style(
            ProgressStyle::default_bar()
                .template("Sys:   {msg}")
                .unwrap(),
        );
        resource_bar.set_message("collecting…");
        resource_monitor::spawn_into_bar(resource_bar.clone(), Duration::from_secs(1));

        Self {
            _multi: multi,
            main_bar,
            stats_bar,
            resource_bar,
            _start_time: Instant::now(),
            mode: ProgressMode::Records,
        }
    }

    /// Update progress with current statistics (byte-based)
    pub fn update(&self, bytes: u64, total: usize, duplicates: usize, filtered: usize) {
        match self.mode {
            ProgressMode::Bytes => {
                self.main_bar.set_position(bytes);
            }
            ProgressMode::Records => {
                self.main_bar.set_position(total as u64);
            }
        }

        self.main_bar.set_message("Processing...");

        let stats_msg = if duplicates > 0 && filtered > 0 {
            format!(
                "{} total | {} duplicates ({:.1}%) | {} filtered ({:.1}%)",
                Self::format_number(total),
                Self::format_number(duplicates),
                (duplicates as f64 / total as f64) * 100.0,
                Self::format_number(filtered),
                (filtered as f64 / total as f64) * 100.0
            )
        } else if duplicates > 0 {
            format!(
                "{} total | {} duplicates ({:.1}%)",
                Self::format_number(total),
                Self::format_number(duplicates),
                (duplicates as f64 / total as f64) * 100.0
            )
        } else if filtered > 0 {
            format!(
                "{} total | {} filtered ({:.1}%)",
                Self::format_number(total),
                Self::format_number(filtered),
                (filtered as f64 / total as f64) * 100.0
            )
        } else {
            format!("{} total", Self::format_number(total))
        };

        self.stats_bar.set_message(stats_msg);
    }

    /// Finish progress reporting
    pub fn finish(&self) {
        self.main_bar.finish_with_message("Complete!");
        self.stats_bar.finish();
        self.resource_bar.finish();
    }

    /// Format large numbers with thousand separators
    fn format_number(n: usize) -> String {
        if n >= 1_000_000 {
            format!("{:.1}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.1}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }
}

/// Print a formatted summary report
pub fn print_summary_report(
    input: &Path,
    output: Option<&Path>,
    total: usize,
    unique: usize,
    duplicates: usize,
    filtered: usize,
) {
    println!("\n{}", "═".repeat(60));
    println!("Dataset Deduplication Complete");
    println!("{}", "═".repeat(60));
    println!("Input:              {}", input.display());

    if let Some(output_path) = output {
        println!("Output:             {}", output_path.display());
    } else {
        println!("Output:             (dry run - no output written)");
    }

    println!("Total records:      {}", format_with_commas(total));

    if duplicates > 0 {
        println!(
            "Duplicates removed: {} ({:.1}%)",
            format_with_commas(duplicates),
            (duplicates as f64 / total as f64) * 100.0
        );
    }

    if filtered > 0 {
        println!(
            "Quality filtered:   {} ({:.1}%)",
            format_with_commas(filtered),
            (filtered as f64 / total as f64) * 100.0
        );
    }

    println!(
        "Final dataset:      {} ({:.1}%)",
        format_with_commas(unique),
        (unique as f64 / total as f64) * 100.0
    );

    println!("{}", "═".repeat(60));
}

/// Format number with thousand separators
fn format_with_commas(n: usize) -> String {
    n.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(ProgressReporter::format_number(42), "42");
        assert_eq!(ProgressReporter::format_number(1_234), "1.2K");
        assert_eq!(ProgressReporter::format_number(1_234_567), "1.2M");
    }

    #[test]
    fn test_format_with_commas() {
        assert_eq!(format_with_commas(1234), "1,234");
        assert_eq!(format_with_commas(1234567), "1,234,567");
        assert_eq!(format_with_commas(42), "42");
    }
}
