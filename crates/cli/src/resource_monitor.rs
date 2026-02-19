//! Process resource sampling: RSS memory and CPU usage.
//!
//! Reads from `/proc/self/status` (RSS) and `/proc/self/stat` (CPU jiffies)
//! on Linux.  Runs in a background thread and sends updates over the TUI's
//! existing `mpsc` channel every second until the sender is dropped.

use std::fs;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

use indicatif::ProgressBar;

use crate::tui::ProgressMsg;

/// Spawn a background thread that samples memory and CPU every `interval`
/// and sends `ProgressMsg::ResourceUpdate` over `tx`.
///
/// The thread exits automatically when the receiver end of the channel is
/// dropped (i.e. when the run finishes).
pub fn spawn(tx: Sender<ProgressMsg>, interval: Duration) {
    thread::spawn(move || {
        let mut prev_cpu = read_cpu_jiffies();
        let mut prev_time = Instant::now();

        loop {
            thread::sleep(interval);

            let (memory_mb, cpu_pct) = sample(&mut prev_cpu, &mut prev_time);

            if tx
                .send(ProgressMsg::ResourceUpdate {
                    memory_mb,
                    cpu_pct,
                })
                .is_err()
            {
                // Receiver dropped — run finished, exit thread.
                break;
            }
        }
    });
}

/// Spawn a background thread that samples memory and CPU every `interval`
/// and writes a formatted string into `bar`'s message.
///
/// Used by the CLI (non-TUI) progress reporter.  The thread exits when `bar`
/// is finished (i.e. `bar.is_finished()` returns true).
pub fn spawn_into_bar(bar: ProgressBar, interval: Duration) {
    thread::spawn(move || {
        let mut prev_cpu = read_cpu_jiffies();
        let mut prev_time = Instant::now();

        loop {
            thread::sleep(interval);

            if bar.is_finished() {
                break;
            }

            let (memory_mb, cpu_pct) = sample(&mut prev_cpu, &mut prev_time);

            let mem_str = if memory_mb == 0 {
                "—".to_string()
            } else if memory_mb >= 1024 {
                format!("{:.1} GB", memory_mb as f64 / 1024.0)
            } else {
                format!("{} MB", memory_mb)
            };

            bar.set_message(format!("{mem_str} RAM | {cpu_pct:.1}% CPU"));
        }
    });
}

// ── Core sampling ─────────────────────────────────────────────────────────────

/// Take one sample, updating `prev_cpu` / `prev_time` in-place.
/// Returns `(memory_mb, cpu_pct)`.
fn sample(prev_cpu: &mut u64, prev_time: &mut Instant) -> (u64, f64) {
    let memory_mb = read_rss_mb().unwrap_or(0);

    let curr_cpu = read_cpu_jiffies();
    let curr_time = Instant::now();
    let elapsed_secs = curr_time.duration_since(*prev_time).as_secs_f64();

    let cpu_pct = if elapsed_secs > 0.0 && curr_cpu >= *prev_cpu {
        // jiffies are in 1/100 s units; scale to percentage of one core
        let delta_jiffies = (curr_cpu - *prev_cpu) as f64;
        // delta_jiffies / 100.0 → seconds of CPU used; divide by wall-clock elapsed
        (delta_jiffies / 100.0 / elapsed_secs * 100.0).min(num_cpus() as f64 * 100.0)
    } else {
        0.0
    };

    *prev_cpu = curr_cpu;
    *prev_time = curr_time;

    (memory_mb, cpu_pct)
}

// ── /proc helpers ────────────────────────────────────────────────────────────

/// Read the process's RSS from `/proc/self/status` in megabytes.
fn read_rss_mb() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // format: "VmRSS:   123456 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

/// Read the sum of utime + stime from `/proc/self/stat` in jiffies (1/100 s).
fn read_cpu_jiffies() -> u64 {
    let Ok(text) = fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // Fields are space-separated.  The second field is the command name
    // wrapped in parentheses (which can contain spaces), so we find the
    // closing ')' and skip past it.
    let after_paren = match text.rfind(')') {
        Some(i) => &text[i + 1..],
        None => return 0,
    };

    // After the closing ')':  state(1) ppid(2) pgrp(3) session(4) tty(5)
    // tpgid(6) flags(7) minflt(8) cminflt(9) majflt(10) cmajflt(11)
    // utime(12) stime(13) ...
    let fields: Vec<&str> = after_paren.split_whitespace().collect();
    // 0-indexed in `fields` (first field after ')' is state → index 0)
    // utime is field 12 → index 11, stime is field 13 → index 12
    let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    utime + stime
}

/// Number of logical CPUs (for capping the CPU% display).
fn num_cpus() -> usize {
    // Read from /proc/cpuinfo; fall back to 1.
    let Ok(text) = fs::read_to_string("/proc/cpuinfo") else {
        return 1;
    };
    text.lines()
        .filter(|l| l.starts_with("processor"))
        .count()
        .max(1)
}
