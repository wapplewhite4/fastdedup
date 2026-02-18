//! Rendering for all three TUI screens.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Wrap},
    Frame,
};

use super::{App, Mode, Screen};

// ─── Colours ─────────────────────────────────────────────────────────────────

const ACCENT: Color = Color::Cyan;
const FOCUSED: Color = Color::Yellow;
const DIM: Color = Color::DarkGray;
const OK: Color = Color::Green;
const ERR: Color = Color::Red;

// ─── Entry ───────────────────────────────────────────────────────────────────

pub fn render(f: &mut Frame, app: &App) {
    match app.screen {
        Screen::Config => render_config(f, app),
        Screen::Running => render_running(f, app),
        Screen::Results => render_results(f, app),
    }
}

// ─── Config screen ───────────────────────────────────────────────────────────

fn render_config(f: &mut Frame, app: &App) {
    let area = f.area();

    // Outer border
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " dataset-dedup TUI ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(outer, area);

    let inner = shrink(area, 1);

    // Vertical layout: mode bar | files | params | help
    let param_lines: u16 = match app.mode {
        Mode::Fuzzy => 11, // 7 param rows + 2 blank + 2 section headers
        Mode::Exact => 5,
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // mode selector
            Constraint::Length(5), // file paths
            Constraint::Length(param_lines),
            Constraint::Min(0),    // error / padding
            Constraint::Length(2), // help bar
        ])
        .split(inner);

    render_mode_bar(f, app, chunks[0]);
    render_files(f, app, chunks[1]);

    match app.mode {
        Mode::Fuzzy => render_fuzzy_params(f, app, chunks[2]),
        Mode::Exact => render_exact_params(f, app, chunks[2]),
    }

    // Error message
    if let Some(ref err) = app.error {
        let para = Paragraph::new(format!("  ⚠  {}", err))
            .style(Style::default().fg(ERR))
            .wrap(Wrap { trim: true });
        f.render_widget(para, chunks[3]);
    }

    render_help_config(f, app, chunks[4]);
}

fn render_mode_bar(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(DIM));
    f.render_widget(block, area);

    let inner = Rect {
        x: area.x + 2,
        y: area.y + 1,
        width: area.width.saturating_sub(4),
        height: 1,
    };

    let (fuzzy_style, exact_style) = match app.mode {
        Mode::Fuzzy => (
            Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
            Style::default().fg(DIM),
        ),
        Mode::Exact => (
            Style::default().fg(DIM),
            Style::default().fg(Color::Black).bg(ACCENT).add_modifier(Modifier::BOLD),
        ),
    };

    let line = Line::from(vec![
        Span::raw("Mode: "),
        Span::styled(" Fuzzy Dedup ", fuzzy_style),
        Span::raw("  "),
        Span::styled(" Exact Dedup ", exact_style),
        Span::styled("    [m] switch", Style::default().fg(DIM)),
    ]);
    f.render_widget(Paragraph::new(line), inner);
}

fn render_files(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" Files ", Style::default().fg(DIM)));
    f.render_widget(block, area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    field_line(f, rows[0], "Input ", &app.input, app.focused == 0, "");
    field_line(f, rows[1], "Output", &app.output, app.focused == 1, "");
}

fn render_fuzzy_params(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" Fuzzy Dedup Parameters ", Style::default().fg(DIM)));
    f.render_widget(block, area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    // Split into left (label+input) and right (hint) columns
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(inner);

    let left_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // field
            Constraint::Length(1), // threshold
            Constraint::Length(1), // num_hashes
            Constraint::Length(1), // shingle_size
            Constraint::Length(1), // word_shingles
            Constraint::Length(1), // bands
            Constraint::Length(1), // rows_per_band
            Constraint::Min(0),
        ])
        .split(cols[0]);

    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(cols[1]);

    field_line(f, left_rows[0], "Field      ", &app.field, app.focused == 2, "");
    field_line(f, left_rows[1], "Threshold  ", &app.threshold, app.focused == 3, "");
    field_line(f, left_rows[2], "Num Hashes ", &app.num_hashes, app.focused == 4, "");
    field_line(f, left_rows[3], "Shingle Size", &app.shingle_size, app.focused == 5, "");
    bool_line(f, left_rows[4], "Word Shingles", app.word_shingles, app.focused == 6);
    field_line(
        f,
        left_rows[5],
        "Bands      ",
        if app.bands.is_empty() { "auto" } else { &app.bands },
        app.focused == 7,
        "",
    );
    field_line(f, left_rows[6], "Rows/Band  ", &app.rows_per_band, app.focused == 8, "");

    // Hints column
    hint(f, right_rows[0], "JSON field containing the text");
    hint(f, right_rows[1], "0.0–1.0  ↑↓ to adjust");
    hint(f, right_rows[2], "more = accurate, slower  ↑↓");
    hint(f, right_rows[3], "char n-gram size  ↑↓");
    hint(f, right_rows[4], "Space to toggle  (char n-grams default)");
    hint(f, right_rows[5], "leave blank for auto (hashes÷rows)");
    hint(f, right_rows[6], "↑↓ to adjust  (bands×rows = hashes)");
}

fn render_exact_params(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(" Exact Dedup Parameters ", Style::default().fg(DIM)));
    f.render_widget(block, area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(inner);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)])
        .split(cols[0]);
    let hint_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)])
        .split(cols[1]);

    field_line(f, rows[0], "Field    ", &app.field, app.focused == 2, "");
    bool_line(f, rows[1], "Normalize", app.normalize, app.focused == 3);
    hint(f, hint_rows[0], "leave blank to hash entire record");
    hint(f, hint_rows[1], "Space to toggle");
}

fn render_help_config(f: &mut Frame, app: &App, area: Rect) {
    let numeric_hint = if app.is_numeric_field() {
        "  ↑↓ adjust value"
    } else {
        "  ↑↓ next/prev field"
    };
    let bool_hint = if app.is_bool_field() { "  Space toggle" } else { "" };

    let line = Line::from(vec![
        Span::styled("[Enter]", Style::default().fg(OK).add_modifier(Modifier::BOLD)),
        Span::raw(" Run  "),
        Span::styled("[Tab]", Style::default().fg(ACCENT)),
        Span::raw(" Next field  "),
        Span::styled("[Shift+Tab]", Style::default().fg(ACCENT)),
        Span::raw(" Prev"),
        Span::styled(numeric_hint, Style::default().fg(ACCENT)),
        Span::styled(bool_hint, Style::default().fg(ACCENT)),
        Span::raw("  "),
        Span::styled("[q]", Style::default().fg(ERR)),
        Span::raw(" Quit"),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

// ─── Running screen ──────────────────────────────────────────────────────────

fn render_running(f: &mut Frame, app: &App) {
    let area = f.area();

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " Running… ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(outer, area);

    let inner = shrink(area, 1);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // gauge
            Constraint::Length(7), // stats
            Constraint::Min(0),
            Constraint::Length(1), // help
        ])
        .split(inner);

    if let Some(ref p) = app.progress {
        // Progress gauge
        let ratio = match p.total_records {
            Some(t) if t > 0 => (p.processed as f64 / t as f64).min(1.0),
            _ => 0.0,
        };
        let pct = (ratio * 100.0) as u16;

        let label = match p.total_records {
            Some(t) => format!(
                "{} / {} records  ({pct}%)",
                fmt_num(p.processed),
                fmt_num(t)
            ),
            None => format!("{} records processed", fmt_num(p.processed)),
        };

        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::NONE))
            .gauge_style(Style::default().fg(ACCENT).bg(Color::DarkGray))
            .ratio(ratio)
            .label(label);

        // Gauge needs a 3-line area; put it inside a padded block
        let gauge_area = Rect {
            x: chunks[0].x,
            y: chunks[0].y + 1,
            width: chunks[0].width,
            height: 1,
        };
        f.render_widget(gauge, gauge_area);

        // Stats
        let elapsed = p.start.elapsed();
        let rate = if elapsed.as_secs_f64() > 0.0 {
            p.processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let eta = if rate > 0.0 {
            match p.total_records {
                Some(t) if t > p.processed => {
                    let secs = ((t - p.processed) as f64 / rate) as u64;
                    format_duration(secs)
                }
                _ => "—".to_string(),
            }
        } else {
            "—".to_string()
        };

        let dup_pct = if p.processed > 0 {
            p.duplicates as f64 / p.processed as f64 * 100.0
        } else {
            0.0
        };

        let stats_text = vec![
            Line::from(vec![
                Span::styled("  Duplicates removed  ", Style::default().fg(DIM)),
                Span::styled(
                    format!("{} ({:.2}%)", fmt_num(p.duplicates), dup_pct),
                    Style::default().fg(OK),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Rate                ", Style::default().fg(DIM)),
                Span::raw(format!("{:.0} records/s", rate)),
            ]),
            Line::from(vec![
                Span::styled("  Elapsed             ", Style::default().fg(DIM)),
                Span::raw(format_duration(elapsed.as_secs())),
            ]),
            Line::from(vec![
                Span::styled("  ETA                 ", Style::default().fg(DIM)),
                Span::raw(eta),
            ]),
        ];

        f.render_widget(Paragraph::new(stats_text), chunks[1]);
    }

    let help = Line::from(vec![
        Span::styled("[Ctrl+C]", Style::default().fg(ERR)),
        Span::raw(" or "),
        Span::styled("[q]", Style::default().fg(ERR)),
        Span::raw(" abort"),
    ]);
    f.render_widget(Paragraph::new(help), chunks[3]);
}

// ─── Results screen ──────────────────────────────────────────────────────────

fn render_results(f: &mut Frame, app: &App) {
    let area = f.area();

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(OK))
        .title(Span::styled(
            " Done ",
            Style::default().fg(OK).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(outer, area);

    let inner = shrink(area, 1);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1), // help
        ])
        .split(inner);

    if let Some(ref r) = app.results {
        let dup_pct = if r.total > 0 {
            r.duplicates as f64 / r.total as f64 * 100.0
        } else {
            0.0
        };
        let unique_pct = 100.0 - dup_pct;

        let mut lines = vec![
            Line::from(""),
            stat_line("  Total records       ", &fmt_num(r.total), Color::White),
            stat_line(
                "  Unique (kept)       ",
                &format!("{} ({:.2}%)", fmt_num(r.unique), unique_pct),
                OK,
            ),
            stat_line(
                "  Duplicates removed  ",
                &format!("{} ({:.2}%)", fmt_num(r.duplicates), dup_pct),
                if r.duplicates > 0 { Color::Yellow } else { OK },
            ),
        ];

        if let Some(prec) = r.lsh_precision {
            lines.push(stat_line(
                "  LSH precision       ",
                &format!("{:.1}%", prec),
                DIM,
            ));
        }

        lines.push(stat_line(
            "  Elapsed             ",
            &format_duration(r.elapsed.as_secs()),
            DIM,
        ));

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  Output  ", Style::default().fg(DIM)),
            Span::styled(&r.output_path, Style::default().fg(ACCENT)),
        ]));

        if let Some(ref rp) = r.removed_path {
            lines.push(Line::from(vec![
                Span::styled("  Removed ", Style::default().fg(DIM)),
                Span::styled(rp.as_str(), Style::default().fg(ACCENT)),
            ]));
        }

        f.render_widget(Paragraph::new(lines), chunks[0]);
    }

    let help = Line::from(vec![
        Span::styled("[Enter]", Style::default().fg(ACCENT)),
        Span::raw(" or "),
        Span::styled("[r]", Style::default().fg(ACCENT)),
        Span::raw(" run again  "),
        Span::styled("[q]", Style::default().fg(ERR)),
        Span::raw(" quit"),
    ]);
    f.render_widget(Paragraph::new(help), chunks[1]);
}

// ─── Widget helpers ──────────────────────────────────────────────────────────

/// Single-line field: "Label   [value___]"
fn field_line(f: &mut Frame, area: Rect, label: &str, value: &str, focused: bool, _hint: &str) {
    let fg = if focused { FOCUSED } else { Color::White };
    let bracket_style = if focused {
        Style::default().fg(FOCUSED).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(DIM)
    };
    let cursor = if focused { "▌" } else { "" };

    let line = Line::from(vec![
        Span::styled(format!("  {:13}", label), Style::default().fg(DIM)),
        Span::styled("[", bracket_style),
        Span::styled(
            format!("{:<20}", format!("{}{}", value, cursor)),
            Style::default().fg(fg),
        ),
        Span::styled("]", bracket_style),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// Single-line bool toggle: "Label   [x] on / [ ] off"
fn bool_line(f: &mut Frame, area: Rect, label: &str, value: bool, focused: bool) {
    let bracket_style = if focused {
        Style::default().fg(FOCUSED).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(DIM)
    };
    let (mark, state_str, state_color) = if value {
        ("x", "on ", OK)
    } else {
        (" ", "off", DIM)
    };

    let line = Line::from(vec![
        Span::styled(format!("  {:13}", label), Style::default().fg(DIM)),
        Span::styled("[", bracket_style),
        Span::styled(mark, Style::default().fg(if value { OK } else { DIM })),
        Span::styled("]", bracket_style),
        Span::raw(" "),
        Span::styled(state_str, Style::default().fg(state_color)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// Right-aligned dim hint text
fn hint(f: &mut Frame, area: Rect, text: &str) {
    let para = Paragraph::new(Span::styled(text, Style::default().fg(DIM)))
        .alignment(Alignment::Right);
    f.render_widget(para, area);
}

/// Coloured stat row for the results screen
fn stat_line(label: &str, value: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(label.to_string(), Style::default().fg(DIM)),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

// ─── Utilities ───────────────────────────────────────────────────────────────

fn shrink(r: Rect, by: u16) -> Rect {
    Rect {
        x: r.x + by,
        y: r.y + by,
        width: r.width.saturating_sub(by * 2),
        height: r.height.saturating_sub(by * 2),
    }
}

fn fmt_num(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}
