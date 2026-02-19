//! Terminal UI for dataset-dedup
//!
//! Three-screen workflow: Config → Running → Results

pub mod runner;
pub mod ui;

use crate::resource_monitor;

use std::io::Stdout;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

// ─── Progress protocol ───────────────────────────────────────────────────────

pub enum ProgressMsg {
    Update {
        processed: u64,
        duplicates: u64,
        total: Option<u64>,
    },
    ResourceUpdate {
        memory_mb: u64,
        cpu_pct: f64,
    },
    Done(RunResults),
    Error(String),
}

pub struct RunResults {
    pub total: u64,
    pub unique: u64,
    pub duplicates: u64,
    pub elapsed: Duration,
    pub lsh_precision: Option<f64>,
    pub output_path: String,
    pub removed_path: Option<String>,
}

// ─── App state ───────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
pub enum Screen {
    Config,
    Running,
    Results,
}

#[derive(Clone, PartialEq)]
pub enum Mode {
    Fuzzy,
    Exact,
}

pub struct RunProgress {
    pub total_records: Option<u64>,
    pub processed: u64,
    pub duplicates: u64,
    pub start: Instant,
    /// Resident Set Size in MB (updated by resource monitor thread)
    pub memory_mb: u64,
    /// CPU usage percentage across all cores (updated by resource monitor thread)
    pub cpu_pct: f64,
}

// Focusable field indices per mode
// Fuzzy: 0=input 1=output 2=field 3=threshold 4=num_hashes 5=shingle_size
//        6=word_shingles(bool) 7=bands 8=rows_per_band
// Exact: 0=input 1=output 2=field 3=normalize(bool)
const FUZZY_FIELD_COUNT: usize = 9;
const EXACT_FIELD_COUNT: usize = 4;

pub struct App {
    pub screen: Screen,
    pub mode: Mode,
    // file fields
    pub input: String,
    pub output: String,
    // shared
    pub field: String,
    // fuzzy params
    pub threshold: String,
    pub num_hashes: String,
    pub shingle_size: String,
    pub word_shingles: bool,
    pub bands: String, // empty → auto
    pub rows_per_band: String,
    // exact params
    pub normalize: bool,
    // UI
    pub focused: usize,
    pub error: Option<String>,
    pub should_quit: bool,
    // run state
    pub progress: Option<RunProgress>,
    pub results: Option<RunResults>,
    pub progress_rx: Option<mpsc::Receiver<ProgressMsg>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            screen: Screen::Config,
            mode: Mode::Fuzzy,
            input: String::new(),
            output: String::new(),
            field: "text".to_string(),
            threshold: "0.80".to_string(),
            num_hashes: "128".to_string(),
            shingle_size: "3".to_string(),
            word_shingles: false,
            bands: String::new(),
            rows_per_band: "8".to_string(),
            normalize: false,
            focused: 0,
            error: None,
            should_quit: false,
            progress: None,
            results: None,
            progress_rx: None,
        }
    }

    pub fn field_count(&self) -> usize {
        match self.mode {
            Mode::Fuzzy => FUZZY_FIELD_COUNT,
            Mode::Exact => EXACT_FIELD_COUNT,
        }
    }

    pub fn next_field(&mut self) {
        self.focused = (self.focused + 1) % self.field_count();
    }

    pub fn prev_field(&mut self) {
        if self.focused == 0 {
            self.focused = self.field_count() - 1;
        } else {
            self.focused -= 1;
        }
    }

    pub fn is_bool_field(&self) -> bool {
        match self.mode {
            Mode::Fuzzy => self.focused == 6,
            Mode::Exact => self.focused == 3,
        }
    }

    pub fn is_numeric_field(&self) -> bool {
        match self.mode {
            Mode::Fuzzy => matches!(self.focused, 3 | 4 | 5 | 8),
            Mode::Exact => false,
        }
    }

    pub fn toggle_bool(&mut self) {
        match self.mode {
            Mode::Fuzzy => {
                if self.focused == 6 {
                    self.word_shingles = !self.word_shingles;
                }
            }
            Mode::Exact => {
                if self.focused == 3 {
                    self.normalize = !self.normalize;
                }
            }
        }
    }

    pub fn adjust_up(&mut self) {
        match self.focused {
            3 => {
                if let Ok(v) = self.threshold.parse::<f64>() {
                    self.threshold = format!("{:.2}", (v + 0.05).min(1.0));
                }
            }
            4 => {
                if let Ok(v) = self.num_hashes.parse::<usize>() {
                    self.num_hashes = (v + 64).min(1024).to_string();
                }
            }
            5 => {
                if let Ok(v) = self.shingle_size.parse::<usize>() {
                    self.shingle_size = (v + 1).min(10).to_string();
                }
            }
            8 => {
                if let Ok(v) = self.rows_per_band.parse::<usize>() {
                    self.rows_per_band = (v + 1).min(32).to_string();
                }
            }
            _ => {}
        }
    }

    pub fn adjust_down(&mut self) {
        match self.focused {
            3 => {
                if let Ok(v) = self.threshold.parse::<f64>() {
                    self.threshold = format!("{:.2}", (v - 0.05).max(0.0));
                }
            }
            4 => {
                if let Ok(v) = self.num_hashes.parse::<usize>() {
                    self.num_hashes = v.saturating_sub(64).max(16).to_string();
                }
            }
            5 => {
                if let Ok(v) = self.shingle_size.parse::<usize>() {
                    self.shingle_size = v.saturating_sub(1).max(1).to_string();
                }
            }
            8 => {
                if let Ok(v) = self.rows_per_band.parse::<usize>() {
                    self.rows_per_band = v.saturating_sub(1).max(1).to_string();
                }
            }
            _ => {}
        }
    }

    pub fn type_char(&mut self, c: char) {
        if let Some(s) = self.focused_string() {
            s.push(c);
        }
    }

    pub fn backspace(&mut self) {
        if let Some(s) = self.focused_string() {
            s.pop();
        }
    }

    fn focused_string(&mut self) -> Option<&mut String> {
        match self.mode {
            Mode::Fuzzy => match self.focused {
                0 => Some(&mut self.input),
                1 => Some(&mut self.output),
                2 => Some(&mut self.field),
                3 => Some(&mut self.threshold),
                4 => Some(&mut self.num_hashes),
                5 => Some(&mut self.shingle_size),
                6 => None,
                7 => Some(&mut self.bands),
                8 => Some(&mut self.rows_per_band),
                _ => None,
            },
            Mode::Exact => match self.focused {
                0 => Some(&mut self.input),
                1 => Some(&mut self.output),
                2 => Some(&mut self.field),
                3 => None,
                _ => None,
            },
        }
    }

    pub fn poll_progress(&mut self) {
        let rx = match &self.progress_rx {
            Some(r) => r,
            None => return,
        };
        loop {
            match rx.try_recv() {
                Ok(ProgressMsg::Update {
                    processed,
                    duplicates,
                    total,
                }) => {
                    if let Some(p) = &mut self.progress {
                        p.processed = processed;
                        p.duplicates = duplicates;
                        if let Some(t) = total {
                            p.total_records = Some(t);
                        }
                    }
                }
                Ok(ProgressMsg::ResourceUpdate { memory_mb, cpu_pct }) => {
                    if let Some(p) = &mut self.progress {
                        p.memory_mb = memory_mb;
                        p.cpu_pct = cpu_pct;
                    }
                }
                Ok(ProgressMsg::Done(r)) => {
                    self.results = Some(r);
                    self.progress = None;
                    self.progress_rx = None;
                    self.screen = Screen::Results;
                    break;
                }
                Ok(ProgressMsg::Error(e)) => {
                    self.error = Some(e);
                    self.progress = None;
                    self.progress_rx = None;
                    self.screen = Screen::Config;
                    break;
                }
                Err(_) => break,
            }
        }
    }

    pub fn start_run(&mut self) {
        self.error = None;
        if self.input.is_empty() {
            self.error = Some("Input file path is required".to_string());
            return;
        }
        if self.output.is_empty() {
            self.error = Some("Output file path is required".to_string());
            return;
        }

        let (tx, rx) = mpsc::channel::<ProgressMsg>();
        self.progress_rx = Some(rx);
        self.progress = Some(RunProgress {
            total_records: None,
            processed: 0,
            duplicates: 0,
            start: Instant::now(),
            memory_mb: 0,
            cpu_pct: 0.0,
        });
        self.screen = Screen::Running;

        let input = PathBuf::from(&self.input);
        let output = PathBuf::from(&self.output);
        let mode = self.mode.clone();
        let field = self.field.clone();
        let threshold = self.threshold.parse::<f64>().unwrap_or(0.8);
        let num_hashes = self.num_hashes.parse::<usize>().unwrap_or(128);
        let shingle_size = self.shingle_size.parse::<usize>().unwrap_or(3);
        let word_shingles = self.word_shingles;
        let bands = self.bands.parse::<usize>().ok();
        let rows_per_band = self.rows_per_band.parse::<usize>().ok();
        let normalize = self.normalize;

        // Spawn resource-monitoring thread (samples every 1 s until run ends)
        resource_monitor::spawn(tx.clone(), Duration::from_secs(1));

        std::thread::spawn(move || match mode {
            Mode::Fuzzy => runner::run_fuzzy(
                tx,
                input,
                output,
                field,
                threshold,
                num_hashes,
                shingle_size,
                word_shingles,
                bands,
                rows_per_band,
            ),
            Mode::Exact => runner::run_exact(tx, input, output, field, normalize),
        });
    }

    pub fn reset_to_config(&mut self) {
        self.screen = Screen::Config;
        self.progress = None;
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

pub fn run_tui() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = event_loop(&mut terminal, &mut app);

    // Always restore the terminal, even on error
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui::render(f, app))?;

        // 50 ms tick gives smooth progress updates without burning CPU
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                // Ctrl-C always quits
                if key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                {
                    break;
                }

                match app.screen {
                    Screen::Config => handle_config_key(app, key.code),
                    Screen::Running => {
                        if matches!(key.code, KeyCode::Char('q')) {
                            break;
                        }
                    }
                    Screen::Results => handle_results_key(app, key.code),
                }
            }
        }

        if app.screen == Screen::Running {
            app.poll_progress();
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}

fn handle_config_key(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Tab => app.next_field(),
        KeyCode::BackTab => app.prev_field(),
        KeyCode::Down => {
            if app.is_numeric_field() {
                app.adjust_down();
            } else {
                app.next_field();
            }
        }
        KeyCode::Up => {
            if app.is_numeric_field() {
                app.adjust_up();
            } else {
                app.prev_field();
            }
        }
        KeyCode::Char(' ') => {
            if app.is_bool_field() {
                app.toggle_bool();
            } else {
                app.type_char(' ');
            }
        }
        KeyCode::Char('m') => {
            app.mode = match app.mode {
                Mode::Fuzzy => Mode::Exact,
                Mode::Exact => Mode::Fuzzy,
            };
            app.focused = 0;
        }
        KeyCode::Enter => app.start_run(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Char(c) => app.type_char(c),
        _ => {}
    }
}

fn handle_results_key(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Enter | KeyCode::Char('r') => app.reset_to_config(),
        _ => {}
    }
}
