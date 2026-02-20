# Contributing to fastdedup

Thanks for your interest in contributing. Bug reports, feature requests, and
pull requests are all welcome.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when
opening an issue. The more detail you provide upfront (dataset size, file
format, command you ran, error output), the faster it gets resolved.

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes. Run the test suite before submitting:
   ```bash
   cargo test --workspace
   ```
3. If you're adding a new feature, add at least one test covering it.
4. Open a PR with a clear description of what it does and why.

## Development setup

Requires Rust 1.70+.

```bash
git clone https://github.com/wapplewhite4/fastdedup.git
cd fastdedup

# Build (dev)
cargo build

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --package fastdedup-core

# Install locally for manual testing
cargo install --path crates/cli
```

## Project structure

```
crates/
  core/      # MinHash, LSH, exact dedup engine
  formats/   # JSONL and Parquet readers/writers
  filters/   # Language detection and quality scoring
  cli/       # Binary, TUI, argument parsing
```

## Code style

- Standard `rustfmt` formatting (`cargo fmt`)
- No warnings — keep `cargo build` clean
- Prefer clarity over cleverness in hot paths; add a comment explaining the
  trade-off when you do something non-obvious for performance

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
