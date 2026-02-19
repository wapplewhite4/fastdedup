# Tiered Signature Storage — Implementation Plan

## Problem

At 15M records the fuzzy dedup LSH index consumes ~60 GB RAM because:

- **Signatures**: `Vec<Option<MinHashSignature>>` — 15M × 1,048 bytes ≈ **15 GB**
- **Band tables**: 16 × `AHashMap<u64, Vec<usize>>` — ID arrays + HashMap overhead ≈ **8–12 GB**
- **`field_values` HashMap** (CLI layer): stores text of every kept record ≈ **variable, multi-GB**
- Allocator overhead, Vec resize headroom, Parquet buffers ≈ rest

## Strategy

Tier **signatures only** to disk.  Band tables stay in memory (~10 GB at 15M,
acceptable).  `field_values` in the CLI layer is a separate concern addressed
with a targeted fix.

Expected peak RAM after: **~12–15 GB** at 15M records (down from 60 GB).

---

## Step 1 — New file: `crates/core/src/signature_store.rs`

Create a `TieredSignatureStore` that owns the signature data behind the same
logical interface the LSH index currently uses.

```rust
pub struct TieredSignatureStore {
    /// Hot cache: most recent signatures, keyed by document ID.
    hot: HashMap<usize, MinHashSignature>,   // bounded to max_hot entries
    /// Cold storage: older signatures serialized to sled.
    cold: sled::Db,
    /// Maximum hot-cache entries before eviction.
    max_hot: usize,
    /// Insertion-order tracking for FIFO eviction.
    insertion_order: VecDeque<usize>,
    /// Total number of signatures stored (hot + cold).
    count: usize,
}
```

Public API:

| Method | Replaces |
|--------|----------|
| `insert(id, sig)` | `Vec` index-assign + resize |
| `get(id) -> Result<Option<MinHashSignature>>` | `Vec::get().and_then()` — returns **owned** |
| `contains(id) -> bool` | `signatures[id].is_some()` |
| `remove(id)` | setting slot to `None` |
| `len() -> usize` | `sig_count` |
| `clear()` | clearing Vec |

Key decisions:

- **Serialization**: each signature is 128 × u64 = 1,024 bytes.  Store as raw
  little-endian bytes in sled (key = id as 8-byte BE, value = 1,024 bytes).
  No serde overhead.
- **Eviction**: FIFO via `VecDeque<usize>`.  When `hot.len() >= max_hot`,
  evict the oldest 10% to cold storage in a single `sled::Batch`.
- **Default `max_hot`**: 2,000,000 entries ≈ 2 GB RAM.  Configurable.
- **`get()` returns owned** `MinHashSignature`, not a reference.  This is the
  key API change that ripples outward.
- **sled path**: derived from a configurable base directory, defaulting to a
  temp directory that is cleaned up on `Drop`.

## Step 2 — Update `LSHIndex` in `minhash.rs`

Replace the current signature storage with `TieredSignatureStore`:

```diff
 pub struct LSHIndex {
     num_bands: usize,
     rows_per_band: usize,
     bands: Vec<AHashMap<u64, Vec<usize>>>,
-    signatures: Vec<Option<MinHashSignature>>,
-    sig_count: usize,
+    signatures: TieredSignatureStore,
     band_hash_builder: RandomState,
     insertions_since_compact: usize,
     compact_interval: usize,
 }
```

Changes to methods:

| Method | Change |
|--------|--------|
| `new()` | Instantiate `TieredSignatureStore::new(max_hot, temp_path)` |
| `with_capacity()` | Forward capacity hint to store |
| `insert()` | `self.signatures.insert(id, sig)` (now returns `Result`) |
| `get_signature()` | Signature change: `-> Result<Option<MinHashSignature>>` (owned) |
| `query()` | Use `self.signatures.contains(id)` for stale-ID filtering (no full fetch needed) |
| `remove_signature()` | `self.signatures.remove(id)` |
| `compact()` | Use `self.signatures.contains(id)` to test liveness |
| `len()` | `self.signatures.len()` |
| `clear()` | `self.signatures.clear()` |

The `query()` method does NOT need to fetch signatures — it only checks whether
an ID is still live.  The `contains()` check on the hot HashMap is O(1); for
cold entries it's a sled `contains_key` which is fast (key-only, no value read).

## Step 3 — Update `FuzzyDeduplicator` in `fuzzy_dedup.rs`

Three internal methods call `lsh_index.get_signature()` in a verification loop:

1. `process_record()` (line 196)
2. `find_duplicates()` (line 259)
3. `process_prepared()` (line 340)

All three follow the same pattern:

```rust
// BEFORE (borrowed reference, infallible)
if let Some(candidate_sig) = self.lsh_index.get_signature(candidate_id) {
    let similarity = signature.jaccard_similarity(candidate_sig);

// AFTER (owned value, fallible — skip on I/O error)
match self.lsh_index.get_signature(candidate_id) {
    Ok(Some(candidate_sig)) => {
        let similarity = signature.jaccard_similarity(&candidate_sig);
    }
    Ok(None) => { /* stale ID, skip */ }
    Err(_) => { /* cold-storage read error, skip candidate */ }
}
```

**Public API of `FuzzyDeduplicator` does NOT change.**  The `Result` from sled
reads is handled internally — a cold-storage I/O error on a single candidate
just skips that candidate (logs a warning).  This keeps the CLI call sites
untouched.

## Step 4 — Wire up configuration

Add `max_hot_signatures` and `signature_store_path` to `FuzzyDedupConfig`:

```rust
pub struct FuzzyDedupConfig {
    // ... existing fields ...
    /// Maximum signatures to keep in memory (default: 2_000_000 ≈ 2 GB).
    pub max_hot_signatures: usize,
    /// Directory for cold signature storage (default: temp dir).
    pub signature_store_path: Option<String>,
}
```

No new CLI flags needed for v1 — the defaults (2M hot, temp dir) are sane.
A `--max-memory` flag can be added later.

## Step 5 — Address `field_values` HashMap in CLI layer

`main.rs:429` and `tui/runner.rs:99` both maintain:

```rust
let mut field_values: HashMap<usize, String> = HashMap::new();
```

This stores the full text of every **kept** record for the `matched_value`
field in the removed log.  At 15M records with an average text length of 1 KB,
this is ~15 GB on its own.

Fix: write kept field values to a temporary sled database instead of a HashMap.
Same pattern as the signature store — recent values in a bounded HashMap, older
values on disk.  Only needed when `write_output` is true.

This is a CLI-layer change only, no core API impact.

## Step 6 — Update `lib.rs` exports, tests

- Add `pub mod signature_store;` to `crates/core/src/lib.rs`.
- Update all existing `minhash.rs` tests to handle `Result` from `get_signature()`.
- Add unit tests for `TieredSignatureStore`: insert/get, eviction, cold
  round-trip, clear, remove, large-scale.
- Existing `fuzzy_dedup.rs` tests should pass unchanged (public API is stable).

## Files changed

| File | Nature of change |
|------|-----------------|
| `crates/core/src/signature_store.rs` | **New** — TieredSignatureStore |
| `crates/core/src/lib.rs` | Add `pub mod signature_store` |
| `crates/core/src/minhash.rs` | Replace `Vec<Option<>>` with `TieredSignatureStore`; `get_signature` returns `Result<Option<MinHashSignature>>` |
| `crates/core/src/fuzzy_dedup.rs` | Handle `Result` from `get_signature` in 3 verification loops |
| `crates/cli/src/main.rs` | Replace `field_values` HashMap with disk-backed store |
| `crates/cli/src/tui/runner.rs` | Same `field_values` fix |
| `crates/core/Cargo.toml` | sled is already a dependency (used by hash_storage) — no change |

## What does NOT change

- `FuzzyDeduplicator` public API signatures (`process_record`, `find_duplicates`,
  `process_prepared`, `prepare_signature`) — no `Result` added
- CLI flag surface — no new required flags
- Band hash tables — remain fully in-memory
- MinHash algorithm, LSH math, shingle logic — untouched
- Exact dedup — untouched
