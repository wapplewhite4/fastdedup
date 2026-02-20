#!/usr/bin/env python3
"""
Exact dedup baseline using DuckDB.
Streams the parquet file, hashes the text field with SHA-256,
keeps the first occurrence of each hash. Memory-efficient — does
not load the full dataset into RAM.
"""
import sys, time
import duckdb

input_path = sys.argv[1]
output_path = sys.argv[2]
field = sys.argv[3] if len(sys.argv) > 3 else "text"

print(f"Input  : {input_path}", flush=True)
print(f"Output : {output_path}", flush=True)
print(f"Field  : {field}", flush=True)

t0 = time.perf_counter()

# Count rows first for reporting
row_count = duckdb.sql(f"SELECT count(*) FROM read_parquet('{input_path}')").fetchone()[0]
print(f"Rows   : {row_count:,}", flush=True)

t_count = time.perf_counter()
print(f"Count  : {t_count - t0:.2f}s", flush=True)

# Deduplicate: keep first occurrence of each sha256(field) hash.
# DuckDB streams the file and uses spill-to-disk for large aggregations
# so memory usage stays bounded regardless of dataset size.
duckdb.sql(f"""
COPY (
    SELECT * EXCLUDE (_hash, _rn)
    FROM (
        SELECT *,
               sha256({field}::VARCHAR)        AS _hash,
               row_number() OVER (
                   PARTITION BY sha256({field}::VARCHAR)
                   ORDER BY (SELECT NULL)
               ) AS _rn
        FROM read_parquet('{input_path}')
    )
    WHERE _rn = 1
) TO '{output_path}' (FORMAT PARQUET)
""")

t_done = time.perf_counter()
total = t_done - t0
dedup_time = t_done - t_count

# Count output rows
out_count = duckdb.sql(f"SELECT count(*) FROM read_parquet('{output_path}')").fetchone()[0]
removed = row_count - out_count

print(f"Deduped: {row_count:,} -> {out_count:,} ({removed:,} removed) in {dedup_time:.2f}s", flush=True)
print(f"Total  : {total:.2f}s  |  {row_count / total:,.0f} records/sec", flush=True)
