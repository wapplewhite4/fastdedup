#!/usr/bin/env python3
"""
Baseline: Streaming deduplication
Memory-efficient Python implementation
"""
import pyarrow.parquet as pq
import pyarrow as pa
import hashlib
import time
import sys

def streaming_dedup(input_file, output_file, text_field='text', batch_size=10000):
    """Streaming deduplication with constant memory"""
    print(f"Reading {input_file} in batches of {batch_size:,}...")

    seen_hashes = set()
    batches = []
    total_records = 0
    duplicates = 0

    start = time.time()

    # Read in batches
    parquet_file = pq.ParquetFile(input_file)

    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        batch_df = batch.to_pandas()
        total_records += len(batch_df)

        # Compute hashes
        hashes = batch_df[text_field].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()
        )

        # Filter duplicates
        mask = ~hashes.isin(seen_hashes)
        unique_batch = batch_df[mask]

        # Update seen hashes
        seen_hashes.update(hashes[mask])

        duplicates += len(batch_df) - len(unique_batch)

        if len(unique_batch) > 0:
            batches.append(pa.RecordBatch.from_pandas(unique_batch))

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {total_records:,} records, {duplicates:,} duplicates found...")

    total_time = time.time() - start

    # Write all batches
    print("Writing output...")
    schema = batches[0].schema
    writer = pq.ParquetWriter(output_file, schema)
    for batch in batches:
        writer.write_batch(batch)
    writer.close()

    return {
        'total_records': total_records,
        'duplicates': duplicates,
        'final_count': total_records - duplicates,
        'total_time': total_time
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python streaming_dedup.py <input.parquet> <output.parquet>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("=" * 60)
    print("Streaming Exact Deduplication Baseline")
    print("=" * 60)

    stats = streaming_dedup(input_file, output_file)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total records:     {stats['total_records']:,}")
    print(f"  Duplicates found:  {stats['duplicates']:,} ({stats['duplicates']/stats['total_records']*100:.1f}%)")
    print(f"  Final count:       {stats['final_count']:,}")
    print(f"  Total time:        {stats['total_time']:.2f}s")
    print(f"  Throughput:        {stats['total_records']/stats['total_time']:,.0f} records/sec")
    print("=" * 60)
