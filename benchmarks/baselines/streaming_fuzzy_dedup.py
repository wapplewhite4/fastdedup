#!/usr/bin/env python3
"""
Baseline: Streaming fuzzy deduplication using MinHash + LSH
Memory-efficient Python implementation
"""
import pyarrow.parquet as pq
import pyarrow as pa
import time
import sys
import re
from datasketch import MinHash, MinHashLSH

def normalize_text(text):
    """Normalize text for comparison"""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_minhash(text, num_perm=128):
    """Create MinHash signature for text"""
    tokens = text.split()
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode('utf8'))
    return m

def streaming_fuzzy_dedup(input_file, output_file, text_field='text', threshold=0.8, num_perm=128, batch_size=10000):
    """Streaming fuzzy deduplication with MinHash + LSH"""
    print(f"Reading {input_file} in batches of {batch_size:,}...")
    print(f"Parameters: threshold={threshold}, num_perm={num_perm}")

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    batches = []
    total_records = 0
    duplicates = 0
    record_id = 0

    start = time.time()

    # Read in batches
    parquet_file = pq.ParquetFile(input_file)

    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        batch_df = batch.to_pandas()
        total_records += len(batch_df)

        # Process each record in the batch
        keep_mask = []

        for idx, row in batch_df.iterrows():
            text = normalize_text(row[text_field])
            minhash = create_minhash(text, num_perm)

            # Query for similar items
            result = lsh.query(minhash)

            if len(result) == 0:
                # No duplicates found, keep this record
                lsh.insert(str(record_id), minhash)
                keep_mask.append(True)
            else:
                # Duplicate found, skip this record
                keep_mask.append(False)
                duplicates += 1

            record_id += 1

        # Filter batch to unique records
        unique_batch = batch_df[keep_mask].reset_index(drop=True)

        if len(unique_batch) > 0:
            batches.append(pa.RecordBatch.from_pandas(unique_batch, preserve_index=False))

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {total_records:,} records, {duplicates:,} duplicates found...")

    total_time = time.time() - start

    # Write all batches
    print("Writing output...")
    if len(batches) > 0:
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
    if len(sys.argv) < 3:
        print("Usage: python streaming_fuzzy_dedup.py <input.parquet> <output.parquet> [threshold] [num_perm]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    num_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    print("=" * 60)
    print("Streaming Fuzzy Deduplication Baseline (MinHash + LSH)")
    print("=" * 60)

    stats = streaming_fuzzy_dedup(input_file, output_file, threshold=threshold, num_perm=num_perm)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total records:     {stats['total_records']:,}")
    print(f"  Duplicates found:  {stats['duplicates']:,} ({stats['duplicates']/stats['total_records']*100:.1f}%)")
    print(f"  Final count:       {stats['final_count']:,}")
    print(f"  Total time:        {stats['total_time']:.2f}s")
    print(f"  Throughput:        {stats['total_records']/stats['total_time']:,.0f} records/sec")
    print("=" * 60)
