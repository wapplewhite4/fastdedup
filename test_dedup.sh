#!/bin/bash
set -e

echo "🚀 fastdedup Testing Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test data file
TEST_FILE="/tmp/test_dataset.jsonl"

echo "Step 1: Generate test dataset"
echo "------------------------------"
cargo run --package fastdedup-cli --example generate_test_data "$TEST_FILE"
echo ""

echo "Step 2: Show original dataset"
echo "------------------------------"
echo "Records in original file:"
wc -l "$TEST_FILE"
echo ""
echo "Sample records:"
head -5 "$TEST_FILE"
echo "..."
echo ""

echo "Step 3: Run EXACT deduplication"
echo "--------------------------------"
cargo run --release --example dedupe_file "$TEST_FILE" exact
echo ""

echo "Step 4: Run FUZZY deduplication"
echo "--------------------------------"
cargo run --release --example dedupe_file "$TEST_FILE" fuzzy
echo ""

echo "Step 5: Run EXACT-NORMALIZED deduplication"
echo "-------------------------------------------"
cargo run --release --example dedupe_file "$TEST_FILE" exact-normalized
echo ""

echo "📊 Results Summary"
echo "=================="
echo ""
printf "${YELLOW}Original file:${NC}\n"
wc -l "$TEST_FILE"
echo ""

if [ -f "${TEST_FILE}.deduped.jsonl" ]; then
    printf "${GREEN}Exact dedup:${NC}\n"
    wc -l "${TEST_FILE}.deduped.jsonl"
    echo ""
fi

echo "✓ Test complete!"
echo ""
echo "Output files created:"
echo "  - ${TEST_FILE}.deduped.jsonl (from last run)"
echo ""
echo "To test with your own data:"
echo "  cargo run --release --example dedupe_file /path/to/your/file.jsonl exact"
echo "  cargo run --release --example dedupe_file /path/to/your/file.jsonl fuzzy"
echo "  cargo run --release --example dedupe_file /path/to/your/file.jsonl exact-normalized"
