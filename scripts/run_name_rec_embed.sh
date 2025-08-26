#!/bin/bash

# Script to embed the name_rec_for_embedding table using embed.py
# This script uses the existing embed.py CLI to generate embeddings for name recognition data

# Set default values
INPUT_TABLE="name_rec_for_embedding"
OUTPUT_TABLE_PREFIX="name_rec_embeddings"
ENDPOINT="http://ec2-3-231-68-18.compute-1.amazonaws.com:8080/embed"
INSTRUCTION="INSTRUCT: Given a search query with a person's name, retrieve relevant passages that has the person mentioned. QUERY: "

# Change to the project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the embedding script
echo "Starting embedding process for name recognition data..."
echo "Input table: $INPUT_TABLE"
echo "Output table prefix: $OUTPUT_TABLE_PREFIX"
echo "Endpoint: $ENDPOINT"
echo "Instruction: $INSTRUCTION"
echo ""

python src/wc_simd/embed.py \
    --input-table "$INPUT_TABLE" \
    --output-table-prefix "$OUTPUT_TABLE_PREFIX" \
    --endpoint "$ENDPOINT" \
    --instruction "$INSTRUCTION" \
    --batch-size 32

echo "Embedding process completed!"
