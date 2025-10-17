#!/bin/bash

set -e

# Set working directories
ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/datasets"
GENERATED_DIR="${DATASET_DIR}/output"

# Create dataset and generated directories
mkdir -p "$GENERATED_DIR"

# Download the DRKG tar.gz archive
echo "📥 Downloading DRKG dataset..."
wget -q --show-progress https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz -O "$DATASET_DIR/drkg.tar.gz"
tput cuu1 && tput el

# Extract the contents
echo "📦 Extracting DRKG archive..."
tar -xzf "$DATASET_DIR/drkg.tar.gz" -C "$DATASET_DIR"

# Check if required files exist in ./datasets/auxiliary
if [ ! -d "$DATASET_DIR/auxiliary" ]; then
    echo "❌ Error: $DATASET_DIR/auxiliary directory not found!"
    exit 1
fi

echo "✅ Found auxiliary files in $DATASET_DIR/auxiliary"

# Check if DRKG file exists
if [ ! -f "$DATASET_DIR/drkg/drkg.tsv" ]; then
    echo "❌ Error: $DATASET_DIR/drkg/drkg.tsv not found!"
    exit 1
fi

echo "✅ Found DRKG dataset in $DATASET_DIR/drkg/drkg.tsv"

if [ ! -d "$GENERATED_DIR" ]; then
    echo "❌ Error: $GENERATED_DIR directory not found!"
    echo "creating $GENERATED_DIR directory..."
    mkdir $GENERATED_DIR
fi
# Run the cleaning pipeline
echo "🚀 Running DRKG to VitaGraph conversion..."
python drkg_to_vitagraph.py

echo "✅ Done!"