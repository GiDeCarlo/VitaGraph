#!/bin/bash

set -e

# Create dataset directory if it doesn't exist
mkdir -p dataset
cd dataset || exit 1
mkdir -p generated

# Download the DRKG tar.gz archive
echo "📥 Downloading DRKG dataset..."
wget -q --show-progress https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
tput cuu1 && tput el

# Extract the contents
echo "📦 Extracting DRKG archive..."
tar -xzf drkg.tar.gz

# Unzip auxiliary.zip if it exists
if [ -f "auxiliary.zip" ]; then
    echo "🗂 Unzipping auxiliary.zip..."
    unzip -q auxiliary.zip
    [ -d "__MACOSX/" ] && rm -rf "__MACOSX/"

    # Clean up unwanted files
    echo "🧼 Removing unnecessary files..."
    rm -f drkg.tar.gz
    rm -rf embed relation_glossary.tsv entity2src.tsv

    # Return to root directory
    cd ..

    # Run the cleaning pipeline
    echo "🚀 Running DRKG to VitaGraph conversion..."
    python drkg_to_vitagraph.py

    echo "✅ Done!"
else
    echo "⚠️ auxiliary.zip not found. Aborting VitaGraph generation. Make sure to have the correct files."
    echo "🧼 Removing unnecessary files..."
    rm -f drkg.tar.gz
    rm -rf embed relation_glossary.tsv entity2src.tsv
    echo "❌ Exiting..."
    exit 1
fi
