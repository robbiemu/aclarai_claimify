#!/bin/bash

# Script to generate 1200 samples using the Data Scout Agent
# This will generate the samples as specified in STEP 1

set -e  # Exit on any error

echo "=== Generating 1200 Samples with Data Scout Agent ==="
echo "Using mission plan: settings/mission_config.yaml"
echo "Model: ollama/gpt-oss:20b"
echo ""

# Create output directories if they don't exist
mkdir -p examples/data/datasets/tier1
mkdir -p examples/data/datasets/tier2
mkdir -p examples/

# Check if Ollama is running and model is available
echo "Checking Ollama setup..."
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed"
    exit 1
fi

if ! ollama list | grep -q "gpt-oss:20b"; then
    echo "ERROR: gpt-oss:20b model is not available in Ollama"
    echo "Please run: ollama pull gpt-oss:20b"
    exit 1
fi

echo "✓ Ollama is installed and gpt-oss:20b model is available"
echo ""

# Backup existing output files if they exist
if [ -d "examples/data/datasets/tier1" ] && [ "$(ls -A examples/data/datasets/tier1)" ]; then
    echo "Backing up existing Tier 1 data..."
    mv examples/data/datasets/tier1 examples/data/datasets/tier1.backup.$(date +%s)
    mkdir -p examples/data/datasets/tier1
fi

if [ -d "examples/data/datasets/tier2" ] && [ "$(ls -A examples/data/datasets/tier2)" ]; then
    echo "Backing up existing Tier 2 data..."
    mv examples/data/datasets/tier2 examples/data/datasets/tier2.backup.$(date +%s)
    mkdir -p examples/data/datasets/tier2
fi

if [ -f "examples/PEDIGREE.md" ]; then
    echo "Backing up existing PEDIGREE.md..."
    mv examples/PEDIGREE.md examples/PEDIGREE.md.backup.$(date +%s)
fi

echo "Starting Data Scout Agent..."
echo "This will generate exactly 1200 samples (150 production + 250 research = 400 per characteristic * 3 characteristics)"
echo "The process may take several hours to complete."
echo ""

# Run the agent for a specific mission
echo "Running agent for the 'research_dataset' mission..."
echo "This will run non-interactively and generate samples until the mission target is met."

aclarai-claimify-scout --mission research_dataset

echo ""
echo "=== Data Generation Completed ==="
echo "Check the following locations for output:"
echo "  - Tier 1 corpus: examples/data/datasets/tier1/"
echo "  - Tier 2 corpus: examples/data/datasets/tier2/"
echo "  - Audit trail: examples/PEDIGREE.md"
echo ""

# Show what was generated
echo "Generated files:"
if [ -d "examples/data/datasets/tier1" ]; then
    echo "Tier 1 files:"
    ls -la examples/data/datasets/tier1/ | head -10
fi

if [ -d "examples/data/datasets/tier2" ]; then
    echo "Tier 2 files:"
    ls -la examples/data/datasets/tier2/ | head -10
fi

if [ -f "examples/PEDIGREE.md" ]; then
    echo "PEDIGREE.md size: $(wc -l < examples/PEDIGREE.md) lines"
fi

echo ""
echo "STEP 1 COMPLETED: 1200 samples generation process finished"