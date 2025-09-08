#!/bin/bash

# Script to complete STEP 1: Generate 1200 samples using the Data Scout Agent
# This script runs the Data Scout Agent with the mission plan to generate the required samples

set -e  # Exit on any error

# Default values
RESUME_FROM=""
MISSION_NAME="research_dataset" # Default mission to run

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --resume-from)
        RESUME_FROM="$2"
        shift # past argument
        shift # past value
        ;;
        --mission)
        MISSION_NAME="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

echo "=== STEP 1: Generate and Integrate Default Production Artifacts ==="

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

echo "Starting Data Scout Agent..."
echo "This will generate samples as specified in the mission plan."
echo "The agent will save results to:"
echo "  - output/approved_books/"
echo "  - examples/PEDIGREE.md (audit trail)"

# Run the agent
COMMAND="aclarai-claimify-scout"
if [ -n "$RESUME_FROM" ]; then
    echo "🔄 Resuming mission from ID: $RESUME_FROM"
    COMMAND="$COMMAND --resume-from $RESUME_FROM"
else
    echo "🚀 Starting new mission: $MISSION_NAME"
    COMMAND="$COMMAND --mission $MISSION_NAME"
fi

echo "Running command: $COMMAND"
echo "This may take several hours to complete..."

eval $COMMAND

echo ""
echo "=== STEP 1 COMPLETED ==="
echo "Data Scout Agent has finished the run."


echo "Generated files:"
echo "  - examples/data/datasets/tier1/ (raw extracted content)"
echo "  - Tier 2 corpus: examples/data/datasets/tier2/ (curated content)"
echo "  - examples/PEDIGREE.md (audit trail)"
echo ""
echo "These files are now ready for use in STEP 2: Generate the 'Gold Standard' Datasets"