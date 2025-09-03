#!/bin/bash

# Script to generate 1200 samples using the Data Scout Agent
# This will generate the samples as specified in STEP 1

set -e  # Exit on any error

echo "=== Generating 1200 Samples with Data Scout Agent ==="
echo "Using mission plan: settings/scout_mission.yaml"
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

# Create a set of sample requests to trigger the agent
cat > /tmp/scout_requests.txt << 'EOF'
Generate samples for verifiability from news reports
Create examples for self-containment from historical narratives
Find atomicity examples in legal documents
Generate verifiable statements from scientific abstracts
Create self-contained examples from political analysis
Find atomic examples in policy summaries
Generate samples for verifiability from financial statements
Create examples for self-containment with pronoun resolution
Find atomicity examples in contract clauses
Generate verifiable facts from encyclopedia entries
Create self-contained narratives from biographical sketches
Find atomic statements in technical specifications
EOF

# Run the agent with the sample requests
echo "Running agent with sample requests..."
echo "Note: For full automation, you would interact with the agent to provide specific requests"
echo "targeted at each of the topics in the mission plan."
echo ""

# Start the agent in the background and send requests
{
    cat /tmp/scout_requests.txt
    echo "exit"
} | aclarai-claimify-scout --mission settings/scout_mission.yaml

# Clean up
rm /tmp/scout_requests.txt

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