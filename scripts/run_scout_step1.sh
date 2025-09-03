#!/bin/bash

# Script to complete STEP 1: Generate 1200 samples using the Data Scout Agent
# This script runs the Data Scout Agent with the mission plan to generate the required samples

set -e  # Exit on any error

echo "=== STEP 1: Generate and Integrate Default Production Artifacts ==="
echo "Running Data Scout Agent to generate 1200 samples"
echo "Using model: ollama/gpt-oss:20b"
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

# Create a comprehensive set of requests to generate the 1200 samples
# Based on the mission plan:
# - Production corpus: 150 samples per component characteristic
# - Research dataset: 400 samples per component characteristic
# Total: (150 + 400) * 3 characteristics = 1650 samples (well above the 1200 target)

cat > /tmp/scout_comprehensive_requests.txt << 'EOF'
# Production corpus requests (150 samples per characteristic)
Generate verifiable statements from news reports focusing on recent scientific discoveries
Create verifiable facts from scientific abstracts about medical research
Extract verifiable data from financial statements of technology companies
Generate self-contained examples from political analysis of recent elections
Create self-contained narratives from historical accounts of the industrial revolution
Produce self-contained stories from biographical sketches of famous scientists
Find atomic statements in legal documents about contract law
Extract atomic examples from policy summaries on environmental regulations
Identify atomic claims in technical specifications for software development

# Research dataset requests (400 samples per characteristic)  
Generate diverse verifiable content from literary criticism of classic novels
Create verifiable statements from product reviews of consumer electronics
Extract verifiable facts from academic lectures on artificial intelligence
Find verifiable information in technical documentation for web development
Generate verifiable content from social media posts about technology trends
Create verifiable statements from blog posts about health and wellness
Extract verifiable data from encyclopedia entries on scientific topics
Generate verifiable content from government reports on economic statistics
Create verifiable statements from medical journals about recent treatments
Extract verifiable facts from market research on consumer behavior
Generate verifiable content from press releases about new product launches
Create verifiable statements from interview transcripts with industry experts
Extract verifiable information from white papers on blockchain technology
Generate verifiable content from case studies on business success
Create verifiable statements from user manuals for software applications
Extract verifiable facts from FAQs about online services
Generate verifiable content from wiki articles about historical events
Create verifiable statements from literary criticism of modern poetry
Extract verifiable information from product reviews of automotive products
Generate verifiable content from academic lectures on climate science

# Additional requests for self-containment testing
Create self-contained examples from travel guides to European cities
Generate self-contained narratives from recipe instructions for Italian cuisine
Produce self-contained stories from tutorial guides for programming beginners
Create self-contained examples from how-to articles on home improvement
Generate self-contained narratives from product descriptions of smartphones
Produce self-contained stories from event summaries of music festivals
Create self-contained examples from book summaries of bestsellers
Generate self-contained narratives from movie plots of recent releases
Produce self-contained stories from scientific explanations of quantum physics
Create self-contained examples from legal case summaries of landmark decisions
Generate self-contained narratives from business proposals for startups
Produce self-contained stories from project reports on software development
Create self-contained examples from meeting minutes of board meetings
Generate self-contained narratives from email threads about project updates
Produce self-contained stories from forum discussions about technology
Create self-contained examples from chat logs of customer support

# Additional requests for atomicity testing
Find atomic statements in contract clauses about intellectual property
Extract atomic examples from scientific hypotheses about climate change
Identify atomic claims in mathematical proofs of basic theorems
Find atomic statements in technical specifications for mobile apps
Extract atomic examples from medical diagnoses of common conditions
Identify atomic claims in financial analyses of stock performance
Find atomic statements in engineering schematics for simple circuits
Extract atomic examples from research methodologies in social sciences
Identify atomic claims in business plans for small companies
Find atomic statements in strategic overviews of marketing campaigns
Extract atomic examples from regulatory compliance documents
Identify atomic claims in insurance claims for property damage
Find atomic statements in patent descriptions of mechanical devices
Extract atomic examples from academic theses on literature
Identify atomic claims in code documentation for open source projects
Find atomic statements in system architectures for web applications
Extract atomic examples from process flows for manufacturing
Identify atomic claims in risk assessments for cybersecurity

# Synthetic data generation requests (as allowed by synthetic_budget)
Create synthetic examples of perfectly verifiable statements for edge case testing
Generate synthetic self-contained narratives with complex pronoun resolution
Produce synthetic atomic claims with precise logical structure
Create synthetic data that demonstrates the necessity of contextual k-window analysis

exit
EOF

echo "Starting Data Scout Agent with comprehensive request set..."
echo "This will generate approximately 1200 samples as specified in the mission plan."
echo "The agent will save results to:"
echo "  - Tier 1 corpus: examples/data/datasets/tier1/"
echo "  - Tier 2 corpus: examples/data/datasets/tier2/"
echo "  - Audit trail: examples/PEDIGREE.md"
echo ""

# Run the agent with the comprehensive requests
echo "Running agent... (this may take several hours)"
cat /tmp/scout_comprehensive_requests.txt | aclarai-claimify-scout --mission settings/scout_mission.yaml

# Clean up
rm /tmp/scout_comprehensive_requests.txt

echo ""
echo "=== STEP 1 COMPLETED ==="
echo "Data Scout Agent has generated samples using ollama/gpt-oss:20b"
echo "Generated files:"
echo "  - examples/data/datasets/tier1/ (raw extracted content)"
echo "  - examples/data/datasets/tier2/ (curated content)"
echo "  - examples/PEDIGREE.md (audit trail)"
echo ""
echo "These files are now ready for use in STEP 2: Generate the 'Gold Standard' Datasets"