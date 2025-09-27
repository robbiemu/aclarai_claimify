#!/bin/bash

# This script reruns the prospector on a specific component for the files that failed in the previous run.
# It first calls the verify_prospects.py script to generate a list of failed files, then copies those files to a temporary directory, and finally runs the prospector on that directory.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <component>"
    exit 1
fi

COMPONENT=$1
RETRY_LIST="rerun_prospects.txt"
TEMP_DIR="temp_rerun"

# Create a list of files to rerun for the specified component
python3 scripts/verify_prospects.py examples/data/prospects --component $COMPONENT --compose $RETRY_LIST

# Create a temporary directory
mkdir -p $TEMP_DIR/$COMPONENT

# Copy the files to the temporary directory
while IFS= read -r file; do
    cp "$file" "$TEMP_DIR/$COMPONENT/"
done < "$RETRY_LIST"

# Rerun the prospector on the temporary directory
uv run python3 scripts/prospector.py $COMPONENT \
  -i $TEMP_DIR/$COMPONENT \
  -o examples/data/prospects/$COMPONENT \
  --model openai/gpt-5-mini --concurrency 10

# Clean up
rm -rf $TEMP_DIR
rm $RETRY_LIST
