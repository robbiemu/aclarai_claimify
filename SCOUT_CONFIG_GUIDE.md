# Scout Agent Mission Configuration Guide

## Overview

The Data Scout Agent configuration has been separated from the main `aclarai_claimify` configuration to provide better separation of concerns. This allows you to:

1. **Use custom configurations** without modifying the main config
2. **Specify configuration files** via command-line flags
3. **Have proper path control** for where samples are generated

## Changes Made

### 1. New Mission Configuration File

Created `scout_config.yaml` in the project root containing all mission parameters and scout settings:

```yaml
# Mission Configuration for Data Scout Agent
search_provider: "brave/search"
recursion_per_sample: 30
initial_prompt: |
  Begin the data prospecting mission based on your configured mission plan.

nodes:
  research:
    max_iterations: 7

mission_plan:
  goal: "Find reliable, verifiable data sources..."
  nodes:
    - name: "supervisor"
      model: "openai/gpt-5-mini"
      temperature: 1
      max_tokens: 65536
    # ... other nodes

writer:
  tier1_path: "examples/data/datasets/tier1"
  tier2_path: "examples/data/datasets/tier2"
  audit_trail_path: "examples/PEDIGREE.md"

checkpointer_path: ".checkpointer.sqlite"
```

### 2. New Configuration Loader

Created `aclarai_claimify/scout/config.py` with a simple `load_scout_config()` function that:
- Defaults to `scout_config.yaml` in the current directory
- Accepts custom config file paths
- Returns sensible defaults if the config file is missing

### 3. Command-Line Flag Support

All scout commands now support `--config` (`-c`) flag:

```bash
# CLI usage
aclarai-claimify-scout --mission research_dataset --config my_scout_config.yaml

# TUI usage  
aclarai-claimify-datascout-tui --config my_scout_config.yaml

# Step 1 script usage
./scripts/run_scout_step1.sh --mission research_dataset --config my_scout_config.yaml
```

### 4. Fixed Path Discrepancy

**Previously**: Samples were going to `output/approved_books/` (hardcoded) while the config specified `examples/data/datasets/tier1/`

**Now**: The archive node properly reads the `writer.tier1_path` from the scout config and saves samples there.

## Usage Examples

### Basic Usage (Default Config)
```bash
# Uses scout_config.yaml in current directory
aclarai-claimify-scout --mission research_dataset
```

### Custom Configuration
```bash
# Uses a custom config file
aclarai-claimify-scout --mission research_dataset --config /path/to/my_config.yaml
```

### Step 1 Script with Custom Config
```bash
./scripts/run_scout_step1.sh --mission research_dataset --config my_custom_mission.yaml
```

### TUI with Custom Config
```bash
aclarai-claimify-datascout-tui --config my_config.yaml --log scout.log
```

## Configuration Structure

The scout config file supports these main sections:

- **`search_provider`**: Search service to use (e.g., "brave/search", "duckduckgo/search")
- **`recursion_per_sample`**: Maximum iterations per sample generation cycle
- **`initial_prompt`**: Starting prompt for the agent
- **`observability`**: LangSmith configuration for tracing and monitoring
  - **`api_key`**: LangSmith API key (recommended to use environment variables instead)
  - **`tracing`**: Enable/disable tracing (boolean)
  - **`endpoint`**: Custom LangSmith endpoint for self-hosted deployments
  - **`project`**: Project name for organizing traces
- **`nodes.research.max_iterations`**: ReAct loop iterations for research
- **`mission_plan`**: LLM model configurations for each agent node
- **`writer`**: Output path configurations
  - `tier1_path`: Where raw samples are saved
  - `tier2_path`: Where curated samples go
  - `audit_trail_path`: Where PEDIGREE.md is written
- **`checkpointer_path`**: SQLite database for agent state persistence

## Migration from Old Config

If you have existing `scout_agent` configuration in `aclarai_claimify/settings/config.yaml`, you can:

1. **Copy the configuration** to a new `scout_config.yaml` file
2. **Flatten the structure** (remove the `scout_agent:` wrapper)
3. **Use the --config flag** to specify your new file

Example migration:
```yaml
# Old: aclarai_claimify/settings/config.yaml
scout_agent:
  writer:
    tier1_path: "examples/data/datasets/tier1"
  # ... other settings

# New: scout_config.yaml  
writer:
  tier1_path: "examples/data/datasets/tier1"
# ... other settings
```

## Benefits

1. **Clean Separation**: Scout configuration is independent from main claimify config
2. **Flexible Deployment**: Use different configs for different environments
3. **Correct Paths**: Samples now go to the configured paths, not hardcoded ones
4. **Better Defaults**: Sensible fallbacks if config file is missing
5. **CLI Integration**: All tools support the --config flag consistently

## LangSmith Observability Configuration

The scout agent supports LangSmith tracing for monitoring and debugging your missions. You can configure LangSmith in two ways:

### Environment Variables (Recommended)

Set these environment variables for LangSmith configuration:

```bash
# Authentication (required for tracing)
export LANGCHAIN_API_KEY="your-langsmith-api-key"
# or
export LANGSMITH_API_KEY="your-langsmith-api-key"

# Enable tracing (optional, enabled by default when API key is present)
export LANGCHAIN_TRACING_V2="true"
# or
export LANGSMITH_TRACING="true"

# Custom endpoint for self-hosted deployments (optional)
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# or
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Project name for organizing traces (optional)
export LANGCHAIN_PROJECT="aclarai-claimify-scout"
# or
export LANGSMITH_PROJECT="aclarai-claimify-scout"
```

### Configuration File

Alternatively, you can configure LangSmith in your `scout_config.yaml`:

```yaml
observability:
  # api_key: "your-langsmith-api-key"  # Not recommended in config files
  tracing: true
  endpoint: "https://api.smith.langchain.com"
  project: "aclarai-claimify-scout"
```

Note: It's recommended to use environment variables for sensitive information like API keys rather than including them in configuration files.

When LangSmith is properly configured, you'll see tracing information in your LangSmith dashboard, which can help you monitor and debug your scout missions.

## Testing the Changes

To verify everything works:

To verify everything works:

1. **Test with default config**:
   ```bash
   aclarai-claimify-scout --mission production_corpus
   # Should use scout_config.yaml and save to examples/data/datasets/tier1/
   ```

2. **Test with custom config**:
   ```bash
   cp scout_config.yaml my_config.yaml
   # Edit my_config.yaml to change tier1_path
   aclarai-claimify-scout --mission production_corpus --config my_config.yaml
   # Should save to your custom path
   ```

3. **Verify paths**: Check that samples appear in the configured `tier1_path`, not in `output/approved_books/`
