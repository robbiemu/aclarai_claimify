# Aclarai-Claimify

[![PyPI Version](https://img.shields.io/pypi/v/aclarai-claimify.svg)](https://pypi.org/project/aclarai-claimify/)
[![Build Status](https://img.shields.io/travis/com/your-username/aclarai-claimify.svg)](https://travis-ci.com/your-username/aclarai-claimify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aclarai-Claimify is a powerful and flexible Python library for extracting atomic, verifiable claims from unstructured text. It provides a toolkit of composable components that perform a three-stage process: selecting relevant sentences, disambiguating them for clarity, and decomposing them into high-quality, standalone claims.

Built for reliability and adaptability, Claimify is framework-agnostic and features a data-driven optimization engine powered by **DSPy**.

## Key Features

*   **Stateless, Composable Components:** Instead of a rigid pipeline, Claimify provides independent components for selection, disambiguation, and decomposition that you can orchestrate in any way you choose.
*   **Pydantic-Powered Data Models:** All inputs, outputs, and state objects are defined with Pydantic, ensuring robust data validation, clear schemas, and excellent developer experience.
*   **DSPy-Powered Optimization:** Don't just hand-tune prompts. Use our built-in tools to "compile" your prompts against your own data, programmatically improving performance and reducing costs by enabling smaller, faster models to perform like larger ones.
*   **Framework Agnostic:** As a toolkit of stateless functions, Claimify can be easily integrated into any agentic framework, including **LlamaIndex**, **LangGraph**, or your own custom orchestration logic.

## Installation

The project uses `uv` for package management.

```bash
# Install the latest stable release
uv pip install aclarai-claimify

# Or install from source for development
git clone https://github.com/your-username/aclarai-claimify.git
cd aclarai-claimify
uv pip install -e .
```

## Quickstart

This example shows the basic flow of using the components in sequence to process a single sentence.

```python
import dspy
from aclarai_claimify.components import SelectionComponent, DisambiguationComponent, DecompositionComponent
from aclarai_claimify.data_models import SentenceChunk, ClaimifyContext, ClaimifyState

# 1. Configure an LLM for DSPy
# Replace with your model and API key
llm = dspy.OpenAI(model='gpt-4o', max_tokens=4000)
dspy.settings.configure(lm=llm)

# 2. Initialize the components
# By default, they will load the pre-compiled, optimized prompts
selection_comp = SelectionComponent()
disambiguation_comp = DisambiguationComponent()
decomposition_comp = DecompositionComponent()

# 3. Prepare the input data
# The sentence we want to process and its surrounding context
sentence = "It failed due to a memory error, which was a surprise to the team."
context_window = [
    "The system had been stable for weeks.",
    sentence,
    "A full diagnostic was initiated immediately."
]

state = ClaimifyState(
    original_chunk=SentenceChunk(text=sentence),
    context=ClaimifyContext(text_window=context_window)
)

# 4. Run the components in sequence
# In a real application, this would be managed by an orchestrator like LangGraph
state = selection_comp(state)

if state.was_selected:
    state = disambiguation_comp(state)
    state = decomposition_comp(state)

# 5. Inspect the results
if state.final_claims:
    print("Extracted Claims:")
    for claim in state.final_claims:
        print(f"- {claim.text}")
else:
    print("No claims were extracted from the sentence.")

# Expected Output:
# Extracted Claims:
# - The system failed due to a memory error.
# - The failure was a surprise to the team.
```

## Examples

### Agent-Based Pipeline Example
The `agent_pipeline_runtime.py` script shows how to run the original, stateful agent-based pipeline. This is useful for understanding the legacy architecture.

Run the script (ensure you have a compatible LLM server running):
```bash
python examples/agent_pipeline_runtime.py
```

### Running the LangGraph Example

This example shows how to run the Claimify pipeline as a graph using LangGraph. It can process text from a file or standard input.

First, install the required optional dependencies:

```bash
uv pip install .[langgraph_example]
```

To run the script on a text file, use the `--input-file` argument. You can also specify the model to use:

```bash
# Create a sample input file
echo "The system returned error code 500. It was unexpected." > input.txt

# Run the example using a local Ollama model
python examples/langgraph_runtime.py --input-file input.txt --model ollama/gemma:2b
```

You can also pipe input directly into the script:

```bash
cat input.txt | python examples/langgraph_runtime.py --model ollama/gemma:2b
```

To use models that require API keys (like OpenAI), you can pass them via the `--model-params` argument:

```bash
python examples/langgraph_runtime.py --input-file input.txt \
  --model gpt-3.5-turbo \
  --model-params '{"api_key": "YOUR_OPENAI_API_KEY"}'
```

## Core Concepts

### Stateless Components

The library's logic is broken into three main, reusable components:

*   **`SelectionComponent`**: Determines if a sentence likely contains a verifiable, factual claim.
*   **`DisambiguationComponent`**: Rewrites a sentence to be self-contained, resolving pronouns and ambiguities using the provided context.
*   **`DecompositionComponent`**: Takes a clean, disambiguated sentence and breaks it down into one or more atomic claims.

### The `ClaimifyState` Object

This Pydantic model is the central data structure that flows through the components. It holds the original input, the intermediate results from each stage, and the final extracted claims. You are responsible for passing this state object from one component to the next.

## Configuration

Claimify uses a cascading configuration system that allows you to easily override default settings.

1.  **Default Settings**: The library ships with a default `settings` directory containing `config.yaml` (for the main library) and `optimization.yaml` (for the DSPy optimizer).
2.  **Local Overrides**: You can override these defaults by creating your own `settings` directory in your project's root.

### The `init` Command

To get started with custom configurations, run the `init` command:

```bash
optimizer init
```

This will create a `settings/` folder in your current directory with copies of the default configuration files. You can then edit these files to change any settings you need. For example, to change the default model, you would edit `settings/config.yaml` and change the `default_model` value.

When you run `aclarai-claimify`, it will automatically detect your local `settings/` directory and use it to override the default settings.

## Performance Optimization with DSPy (User Guide)

The real power of Claimify comes from its ability to adapt to your specific data and models. You can compile your own optimized prompts to improve accuracy and reduce costs.

### Step 1: Create a "Gold Standard" Dataset

Create a JSONL file with high-quality examples of inputs and desired outputs for a component. For the decomposition component, it might look like this:

**`my_decomposition_trainset.jsonl`**
```json
{"disambiguated_text": "The system failed due to a memory error, and this failure was a surprise to the team.", "claim_candidates_json": "[{\"text\": \"The system failed due to a memory error.\"}, {\"text\": \"The failure was a surprise to the team.\"}]"}
{"disambiguated_text": "Project Apollo's primary mission was to land humans on the Moon.", "claim_candidates_json": "[{\"text\": \"Project Apollo's primary mission was to land humans on the Moon.\"}]"}
```

### Step 2: Run the Compilation Tool

Use the built-in CLI to generate a new, optimized prompt artifact. This command will run the DSPy optimization process using your data and models.

The `compile` command automatically uses the configuration from `settings/config.yaml`. If you want to use a custom optimizer configuration, you can use the `--config` flag.

```bash
# Basic compilation
optimizer compile \
    --component decomposition \
    --trainset ./my_decomposition_trainset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --output-path ./custom_prompts/my_compiled_decomposition.json

# With a custom optimizer config
optimizer compile \
    --component decomposition \
    --trainset ./my_decomposition_trainset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --config ./my_custom_optimizer.yaml \
    --output-path ./custom_prompts/my_compiled_decomposition.json
```

#### GEPA concurrency

GEPA supports batched evaluation through its `num_threads` parameter. Setting `num_threads` greater than one in your optimizer YAML will run student model evaluations in parallel, so ensure your rate limits and API quotas can absorb the additional simultaneous calls.

#### GEPA logging

Add `log_dir` to your GEPA optimizer params to persist the optimizer’s artifacts. Each run creates a unique subdirectory under that base (e.g., `logs/<uuid>_log/`), and the CLI prints the exact path when verbose output is enabled.

### Step 3: Use Your Compiled Prompt

Now, simply point the component to your newly created artifact during initialization. It will load your custom-tuned prompt instead of the library's default.

```python
from aclarai_claimify.components import DecompositionComponent

# Initialize the component with your custom, optimized prompt
custom_decomposition_comp = DecompositionComponent(
    compiled_prompt_path="./custom_prompts/my_compiled_decomposition.json"
)

# Use it just like before
# state = custom_decomposition_comp(state)
```

## Generate Gold Standard Datasets

To make it easier to create training datasets for optimization, Claimify provides a `generate-dataset` CLI command that uses a powerful teacher model to automatically generate structured training examples from raw text.

### Step 1: Prepare Raw Text Input

Create a simple text file with one sentence per line:

**`my_raw_sentences.txt`**
```
The system failed due to a memory error, and this failure was a surprise to the team.
Project Apollo's primary mission was to land humans on the Moon.
The database migration began at midnight and completed in 4.5 hours.
```

### Step 2: Generate the Dataset

Use the built-in CLI to generate a structured dataset for a specific component:

```bash
optimizer generate-dataset \
    --input-path ./my_raw_sentences.txt \
    --output-file ./my_decomposition_dataset.jsonl \
    --component decomposition \
    --teacher-model gpt-4o

# For components that use context (selection, disambiguation), you can
# control the context window size with the --k-window-size flag.
optimizer generate-dataset \
    --input-path ./my_raw_sentences.txt \
    --output-file ./my_selection_dataset.jsonl \
    --component selection \
    --teacher-model gpt-4o \
    --k-window-size 3 # Use 3 sentences before and 3 after

# When working from curated JSON prospects, enable parallel processing.
optimizer generate-dataset \
    --input-path ./examples/data/prospects/selection \
    --output-file ./my_selection_dataset.jsonl \
    --component selection \
    --teacher-model gpt-4o \
    --curated \
    --concurrency 8

# Curated disambiguation/decomposition runs can also request labelled negatives.
optimizer generate-dataset \
    --input-path ./examples/data/prospects/disambiguation \
    --output-file ./my_disambiguation_dataset.jsonl \
    --component disambiguation \
    --teacher-model gpt-4o \
    --curated \
    --include-negatives \
    --negative-quota 1
```

⚠️ **Warning**: This operation may incur costs depending on your API provider and usage!

#### Helpful Flags

- `--curated`: tells the generator that the input directory contains curated JSON files
  (each file may hold positive/negative examples). Files without usable entries are
  skipped automatically.
- `--clean-markdown`: strips markdown formatting when reading `.md` sources.
- `--concurrency`: caps simultaneous teacher-model calls; increase cautiously to stay
  within provider rate limits.
- `--include-negatives`: (disambiguation & decomposition) synthesize negative samples labelled
  by failure mode using curated prospects.
- `--negative-quota`: enforce a minimum count per failure mode when generating negatives; the
  command aborts if coverage is incomplete.

### Step 3: Review and Refine

The generated dataset is a good starting point, but you should review it for quality and make any necessary corrections before using it for compilation.

### Step 4: Compile with Your Dataset

Now you can use your generated dataset with the standard compilation process:

```bash
optimizer compile \
    --component decomposition \
    --trainset ./my_decomposition_dataset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --output-path ./custom_prompts/my_compiled_decomposition.json
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

During testing, you may notice some deprecation warnings from external dependencies. These are documented in [EXTERNAL_DEPENDENCY_WARNINGS.md](docs/EXTERNAL_DEPENDENCY_WARNINGS.md) and are outside the scope of this project to fix.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
