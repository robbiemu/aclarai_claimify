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
aclarai-claimify init
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
aclarai-claimify compile \
    --component decomposition \
    --trainset ./my_decomposition_trainset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --output-path ./custom_prompts/my_compiled_decomposition.json

# With a custom optimizer config
aclarai-claimify compile \
    --component decomposition \
    --trainset ./my_decomposition_trainset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --config ./my_custom_optimizer.yaml \
    --output-path ./custom_prompts/my_compiled_decomposition.json
```

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
aclarai-claimify generate-dataset \
    --input-file ./my_raw_sentences.txt \
    --output-file ./my_decomposition_dataset.jsonl \
    --component decomposition \
    --teacher-model gpt-4o

# For components that use context (selection, disambiguation), you can
# control the context window size with the --k-window-size flag.
aclarai-claimify generate-dataset \
    --input-file ./my_raw_sentences.txt \
    --output-file ./my_selection_dataset.jsonl \
    --component selection \
    --teacher-model gpt-4o \
    --k-window-size 3 # Use 3 sentences before and 3 after
```

⚠️ **Warning**: This operation may incur costs depending on your API provider and usage!

### Step 3: Review and Refine

The generated dataset is a good starting point, but you should review it for quality and make any necessary corrections before using it for compilation.

### Step 4: Compile with Your Dataset

Now you can use your generated dataset with the standard compilation process:

```bash
aclarai-claimify compile \
    --component decomposition \
    --trainset ./my_decomposition_dataset.jsonl \
    --student-model gpt-3.5-turbo \
    --teacher-model gpt-4o \
    --output-path ./custom_prompts/my_compiled_decomposition.json
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.