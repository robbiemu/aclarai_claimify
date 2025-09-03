# Guide: Curating Datasets with the Data Scout Agent

## 1. Introduction

Manually collecting, cleaning, and auditing a high-quality dataset for training and evaluating LLM systems is a significant bottleneck. The **Data Scout Agent** is an automated tool designed to solve this problem. It transforms the open-ended task of "finding data" into a systematic, goal-directed, and reproducible process.

This guide will walk you through how to use the agent to curate a custom corpus tailored to your specific needs.

**What the Agent Does:**

*   **Deconstructs Goals:** Translates a high-level mission into a concrete plan.
*   **Finds and Fetches Data:** Systematically searches the web for relevant text.
*   **Cleans and Validates:** Extracts core content and runs "fitness checks" to ensure the data is useful.
*   **Creates an Audit Trail:** Automatically generates a `PEDIGREE.md` file to track the provenance of every piece of data.

## 2. Prerequisites

Before you begin, ensure you have installed the project with the optional `[agent]` dependencies. This provides the agent with all the necessary libraries, including LangGraph.

```bash
# From the root of the project repository
pip install -e .[agent]
```

## 3. The Core Concept: The Mission Plan

The entire behavior of the Data Scout Agent is controlled by a single YAML configuration file: the **Mission Plan**. Think of this file as the "brain" of the agent. You don't write code to change the agent's goals; you simply describe your desired outcome in the mission plan, and the agent adapts its strategy to achieve it.

The default mission plan is located at `settings/scout_mission.yaml`.

## 4. Quick Start: A Simple Example

Let's run the agent with a simple mission to see it in action.

#### **Step 1: Define a Simple Mission**

Create a new file named `settings/simple_mission.yaml` and add the following content. This mission tells the agent to find just 10 text samples related to "Quantum Computing."

```yaml
# in settings/simple_mission.yaml
missions:
  - name: "quantum_computing_corpus"
    target_size: 10 # Find 10 total samples
    synthetic_budget: 0.1 # Allow 10% (1 sample) to be invented if needed
    goals:
      - characteristic: "Verifiability"
        topics: ["Quantum Computing Research Abstracts"]
```

#### **Step 2: Execute the Agent**

Run the agent from your terminal, pointing it to your new mission plan.

```bash
aclarai-claimify-scout --mission-plan settings/simple_mission.yaml
```

The agent will now begin its work. You will see logs in your terminal as it brainstorms a strategy, searches for sources, extracts text, and validates the content.

#### **Step 3: Review the Output**

Once the agent completes its mission, you will find the following new files and directories:

*   **`examples/data/datasets/tier1/`**: Contains the raw, cleaned text files downloaded from the web. Each file is named with the date and topic, e.g., `2025-09-05_Quantum_Computing_Research_Abstracts.txt`.
*   **`examples/data/datasets/tier2/`**: Contains the final, curated corpus files, created by sampling from the Tier 1 data. This is the data you will use in the next steps of the pipeline.
*   **`examples/PEDIGREE.md`**: The audit trail. This file contains a detailed log of where each piece of data came from, when it was sourced, and for what purpose.

## 5. Deep Dive: The `scout_mission.yaml` File

The mission plan is highly configurable. Here is a breakdown of the key parameters you can use to control the agent's behavior.

```yaml
# The root of the file is a list of missions. The agent will execute them sequentially.
missions:
  - name: "production_corpus" # A descriptive name for the mission.

    # The total number of text blocks to collect per component/characteristic.
    target_size: 150

    # The proportion of the target_size that the agent is allowed to
    # invent using an LLM if it cannot find good examples online.
    # 0.2 means 20% of the samples can be synthetic.
    synthetic_budget: 0.2

    # A list of specific goals. The agent will work to fulfill each one.
    goals:
      - # Goal 1: Find text to test the "Verifiability" characteristic.
        characteristic: "Verifiability"

        # A list of specific topics to search for within this goal.
        topics:
          - "news reports"
          - "scientific abstracts"
          - "financial statements"
          - "opinion editorials" # Good for negative examples

      - # Goal 2: Find text to test "Self-containment" (Disambiguation).
        characteristic: "Self-containment"
        topics:
          - "political analysis"
          - "historical narratives"
          - "multi-paragraph news stories"
      
      # ... and so on for other goals.```

## 6. The Full Workflow: From Scout to Compiled Artifact

The Data Scout Agent is the first step in a three-step pipeline to create a final, optimized DSPy artifact.

**Step 1: Curate the Corpus with the Scout Agent**
Run the agent with your configured mission plan to produce the Tier 2 raw text corpus.

```bash
aclarai-claimify-scout --mission-plan settings/scout_mission.yaml
```

**Step 2: Generate the "Gold Standard" Dataset**
Use the `generate-dataset` command to convert the raw text from the agent into a structured `.jsonl` training file. This step uses a powerful "teacher" model to create the ideal outputs for each example.

```bash
aclarai-claimify generate-dataset \
  --input-file examples/data/datasets/tier2/selection_raw.txt \
  --output-file data/selection_train.jsonl \
  --teacher-model gpt-4o \
  --k-window-size 2
```

**Step 3: Compile the Production Artifact**
Finally, use the `compile` command to take the gold-standard training set and produce the final, optimized `.json` artifact.

```bash
aclarai-claimify compile \
  --config configs/selection_config.yaml \
  --trainset data/selection_train.jsonl \
  --output-path aclarai_claimify/compiled_prompts/selection.json
```

## 7. Conclusion

The Data Scout Agent is a powerful tool for creating high-quality, auditable datasets. By investing time in creating a thoughtful **Mission Plan**, you can automate the most time-consuming part of the optimization process. Remember to start with small, targeted missions and iterate as you refine your data requirements.