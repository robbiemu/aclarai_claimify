# nodes.py
"""
Agent nodes for the Data Scout graph.
"""

import json
import json_repair
import yaml
from datetime import datetime
import time
from typing import Dict, List, Optional

import random

from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage

from .tools import (
    get_tools_for_role,
    write_file,
    _truncate_response_for_role,
)
from .utils import (
    get_characteristic_context,
    get_claimify_strategy_block,
    append_to_pedigree,
    strip_reasoning_block,
)

from .state import DataScoutState
from .models import FitnessReport
from ..config import ClaimifyConfig, load_claimify_config
from .config import load_scout_config
from ..data_models import ScoutAgentMissionPlanNodeConfig
from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """Defines the structured decision output for the supervisor LLM."""

    next_agent: str = Field(
        ...,
        description="The name of the next agent to route to (e.g., 'research', 'fitness').",
    )
    new_task: Optional[Dict] = Field(
        None,
        description="Optional: A new task to assign if the supervisor decides to switch focus.",
    )


def create_llm(config: ClaimifyConfig, role: str) -> ChatLiteLLM:
    """Creates a configured ChatLiteLLM instance for a given agent role."""
    node_config = None
    if config.scout_agent and config.scout_agent.mission_plan:
        # Use the helper function for a clean, exact match
        node_config = config.scout_agent.mission_plan.get_node_config(role)

    # Use node-specific config if available, otherwise fall back to defaults
    if node_config:
        model = node_config.model
        temperature = node_config.temperature
        max_tokens = node_config.max_tokens
    else:
        # Fallback to the default model from the main config
        model = config.default_model
        temperature = config.temperature or 0.1
        max_tokens = config.max_tokens or 2000

    return ChatLiteLLM(model=model, temperature=temperature, max_tokens=max_tokens)


def create_agent_runnable(llm: ChatLiteLLM, system_prompt: str, role: str):
    """Factory to create a new agent node's runnable."""
    tools = get_tools_for_role(role)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


def _get_next_task_from_progress(
    progress: Dict, exclude: Optional[List[tuple[str, str]]] = None
) -> Optional[Dict]:
    """Finds the next uncompleted characteristic/topic pair."""
    if not progress:
        return None

    mission_name = list(progress.keys())[0]
    eligible_tasks = []
    for char, char_data in progress[mission_name].items():
        if char_data["collected"] < char_data["target"]:
            for topic, topic_data in char_data["topics"].items():
                if topic_data["collected"] < topic_data["target"]:
                    if not exclude or (char, topic) not in exclude:
                        eligible_tasks.append({"characteristic": char, "topic": topic})

    if not eligible_tasks:
        return None

    return random.choice(eligible_tasks)


def supervisor_node(state: DataScoutState) -> Dict:
    """The supervisor node, now driven by a progress tracker."""
    config = load_claimify_config()
    llm = create_llm(config, "supervisor")

    # --- 1. Load State ---
    progress = state.get("progress", {})
    _current_mission = state.get("current_mission", "production_corpus")
    current_task = state.get("current_task")

    # --- 3. Select Next Task (if necessary) ---
    # We only select a new task if there's no current task.
    if not current_task:
        next_task = _get_next_task_from_progress(progress)
    else:
        next_task = current_task

    # --- 4. Decide What to Do ---
    if not next_task:
        # All tasks are complete.
        print("🎉 Supervisor: All tasks in the mission are complete!")
        return {"next_agent": "end", "progress": progress, "strategy_block": ""}

    # We have a task. Now, we determine the strategy block for it.
    characteristic = next_task.get("characteristic", "Verifiability")
    topic = next_task.get("topic", "general domain")
    try:
        with open("settings/mission_config.yaml", "r") as f:
            content = f.read()
            if content.startswith("#"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_config = yaml.safe_load(content)
        characteristic_context = get_characteristic_context(next_task, mission_config)
    except Exception:
        characteristic_context = None

    if characteristic_context:
        print(
            f"   ✅ Supervisor: Using dynamic context for '{characteristic}' from mission plan."
        )
        strategy_block = (
            f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
        )
    else:
        print(
            f"   ⚠️  Supervisor: Could not find dynamic context for '{characteristic}'. Using built-in fallback."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    decision_history = state.get("decision_history", [])
    tool_execution_failures = state.get("tool_execution_failures", 0)
    research_attempts = state.get("research_attempts", 0)
    consecutive_failures = state.get("consecutive_failures", 0)
    last_action_status = state.get("last_action_status", "success")
    last_action_agent = state.get("last_action_agent", "")

    # If the last action was a successful archive, end the current cycle.
    if last_action_agent == "archive":
        print(
            "✅ Supervisor: Detected a successful archival. Ending the current cycle."
        )
        # Pass the full state along with the decision to end.
        return {
            "next_agent": "end",
            "progress": state.get("progress", {}),
            "current_task": state.get("current_task"),
            "strategy_block": state.get("strategy_block", ""),
            "decision_history": state.get("decision_history", []),
            "tool_execution_failures": state.get("tool_execution_failures", 0),
            "research_attempts": state.get("research_attempts", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "synthetic_samples_generated": state.get("synthetic_samples_generated", 0),
            "research_samples_generated": state.get("research_samples_generated", 0),
            "synthetic_budget": state.get("synthetic_budget", 0.2),
            "fitness_report": None,
        }

    synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
    research_samples_generated = state.get("research_samples_generated", 0)
    synthetic_budget = state.get("synthetic_budget", 0.2)
    total_samples_generated = synthetic_samples_generated + research_samples_generated

    messages = state.get("messages", [])

    # Check if we have a new research report first (this takes priority over old fitness reports)
    last_message_content = str(messages[-1].content) if messages else ""
    has_new_research_report = (
        decision_history
        and decision_history[-1] == "research"
        and "# Data Prospecting Report" in last_message_content
    )

    # Only check for old fitness reports if we don't have a new research report
    fitness_report = (
        state.get("fitness_report") if not has_new_research_report else None
    )

    if fitness_report:
        if fitness_report.passed:
            print(
                "✅ Supervisor: Detected PASSED fitness report in state. Deterministically routing to 'archive'."
            )
            # Clear the fitness report from state
            state_dict = dict(state)
            state_dict["fitness_report"] = None

            return {
                "next_agent": "archive",
                "progress": progress,
                "current_task": next_task,
                "strategy_block": strategy_block,
                "fitness_report": None,
                "decision_history": decision_history + ["archive"],
                "tool_execution_failures": tool_execution_failures,
                "research_attempts": research_attempts,
                "consecutive_failures": 0,
                "last_action_status": "success",
                "last_action_agent": "supervisor",
                "synthetic_samples_generated": synthetic_samples_generated,
                "research_samples_generated": research_samples_generated,
                "synthetic_budget": synthetic_budget,
                "research_findings": state.get("research_findings", []),
            }
        else:
            print(
                "❌ Supervisor: Detected FAILED fitness report in state. Proposing a new task to the supervisor."
            )
            # Clear the fitness report from state
            state_dict = dict(state)
            state_dict["fitness_report"] = None

            # --- NEW: Log the failure ---
            task_history = state.get("task_history", [])
            current_task = state.get("current_task")
            if current_task:
                task_history.append(
                    (
                        current_task.get("characteristic"),
                        current_task.get("topic"),
                        fitness_report.reason,
                    )
                )

            # --- NEW: Give the supervisor the option to change tasks ---
            excluded_tasks = [(t[0], t[1]) for t in task_history]
            alt_task = _get_next_task_from_progress(progress, exclude=excluded_tasks)

            if alt_task and alt_task != next_task:
                alt_characteristic = alt_task.get("characteristic", "N/A")
                alt_topic = alt_task.get("topic", "N/A")
                synthetic_budget = state.get("synthetic_budget", 0.2)
                total_samples_target = 0
                if progress:
                    mission_name = list(progress.keys())[0]
                    for char_data in progress[mission_name].values():
                        total_samples_target += char_data["target"]
                max_synthetic_samples = int(total_samples_target * synthetic_budget)
                synthetic_samples_generated = state.get(
                    "synthetic_samples_generated", 0
                )

                last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** fitness
   - **Reason:** The agent rejected the previous submission: {fitness_report.reason}
   - **Guidance:** You need to decide the next action based on the complete history you see. It may be that the current task is difficult to research, and we could more easily make progress on a different task. You have three options:
     1. Delegate to `research` to retry the current task (`{next_task["characteristic"]}` / `{next_task["topic"]}`).
     2. Delegate to 'synthetic' to complete the task (we have only {max_synthetic_samples - synthetic_samples_generated} of {max_synthetic_samples} submissions remaining that should ideally be synthetic)
     3. Switch to a different, uncompleted task, such as (`{alt_characteristic}` / `{alt_topic}`), by setting the `new_task` field in your response. If you switch, the researcher's memory will be cleared."""
            else:
                last_action_analysis = f"""**3. Last Action Analysis:** FAILURE
   - **Agent:** fitness
   - **Reason:** The agent rejected the previous submission: {fitness_report.reason}
   - **Guidance:** The current task is the only one remaining. You must delegate to `research` to retry it, but consider suggesting a new strategy."""

            # We will now fall through to the standard supervisor prompt, but with the special failure guidance.

    last_message_content = str(messages[-1].content) if messages else ""
    if (
        decision_history
        and decision_history[-1] == "research"
        and "# Data Prospecting Report" in last_message_content
    ):
        print(
            "✅ Supervisor: Detected a completed 'Data Prospecting Report'. Deterministically routing to 'fitness'."
        )

        # --- FIX: Extract content for the fitness node ---
        # Default values
        source_url = None
        provenance = "synthetic"
        research_findings = []

        # Safely extract the report content
        report_content = strip_reasoning_block(last_message_content)

        # --- FIX: Handle cache references ---
        research_cache = state.get("research_session_cache", [])
        if "[CACHE_REFERENCE:" in report_content and research_cache:
            try:
                # Extract call_id from the report
                call_id = report_content.split("[CACHE_REFERENCE:")[1].split("]")[0]

                # Find the corresponding evidence in the cache
                for evidence in research_cache:
                    if evidence.get("call_id") == call_id:
                        # Replace the token with the actual tool output
                        tool_output = evidence.get("output", "")
                        report_content = str(tool_output)
                        print(
                            f"✅ Supervisor: Resolved cache reference for call_id '{call_id}'."
                        )
                        break
            except IndexError:
                print("⚠️ Supervisor: Malformed cache reference found.")
                pass  # Keep the original report_content if reference is malformed

        research_findings.append(report_content)

        # Try to parse the URL from the report
        for line in report_content.split("\n"):
            if "**Source URL**:" in line:
                try:
                    source_url = line.split("`")
                    if len(source_url) > 1:
                        source_url = source_url[1]
                    else:
                        source_url = None
                except IndexError:
                    pass  # Keep source_url as None
                break

        # Verify provenance if a URL was found
        research_cache = state.get("research_session_cache", [])
        if source_url and research_cache:
            for evidence in research_cache:
                is_valid_evidence = (
                    isinstance(evidence, dict)
                    and "args" in evidence
                    and isinstance(evidence["args"], dict)
                )
                if is_valid_evidence:
                    args = evidence["args"]
                    found_url = (
                        args.get("url") or args.get("base_url") or args.get("start_url")
                    )
                    if found_url == source_url:
                        output = evidence.get("output", {})
                        if isinstance(output, dict) and output.get("status") == "ok":
                            provenance = "researched"
                            print(
                                f"✅ Supervisor: Provenance VERIFIED as 'researched' for URL: {source_url}"
                            )
                            break  # Found definitive proof
            if provenance == "synthetic":
                print(
                    f"⚠️  Supervisor: Provenance could NOT be verified for URL: {source_url}. Defaulting to 'synthetic'."
                )
        else:
            print(
                "ℹ️ Supervisor: No source URL or research cache found, defaulting provenance to 'synthetic'."
            )

        return {
            "next_agent": "fitness",
            "decision_history": decision_history + ["fitness"],
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": "supervisor",
            "current_sample_provenance": provenance,
            "progress": progress,
            "current_task": next_task,
            "strategy_block": strategy_block,
            "tool_execution_failures": tool_execution_failures,
            "research_attempts": research_attempts,
            "synthetic_samples_generated": synthetic_samples_generated,
            "research_samples_generated": research_samples_generated,
            "synthetic_budget": synthetic_budget,
            # Pass the extracted content to the next node
            "research_findings": research_findings,
            # Clear any old fitness report when routing to fitness for a new report
            "fitness_report": None,
        }

    base_prompt = """You are the supervisor of a team of Data Prospecting agents. Your role is to analyze the current mission status and decide which agent should act next.

Available Agents:
- `research`: Finds source documents from the web.
- `fitness`: Evaluates the quality of a source document.
- `archive`: Saves an approved document.
- `synthetic`: Generates a document from scratch.
- `end`: Finishes the mission."""

    characteristic = next_task.get("characteristic", "N/A")
    topic = next_task.get("topic", "N/A")
    current_task_str = f"   - Find sources for the characteristic '{characteristic}' in the topic '{topic}'."

    total_samples_target = 0
    if progress:
        mission_name = list(progress.keys())[0]
        for char_data in progress[mission_name].values():
            total_samples_target += char_data["target"]

    max_synthetic_samples = int(total_samples_target * synthetic_budget)
    synthetic_status = f"{synthetic_samples_generated} of a target {max_synthetic_samples} (Budget: {synthetic_budget:.0%})"

    strategic_guidance = f"""
**4. Strategic Reasoning Guidance:**
Your goal is to guide the team to produce a final corpus of approximately {total_samples_target} samples, of which roughly {max_synthetic_samples} should be synthetic.

When deciding between `research` and `synthetic`, you must manage the synthetic budget thoughtfully:
- **Current Status**: You have generated **{synthetic_samples_generated} synthetic samples** out of {total_samples_generated} total samples so far.
- **Mission Target**: Your mission allows for up to **{max_synthetic_samples} synthetic samples** out of {total_samples_target} total samples.
- **Remaining Budget**: You still have **{max_synthetic_samples - synthetic_samples_generated} synthetic samples** available in your budget.
- **Remaining Work**: You need to generate **{total_samples_target - total_samples_generated} more samples** to complete the mission.

Even if your current synthetic rate seems high, consider whether you still have room in your overall budget to generate synthetic samples, especially if research is proving challenging. Focus on the remaining budget rather than just the current rate.

A perfect final ratio is not required, but you should guide the process toward the mission's goal while respecting your remaining budget."""

    last_action_analysis = ""
    last_message_content = (
        str(messages[-1].content) if messages else "No recent messages."
    )

    if last_action_status == "failure":
        failure_agent = last_action_agent if last_action_agent else "unknown"
        if failure_agent == "research":
            last_action_analysis = """
**3. Last Action Analysis:** FAILURE
   - **Agent:** research
   - **Reason:** The agent failed to produce a valid Data Prospecting Report.
   - **Guidance:** The current research approach is not working. You have two options:
     1. Delegate to `research` again, but you should suggest a significantly different search strategy.
     2. If this has failed and seems likely to continue to fail (see Decision History), consider escalating to `synthetic`. """
        elif failure_agent == "archive":
            error_snippet = last_message_content.replace("\n", " ").strip()[:150]
            last_action_analysis = f"""
**3. Last Action Analysis:** FAILURE
   - **Agent:** archive
   - **Reason:** A tool error occurred during file saving: '{error_snippet}...'
   - **Guidance:** This is a system error. You should probably `end` the mission so a human can investigate. Retrying is unlikely to succeed."""
        else:
            last_action_analysis = f"""
**3. Last Action Analysis:** FAILURE
   - **Agent:** {failure_agent}
   - **Reason:** The last action resulted in an error.
   - **Guidance:** Analyze the mission history and decide the best recovery path."""
    else:
        success_agent = last_action_agent if last_action_agent else "initial_start"
        last_action_analysis = f"""
**3. Last Action Analysis:** SUCCESS
   - **Agent:** {success_agent}
   - **Guidance:** The previous task was completed. It is time to start the next task. Analyze the full mission context and decide on the next agent."""

    mission_context = f"""
---
**MISSION CONTEXT**

**1. Current Task:**
{current_task_str}

**2. Mission History & Status:**
   - **Decision History (Last 5):** {decision_history[-5:]}
   - **Consecutive Failures:** {consecutive_failures}
   - **Synthetic Sample Status:** {synthetic_status}

{last_action_analysis}

{strategic_guidance}
---
Your response MUST be a JSON object matching the required schema, with a single key "next_agent" and the name of the agent as the value."""

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_prompt),
            ("human", mission_context),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    agent = agent_prompt | llm
    raw_result = agent.invoke({"messages": messages})

    try:
        dethought = strip_reasoning_block(raw_result.content)
        repaired_data = json_repair.loads(dethought)
        decision = SupervisorDecision.model_validate(repaired_data)
        next_agent = decision.next_agent.lower()

        # --- NEW: Handle task switching ---
        if decision.new_task:
            print(f"✅ Supervisor: Switching to new task: {decision.new_task}")
            next_task = decision.new_task
            # Recalculate strategy block for the new task
            characteristic = next_task.get("characteristic", "Verifiability")
            try:
                with open("settings/mission_config.yaml", "r") as f:
                    content = f.read()
                    if content.startswith("#"):
                        first_newline = content.find("\n")
                        if first_newline != -1:
                            content = content[first_newline + 1 :]
                    mission_config = yaml.safe_load(content)
                characteristic_context = get_characteristic_context(
                    next_task, mission_config
                )
            except Exception:
                characteristic_context = None

            if characteristic_context:
                strategy_block = f"**Strategic Focus for '{characteristic}':**\n{characteristic_context}"
            else:
                strategy_block = get_claimify_strategy_block(characteristic)

            # Clear messages for the researcher
            messages = []
    except Exception as parse_error:
        print(f"⚠️ Supervisor: JSON parsing failed: {parse_error}")
        print(f"   Raw content: '{raw_result.content}'")
        next_agent = "research"

    return {
        "next_agent": next_agent,
        "progress": progress,
        "current_task": next_task,
        "strategy_block": strategy_block,
        "decision_history": decision_history + [next_agent],
        "tool_execution_failures": tool_execution_failures,
        "research_attempts": research_attempts + (1 if next_agent == "research" else 0),
        "consecutive_failures": consecutive_failures,
        "last_action_status": "success",
        "last_action_agent": "supervisor",
        "synthetic_samples_generated": synthetic_samples_generated,
        "research_samples_generated": research_samples_generated,
        "synthetic_budget": synthetic_budget,
        # Clear any fitness report when falling through to standard logic
        "fitness_report": None,
    }


def research_node(state: "DataScoutState") -> Dict:
    """
    Verifiable Research Workflow:
    - Dynamically scoped ReAct loop (discover → extract → synthesize)
    - Cache ALL successful tool outputs as evidence in research_session_cache
    - Return a single 'Data Prospecting Report' as the final submission
    - Do NOT mutate research_findings here (Supervisor handles parsing)
    """
    # --- Setup ---
    config = load_claimify_config()
    llm = create_llm(config, "research")

    print("🔍 RESEARCH NODE (Verifiable Research Workflow)")
    print(f"   Incoming messages: {len(state.get('messages', []))}")
    print(f"   LLM model: {getattr(llm, 'model', 'unknown')}")

    # --- Extract the latest human question ---
    user_question = None
    for msg in reversed(state.get("messages", [])):
        # LangChain HumanMessage or any object with type == "human"
        if getattr(msg, "content", None):
            if getattr(msg, "type", None) == "human" or "HumanMessage" in str(
                type(msg)
            ):
                user_question = msg.content
                break

    if not user_question:
        print("   ❗ No user question found; returning request for clarification.")
        no_question_msg = AIMessage(
            content="No clear research question found in conversation history. Please provide a specific question for me to research."
        )
        return {
            "messages": [no_question_msg],
            # Return unchanged cache if any exists
            "research_session_cache": state.get("research_session_cache", []),
        }

    # --- Evidence cache (auditable log of work for Supervisor) ---
    session_cache = list(
        state.get("research_session_cache", [])
    )  # make a copy we can append to
    print(f"   Session cache (pre-run): {len(session_cache)} items")

    # --- Tools & Mission Context ---
    all_research_tools = get_tools_for_role("research")
    print(f"   Tools available (global): {[t.name for t in all_research_tools]}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")
    if current_task:
        characteristic = current_task.get("characteristic", "Verifiability")
        topic = current_task.get("topic", "general domain")
        print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")
    else:
        characteristic = "Verifiability"
        topic = "general domain"
        print("   🎯 No specific task queued; using default mission focus.")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    # --- System prompt (mission-specific) ---
    system_prompt = f"""You are a Data Prospector, a specialist in identifying high-quality raw text for data extraction pipelines. You operate using a ReAct (Reasoning and Acting) methodology.

Your Mission: Your goal is to find and retrieve a source document from the **{topic}** domain whose writing style and structure make it an exceptionally good source for extracting factual claims that exemplify the principle of **"{characteristic}"**.

You are not extracting the final claims. You are finding the *ore*. You must find a document that is naturally rich in sentences that a downstream agent could easily turn into high-quality claims with the desired characteristic.

---
{strategy_block}
---

### ReAct Process & Tool Usage

Your workflow is a two-step process: **Discover, then Extract.**

1.  **REASON:** Based on your strategic focus, formulate a search plan.
2.  **ACT (Discover):** Use the search tools (`web_search`, `arxiv_search`, `wikipedia_search`) to find promising URLs. The output of these tools is just a list of links and snippets; it is **not** the final document.
3.  **OBSERVE:** Analyze the search results. Identify the single most promising URL that is likely to contain the full source document.
4.  **ACT (Extract):** Use the `url_to_markdown` or `documentation_crawler` tool on that single URL to retrieve the **full text** of the candidate document.
5.  **REPEAT:** If the extracted document is low-quality or irrelevant, discard it and refine your search. Continue until you find one high-quality source document that is a strong match.

### Content Curation (Expert Refinement)

Your goal is to maximize the signal-to-noise ratio for the next agent.

- **To submit a specific, high-value excerpt:** If a specific section of a document is highly relevant, you should extract and submit ONLY that section in the `Retrieved Content (Markdown)` block. You may use ellipses `(...)` on their own line to indicate where you have removed irrelevant surrounding text.
- **To submit the entire document:** If the whole document is a good fit, you do NOT need to copy its contents. Instead, use a special token to reference the cached tool output. Find the `call_id` from the successful tool call in your history and place it in the report like this:

`[CACHE_REFERENCE: call_...]`

This tells the supervisor to fetch the full content from the cache, saving context space.

When you have successfully found and extracted a suitable source, you MUST output a single, structured 'Data Prospecting Report' exactly in the format below—no extra commentary. Your response must start with `# Data Prospecting Report`.

# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `[The specific URL of the retrieved content]`
**Source Title**: `"[The title of the web page or document]"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: [Explain in detail *why* the writing style, sentence structure, and overall format of this document make it an excellent source for extracting claims that will have the property of `{characteristic}`. Refer back to your strategic focus in your reasoning.]
* **Potential for High Yield**: [Briefly explain why you believe this document will provide a large number of usable examples for the downstream agents.]

---

## Retrieved Content (Markdown)

`[Either paste the curated excerpt OR provide the [CACHE_REFERENCE: ...] token here.]`
"""

    # --- Base prompt template (system is dynamic per-iteration) ---
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt_for_iteration}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # --- Seed conversation for the agent (internal to this node) ---
    react_messages = []
    if state.get("messages"):
        react_messages.extend(state["messages"])

    # --- ReAct loop with dynamic tool scoping ---
    max_iterations = int(
        getattr(config.scout_agent.nodes.research, "max_iterations", 8)
    )
    print(f"   🔄 ReAct loop starting (max_iterations={max_iterations})")

    final_report_msg = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"   ▶ Iteration {iteration}/{max_iterations}")

        # <<< NEW DEBUGGING >>>
        history_char_count = sum(len(str(m.content)) for m in react_messages)
        print(
            f"      📊 CONTEXT: {len(react_messages)} messages, ~{history_char_count} chars"
        )
        # <<< END NEW DEBUGGING >>>

        # Dynamic scoping + warnings
        warning_message = ""
        current_tools = all_research_tools

        if iteration == max_iterations - 2:
            warning_message = (
                "\n\n**SYSTEM WARNING:** You have 3 iterations remaining."
                "Begin concluding discovery and prepare to select a final document for extraction. This is the last turn that you may use the search tools (`web_search`, `arxiv_search`, `wikipedia_search`)."
            )
        elif iteration == max_iterations - 1:
            # Disable discovery; force extraction
            current_tools = [
                t
                for t in all_research_tools
                if t.name not in ["web_search", "arxiv_search", "wikipedia_search"]
            ]
            warning_message = (
                "\n\n**SYSTEM WARNING:** You have 2 iterations remaining. "
                "The search tools (`web_search`, `arxiv_search`, `wikipedia_search`) have been disabled. Analyze current leads and extract a full document. This is the last turn that you may use the tools you are being provided."
            )
        elif iteration == max_iterations:
            # Disable all tools; force synthesis
            current_tools = []
            warning_message = (
                "\n\n**SYSTEM WARNING:** FINAL iteration. All tools disabled. "
                "Write the final 'Data Prospecting Report' now."
            )

        system_prompt_for_iteration = system_prompt + warning_message

        # <<< NEW DEBUGGING (for final iteration only) >>>
        if iteration == max_iterations:
            print(
                "      ❗ FINAL PROMPT: The following system prompt is being sent to the LLM for its last chance."
            )
            print("      " + "-" * 20)
            print(f"      {system_prompt_for_iteration[-500:]}")  # Print last 500 chars
            print("      " + "-" * 20)
        # <<< END NEW DEBUGGING >>>

        # Bind tools as scoped this iteration
        llm_with_tools = llm.bind_tools(current_tools) if current_tools else llm
        print(
            f"      Tools this iteration: {[t.name for t in current_tools] if current_tools else '[]'}"
        )

        # Build runnable and invoke
        react_agent = (
            prompt_template.partial(
                system_prompt_for_iteration=system_prompt_for_iteration
            )
            | llm_with_tools
        )

        try:
            result = react_agent.invoke({"messages": react_messages})

            # <<< NEW DEBUGGING >>>
            raw_content = getattr(result, "content", "[NO CONTENT]")
            print("      📝 --- START RAW LLM RESPONSE ---")
            print(f"{raw_content.strip()}")
            print("      📝 ---  END RAW LLM RESPONSE  ---")
            # <<< END NEW DEBUGGING >>>

            # RECOVERY: Handle empty responses on final iteration with sleep-and-retry
            if iteration == max_iterations and (
                not raw_content or not raw_content.strip()
            ):
                print(
                    "      🚨 CRITICAL: Empty response on final iteration - attempting retry after brief pause"
                )

                try:
                    print("      ⏳ Waiting 3 seconds for model to stabilize...")
                    import time

                    time.sleep(3)

                    print("      🔄 Retrying final iteration with same prompt...")
                    # Use the exact same agent and prompt - just retry
                    retry_result = react_agent.invoke({"messages": react_messages})
                    retry_content = getattr(retry_result, "content", "")

                    print("      📝 --- RETRY RESPONSE ---")
                    print(f"{retry_content.strip()}")
                    print("      📝 --- END RETRY ---")

                    # If retry produced content, use it
                    if retry_content and retry_content.strip():
                        result.content = retry_content
                        print("      ✅ Retry successful - using response")
                    else:
                        print("      ⚠️ Retry also failed - will use fallback")

                except Exception as retry_error:
                    print(f"      ❌ Retry attempt failed: {retry_error}")
                    # Continue with the empty result, fallback will handle it

            react_messages.append(result)

            # Successful termination: exact string check per spec
            print(
                f"      🕵️‍♀️ Checking for final report in content: {getattr(result, 'content', '')[:100]}..."
            )
            if (
                getattr(result, "content", "")
                and "# Data Prospecting Report" in result.content
            ):
                print("      🏁 Final submission detected ('Data Prospecting Report').")
                final_report_msg = result
                break

            # Process tool calls (cache ALL successful outputs as evidence)
            tool_calls = getattr(result, "tool_calls", None)
            if tool_calls:
                print(f"      🔧 Tool calls: {len(tool_calls)}")
                for idx, tool_call in enumerate(tool_calls):
                    try:
                        # Normalize tool call access (dict or object)
                        tool_name = (
                            tool_call.get("name")
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "name", None)
                        )
                        tool_args = (
                            tool_call.get("args", {})
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "args", {}) or {}
                        )
                        tool_id = (
                            tool_call.get("id")
                            if isinstance(tool_call, dict)
                            else getattr(tool_call, "id", f"call_{iteration}_{idx}")
                        )

                        matching_tool = next(
                            (t for t in current_tools if t.name == tool_name), None
                        )
                        if not matching_tool:
                            print(
                                f"         ⚠️ Tool '{tool_name}' not found in current scope; skipping."
                            )
                            continue

                        print(
                            f"         ▶ Executing {tool_name} with args: {str(tool_args)[:200]} ..."
                        )
                        tool_result = matching_tool.invoke(tool_args)

                        # Validate search tool results to ensure URLs are accessible
                        if tool_name in [
                            "web_search",
                            "arxiv_search",
                            "wikipedia_search",
                        ] and isinstance(tool_result, dict):
                            if tool_result.get("status") == "ok" and tool_result.get(
                                "results"
                            ):
                                from .scout_utils import _validate_search_results

                                print(f"         🔍 Validating {tool_name} results...")
                                validation_result = _validate_search_results(
                                    tool_result["results"],
                                    tool_name,
                                    tool_args,
                                    matching_tool,
                                )

                                tool_result["results"] = validation_result["results"]
                                tool_result["validation_info"] = validation_result
                                print(
                                    f"         ✅ {tool_name} results validated ({len(validation_result['results'])} results)"
                                )

                                # Log retry information if performed
                                if validation_result.get("retry_performed"):
                                    if validation_result.get("retry_successful"):
                                        print(
                                            f"         🔄 {tool_name}: Auto-retry successful"
                                        )
                                    else:
                                        print(
                                            f"         🔄 {tool_name}: Auto-retry attempted but unsuccessful"
                                        )

                        # Truncate the tool result based on the research role's max_tokens setting
                        if isinstance(tool_result, dict):
                            tool_result = _truncate_response_for_role(
                                tool_result, "research"
                            )

                        # Append ToolMessage for agent observation
                        react_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id,
                            )
                        )

                        # === EVIDENCE CACHING (unconditional for successful calls) ===
                        session_cache.append(
                            {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "iteration": iteration,
                                "call_id": tool_id,
                                "tool": tool_name,
                                "args": tool_args,  # original arguments
                                "output": tool_result,  # full raw output (JSON/dict/string)
                                "user_question": user_question,
                            }
                        )
                        print("         ✅ Evidence cached.")

                    except Exception as tool_error:
                        print(f"         ❌ Tool execution error: {tool_error}")
                        # Provide an observation to the agent but DO NOT cache as successful evidence
                        react_messages.append(
                            ToolMessage(
                                content=f"Tool '{tool_name}' failed: {tool_error}",
                                tool_call_id=tool_id
                                if "tool_id" in locals()
                                else f"error_{iteration}_{idx}",
                            )
                        )
                # Let the loop continue so the model can reason over observations
                continue

            else:  # This else block is the key change
                # No tools were called and no final report yet; continue  iterations
                print("      ℹ️ No tool calls this step; continuing.")
                continue

        except Exception as iter_error:
            print(f"      ❌ Iteration error: {iter_error}")
            # Continue to fallback after loop

    print(f"   ✅ Loop ended after {iteration} iterations")

    # --- Fallback: always return a 'Data Prospecting Report' (even if empty) ---
    if not final_report_msg:
        # [EXISTING CODE] print("   ⚠️ No final report produced; generating a minimal report to satisfy contract.")

        # <<< NEW DEBUGGING >>>
        print(
            "      ☠️ FAILURE ANALYSIS: The agent completed all iterations without producing a final report."
        )
        print("      Last 3 messages in history:")
        for msg in react_messages[-3:]:
            print(
                f"         - [{getattr(msg, 'type', 'UNKNOWN').upper()}]: {str(getattr(msg, 'content', ''))[:150]}..."
            )
        # <<< END NEW DEBUGGING >>>

        # Produce an honest, minimal report so Supervisor can proceed
        fallback_report = f"""# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `None`
**Source Title**: `"No qualifying source selected"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: The agent did not produce a final selection within the allotted iterations.
* **Potential for High Yield**: Unable to assess due to missing final selection.

---

## Retrieved Content (Markdown)

`No extracted content. See research_session_cache for all gathered evidence.`
"""
        final_report_msg = AIMessage(content=fallback_report)
    elif not final_report_msg.content.startswith("# Data Prospecting Report"):
        final_report_msg.content = (
            "# Data Prospecting Report\n\n" + final_report_msg.content
        )

    # --- Return only the final submission + full evidence cache ---
    print(
        f"   🧾 Returning final submission + evidence (cache size: {len(session_cache)})"
    )
    return {
        "messages": [final_report_msg],  # Append the final Data Prospecting Report ONLY
        "research_session_cache": session_cache,  # Full, updated evidence cache (no clearing)
    }


def archive_node(state: "DataScoutState") -> Dict:
    """
    The archive node, responsible for saving data and updating the audit trail
    using a procedural approach.
    """
    # Load scout config instead of main config for writer paths
    scout_config = load_scout_config()
    config = load_claimify_config()
    llm = create_llm(config, "archive")

    # --- FIX: Get content from research_findings, not messages ---
    provenance = state.get("current_sample_provenance", "synthetic")
    messages = state.get("messages", [])
    research_findings = state.get("research_findings", [])

    # Robustly find the document content
    document_content = None
    if research_findings:
        # Content is now a list of strings, so we join them.
        document_content = "\n\n---\n\n".join(research_findings)
        print("   ✅ Archive: Found content in 'research_findings'.")
    else:
        # Fallback for older states or different paths
        for msg in reversed(messages):
            if hasattr(msg, "content") and "# Data Prospecting Report" in msg.content:
                document_content = msg.content
                print("   ⚠️  Archive: Found content via message search fallback.")
                break

    if not document_content:
        error_message = AIMessage(
            content="Archive Error: Could not find a 'Data Prospecting Report' in the conversation history to save."
        )
        return {"messages": messages + [error_message]}

    # Get task details for naming
    current_task = state.get("current_task")
    characteristic = "unknown"
    topic = "unknown"
    if current_task:
        characteristic = (
            current_task.get("characteristic", "unknown").lower().replace(" ", "_")
        )
        topic = current_task.get("topic", "unknown").lower().replace(" ", "_")

    # --- FIX: Remove JSON and ask for the raw markdown string directly ---
    system_prompt = f"""You are the Library Cataloger in the Claimify data pipeline.

Your task is to generate a concise pedigree catalog entry in Markdown format based on the provided document.

The entry MUST include:
- The sample’s provenance: **'{provenance}'**
- The source URL from the document.
- The target characteristic from the document.

Respond ONLY with the Markdown content for the pedigree entry. Do NOT include any other text, greetings, or JSON formatting.

**Example Response:**
### YYYY-MM-DD — Sample Archived
- **Source Type:** {provenance}
- **Source URL:** [source url]
- **Target Characteristic:** {characteristic}
"""
    agent_runnable = create_agent_runnable(llm, system_prompt, "archive")
    llm_result = agent_runnable.invoke({"messages": messages})

    # The LLM's raw output is now our entry markdown. No parsing needed.
    entry_markdown = llm_result.content

    # --- Procedural control flow ---

    # Generate a unique timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Construct the deterministic filepath using config
    filename = f"{characteristic}_{topic}_{timestamp}.md"
    writer_config = scout_config.get("writer", {})
    tier1_path = writer_config.get("tier1_path", "examples/data/datasets/tier1")
    filepath = f"{tier1_path}/{filename}"

    # Now use this 'filepath' variable in the write_file tool.
    write_result = write_file.invoke(
        {"filepath": filepath, "content": document_content}
    )

    if write_result.get("status") == "ok":
        run_id = state.get("run_id")
        # Use pedigree path from scout config if available, otherwise from state
        pedigree_path = state.get("pedigree_path") or writer_config.get(
            "audit_trail_path", "examples/PEDIGREE.md"
        )
        append_to_pedigree(
            pedigree_path=pedigree_path,
            entry_markdown=entry_markdown,
            run_id=run_id,
        )
    else:
        error_message = AIMessage(
            content=f"Archive Error: Failed to write file to '{filepath}'. Error: {write_result.get('error')}"
        )
        return {"messages": messages + [error_message]}

    # --- Progress counters ---
    is_synthetic_sample = provenance == "synthetic"
    synthetic_samples_generated = state.get("synthetic_samples_generated", 0)
    research_samples_generated = state.get("research_samples_generated", 0)

    if is_synthetic_sample:
        synthetic_samples_generated += 1
    else:
        research_samples_generated += 1

    samples_generated = state.get("samples_generated", 0) + 1
    total_target = state.get("total_samples_target", 1200)
    total_samples = synthetic_samples_generated + research_samples_generated
    synthetic_pct = (
        (synthetic_samples_generated / total_samples) if total_samples > 0 else 0.0
    )
    progress_pct = (samples_generated / total_target) * 100 if total_target > 0 else 0
    source_type = "synthetic" if is_synthetic_sample else "research"

    print(
        f"   📊 Sample #{samples_generated} archived ({progress_pct:.1f}% complete) - Source: {source_type}"
    )
    print(
        f"   📈 Updated ratios - Research: {research_samples_generated}, Synthetic: {synthetic_samples_generated} ({synthetic_pct:.1%})"
    )

    confirmation_message = AIMessage(
        content=f"Successfully archived document to '{filepath}' and updated the pedigree."
    )
    return {
        "messages": messages + [confirmation_message],
        "samples_generated": samples_generated,
        "synthetic_samples_generated": synthetic_samples_generated,
        "research_samples_generated": research_samples_generated,
        "last_action_agent": "archive",
    }


def fitness_node(state: "DataScoutState") -> Dict:
    """The fitness node, responsible for evaluating content and producing a structured report."""
    config = load_claimify_config()
    llm = create_llm(config, "fitness")

    # --- START: PROVENANCE-AWARE LOGIC (Part 3) ---
    # 1. Retrieve the provenance flag from the state
    provenance = state.get("current_sample_provenance", "unknown")

    # 2. Construct dynamic guidance based on provenance
    provenance_guidance = ""
    if provenance == "researched":
        provenance_guidance = "**Provenance Note:** The source of this document has been programmatically verified."
    elif provenance == "synthetic":
        provenance_guidance = """
**Provenance Note:** The indicated source for this document could not be programmatically verified, suggesting the content may be synthetic. Please evaluate its content and style on its own merits against the mission rubric, focusing on its potential value as a training example rather than its real-world origin."""
    # --- END: PROVENANCE-AWARE LOGIC ---

    # Get the current mission context (characteristic, topic) from the task queue.
    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")

    if not current_task:
        characteristic = "the target"
        topic = "the correct"
        strategy_block = "No specific strategy defined."
    else:
        characteristic = current_task.get("characteristic", "the target")
        topic = current_task.get("topic", "the correct")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    research_findings = state.get("research_findings", [])
    # Escape braces in research findings to prevent template errors
    escaped_research_findings = (
        str(research_findings).replace("{", "{{").replace("}", "}}")
    )
    fitness_schema = json.dumps(FitnessReport.model_json_schema(), indent=2)

    # Escape curly braces in the JSON schema to prevent f-string template conflicts
    escaped_fitness_schema = fitness_schema.replace("{", "{{").replace("}", "}}")

    # --- START: REVISED PROMPT AND RUNNABLE CONSTRUCTION ---

    # 3. Define the system prompt with all dynamic content properly escaped
    system_prompt = f"""You are a Quality Inspector in the Claimify data pipeline. Your role is to evaluate whether a 'book' (a source document) found by the Research Agent is a high-quality source for our mission.

Your job is to inspect the **entire book** and decide if it's worth keeping. A downstream process, the 'Copy Editor', will later extract the specific 'quotes' (claims). You are NOT extracting quotes, only approving the source.

---
**Current Mission Context**
- **Target Characteristic:** {characteristic}
- **Search Domain:** {topic}
---
**Quality Standards for this Mission**

To be approved, the document's writing style and structure must align with the strategic focus for '{characteristic}'. Here is your rubric:

---
{strategy_block}
---

**Your Task**

{provenance_guidance}

The Research Agent has returned the following document(s):
{escaped_research_findings}

Evaluate the retrieved content against the mission. Is this document a goldmine for the Copy Editor, or a waste of time?

**CRITICAL: Your response must be ONLY a valid JSON object with no additional text, explanations, or formatting. Do not include any text before or after the JSON. Start your response directly with the opening brace {{{{ and end with the closing brace }}}}.**

**JSON Schema to follow:**
```json
{escaped_fitness_schema}
```"""

    agent_runnable = create_agent_runnable(llm, system_prompt, "fitness")
    raw_result = agent_runnable.invoke({"messages": state["messages"]})

    try:
        dethought = strip_reasoning_block(raw_result.content)
        repaired_data = json_repair.loads(dethought)
        report = FitnessReport.model_validate(repaired_data)
    except Exception as parse_error:
        print(f"⚠️ Fitness Node: JSON parsing failed: {parse_error}")
        print(f"   Raw content: '{raw_result.content}'")
        report = FitnessReport(
            passed=False,
            reason="The quality inspector LLM failed to produce a valid structured evaluation. The source document could not be reliably assessed.",
        )

    # --- END: REVISED PROMPT AND RUNNABLE CONSTRUCTION ---

    # The rest of the function remains the same, but is now more robust.
    if not isinstance(report, FitnessReport):
        print(
            "❌ Fitness Node: LLM failed to return a valid FitnessReport object. Treating as REJECTED."
        )
        report = FitnessReport(
            passed=False,
            reason="The quality inspector LLM failed to produce a valid structured evaluation. The source document could not be reliably assessed.",
        )

    status = "APPROVED" if report.passed else "REJECTED"
    log_message_content = (
        f"**Inspection Report**\n- **Status:** {status}\n- **Reason:** {report.reason}"
    )

    return {
        "messages": [AIMessage(content=log_message_content)],
        "fitness_report": report,
    }


def synthetic_node(state: DataScoutState) -> Dict:
    """The synthetic node, responsible for generating synthetic data when syntethic results are preferable or when research fails."""
    config = load_claimify_config()
    llm = create_llm(config, "synthetic")

    print("🎨 SYNTHETIC NODE: Starting synthetic data generation")
    print(f"   State messages: {len(state.get('messages', []))}")
    print(f"   Decision history: {state.get('decision_history', [])[-3:]}")

    current_task = state.get("current_task")
    strategy_block = state.get("strategy_block", "")
    if current_task:
        characteristic = current_task.get("characteristic", "Verifiability")
        topic = current_task.get("topic", "general domain")
        print(f"   🎯 Task selected: characteristic={characteristic} topic={topic}")
    else:
        characteristic = "Verifiability"
        topic = "general domain"
        print("   🎯 No specific task queued; using default mission focus.")

    if not strategy_block:
        print(
            f"   ⚠️  No strategy block found in state. Using built-in fallback for '{characteristic}'."
        )
        strategy_block = get_claimify_strategy_block(characteristic)

    system_prompt = f"""You are a Synthetic Book Author in the Claimify data pipeline. Your role is to create high-quality synthetic books (source documents) when the Librarian has failed to find suitable real books from the internet.

When the Librarian has been unable to find good books for a specific characteristic and domain, you step in to author a synthetic book that would be perfect for the Copy Editor to extract quotes from.

Analyze the conversation history to understand:
1. What Claimify characteristic was being targeted ({characteristic})
2. What topic domain was being searched ({topic})
3. Why the real book search failed

**Your synthetic book should be crafted to maximize signal-to-noise ratio for quote extraction:**

{strategy_block}

**Output Format (mimic the Librarian's Data Prospecting Report):**

# Data Prospecting Report

**Target Characteristic**: `{characteristic}`
**Search Domain**: `{topic}`

**Source URL**: `https://synthetic-library.generated/[relevant-path]`
**Source Title**: `"[Title of your synthetic book]"`

---

## Justification for Selection

* **Alignment with `{characteristic}`**: [Why this synthetic book is perfect for the characteristic]
* **Potential for High Yield**: [Why the Copy Editor will find many excellent quotes here]

---

## Retrieved Content (Markdown)

`[Your substantial, realistic synthetic book content - rich with extractable quotes]`

Focus on creating a book that will be a goldmine for the Copy Editor to extract high-quality sentences from."""

    agent_runnable = create_agent_runnable(llm, system_prompt, "synthetic")
    result = agent_runnable.invoke({"messages": state["messages"]})

    # --- FIX: Align output with the new data flow ---
    report_content = strip_reasoning_block(result.content)

    print(f"🎨 SYNTHETIC NODE: Generated response of {len(report_content)} characters")

    # Log the generated content in the same format as research node for TUI display
    # This creates a message that the TUI can parse and extract sample content from
    display_message = AIMessage(
        content=(
            "      📝 --- START LLM RESPONSE ---\n"
            f"🎨 SYNTHETIC GENERATION: Created synthetic content for {characteristic} - {topic}\n\n"
            f"## Generated Content (Synthetic)\n\n{report_content}\n\n"
            "      📝 ---  END LLM RESPONSE  ---"
        )
    )

    # The synthetic node bypasses fitness and routes directly to archive.
    # We must populate the state in the same way the supervisor would when routing
    # to the fitness node, so the archive node can find the content.
    return {
        "messages": [display_message],  # Use display message for TUI logging
        "research_findings": [report_content],  # Pass content to the archive node
        "current_sample_provenance": "synthetic",  # Explicitly set provenance
    }


def get_node_config(node_name: str) -> Optional[ScoutAgentMissionPlanNodeConfig]:
    """Retrieve the configuration for a specific node by name."""
    config = load_claimify_config()
    if config.scout_agent and config.scout_agent.mission_plan:
        return config.scout_agent.mission_plan.get_node_config(node_name)
    return None
