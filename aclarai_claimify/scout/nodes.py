import os
import json
import litellm
from .state import DataScoutState
from aclarai_claimify.config import load_claimify_config

# --- Prompts ---

DECONSTRUCT_PROMPT = """
Deconstruct the following high-level goal into a structured JSON object.
The JSON object should contain:
1.  "entities": A list of key entities or topics to research.
2.  "scope": A description of what is in-scope and out-of-scope.
3.  "success_criteria": A list of questions that need to be answered to fulfill the goal.

Goal: {goal}

Return only the JSON object.
"""

PLAN_PROMPT = """
Based on the deconstructed goal, create a web search plan.
The plan should be a JSON object with a list of search queries.
Each query should be targeted to find specific information related to the success criteria.

Deconstructed Goal: {deconstructed_goal}

Return only the JSON object with a "queries" key.
"""

FITNESS_CHECK_PROMPT = """
Evaluate the following text based on the deconstructed goal.
Determine if the text is relevant and reliable for the data-gathering mission.
Provide a JSON object with the following keys:
1.  "is_fit": boolean, true if the text is relevant and reliable.
2.  "reasoning": a brief explanation of your decision.
3.  "score": a float from 0.0 to 1.0 indicating the quality of the text.

Deconstructed Goal: {deconstructed_goal}
Text to Evaluate:
---
{text}
---

Return only the JSON object.
"""


def get_node_config(node_name: str):
    """Helper to get node configuration."""
    config = load_claimify_config()
    if config.scout_agent:
        for node in config.scout_agent.mission_plan.nodes:
            if node.name == node_name:
                return node
    return None


def deconstruct_goal_node(state: DataScoutState) -> DataScoutState:
    print("---DECONSTRUCT GOAL---")
    goal = state['mission_goal']
    node_config = get_node_config("DeconstructGoalNode")
    if not node_config:
        raise ValueError("Config for DeconstructGoalNode not found.")

    prompt = DECONSTRUCT_PROMPT.format(goal=goal)
    response = litellm.completion(
        model=node_config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=node_config.temperature,
        max_tokens=node_config.max_tokens,
    )

    try:
        deconstructed_goal = json.loads(response.choices[0].message.content)
        state['deconstructed_goal'] = deconstructed_goal
    except (json.JSONDecodeError, KeyError):
        print("Error: Failed to decode JSON from deconstruction response.")
        state['deconstructed_goal'] = {"error": "Failed to parse LLM response."}

    return state


def plan_node(state: DataScoutState) -> DataScoutState:
    print("---PLANNING---")
    deconstructed_goal = state['deconstructed_goal']
    node_config = get_node_config("PlanNode")
    if not node_config:
        raise ValueError("Config for PlanNode not found.")

    prompt = PLAN_PROMPT.format(deconstructed_goal=json.dumps(deconstructed_goal, indent=2))
    response = litellm.completion(
        model=node_config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=node_config.temperature,
        max_tokens=node_config.max_tokens,
    )

    try:
        search_plan = json.loads(response.choices[0].message.content)
        state['search_plan'] = search_plan
        state['search_queries'] = search_plan.get("queries", [])
    except (json.JSONDecodeError, KeyError):
        print("Error: Failed to decode JSON from planning response.")
        state['search_plan'] = {"error": "Failed to parse LLM response."}
        state['search_queries'] = []

    return state


def web_search_node(state: DataScoutState) -> DataScoutState:
    print("---WEB SEARCH---")
    # This is a placeholder for a real web search tool.
    # In a real implementation, this would use a library like `requests` and `BeautifulSoup`,
    # or an API like Tavily or SearxNG.
    queries = state.get("search_queries", [])
    if not queries:
        print("No search queries found. Skipping web search.")
        state['search_results'] = []
        state['extraction_results'] = []
        return state

    # For now, we'll just return a dummy result.
    query = queries.pop(0) # Process one query per iteration
    print(f"Executing dummy search for: {query}")
    state['search_results'] = [{"url": f"http://example.com/search?q={query}", "content": "This is a placeholder content for the search query: " + query}]
    state['extraction_results'] = ["This is the extracted text from the dummy search result."]
    state['search_queries'] = queries # Update the list of remaining queries
    return state


def fitness_check_node(state: DataScoutState) -> DataScoutState:
    print("---FITNESS CHECK---")
    deconstructed_goal = state['deconstructed_goal']
    extraction_results = state.get("extraction_results", [])
    if not extraction_results:
        print("No extraction results to check. Skipping fitness check.")
        state['fitness_check_results'] = []
        return state

    node_config = get_node_config("FitnessCheckNode")
    if not node_config:
        raise ValueError("Config for FitnessCheckNode not found.")

    # For now, we only check the first result.
    text_to_evaluate = extraction_results[0]
    prompt = FITNESS_CHECK_PROMPT.format(deconstructed_goal=json.dumps(deconstructed_goal, indent=2), text=text_to_evaluate)

    response = litellm.completion(
        model=node_config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=node_config.temperature,
        max_tokens=node_config.max_tokens,
    )

    try:
        fitness_result = json.loads(response.choices[0].message.content)
        state['fitness_check_results'] = [fitness_result]
    except (json.JSONDecodeError, KeyError):
        print("Error: Failed to decode JSON from fitness check response.")
        state['fitness_check_results'] = [{"error": "Failed to parse LLM response."}]

    return state


def archiving_node(state: DataScoutState) -> DataScoutState:
    print("---ARCHIVING---")

    config = load_claimify_config()
    if not config.scout_agent:
        print("Warning: Scout agent config not found. Skipping archiving.")
        return state

    writer_config = config.scout_agent.writer

    # Create directories if they don't exist
    os.makedirs(writer_config.tier1_path, exist_ok=True)
    os.makedirs(writer_config.tier2_path, exist_ok=True)

    # --- Tier 1: Raw Output ---
    raw_content = "\n".join(state.get("extraction_results", []))
    iteration = state.get("iteration", 0)
    if raw_content:
        tier1_filename = os.path.join(writer_config.tier1_path, f"raw_output_{iteration}.txt")
        with open(tier1_filename, "w") as f:
            f.write(raw_content)
        print(f"Tier 1 data written to {tier1_filename}")

    # --- Tier 2: Curated Output & Audit Trail ---
    fitness_results = state.get("fitness_check_results", [])
    if fitness_results and fitness_results[0].get("is_fit"):
        # Write to Tier 2
        tier2_filename = os.path.join(writer_config.tier2_path, f"curated_output_{iteration}.json")
        with open(tier2_filename, "w") as f:
            json.dump(state["extraction_results"], f, indent=2)
        print(f"Tier 2 data written to {tier2_filename}")

        # Update Audit Trail
        with open(writer_config.audit_trail_path, "a") as f:
            f.write(f"## Iteration {iteration}\n\n")
            f.write(f"**Source URL:** {state.get('search_results', [{}])[0].get('url', 'N/A')}\n")
            f.write(f"**Fitness Check Result:** {fitness_results[0].get('reasoning', 'N/A')}\n")
            f.write(f"**Score:** {fitness_results[0].get('score', 'N/A')}\n\n")
        print(f"Audit trail updated in {writer_config.audit_trail_path}")

    state['iteration'] = iteration + 1
    state['archived_data'].append({"message": "Data archived"})
    return state
