# nodes.py
"""
Agent nodes for the Data Scout graph.
"""
import pydantic
from typing import List, Dict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import DataScoutState
from .tools import get_tools_for_role
from ..config import ClaimifyConfig, load_claimify_config

# --- LLM and Agent Creation ---

def create_llm(config: ClaimifyConfig, role: str) -> ChatOpenAI:
    """Creates a configured ChatOpenAI instance for a given agent role."""
    # This is a simplified config lookup. A real implementation would have more
    # detailed, role-specific settings in the config.yaml.
    default_model = config.default_model or "gpt-4o-mini"
    temperature = config.temperature or 0.1
    max_tokens = config.max_tokens or 2000

    return ChatOpenAI(model=default_model, temperature=temperature, max_tokens=max_tokens)

def create_agent_runnable(llm: ChatOpenAI, system_prompt: str, role: str):
    """Factory to create a new agent node's runnable."""
    tools = get_tools_for_role(role)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

# --- Pydantic Models for Structured Output ---

class SupervisorDecision(pydantic.BaseModel):
    """The supervisor's decision on the next agent to run."""
    next_agent: str = pydantic.Field(
        description="The name of the next agent to run. Must be one of 'research', 'archive', 'fitness', or 'end'."
    )

# --- Agent Node Implementations ---

def supervisor_node(state: DataScoutState) -> Dict:
    """The supervisor node, responsible for routing tasks using structured output."""
    config = load_claimify_config()
    system_prompt = """You are the supervisor of a team of agents performing data scouting.
Your role is to analyze the user's request, the conversation history, and the current state, then decide which agent should act next.

The available agents and their roles are:
- research: Finds information from the web using search tools. Call this agent when the user asks a question or requests information.
- archive: Writes information to files and updates the audit trail. Call this agent when research is complete and the findings need to be saved.
- fitness: Evaluates the quality and relevance of the research findings. Call this agent after the research agent has produced some findings.
- end: If the user's request has been fully addressed and no more actions are needed, you can end the conversation.

Here is the current state:
- Task Queue: {task_queue}
- Research Findings: {research_findings}

Based on the latest user message and the current state, decide which agent should act next.
Your response MUST be a JSON object matching the required schema."""

    llm = create_llm(config, "supervisor").with_structured_output(SupervisorDecision)

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(
        task_queue=str(state['task_queue']),
        research_findings=str(state['research_findings'])
    )

    agent = agent_prompt | llm
    result = agent.invoke({"messages": state["messages"]})
    return {"next_agent": result.next_agent.lower()}


def research_node(state: DataScoutState) -> Dict:
    """The research node, responsible for finding information."""
    config = load_claimify_config()
    system_prompt = """You are a professional researcher. Your goal is to find information to answer the user's request, which is in the message history.
Review the conversation history. If the user has a specific question, use your tools to find the answer.
If the user has a general topic, use your tools to find relevant documentation or articles.
You can use `url_to_markdown` for single pages or `documentation_crawler` for entire documentation sites.
Once you have gathered the information, another agent will handle saving it. Just return the results of your research."""
    llm = create_llm(config, "research")
    agent_runnable = create_agent_runnable(llm, system_prompt, "research")
    result = agent_runnable.invoke(state)
    return {"messages": [result]}


def archive_node(state: DataScoutState) -> Dict:
    """The archive node, responsible for saving data and updating the audit trail."""
    config = load_claimify_config()
    system_prompt = """You are a meticulous archivist. Your role is to save research findings and maintain a clean audit trail.
You have two primary tools:
1. `write_file`: To save content to a file. You should generate a suitable filepath, e.g., `output/research_results.md`.
2. `append_to_pedigree`: To record how the data was obtained. The markdown for this entry is often provided by other agents.

When another agent provides you with content to save (`full_markdown` from the crawler or `markdown` from the url tool), you MUST:
1. Call `write_file` to save the content.
2. Call `append_to_pedigree` with the `pedigree_entry` that was provided along with the content."""
    llm = create_llm(config, "archive")
    agent_runnable = create_agent_runnable(llm, system_prompt, "archive")
    result = agent_runnable.invoke({"messages": state["messages"]})

    if result.tool_calls:
        for tool_call in result.tool_calls:
            if tool_call["name"] == "append_to_pedigree":
                tool_call["args"]["pedigree_path"] = state["pedigree_path"]

    return {"messages": [result]}


def fitness_node(state: DataScoutState) -> Dict:
    """The fitness node, responsible for evaluating content."""
    config = load_claimify_config()
    system_prompt = """You are a quality assurance agent. Your role is to evaluate the quality and relevance of research findings.
Look at the last message in the conversation. If it is a research finding, evaluate it against the original user request.
Respond with a critique, suggestions for improvement, or an approval. For example:
- "The research on LangGraph is good, but it's missing details about the checkpointer."
- "The article about web scraping is not relevant to the user's request about data processing."
- "The findings are excellent and directly answer the user's question."
"""
    llm = create_llm(config, "fitness")
    agent_runnable = create_agent_runnable(llm, system_prompt, "fitness")
    result = agent_runnable.invoke(state)
    return {"messages": [result]}
