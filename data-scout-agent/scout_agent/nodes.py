# nodes.py
"""
Agent nodes for the Data Scout graph.
"""
import pydantic
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from scout_agent.state import DataScoutState
from scout_agent.tools import get_tools_for_role

# A basic LLM for all agents to use
# In a real app, you might have different models for different agents
llm = ChatOpenAI(model="gpt-4o")

class SupervisorDecision(pydantic.BaseModel):
    """The supervisor's decision on the next agent to run."""
    next_agent: str = pydantic.Field(
        description="The name of the next agent to run. Must be one of 'research', 'archive', 'fitness', or 'end'."
    )

def create_agent(llm: ChatOpenAI, system_prompt: str, role: str):
    """Factory to create a new agent node."""
    tools = get_tools_for_role(role)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        agent = prompt | llm.bind_tools(tools)
    else:
        agent = prompt | llm
    return agent

def supervisor_node(state: DataScoutState):
    """The supervisor node, responsible for routing tasks using structured output."""
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
Your response MUST be a JSON object matching the required schema.
"""
    structured_llm = llm.with_structured_output(SupervisorDecision)

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(
        task_queue=str(state['task_queue']),
        research_findings=str(state['research_findings'])
    )

    agent = agent_prompt | structured_llm
    result = agent.invoke({"messages": state["messages"]})

    return {"next_agent": result.next_agent.lower()}

def research_node(state: DataScoutState):
    """The research node, responsible for finding information."""
    system_prompt = """You are a professional researcher. Your goal is to find information to answer the user's request, which is in the message history.
Review the conversation history. If the user has a specific question, use your tools to find the answer.
If the user has a general topic, use your tools to find relevant documentation or articles.
You can use `url_to_markdown` for single pages or `documentation_crawler` for entire documentation sites.
Once you have gathered the information, another agent will handle saving it. Just return the results of your research."""
    agent = create_agent(llm, system_prompt, "research")
    result = agent.invoke(state)
    return {"messages": [result]}

def archive_node(state: DataScoutState):
    """The archive node, responsible for saving data and updating the audit trail."""
    system_prompt = """You are a meticulous archivist. Your role is to save research findings and maintain a clean audit trail.
You have two primary tools:
1. `write_file`: To save content to a file. You should generate a suitable filepath, e.g., `output/research_results.md`.
2. `append_to_pedigree`: To record how the data was obtained. The markdown for this entry is often provided by other agents.

When another agent provides you with content to save (`full_markdown` from the crawler or `markdown` from the url tool), you MUST:
1. Call `write_file` to save the content.
2. Call `append_to_pedigree` with the `pedigree_entry` that was provided along with the content.
"""
    agent = create_agent(llm, system_prompt, "archive")
    result = agent.invoke({"messages": state["messages"]})

    # The agent returns an AIMessage with tool calls. We need to modify the
    # tool call for `append_to_pedigree` to inject the path from the state.
    if result.tool_calls:
        for tool_call in result.tool_calls:
            if tool_call["name"] == "append_to_pedigree":
                tool_call["args"]["pedigree_path"] = state["pedigree_path"]

    return {"messages": [result]}


def fitness_node(state: DataScoutState):
    """The fitness node, responsible for evaluating content."""
    system_prompt = """You are a quality assurance agent. Your role is to evaluate the quality and relevance of research findings.
Look at the last message in the conversation. If it is a research finding, evaluate it against the original user request.
Respond with a critique, suggestions for improvement, or an approval. For example:
- "The research on LangGraph is good, but it's missing details about the checkpointer."
- "The article about web scraping is not relevant to the user's request about data processing."
- "The findings are excellent and directly answer the user's question."
"""
    agent = create_agent(llm, system_prompt, "fitness")
    result = agent.invoke(state)
    return {"messages": [result]}
