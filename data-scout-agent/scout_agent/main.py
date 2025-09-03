# main.py
"""
CLI entry point for the Data Scout Agent.
"""
import os
import uuid
import litellm
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from scout_agent.graph import build_graph
from scout_agent.tools import _HAVE_LIBCRAWLER

# Load environment variables from .env file
load_dotenv()

def setup_observability():
    """Configure litellm for LangSmith tracing if API keys are present."""
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ["LITELLM_CALLBACK"] = "langsmith"
        litellm.success_callback = ["langsmith"]
        print("✅ LangSmith tracing enabled.")
    else:
        print("⚠️ LangSmith tracing is not configured. Set LANGCHAIN_API_KEY to enable.")

def run():
    """Main function to run the CLI interactive loop."""
    setup_observability()

    if not _HAVE_LIBCRAWLER:
        print("\n⚠️  Warning: `libcrawler` is not installed.")
        print("Crawler functionality will be disabled.")
        print("To enable, run: pip install -e .[crawler]\n")

    app = build_graph()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"🤖 Data Scout Agent is ready. Starting new thread: {thread_id}")

    # Check if state for this thread already exists. If not, create it.
    thread_state = app.get_state(config)
    if not thread_state or not thread_state.values().get("messages"):
        print("   -> Creating new state for this thread.")
        initial_state = {
            "messages": [],
            "task_queue": [],
            "research_findings": [],
            "pedigree_path": "output/pedigree.md",
        }
        app.update_state(config, initial_state)

    while True:
        try:
            user_input = input(">> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            inputs = {"messages": [HumanMessage(content=user_input)]}

            for event in app.stream(inputs, config=config):
                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        print("\n🏁 Agent has finished the task.")
                        continue

                    print(f"-> Executing Node: {node_name.upper()}")
                    if "messages" in node_output:
                        message = node_output["messages"][-1]
                        if message.tool_calls:
                            tool_calls_str = ", ".join([f"{tc['name']}(...)" for tc in message.tool_calls])
                            print(f"   - Decided to call tools: {tool_calls_str}")
                        else:
                            print(f"   - Responded: {message.content}")
                    else:
                        print(f"   - Output: {str(node_output)[:300]}...")
                    print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    run()
