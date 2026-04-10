"""LangGraph video agent — generates images and videos via Replicate."""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv

from src.tools.replicate_tool import generate_image, generate_video

load_dotenv()

# ── State ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


# ── Tools ─────────────────────────────────────────────────────────

tools = [generate_image, generate_video]
tool_node = ToolNode(tools)

# ── LLM ──────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(tools)


# ── Nodes ─────────────────────────────────────────────────────────


def call_model(state: AgentState) -> dict:
    """Call Claude with the current messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the agent should call a tool or finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ── Graph ─────────────────────────────────────────────────────────


def create_agent():
    """Build and compile the video agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────

agent = create_agent()


async def run_agent(user_message: str) -> list[BaseMessage]:
    """Run the agent with a user message and return all messages."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}
    )
    return result["messages"]
