"""LangGraph Pruna agent — generates images and videos via Replicate.

4 tools matching the content-engine pipeline:
1. generate_image_fast → seed images, quick iterations
2. generate_image      → high quality production images
3. edit_image          → transform existing images
4. generate_video      → animate images into talking videos
"""

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv

from src.tools.replicate_tool import (
    generate_image_fast,
    generate_image,
    edit_image,
    generate_video,
)

load_dotenv()

# ── State ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]


# ── Tools ─────────────────────────────────────────────────────────

tools = [generate_image_fast, generate_image, edit_image, generate_video]
tool_node = ToolNode(tools)

# ── LLM ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Pruna AI video production assistant. You help users generate images and videos using Pruna's AI models via Replicate.

You have 4 tools:

1. generate_image_fast — Fast seed image generation. Use for quick character/scene creation. ~$0.005.
2. generate_image — Higher quality image. Use when user wants polished output. ~$0.005.
3. edit_image — Edit an existing image (change outfit, background, lighting). Needs an image URL from a previous generation. ~$0.01.
4. generate_video — Animate an image into a talking video with voice. Needs a seed image URL. $0.02/sec.

WORKFLOW for creating a talking video:
1. First generate a seed image with generate_image_fast
2. Then pass that image URL to generate_video with a voice prompt

PROMPT BEST PRACTICES (from our production pipeline):
- For images: include person description, clothing, setting, lighting. Add "NOT a model, NOT glamorous" for realistic look.
- For videos: keep prompts minimal. Scene + camera + "She/He says: 'text'". No gesture instructions — Pruna decides body language naturally.
- For edits: describe WHAT CHANGES, not the full scene. Always add "Preserve facial features, keep same person."

When a user asks for a video, ALWAYS generate the seed image first, then use that URL for the video."""

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
).bind_tools(tools)


# ── Nodes ─────────────────────────────────────────────────────────


def call_model(state: AgentState) -> dict:
    """Call Claude with the system prompt and current messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the agent should call a tool or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── Graph ─────────────────────────────────────────────────────────


def create_agent():
    """Build and compile the Pruna agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Run ───────────────────────────────────────────────────────────

agent = create_agent()


async def run_agent(user_message: str) -> List[BaseMessage]:
    """Run the agent with a user message and return all messages."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}
    )
    return result["messages"]
