"""FastAPI routes for the agent."""

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agents.video_agent import run_agent

app = FastAPI(title="Agent Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None


class ChatResponse(BaseModel):
    messages: List[ChatMessage]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get the full conversation back."""
    try:
        messages = await run_agent(request.message)

        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append(ChatMessage(role="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                # content can be a string or a list of blocks
                text_content = ""
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    text_content = " ".join(
                        block.get("text", "") for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                if text_content.strip():
                    result.append(ChatMessage(role="assistant", content=text_content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        result.append(
                            ChatMessage(
                                role="assistant",
                                content=f"Calling {tc['name']}...",
                                tool_name=tc["name"],
                            )
                        )
            elif isinstance(msg, ToolMessage):
                result.append(
                    ChatMessage(
                        role="tool",
                        content=msg.content,
                        tool_name=msg.name,
                    )
                )

        return ChatResponse(messages=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
