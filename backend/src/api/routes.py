"""FastAPI routes for the agent."""

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agents.video_agent import run_agent
from src.agents.pvideo_agent import create_video

app = FastAPI(title="Agent Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ── P-Video Agent endpoint ────────────────────────────────────────


class VideoRequest(BaseModel):
    script: str
    image_url: Optional[str] = None
    draft: bool = False


class VideoResponse(BaseModel):
    status: str
    rejection_reason: Optional[str] = None
    plan: Optional[dict] = None
    estimated_cost: Optional[float] = None
    seed_image_url: Optional[str] = None
    clip_urls: Optional[List[str]] = None
    final_video_url: Optional[str] = None
    messages: List[str] = []


@app.post("/video", response_model=VideoResponse)
async def video(request: VideoRequest):
    """Send a script to the P-Video agent. Returns a finished video.

    The agent will:
    1. Analyze the script and reject if it requires motion Pruna can't do
    2. Plan clip structure (duration, count, camera angles)
    3. Generate seed image (or use provided one)
    4. Generate P-Video clips
    5. Stitch if multiple clips

    Input: { "script": "the text to speak", "image_url": "optional seed image" }
    Output: { "status": "done|rejected", "final_video_url": "...", "plan": {...} }
    """
    try:
        result = await create_video(request.script, request.image_url, request.draft)
        return VideoResponse(
            status=result.get("status", "unknown"),
            rejection_reason=result.get("rejection_reason"),
            plan=result.get("plan"),
            estimated_cost=result.get("estimated_cost"),
            seed_image_url=result.get("seed_image_url"),
            clip_urls=result.get("clip_urls", []),
            final_video_url=result.get("final_video_url"),
            messages=result.get("messages", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
