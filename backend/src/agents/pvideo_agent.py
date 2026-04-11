"""Sophisticated P-Video Agent — Script to finished video.

Takes a regular script as input. Understands Pruna limitations deeply.
Plans clip structure, generates seed image, creates clips, stitches.
Rejects scripts that require too much motion.

Graph: PLAN → GENERATE_IMAGE → GENERATE_CLIPS → STITCH → DONE
"""

import os
import json
import base64
import urllib.request
import subprocess
import tempfile
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import replicate

load_dotenv()

# ── Pruna Knowledge Base (baked into agent) ──────────────────────

PRUNA_KNOWLEDGE = """You are an expert video production agent that uses Pruna AI models via Replicate.

═══ PRUNA P-VIDEO HARD LIMITS ═══
- Duration: 5-20 seconds per clip. HARD LIMIT.
- Under 5s: video freezes, audio plays but no visible motion. UNUSABLE.
- 8-12s: sweet spot. Best quality.
- 13-15s: acceptable, slight quality drop.
- Over 15s: noticeable quality degradation. Avoid.
- Over 20s: will fail or produce artifacts.
- Resolution: 720p or 1080p. Use 720p for cost efficiency.
- Aspect ratio: 9:16 (portrait/vertical), 16:9, 1:1

═══ SPEECH & TIMING ═══
- Speech rate: ~2.5 words per second
- 20 words = 8s clip, 25 words = 10s, 30 words = 12s
- Short sentences with PERIODS create natural pauses. Commas DON'T pause.
- Voice direction: prompt MUST end with She/He says: "[text]"
- P-Video reads at constant pace. More words = teleprompter energy.

═══ WHAT PRUNA CAN DO (MOTION) ═══
GOOD (stationary subjects):
- Talking head at a desk — gestures, head movements, facial expressions
- Sitting person — natural upper body movement
- Standing person — weight shifting, hand gestures
- Wind in hair, breathing, blinking — subtle motion
- Slight camera movements — push-in, pull-back, dolly

BAD (locomotion / complex motion — REJECT THESE):
- Walking, running, dancing — character stays anchored to starting pose
- Picking up objects, reaching for things — arm movement is unreliable
- Multiple people interacting — faces merge/drift
- Scene transitions within a single clip — impossible
- Rapid head turns or looking away then back — face consistency breaks
- Sports, cooking, driving — any complex physical action

═══ PROMPT BEST PRACTICES ═══
- Keep prompts MINIMAL. Scene + camera + "She says: text". Nothing else.
- NO gesture instructions — Pruna decides body language from the words naturally.
- NO energy descriptors ("says casually", "says excitedly") — they cause inconsistent energy between clips.
- Different camera angle per clip for visual variety: push-in, dolly right, pull-back.
- ALWAYS include room/lighting: "warm tungsten lamp light from the left, dark gray walls, handheld documentary camera feel"

═══ MULTI-CLIP STITCHING ═══
For scripts longer than 20s, split into multiple clips and stitch:
- Each clip: 8-12s (sweet spot)
- Same seed image for all clips (character consistency)
- Dead air trimming: Pruna clips have 0.3-1.0s silence at start/end. Trim using speech timestamps.
- Different camera angle per clip creates natural "edit" feel
- FFmpeg concat for stitching: ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4

═══ SEED IMAGE ═══
- Always generate a seed image first via z-image-turbo
- The seed image anchors the character's appearance across ALL clips
- Anti-beauty prompts for realism: "visible pores, no makeup, slightly wrinkled clothing, NOT a model"
- Position eyes at 40% from top of frame (leave room for text overlays above head)
"""

# ── State ─────────────────────────────────────────────────────────


class VideoState(TypedDict):
    # Input
    script: str
    image_url: Optional[str]  # optional pre-existing seed image

    # Plan (set by PLAN node)
    status: str  # "planning" | "generating" | "stitching" | "done" | "rejected"
    rejection_reason: Optional[str]
    plan: Optional[dict]  # { clips: [{text, duration, prompt}], total_duration, seed_image_prompt }

    # Generation (set by GENERATE nodes)
    seed_image_url: Optional[str]
    clip_urls: List[str]  # URLs of generated clip videos
    clip_paths: List[str]  # local paths after download

    # Output
    final_video_path: Optional[str]
    final_video_url: Optional[str]
    messages: List[str]  # progress log


# ── LLM ──────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


# ── PLAN Node ─────────────────────────────────────────────────────

def plan_video(state: VideoState) -> dict:
    """Analyze script, detect motion issues, plan clip structure."""

    script = state["script"]
    word_count = len(script.split())

    response = llm.invoke([
        SystemMessage(content=PRUNA_KNOWLEDGE),
        HumanMessage(content=f"""Analyze this script and create a video production plan.

SCRIPT:
"{script}"

TASKS:
1. Check if the script requires motion that Pruna CANNOT do (walking, running, picking up objects, multiple people, scene changes). If so, set rejected=true and explain why.

2. If acceptable, split into clips:
   - Count words, calculate duration at 2.5 words/sec
   - Each clip: 8-12s max. If total > 12s, split into multiple clips.
   - Never split mid-sentence. Break at period boundaries.
   - Assign different camera angles per clip.

3. Write the Pruna prompt for each clip:
   - Scene + camera + room lighting + "She says: [text]"
   - No gestures, no energy descriptors
   - Room: "warm tungsten lamp light from the left, dark gray walls, handheld documentary camera feel"

4. Write a seed image prompt for the presenter.

Return ONLY valid JSON:
{{
  "rejected": false,
  "rejection_reason": null,
  "seed_image_prompt": "full prompt for z-image-turbo",
  "clips": [
    {{"text": "what she says in this clip", "duration": 10, "prompt": "full Pruna prompt ending with She says: ..."}}
  ],
  "total_duration": 25,
  "word_count": 60,
  "clip_count": 3
}}

If rejected:
{{
  "rejected": true,
  "rejection_reason": "This script requires walking which Pruna cannot do. Suggest rewriting to a stationary talking head format."
}}""")
    ])

    try:
        text = response.content
        if isinstance(text, list):
            text = " ".join(b.get("text", "") for b in text if isinstance(b, dict) and b.get("type") == "text")

        # Extract JSON
        json_match = None
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            json_match = match.group(1).strip()
        else:
            # Try parsing the whole response as JSON
            json_match = text.strip()

        plan = json.loads(json_match)

        if plan.get("rejected"):
            return {
                "status": "rejected",
                "rejection_reason": plan.get("rejection_reason", "Script not suitable for Pruna"),
                "plan": plan,
                "messages": state.get("messages", []) + [f"REJECTED: {plan.get('rejection_reason')}"],
            }

        return {
            "status": "generating",
            "plan": plan,
            "messages": state.get("messages", []) + [
                f"Plan: {plan.get('clip_count')} clips, {plan.get('total_duration')}s total, {plan.get('word_count')} words"
            ],
        }
    except Exception as e:
        return {
            "status": "rejected",
            "rejection_reason": f"Failed to parse plan: {str(e)}",
            "messages": state.get("messages", []) + [f"Plan error: {str(e)}"],
        }


# ── GENERATE IMAGE Node ──────────────────────────────────────────

def generate_seed_image(state: VideoState) -> dict:
    """Generate the seed character image."""

    if state.get("image_url"):
        return {
            "seed_image_url": state["image_url"],
            "messages": state.get("messages", []) + ["Using provided seed image"],
        }

    prompt = state["plan"]["seed_image_prompt"]

    output = replicate.run(
        "prunaai/z-image-turbo",
        input={
            "prompt": prompt,
            "width": 576,
            "height": 1024,
            "num_inference_steps": 8,
            "guidance_scale": 0,
            "output_format": "jpg",
            "output_quality": 95,
        },
    )
    url = output[0] if isinstance(output, list) else str(output)

    return {
        "seed_image_url": url,
        "messages": state.get("messages", []) + [f"Seed image generated: {url}"],
    }


# ── GENERATE CLIPS Node ──────────────────────────────────────────

def generate_clips(state: VideoState) -> dict:
    """Generate P-Video clips for each segment."""

    seed_url = state["seed_image_url"]
    clips = state["plan"]["clips"]

    # Download and base64 encode seed image
    with urllib.request.urlopen(seed_url) as response:
        image_data = response.read()
    image_b64 = "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")

    clip_urls = []
    messages = list(state.get("messages", []))

    for i, clip in enumerate(clips):
        messages.append(f"Generating clip {i+1}/{len(clips)} ({clip['duration']}s)...")

        output = replicate.run(
            "prunaai/p-video",
            input={
                "prompt": clip["prompt"],
                "image": image_b64,
                "duration": clip["duration"],
                "resolution": "720p",
                "aspect_ratio": "9:16",
                "draft": False,
                "save_audio": True,
                "prompt_upsampling": False,
            },
        )
        url = output if isinstance(output, str) else str(output)
        clip_urls.append(url)
        messages.append(f"Clip {i+1} ready: {url}")

    return {
        "clip_urls": clip_urls,
        "messages": messages,
    }


# ── STITCH Node ───────────────────────────────────────────────────

def stitch_clips(state: VideoState) -> dict:
    """Download clips and stitch with FFmpeg."""

    clip_urls = state["clip_urls"]
    messages = list(state.get("messages", []))

    if len(clip_urls) == 1:
        # Single clip — no stitching needed
        return {
            "status": "done",
            "final_video_url": clip_urls[0],
            "messages": messages + ["Single clip — no stitching needed"],
        }

    # Download all clips
    tmp_dir = tempfile.mkdtemp(prefix="pvideo_")
    clip_paths = []

    for i, url in enumerate(clip_urls):
        path = os.path.join(tmp_dir, f"clip-{i+1}.mp4")
        urllib.request.urlretrieve(url, path)
        clip_paths.append(path)
        messages.append(f"Downloaded clip {i+1}")

    # Create concat list
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    # Stitch with FFmpeg
    output_path = os.path.join(tmp_dir, "final.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-sn", output_path],
            capture_output=True, check=True, timeout=60,
        )
        messages.append(f"Stitched {len(clip_paths)} clips → {output_path}")
    except FileNotFoundError:
        # ffmpeg not in PATH — return last clip URL as fallback
        messages.append("FFmpeg not found — returning individual clip URLs")
        return {
            "status": "done",
            "final_video_url": clip_urls[-1],
            "clip_paths": clip_paths,
            "messages": messages,
        }
    except subprocess.CalledProcessError as e:
        messages.append(f"FFmpeg error: {e.stderr.decode()[:200]}")
        return {
            "status": "done",
            "final_video_url": clip_urls[0],
            "clip_paths": clip_paths,
            "messages": messages,
        }

    return {
        "status": "done",
        "final_video_path": output_path,
        "final_video_url": clip_urls[0],  # also keep remote URL
        "clip_paths": clip_paths,
        "messages": messages,
    }


# ── Routing ───────────────────────────────────────────────────────

def route_after_plan(state: VideoState) -> str:
    if state.get("status") == "rejected":
        return END
    return "generate_image"


def route_after_clips(state: VideoState) -> str:
    if len(state.get("clip_urls", [])) > 1:
        return "stitch"
    return "finish"


def finish(state: VideoState) -> dict:
    """Mark as done for single-clip videos."""
    return {
        "status": "done",
        "final_video_url": state["clip_urls"][0] if state.get("clip_urls") else None,
        "messages": state.get("messages", []) + ["Done!"],
    }


# ── Graph ─────────────────────────────────────────────────────────

def create_pvideo_agent():
    graph = StateGraph(VideoState)

    graph.add_node("plan", plan_video)
    graph.add_node("generate_image", generate_seed_image)
    graph.add_node("generate_clips", generate_clips)
    graph.add_node("stitch", stitch_clips)
    graph.add_node("finish", finish)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", route_after_plan, {"generate_image": "generate_image", END: END})
    graph.add_edge("generate_image", "generate_clips")
    graph.add_conditional_edges("generate_clips", route_after_clips, {"stitch": "stitch", "finish": "finish"})
    graph.add_edge("stitch", END)
    graph.add_edge("finish", END)

    return graph.compile()


pvideo_agent = create_pvideo_agent()


# ── Public API ────────────────────────────────────────────────────

async def create_video(script: str, image_url: str = None) -> dict:
    """Run the P-Video agent pipeline.

    Args:
        script: The text the presenter should say.
        image_url: Optional seed image URL. If not provided, one will be generated.

    Returns:
        dict with: status, plan, seed_image_url, clip_urls, final_video_url, messages
    """
    result = await pvideo_agent.ainvoke({
        "script": script,
        "image_url": image_url,
        "status": "planning",
        "rejection_reason": None,
        "plan": None,
        "seed_image_url": None,
        "clip_urls": [],
        "clip_paths": [],
        "final_video_path": None,
        "final_video_url": None,
        "messages": [],
    })
    return result
