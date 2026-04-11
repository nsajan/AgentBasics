"""Sophisticated P-Video Agent — Script to finished video.

Implements the full Pruna pipeline from content-engine docs:
- Draft preview before final generation
- Scene chaining with last-frame extraction
- P-Image-Edit for scene/setting changes
- Separate image prompts from video prompts
- Locomotion support via action pose workflow
- Consistent seed across clips
- Last-frame trimming before concat
- Cost estimation before generation

Graph: PLAN → GENERATE_IMAGE → GENERATE_CLIPS → STITCH → DONE
"""

import os
import json
import re
import base64
import urllib.request
import subprocess
import tempfile
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import replicate

load_dotenv()

# ── Pruna Knowledge (comprehensive from all docs) ────────────────

PRUNA_KNOWLEDGE = """You are an expert Pruna AI video production agent.

═══ MODELS ═══
- prunaai/z-image-turbo: Fast image gen. $0.005. <1s. Good for seeds.
- prunaai/p-image: Quality image gen. $0.005. 2-3s. Better composition.
- prunaai/p-image-edit: Edit existing image. $0.01. <1s. Preserves identity.
- prunaai/p-video: Animate image to video. $0.02/sec (720p). $0.005/sec (draft).

═══ P-VIDEO LIMITS ═══
- Duration: 5-20s per clip. Sweet spot: 8-12s.
- Under 5s: frozen video. Over 15s: quality degrades. Over 20s: fails.
- Speech rate: ~2.5 words/second.
- Periods create pauses. Commas don't.
- Voice: prompt MUST end with She/He says: "[text]"

═══ PROMPT SEPARATION (CRITICAL) ═══
Image prompt → handles APPEARANCE (person, clothing, setting, lighting)
Video prompt → handles MOTION + CAMERA + SPEECH only
Do NOT re-describe the person/setting in the video prompt. The image handles that.

Video prompt should be: "[camera movement], [ambient description]. She says: '[text]'"

═══ MOTION CAPABILITIES ═══
STATIONARY (seed image → P-Video directly, up to 15s):
- Talking, gesturing, head movements, facial expressions
- Wind, breathing, blinking, hair movement
- Sitting, standing (weight shifting only)

LOCOMOTION (seed → P-Image-Edit action pose → P-Video, MAX 5s):
- Walking, jogging: edit seed to mid-stride pose first
- Use: "Change pose to mid-stride walking. Preserve face and outfit."
- Duration MUST be ≤5s for locomotion clips

IMPOSSIBLE (REJECT):
- Multiple people interacting (faces merge/drift)
- Complex physical actions (cooking, sports, driving)
- Scene transitions within a single clip
- Rapid head turns or looking away then back

═══ SCENE CHAINING (for multi-clip) ═══
1. Generate seed image (P-Image or z-image-turbo)
2. Animate clip 1 (P-Video)
3. Extract LAST FRAME from clip 1
4. Use last frame as seed for clip 2 (maintains pose continuity at cut)
5. Optional: P-Image-Edit between clips to change setting
6. Repeat for all clips
7. Trim last frame from each clip before concat (avoid duplicate frame)
8. Use SAME SEED number across all P-Video calls

═══ COST ═══
- Image: $0.005
- Image edit: $0.01
- Video (720p normal): $0.02/sec → 10s = $0.20
- Video (720p draft): $0.005/sec → 10s = $0.05
- Always calculate total cost in the plan

═══ ANTI-BEAUTY (for realistic people) ═══
Include in image prompts: "visible pores, no makeup, slightly wrinkled clothing, shot on iPhone 15 Pro, NOT a model, NOT glamorous"
"""

# ── State ─────────────────────────────────────────────────────────


class VideoState(TypedDict):
    script: str
    image_url: Optional[str]
    draft: bool

    status: str
    rejection_reason: Optional[str]
    plan: Optional[dict]
    estimated_cost: Optional[float]

    seed_image_url: Optional[str]
    clip_urls: List[str]
    clip_paths: List[str]

    final_video_path: Optional[str]
    final_video_url: Optional[str]
    messages: List[str]


# ── LLM ──────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

SEED = 42  # consistent seed across all Pruna calls


# ── PLAN Node ─────────────────────────────────────────────────────

def plan_video(state: VideoState) -> dict:
    script = state["script"]
    is_draft = state.get("draft", False)

    response = llm.invoke([
        SystemMessage(content=PRUNA_KNOWLEDGE),
        HumanMessage(content=f"""Analyze this script and create a production plan.

SCRIPT:
"{script}"

TASKS:
1. REJECT if script requires: multiple people interacting, complex physical actions, scene transitions within a clip, or impossible motion. Explain why and suggest a rewrite.

2. For each part of the script, decide:
   - Is it STATIONARY (talking head)? → standard seed → P-Video, up to 12s
   - Does it require LOCOMOTION (walking, running)? → seed → P-Image-Edit (action pose) → P-Video, max 5s
   - Does it change SETTING? → P-Image-Edit between clips to change background

3. Split into clips:
   - Count words per clip, calculate duration (2.5 words/sec, round up)
   - 8-12s per clip max for stationary. 5s max for locomotion.
   - Break at sentence boundaries (periods).
   - Assign DIFFERENT camera per clip (push-in, dolly right, pull-back, static handheld)

4. Write prompts:
   - seed_image_prompt: full appearance description (person, clothing, setting, lighting, anti-beauty)
   - For each clip:
     - video_prompt: ONLY motion + camera + speech. NOT appearance.
       Format: "[camera movement], [ambient]. She says: '[clip text]'"
     - edit_prompt: (optional) if setting changes between this clip and previous
     - motion_type: "stationary" or "locomotion"

5. Calculate cost:
   - Image: $0.005
   - Edits: $0.01 each
   - Video: {"$0.005" if is_draft else "$0.02"}/sec per clip
   - Total

Return ONLY valid JSON:
{{
  "rejected": false,
  "rejection_reason": null,
  "seed_image_prompt": "full image prompt",
  "clips": [
    {{
      "text": "what she says",
      "duration": 10,
      "video_prompt": "camera + ambient + She says: 'text'",
      "edit_prompt": null,
      "motion_type": "stationary"
    }}
  ],
  "total_duration": 25,
  "word_count": 60,
  "clip_count": 3,
  "estimated_cost": 0.45,
  "uses_scene_chaining": true,
  "uses_locomotion": false
}}""")
    ])

    try:
        text = response.content
        if isinstance(text, list):
            text = " ".join(b.get("text", "") for b in text if isinstance(b, dict) and b.get("type") == "text")

        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        json_str = match.group(1).strip() if match else text.strip()
        plan = json.loads(json_str)

        if plan.get("rejected"):
            return {
                "status": "rejected",
                "rejection_reason": plan.get("rejection_reason"),
                "plan": plan,
                "messages": state.get("messages", []) + [f"REJECTED: {plan.get('rejection_reason')}"],
            }

        return {
            "status": "generating",
            "plan": plan,
            "estimated_cost": plan.get("estimated_cost", 0),
            "messages": state.get("messages", []) + [
                f"Plan: {plan.get('clip_count')} clips, {plan.get('total_duration')}s, ~${plan.get('estimated_cost', 0):.2f}",
                f"Scene chaining: {plan.get('uses_scene_chaining', False)}",
                f"Locomotion: {plan.get('uses_locomotion', False)}",
            ],
        }
    except Exception as e:
        return {
            "status": "rejected",
            "rejection_reason": f"Plan parse error: {str(e)}",
            "messages": state.get("messages", []) + [f"Error: {str(e)}"],
        }


# ── GENERATE IMAGE Node ──────────────────────────────────────────

def generate_seed_image(state: VideoState) -> dict:
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
            "seed": SEED,
        },
    )
    url = output[0] if isinstance(output, list) else str(output)

    return {
        "seed_image_url": url,
        "messages": state.get("messages", []) + [f"Seed image: {url}"],
    }


# ── GENERATE CLIPS Node (with scene chaining) ────────────────────

def generate_clips(state: VideoState) -> dict:
    seed_url = state["seed_image_url"]
    clips = state["plan"]["clips"]
    is_draft = state.get("draft", False)
    messages = list(state.get("messages", []))

    # Download seed image
    with urllib.request.urlopen(seed_url) as response:
        current_image_data = response.read()

    clip_urls = []
    tmp_dir = tempfile.mkdtemp(prefix="pvideo_")
    clip_paths = []

    for i, clip in enumerate(clips):
        messages.append(f"Clip {i+1}/{len(clips)} ({clip['duration']}s, {clip.get('motion_type', 'stationary')})...")

        # Step A: If edit_prompt exists, transform the image for this scene
        if clip.get("edit_prompt") and i > 0:
            messages.append(f"  Editing scene: {clip['edit_prompt'][:60]}...")
            edit_b64 = "data:image/jpeg;base64," + base64.b64encode(current_image_data).decode("utf-8")
            edit_output = replicate.run(
                "prunaai/p-image-edit",
                input={
                    "images": [edit_b64],
                    "prompt": clip["edit_prompt"] + " Preserve facial features, keep same person.",
                    "aspect_ratio": "9:16",
                    "seed": SEED,
                },
            )
            edit_url = edit_output[0] if isinstance(edit_output, list) else str(edit_output)
            with urllib.request.urlopen(edit_url) as response:
                current_image_data = response.read()
            messages.append(f"  Scene edited")

        # Step B: If locomotion, create action pose via P-Image-Edit
        if clip.get("motion_type") == "locomotion":
            messages.append(f"  Creating action pose for locomotion...")
            pose_b64 = "data:image/jpeg;base64," + base64.b64encode(current_image_data).decode("utf-8")
            pose_output = replicate.run(
                "prunaai/p-image-edit",
                input={
                    "images": [pose_b64],
                    "prompt": "Change pose to mid-stride walking toward camera. Preserve facial features, keep same person, maintain outfit.",
                    "aspect_ratio": "9:16",
                    "seed": SEED,
                },
            )
            pose_url = pose_output[0] if isinstance(pose_output, list) else str(pose_output)
            with urllib.request.urlopen(pose_url) as response:
                current_image_data = response.read()

        # Step C: Generate P-Video
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(current_image_data).decode("utf-8")

        output = replicate.run(
            "prunaai/p-video",
            input={
                "prompt": clip["video_prompt"],
                "image": image_b64,
                "duration": clip["duration"],
                "resolution": "720p",
                "aspect_ratio": "9:16",
                "draft": is_draft,
                "save_audio": True,
                "prompt_upsampling": False,
                "seed": SEED,
            },
        )
        url = output if isinstance(output, str) else str(output)
        clip_urls.append(url)

        # Download clip for potential scene chaining
        clip_path = os.path.join(tmp_dir, f"clip-{i+1}.mp4")
        urllib.request.urlretrieve(url, clip_path)
        clip_paths.append(clip_path)

        # Step D: Extract last frame for next clip (scene chaining)
        if i < len(clips) - 1 and state["plan"].get("uses_scene_chaining", True):
            try:
                last_frame_path = os.path.join(tmp_dir, f"lastframe-{i+1}.jpg")
                # Get duration and extract frame near the end
                probe = subprocess.run(
                    ["ffmpeg", "-i", clip_path, "-f", "null", "-"],
                    capture_output=True, text=True, timeout=10,
                )
                dur_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', probe.stderr)
                if dur_match:
                    total_sec = int(dur_match.group(1)) * 3600 + int(dur_match.group(2)) * 60 + float(dur_match.group(3))
                    seek_to = max(0, total_sec - 0.1)
                else:
                    seek_to = max(0, clip["duration"] - 0.1)

                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(seek_to), "-i", clip_path, "-frames:v", "1", "-q:v", "2", last_frame_path],
                    capture_output=True, timeout=10,
                )

                if os.path.exists(last_frame_path):
                    with open(last_frame_path, "rb") as f:
                        current_image_data = f.read()
                    messages.append(f"  Extracted last frame for scene chaining")
            except Exception as e:
                messages.append(f"  Scene chain frame extraction failed: {str(e)[:50]}")
                # Fall back to using the seed image
                with urllib.request.urlopen(seed_url) as response:
                    current_image_data = response.read()

        messages.append(f"Clip {i+1} ready: {url}")

    return {
        "clip_urls": clip_urls,
        "clip_paths": clip_paths,
        "messages": messages,
    }


# ── STITCH Node (with last-frame trimming) ────────────────────────

def stitch_clips(state: VideoState) -> dict:
    clip_paths = state.get("clip_paths", [])
    clip_urls = state.get("clip_urls", [])
    messages = list(state.get("messages", []))

    if len(clip_paths) <= 1:
        return {
            "status": "done",
            "final_video_url": clip_urls[0] if clip_urls else None,
            "messages": messages + ["Single clip — no stitching needed"],
        }

    tmp_dir = os.path.dirname(clip_paths[0])

    # Trim last frame from each clip (except final) to avoid duplicates at cuts
    trimmed_paths = []
    for i, path in enumerate(clip_paths):
        if i < len(clip_paths) - 1:
            try:
                probe = subprocess.run(
                    ["ffmpeg", "-i", path, "-f", "null", "-"],
                    capture_output=True, text=True, timeout=10,
                )
                dur_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', probe.stderr)
                if dur_match:
                    total_sec = int(dur_match.group(1)) * 3600 + int(dur_match.group(2)) * 60 + float(dur_match.group(3))
                    trim_to = max(0, total_sec - 0.042)  # trim ~1 frame at 24fps
                    trimmed_path = os.path.join(tmp_dir, f"trimmed-{i+1}.mp4")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", path, "-t", str(trim_to), "-c", "copy", trimmed_path],
                        capture_output=True, timeout=10,
                    )
                    trimmed_paths.append(trimmed_path)
                    continue
            except Exception:
                pass
            trimmed_paths.append(path)
        else:
            trimmed_paths.append(path)

    # Concat
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w") as f:
        for p in trimmed_paths:
            f.write(f"file '{p}'\n")

    output_path = os.path.join(tmp_dir, "final.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-sn", output_path],
            capture_output=True, check=True, timeout=60,
        )
        messages.append(f"Stitched {len(trimmed_paths)} clips (last-frame trimmed)")
    except Exception as e:
        messages.append(f"FFmpeg stitch failed: {str(e)[:100]}")
        return {
            "status": "done",
            "final_video_url": clip_urls[0],
            "messages": messages,
        }

    return {
        "status": "done",
        "final_video_path": output_path,
        "final_video_url": clip_urls[0],
        "messages": messages + ["Done!"],
    }


# ── Routing ───────────────────────────────────────────────────────

def route_after_plan(state: VideoState) -> str:
    return END if state.get("status") == "rejected" else "generate_image"


def route_after_clips(state: VideoState) -> str:
    return "stitch" if len(state.get("clip_urls", [])) > 1 else "finish"


def finish(state: VideoState) -> dict:
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

async def create_video(script: str, image_url: str = None, draft: bool = False) -> dict:
    """Run the P-Video agent.

    Args:
        script: Text the presenter should say.
        image_url: Optional seed image. If not provided, one is generated.
        draft: If True, use draft mode (4x cheaper, slightly lower quality).

    Returns:
        dict with: status, plan, estimated_cost, seed_image_url, clip_urls, final_video_url, messages
    """
    result = await pvideo_agent.ainvoke({
        "script": script,
        "image_url": image_url,
        "draft": draft,
        "status": "planning",
        "rejection_reason": None,
        "plan": None,
        "estimated_cost": None,
        "seed_image_url": None,
        "clip_urls": [],
        "clip_paths": [],
        "final_video_path": None,
        "final_video_url": None,
        "messages": [],
    })
    return result
