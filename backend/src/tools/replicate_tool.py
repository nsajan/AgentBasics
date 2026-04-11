"""Pruna AI tools for LangGraph agents via Replicate.

4 tools matching the 4 Pruna models used in content-engine:
1. generate_image_fast  → prunaai/z-image-turbo (fast, cheap, good for seeds)
2. generate_image       → prunaai/p-image (higher quality, slower)
3. edit_image           → prunaai/p-image-edit (transform existing image)
4. generate_video       → prunaai/p-video (animate image with voice)
"""

import os
import base64
import replicate
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def generate_image_fast(prompt: str) -> str:
    """Generate a fast image using Pruna z-image-turbo. Best for seed images, character generation, and quick iterations.

    Cost: ~$0.005 per image. Speed: <1 second.

    Args:
        prompt: Detailed description of the image. Include person description, clothing, setting, lighting, camera angle. End with "9:16 vertical" for portrait.

    Returns:
        URL of the generated image.
    """
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
    url = output[0] if isinstance(output, list) else output
    return str(url)


@tool
def generate_image(prompt: str, aspect_ratio: str = "9:16") -> str:
    """Generate a high-quality image using Pruna p-image. Better quality than fast mode, good for final production images.

    Cost: ~$0.005 per image. Speed: ~2-3 seconds.

    Args:
        prompt: Detailed description of the image.
        aspect_ratio: Aspect ratio - "9:16" (portrait), "16:9" (landscape), "1:1" (square).

    Returns:
        URL of the generated image.
    """
    output = replicate.run(
        "prunaai/p-image",
        input={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "prompt_upsampling": False,
        },
    )
    url = output[0] if isinstance(output, list) else output
    return str(url)


@tool
def edit_image(image_url: str, edit_prompt: str, aspect_ratio: str = "9:16") -> str:
    """Edit an existing image using Pruna p-image-edit. Change outfits, backgrounds, lighting, or add objects while preserving the person's identity.

    Cost: ~$0.01 per edit. Speed: <1 second.

    Always append "Preserve facial features, keep same person" to your edit prompt.

    Args:
        image_url: URL of the image to edit.
        edit_prompt: What to change. e.g. "Change background to a beach at sunset. Preserve facial features, keep same person."
        aspect_ratio: Aspect ratio for output.

    Returns:
        URL of the edited image.
    """
    # Download and base64 encode the image
    import urllib.request
    with urllib.request.urlopen(image_url) as response:
        image_data = response.read()
    b64 = "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")

    output = replicate.run(
        "prunaai/p-image-edit",
        input={
            "images": [b64],
            "prompt": edit_prompt,
            "aspect_ratio": aspect_ratio,
        },
    )
    url = output[0] if isinstance(output, list) else output
    return str(url)


@tool
def generate_video(prompt: str, image_url: str = "", duration: int = 10) -> str:
    """Generate a video using Pruna p-video. Can animate a seed image into a talking video with voice.

    Cost: $0.02/sec at 720p. A 10s video costs $0.20.

    IMPORTANT: For talking head videos, end the prompt with: She/He says: "the spoken text"
    Keep prompts minimal - just scene description + camera + speech. No gesture instructions.

    Pruna limits:
    - 5-20 seconds per clip. Sweet spot is 8-12s.
    - Under 5s: video freezes. Over 15s: quality degrades.
    - ~2.5 words per second for speech.

    Args:
        prompt: Scene description + voice direction. Example: "Medium close-up at a dark studio desk, slow push-in, warm tungsten lighting. She says: 'Hello world this is a test.'"
        image_url: Optional seed image URL for character consistency. If provided, the video will animate this person.
        duration: Video length in seconds (5-20, default 10).

    Returns:
        URL of the generated video (MP4).
    """
    input_params = {
        "prompt": prompt,
        "duration": min(20, max(5, duration)),
        "resolution": "720p",
        "aspect_ratio": "9:16",
        "draft": False,
        "save_audio": True,
        "prompt_upsampling": False,
    }

    if image_url:
        import urllib.request
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
        input_params["image"] = "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")

    output = replicate.run("prunaai/p-video", input=input_params)
    url = output if isinstance(output, str) else str(output)
    return url
