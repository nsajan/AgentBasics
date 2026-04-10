"""Replicate API tool for LangGraph agents."""

import os
import replicate
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def generate_image(prompt: str, width: int = 576, height: int = 1024) -> str:
    """Generate an image using Replicate's Pruna AI model.

    Args:
        prompt: Text description of the image to generate.
        width: Image width in pixels (default 576).
        height: Image height in pixels (default 1024).

    Returns:
        URL of the generated image.
    """
    output = replicate.run(
        "prunaai/z-image-turbo",
        input={
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 8,
            "guidance_scale": 0,
            "output_format": "jpg",
            "output_quality": 95,
        },
    )
    url = output[0] if isinstance(output, list) else output
    return str(url)


@tool
def generate_video(prompt: str, duration: int = 10, image_url: str = "") -> str:
    """Generate a video using Replicate's Pruna P-Video model.

    Args:
        prompt: Text description + voice direction. End with She/He says: "text".
        duration: Video length in seconds (5-20).
        image_url: Optional seed image URL for character consistency.

    Returns:
        URL of the generated video.
    """
    input_params = {
        "prompt": prompt,
        "duration": duration,
        "resolution": "720p",
        "aspect_ratio": "9:16",
        "draft": False,
        "save_audio": True,
        "prompt_upsampling": False,
    }
    if image_url:
        input_params["image"] = image_url

    output = replicate.run("prunaai/p-video", input=input_params)
    url = output if isinstance(output, str) else str(output)
    return url
