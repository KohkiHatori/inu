"""Generate a one-off "subscribe" shot with the three characters.

Content: Three characters (Pop, Pup-A, Pup-B) holding a board written "SUBSCRIBE" in red.
Layout: One puppy on each side of the board, Pop holding the board.
Output path: output/subscribe.mp4
"""

import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

MODEL = "veo-3.1-generate-preview"
RESOLUTION = "1080p"
ASPECT_RATIO = "16:9"
DURATION = 8

REF_IMAGE_PATHS: list[str] = [
    "assets/ref/channels/pup-pop-pup/pop.png",
    "assets/ref/channels/pup-pop-pup/pupa.png",
    "assets/ref/channels/pup-pop-pup/pupb.png",
]

HERO_PREFIX = (
    "Characters: Pop, a photorealistic calm adult golden retriever with warm amber eyes "
    "and a broad snout; Pup-A, a photorealistic small fluffy golden puppy with oversized "
    "paws and floppy ears; Pup-B, a photorealistic slightly smaller golden puppy with a "
    "white chest patch and a wagging tail. The dogs have real fur, real eyes, and real anatomy. "
    "All objects, props, vehicles, and environmental elements around them are stylized like "
    "colorful 3D CG cartoon items — shiny, rounded, oversized, and vibrant. "
)

# Modified style suffix to allow text for the subscribe sign
STYLE_SUFFIX = (
    " 4K, shallow depth of field, warm vibrant lighting, smooth gentle camera motion, "
    "adorable and whimsical tone, no humans. "
    "Audio: natural ambient sounds only — animal sounds, gentle whooshes, soft cartoon-like "
    "sound effects. No music, no instruments, no soundtrack."
)

DESCRIPTION = (
    "Three dogs in a sunny, vibrant green grassy field under a bright blue sky with fluffy white clouds. "
    "In the center, Pop (the adult golden retriever) is sitting up and holding a large, clean white rectangular sign board "
    "with his front paws. The sign has the word 'SUBSCRIBE' written clearly in large, bold red capital letters. "
    "On the left side of the sign, Pup-A (the small fluffy puppy) is sitting in the grass, looking at the camera with a happy expression. "
    "On the right side of the sign, Pup-B (the puppy with the white chest patch) is also sitting in the grass, looking at the camera and wagging its tail. "
    "The background features gently swaying green grass and stylized colorful flowers. "
    "The lighting is bright and cheerful, suggesting a perfect sunny day."
)

def make_ref_image_config(image_bytes: bytes, mime_type: str) -> types.VideoGenerationReferenceImage:
    return types.VideoGenerationReferenceImage(
        image=types.Image(
            image_bytes=base64.b64encode(image_bytes).decode("utf-8"),
            mime_type=mime_type,
        ),
        reference_type="asset",
    )

def _poll_and_download(client: genai.Client, operation):
    """Poll until done, then download and return the video object."""
    while not operation.done:
        print("  Polling...")
        time.sleep(15)
        operation = client.operations.get(operation)

    if operation.response is None:
        error = getattr(operation, "error", None)
        raise RuntimeError(f"Video generation failed (no response): {error}")

    if not operation.response.generated_videos:
        raise RuntimeError(f"Video generation failed (no videos returned): {operation}")

    video = operation.response.generated_videos[0]
    if not USE_VERTEX:
        client.files.download(file=video.video)
    return video.video

def generate_subscribe_shot() -> None:
    client = genai.Client()

    # Verify ref images
    char_refs = []
    for img_path in REF_IMAGE_PATHS:
        path = Path(img_path)
        if not path.exists():
            print(f"Error: reference image not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
        char_refs.append(make_ref_image_config(path.read_bytes(), mime))

    print(f"Loaded {len(char_refs)} character reference images.")

    prompt = f"{HERO_PREFIX}{DESCRIPTION}{STYLE_SUFFIX}"
    print(f"Prompt: {prompt}")

    out_path = Path("output/subscribe.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating subscribe shot...")

    config_kwargs = {
        "aspect_ratio": ASPECT_RATIO,
        "number_of_videos": 1,
        "duration_seconds": DURATION,
        "resolution": RESOLUTION,
        "reference_images": char_refs,
    }

    try:
        operation = client.models.generate_videos(
            model=MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(**config_kwargs),
        )
        video_file = _poll_and_download(client, operation)

        video_file.save(str(out_path))
        print(f"Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    except Exception as e:
        print(f"Error generating video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_subscribe_shot()
