"""Generate video clips for each shot in a story YAML file.

Uses Veo 3.1 via the Gemini API to generate one clip per shot, serially.
Hero shots include a reference image for character consistency.
"""

import argparse
import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

MODEL = "veo-3.1-generate-preview"
RESOLUTION = "4k"  # "720p" | "1080p" | "4k"

HERO_PREFIX = (
    "Characters: Pop, a photorealistic calm adult golden retriever with warm amber eyes "
    "and a broad snout; Pup-A, a photorealistic small fluffy golden puppy with oversized "
    "paws and floppy ears; Pup-B, a photorealistic slightly smaller golden puppy with a "
    "white chest patch and a wagging tail. The dogs have real fur, real eyes, and real anatomy. "
    "All objects, props, vehicles, and environmental elements around them are stylized like "
    "colorful 3D CG cartoon items — shiny, rounded, oversized, and vibrant. "
)

STYLE_SUFFIX = (
    " 4K, shallow depth of field, warm vibrant lighting, smooth gentle camera motion, "
    "adorable and whimsical tone, no humans, no text. "
    "Audio: natural ambient sounds only — animal sounds, gentle whooshes, soft cartoon-like "
    "sound effects. No music, no instruments, no soundtrack."
)

FORMAT_CONFIG = {
    "normal": {"aspect_ratio": "16:9"},
    "short":  {"aspect_ratio": "9:16"},
}

POLL_INTERVAL_SECONDS = 15
DELAY_BETWEEN_SHOTS_SECONDS = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 60
MIN_VALID_BYTES = 1 * 1024 * 1024  # 1 MB — anything smaller is treated as a failed generation


def build_prompt(description: str) -> str:
    return f"{HERO_PREFIX}{description}{STYLE_SUFFIX}"


def generate_shot(
    client: genai.Client,
    prompt: str,
    ref_image_path: str | None,
    aspect_ratio: str,
    duration: int,
) -> bytes:
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": 1,
        "duration_seconds": duration,
        "resolution": RESOLUTION,
    }

    if ref_image_path:
        image_bytes = Path(ref_image_path).read_bytes()
        mime_type = mimetypes.guess_type(ref_image_path)[0] or "image/jpeg"
        config_kwargs["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=types.Image(
                    image_bytes=base64.b64encode(image_bytes).decode("utf-8"),
                    mime_type=mime_type,
                ),
                reference_type="asset",
            ),
        ]

    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(**config_kwargs),
    )

    while not operation.done:
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)

    if operation.response is None:
        error = getattr(operation, "error", None)
        raise RuntimeError(f"Video generation failed: {error}")

    video = operation.response.generated_videos[0]
    if not USE_VERTEX:
        client.files.download(file=video.video)
    return video.video


def process_story(
    story_path: str,
    ref_image_path: str,
    shot_duration: int,
    video_type: str,
    start_shot: int = 1,
    end_shot: int | None = None,
) -> None:
    path = Path(story_path)
    with open(path) as f:
        story = yaml.safe_load(f)

    video_id = path.parent.name
    story_name = path.stem
    out_dir = Path("output") / video_id / "raw_clips" / story_name
    out_dir.mkdir(parents=True, exist_ok=True)

    aspect_ratio = FORMAT_CONFIG[video_type]["aspect_ratio"]
    client = genai.Client()
    shots = [s for s in sorted(story["shots"], key=lambda s: s["id"])
             if s["id"] >= start_shot and (end_shot is None or s["id"] <= end_shot)]

    print(f"Generating {len(shots)} shots for '{story.get('title', story_name)}'")
    print(f"  Type: {video_type} ({aspect_ratio}), Duration: {shot_duration}s/shot")
    if start_shot > 1 or end_shot is not None:
        range_str = f"{start_shot}–{end_shot if end_shot is not None else 'end'}"
        print(f"  Shot range: {range_str}")
    print(f"  Output: {out_dir}/")

    for shot in shots:
        shot_id = shot["id"]
        out_path = out_dir / f"{shot_id}.mp4"

        if out_path.exists():
            print(f"  Shot {shot_id}: already exists, skipping")
            continue

        prompt = build_prompt(shot["description"])
        print(f"  Shot {shot_id}: generating...")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                video_file = generate_shot(
                    client, prompt, ref_image_path,
                    aspect_ratio, shot_duration,
                )
                video_file.save(str(out_path))
                size = out_path.stat().st_size
                if size < MIN_VALID_BYTES:
                    out_path.unlink()
                    raise ValueError(f"Output too small ({size} bytes) — likely a failed generation")
                print(f"  Shot {shot_id}: saved ({size / 1024 / 1024:.1f} MB)")
                success = True
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    print(f"  Shot {shot_id}: rate limited, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    print(f"  Shot {shot_id}: error — {error_msg}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS)

        if not success:
            print(f"  Shot {shot_id}: FAILED after {MAX_RETRIES} attempts — quitting. Resume with --start_shot {shot_id}")
            sys.exit(1)

        if shot["id"] != shots[-1]["id"]:
            time.sleep(DELAY_BETWEEN_SHOTS_SECONDS)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate video clips from a story YAML.")
    parser.add_argument("--story", required=True, help="Path to the story YAML file.")
    parser.add_argument("--ref_image", default=None, help="Optional path to character reference image.")
    parser.add_argument("--shot_duration", default=8, type=int, help="Duration per shot in seconds.")
    parser.add_argument("--type", default="normal", choices=["normal", "short"], dest="video_type",
                        help="Video type: 'normal' (landscape 16:9) or 'short' (portrait 9:16).")
    parser.add_argument("--start_shot", default=1, type=int, help="Shot ID to start from (default: 1).")
    parser.add_argument("--end_shot", default=None, type=int, help="Shot ID to stop at, inclusive (default: last shot).")
    args = parser.parse_args()

    if not Path(args.story).exists():
        print(f"Error: story file not found: {args.story}", file=sys.stderr)
        sys.exit(1)
    if args.ref_image and not Path(args.ref_image).exists():
        print(f"Error: reference image not found: {args.ref_image}", file=sys.stderr)
        sys.exit(1)

    process_story(args.story, args.ref_image, args.shot_duration, args.video_type, args.start_shot, args.end_shot)


if __name__ == "__main__":
    main()
