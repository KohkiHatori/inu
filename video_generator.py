"""Generate video clips using a hybrid T2V / I2V strategy.

Each shot has a mode field ("t2v" or "i2v") set by the story generator:
  - T2V: text-to-video with character reference images. Used at scene transitions
    to re-anchor character identity.
  - I2V: image-to-video using the last frame of the previous shot as the starting
    frame. Used for continuation shots within a scene for smooth visual continuity.
"""

import argparse
import base64
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

MODEL = "veo-3.1-fast-generate-preview"
RESOLUTION = "1080p"  # "720p" | "1080p" | "4k"

# Character reference images — one per character, full-body 3/4 view on white background.
# Set to None to skip for a character.
REF_IMAGE_PATHS: list[str] = [
    "assets/ref/pop.png",
    "assets/ref/pupa.png",
    "assets/ref/pupb.png",
]

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
    "steady camera, slow pace, adorable and whimsical tone, no humans, no text. "
    "Audio: natural ambient sounds only — animal sounds, gentle whooshes, soft cartoon-like "
    "sound effects. No music, no instruments, no soundtrack."
)

NEGATIVE_PROMPT = "distorted, morphing, extra limbs, extra dogs, text, watermarks, blurry, shaky camera, fast motion, chaotic, human hands, human body"

# Used for shots 2+ in I2V mode — replaces HERO_PREFIX entirely.
# We omit character physical descriptions because the I2V starting frame already
# establishes who the dogs are; re-describing them risks Veo generating new dogs
# to match the text rather than continuing from the visual anchor.
CONTINUATION_PREFIX = (
    "Continuing directly from the previous frame. The exact same three dogs already shown "
    "continue the scene — do not introduce any new characters or alter their appearance. "
    "The dogs must remain photorealistic with real fur, real eyes, and real anatomy throughout. "
    "All objects, props, vehicles, and environmental elements are stylized like "
    "colorful 3D CG cartoon items — shiny, rounded, oversized, and vibrant. "
)

FORMAT_CONFIG = {
    "normal": {"aspect_ratio": "16:9"},
    "short":  {"aspect_ratio": "9:16"},
}

POLL_INTERVAL_SECONDS = 15
DELAY_BETWEEN_SHOTS_SECONDS = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 60
MIN_VALID_BYTES = 1 * 1024 * 1024
FRAME_OFFSET_SECONDS = 1.0


def extract_last_frame(video_path: str, offset_from_end: float = FRAME_OFFSET_SECONDS) -> bytes:
    """Extract a frame near the end of a video using ffmpeg. Returns JPEG bytes."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    timestamp = max(0, duration - offset_from_end)

    frame_result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
         "-frames:v", "1", "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
        capture_output=True, check=True,
    )
    return frame_result.stdout


def build_prompt(description: str, is_continuation: bool = False) -> str:
    # For I2V shots, swap out HERO_PREFIX entirely for CONTINUATION_PREFIX so
    # Veo doesn't try to generate fresh dogs to match a text description.
    prefix = CONTINUATION_PREFIX if is_continuation else HERO_PREFIX
    return f"{prefix}{description}{STYLE_SUFFIX}"


def make_ref_image_config(image_bytes: bytes, mime_type: str) -> types.VideoGenerationReferenceImage:
    return types.VideoGenerationReferenceImage(
        image=types.Image(
            image_bytes=base64.b64encode(image_bytes).decode("utf-8"),
            mime_type=mime_type,
        ),
        reference_type="asset",
    )


def generate_shot_t2v(
    client: genai.Client,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    char_refs: list[types.VideoGenerationReferenceImage] | None = None,
) -> bytes:
    """Text-to-video with optional character reference images."""
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": 1,
        "duration_seconds": duration,
        "resolution": RESOLUTION,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    if char_refs:
        config_kwargs["reference_images"] = char_refs

    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(**config_kwargs),
    )
    return _poll_and_download(client, operation)


def generate_shot_i2v(
    client: genai.Client,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    start_frame: bytes,
) -> bytes:
    """Image-to-video — the start_frame becomes the literal first frame."""
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": 1,
        "duration_seconds": duration,
        "resolution": RESOLUTION,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    start_image = types.Image(
        image_bytes=base64.b64encode(start_frame).decode("utf-8"),
        mime_type="image/jpeg",
    )
    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        image=start_image,
        config=types.GenerateVideosConfig(**config_kwargs),
    )
    return _poll_and_download(client, operation)


def _poll_and_download(client: genai.Client, operation):
    """Poll until done, then download and return the video object."""
    while not operation.done:
        time.sleep(POLL_INTERVAL_SECONDS)
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


def process_story(
    story_path: str,
    ref_image_paths: list[str],
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
    all_shots = sorted(story["shots"], key=lambda s: s["id"])
    shots = [s for s in all_shots
             if s["id"] >= start_shot and (end_shot is None or s["id"] <= end_shot)]

    print(f"Generating {len(shots)} shots (frame-continuity mode) for '{story.get('title', story_name)}'")
    print(f"  Type: {video_type} ({aspect_ratio}), Duration: {shot_duration}s/shot")
    if start_shot > 1 or end_shot is not None:
        range_str = f"{start_shot}–{end_shot if end_shot is not None else 'end'}"
        print(f"  Shot range: {range_str}")
    print(f"  Output: {out_dir}/")

    char_refs = []
    for img_path in ref_image_paths:
        mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
        char_refs.append(make_ref_image_config(Path(img_path).read_bytes(), mime))
    if char_refs:
        print(f"  Character references: {len(char_refs)} image(s)")

    for shot in shots:
        shot_id = shot["id"]
        out_path = out_dir / f"{shot_id}.mp4"

        if out_path.exists():
            print(f"  Shot {shot_id}: already exists, skipping")
            continue

        shot_mode = shot.get("mode", "t2v" if shot_id == 1 else "i2v")

        # I2V: extract last frame from previous shot; fall back to T2V if unavailable
        start_frame = None
        if shot_mode == "i2v":
            prev_path = out_dir / f"{shot_id - 1}.mp4"
            if prev_path.exists():
                try:
                    start_frame = extract_last_frame(str(prev_path))
                except Exception as e:
                    print(f"  Shot {shot_id}: warning — frame extraction failed, falling back to T2V: {e}")
                    shot_mode = "t2v"
            else:
                print(f"  Shot {shot_id}: warning — previous shot not found, falling back to T2V")
                shot_mode = "t2v"

        is_continuation = (shot_mode == "i2v")
        prompt = build_prompt(shot["description"], is_continuation=is_continuation)
        print(f"  Shot {shot_id}: generating ({shot_mode.upper()}{'+ref' if shot_mode == 't2v' and char_refs else ''})...")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if shot_mode == "i2v":
                    video_file = generate_shot_i2v(
                        client, prompt, aspect_ratio, shot_duration,
                        start_frame=start_frame,
                    )
                else:
                    video_file = generate_shot_t2v(
                        client, prompt, aspect_ratio, shot_duration,
                        char_refs=char_refs or None,
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
    parser = argparse.ArgumentParser(
        description="Generate video clips with frame-based continuity between shots.")
    parser.add_argument("--story", required=True, help="Path to the story YAML file.")
    parser.add_argument("--shot_duration", default=8, type=int, help="Duration per shot in seconds.")
    parser.add_argument("--type", default="normal", choices=["normal", "short"], dest="video_type",
                        help="Video type: 'normal' (landscape 16:9) or 'short' (portrait 9:16).")
    parser.add_argument("--start_shot", default=1, type=int, help="Shot ID to start from (default: 1).")
    parser.add_argument("--end_shot", default=None, type=int, help="Shot ID to stop at, inclusive (default: last shot).")
    args = parser.parse_args()

    if not Path(args.story).exists():
        print(f"Error: story file not found: {args.story}", file=sys.stderr)
        sys.exit(1)

    ref_images = [p for p in REF_IMAGE_PATHS if p is not None]
    for img in ref_images:
        if not Path(img).exists():
            print(f"Error: reference image not found: {img}", file=sys.stderr)
            sys.exit(1)

    process_story(args.story, ref_images, args.shot_duration, args.video_type, args.start_shot, args.end_shot)


if __name__ == "__main__":
    main()
