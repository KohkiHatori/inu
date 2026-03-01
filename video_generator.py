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

import prompt_builder

MODEL = "veo-3.1-fast-generate-preview"
RESOLUTION = "1080p"  # "720p" | "1080p" | "4k"



POLL_INTERVAL_SECONDS = 15
DELAY_BETWEEN_SHOTS_SECONDS = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 60
MIN_VALID_BYTES = 1 * 1024 * 1024


def extract_last_frame(video_path: str, offset_from_end: float) -> bytes:
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
    channel: dict,
    start_frame: bytes,
) -> bytes:
    """Image-to-video — the start_frame becomes the literal first frame."""
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": 1,
        "duration_seconds": duration,
        "resolution": RESOLUTION,
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
    channel_config: dict,
    shot_duration: int,
    aspect_ratio: str,
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


    client = genai.Client()
    all_shots = sorted(story["shots"], key=lambda s: s["id"])
    shots = [s for s in all_shots
             if s["id"] >= start_shot and (end_shot is None or s["id"] <= end_shot)]

    print(f"Generating {len(shots)} shots (frame-continuity mode) for '{story.get('title', story_name)}'")
    print(f"  Type: {aspect_ratio}, Duration: {shot_duration}s/shot")
    if start_shot > 1 or end_shot is not None:
        range_str = f"{start_shot}–{end_shot if end_shot is not None else 'end'}"
        print(f"  Shot range: {range_str}")
    print(f"  Output: {out_dir}/")

    # Map channel character IDs to their paths
    channel_refs = {}
    for char in channel_config.get("characters", []):
        if "id" in char and "ref_image" in char:
            channel_refs[char["id"]] = char["ref_image"]

    for shot in shots:
        shot_id = shot["id"]
        out_path = out_dir / f"{shot_id}.mp4"

        if out_path.exists():
            print(f"  Shot {shot_id}: already exists, skipping")
            continue

        shot_mode = shot.get("mode", "t2v" if shot_id == 1 else "i2v")

        # Determine Reference Images for this shot
        char_refs = []
        if shot_mode == "t2v" and "reference_images" in shot:
            for ref_id in shot["reference_images"]:
                # Check channel first
                if ref_id in channel_refs:
                    img_path = Path(channel_refs[ref_id])
                else: # Otherwise check dynamic generation folder
                    img_path = Path("assets") / "ref" / video_id / story_name / f"{ref_id}.png"

                if img_path.exists():
                    mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                    char_refs.append(make_ref_image_config(img_path.read_bytes(), mime))
                else:
                    print(f"  Shot {shot_id}: WARNING - Reference image '{ref_id}' not found at {img_path}")

        # I2V: extract last frame from previous shot; fall back to T2V if unavailable
        start_frame = None
        if shot_mode == "i2v":
            prev_path = out_dir / f"{shot_id - 1}.mp4"
            if prev_path.exists():
                try:
                    offset = channel_config.get("video_settings", {}).get("frame_offset_seconds", 1.0)
                    start_frame = extract_last_frame(str(prev_path), offset)
                except Exception as e:
                    print(f"  Shot {shot_id}: warning — frame extraction failed, falling back to T2V: {e}")
                    shot_mode = "t2v"
            else:
                print(f"  Shot {shot_id}: warning — previous shot not found, falling back to T2V")
                shot_mode = "t2v"

        is_continuation = (shot_mode == "i2v")
        if is_continuation:
            prompt = prompt_builder.build_video_continuation_prompt(channel_config, shot["description"])
        else:
            prompt = prompt_builder.build_video_hero_prompt(channel_config, shot["description"])

        print(f"  Shot {shot_id}: generating ({shot_mode.upper()}{'+ref' if shot_mode == 't2v' and char_refs else ''})...")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if shot_mode == "i2v":
                    video_file = generate_shot_i2v(
                        client, prompt, aspect_ratio, shot_duration,
                        channel=channel_config,
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
    parser.add_argument("--channel", default=None, help="Override channel configuration.")
    parser.add_argument("--story", required=True, help="Path to the story YAML file.")
    parser.add_argument("--shot_duration", default=None, type=int, choices=[4, 6, 8], help="Override duration per shot in seconds.")
    parser.add_argument("--aspect_ratio", default=None, help="Override aspect ratio (e.g., 16:9).")
    parser.add_argument("--start_shot", default=1, type=int, help="Shot ID to start from (default: 1).")
    parser.add_argument("--end_shot", default=None, type=int, help="Shot ID to stop at, inclusive (default: last shot).")
    args = parser.parse_args()

    if not Path(args.story).exists():
        print(f"Error: story file not found: {args.story}", file=sys.stderr)
        sys.exit(1)

    with open(args.story) as f:
        story_data = yaml.safe_load(f)

    metadata = story_data.get("metadata", {})
    channel_name = args.channel or metadata.get("channel", "pup-pop-pup")
    shot_duration = args.shot_duration or metadata.get("shot_duration", 8)
    aspect_ratio = args.aspect_ratio or metadata.get("aspect_ratio", "16:9")

    try:
        channel_config = prompt_builder.load_channel_config(channel_name)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    process_story(args.story, channel_config, shot_duration, aspect_ratio, args.start_shot, args.end_shot)


if __name__ == "__main__":
    main()
