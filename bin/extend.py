"""Generate video clips using Veo's video extension for continuity.

Shot 1 is generated from scratch. Each subsequent shot extends the previous
one, giving Veo visual context to maintain continuity across the story.
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
GCS_OUTPUT_URI = os.getenv("GCS_OUTPUT_URI")  # required on Vertex AI for large videos, e.g. gs://my-bucket/output/

MODEL = "veo-3.1-generate-001"
RESOLUTION = "1080p"  # "720p" | "1080p" | "4k"

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


def download_from_gcs(gcs_uri: str, dest_path: str) -> None:
    """Download a file from a gs:// URI to a local path."""
    from google.cloud import storage
    bucket_name, blob_path = gcs_uri[5:].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_path).download_to_filename(dest_path)


def get_video_bytes(client: genai.Client, video) -> bytes:
    """Return video bytes, handling both Gemini API and Vertex AI response formats."""
    if USE_VERTEX and GCS_OUTPUT_URI and getattr(video, "uri", None):
        tmp = Path("/tmp/_veo_tmp.mp4")
        download_from_gcs(video.uri, str(tmp))
        data = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return data
    if not USE_VERTEX:
        client.files.download(file=video)
    return video.video_bytes


def build_prompt(description: str) -> str:
    return f"{HERO_PREFIX}{description}{STYLE_SUFFIX}"


def poll_operation(client: genai.Client, operation) -> object:
    while not operation.done:
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)
    if operation.response is None:
        error = getattr(operation, "error", None)
        raise RuntimeError(f"Video generation failed: {error}")
    return operation


def generate_first_shot(
    client: genai.Client,
    prompt: str,
    ref_image_path: str | None,
    aspect_ratio: str,
    duration: int,
) -> tuple[bytes, object]:
    """Generate the first shot from scratch. Returns (video_bytes, video_object)."""
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "number_of_videos": 1,
        "duration_seconds": duration,
        "resolution": RESOLUTION,
    }
    if USE_VERTEX and GCS_OUTPUT_URI:
        config_kwargs["output_gcs_uri"] = GCS_OUTPUT_URI

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
    operation = poll_operation(client, operation)

    video = operation.response.generated_videos[0]
    clean = types.Video(uri=video.video.uri, mime_type=video.video.mime_type or "video/mp4")
    return get_video_bytes(client, video.video), clean


def extend_from_previous(
    client: genai.Client,
    prompt: str,
    previous_video: object,
    duration: int,
) -> tuple[bytes, object]:
    """Extend a video from the previous shot's output. Returns (video_bytes, video_object)."""
    MAX_EXTENSION_DURATION = 7  # Vertex AI video_extension only supports up to 7s
    config_kwargs = {
        "number_of_videos": 1,
        "duration_seconds": MAX_EXTENSION_DURATION,
        # resolution intentionally omitted — must inherit from the input video automatically
    }
    if USE_VERTEX and GCS_OUTPUT_URI:
        config_kwargs["output_gcs_uri"] = GCS_OUTPUT_URI

    operation = client.models.generate_videos(
        model=MODEL,
        prompt=prompt,
        video=previous_video,
        config=types.GenerateVideosConfig(**config_kwargs),
    )
    operation = poll_operation(client, operation)

    video = operation.response.generated_videos[0]
    clean = types.Video(uri=video.video.uri, mime_type=video.video.mime_type or "video/mp4")
    return get_video_bytes(client, video.video), clean


def process_story(
    story_path: str,
    ref_image_path: str,
    shot_duration: int,
    video_type: str,
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
    shots = sorted(story["shots"], key=lambda s: s["id"])

    print(f"Generating {len(shots)} shots (extension mode) for '{story.get('title', story_name)}'")
    print(f"  Type: {video_type} ({aspect_ratio}), Duration: {shot_duration}s/shot")
    print(f"  Output: {out_dir}/")

    previous_video = None

    for i, shot in enumerate(shots):
        shot_id = shot["id"]
        out_path = out_dir / f"{shot_id}.mp4"
        prompt = build_prompt(shot["description"])
        is_first = (i == 0)

        print(f"  Shot {shot_id}: {'generating from scratch' if is_first else 'extending from previous'}...")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if is_first:
                    video_file, previous_video = generate_first_shot(
                        client, prompt, ref_image_path,
                        aspect_ratio, shot_duration,
                    )
                else:
                    video_file, previous_video = extend_from_previous(
                        client, prompt, previous_video, shot_duration,
                    )

                out_path.write_bytes(video_file)
                size = out_path.stat().st_size
                if size < MIN_VALID_BYTES:
                    out_path.unlink()
                    raise ValueError(f"Output too small ({size} bytes) — likely a failed generation")
                print(f"  Shot {shot_id}: saved ({size / 1024 / 1024:.1f} MB)")
                break
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    print(f"  Shot {shot_id}: rate limited, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    print(f"  Shot {shot_id}: error — {error_msg}")
                    if attempt == MAX_RETRIES:
                        print(f"  Shot {shot_id}: FAILED after {MAX_RETRIES} attempts — stopping chain")
                        return
                    time.sleep(RETRY_BACKOFF_SECONDS)

        if shot["id"] != shots[-1]["id"]:
            time.sleep(DELAY_BETWEEN_SHOTS_SECONDS)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate video clips using Veo extension mode for continuity.")
    parser.add_argument("--story", required=True, help="Path to the story YAML file.")
    parser.add_argument("--ref_image", default=None, help="Path to the character reference image.")
    parser.add_argument("--shot_duration", default=8, type=int, help="Duration per shot in seconds.")
    parser.add_argument("--type", default="normal", choices=["normal", "short"], dest="video_type",
                        help="Video type: 'normal' (landscape 16:9) or 'short' (portrait 9:16).")
    args = parser.parse_args()

    if not Path(args.story).exists():
        print(f"Error: story file not found: {args.story}", file=sys.stderr)
        sys.exit(1)
    if args.ref_image and not Path(args.ref_image).exists():
        print(f"Error: reference image not found: {args.ref_image}", file=sys.stderr)
        sys.exit(1)

    process_story(args.story, args.ref_image, args.shot_duration, args.video_type)


if __name__ == "__main__":
    main()
