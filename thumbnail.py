"""Generate thumbnails for YouTube videos using Gemini.

Generates a thumbnail image based on the story concept and reference images.
"""

import argparse
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

import prompt_builder

load_dotenv()

USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
MODEL = "gemini-3-pro-image-preview"

# Default character references
CHAR_REF_PATHS = [
    Path("assets/ref/pop.png"),
    Path("assets/ref/pupa.png"),
    Path("assets/ref/pupb.png"),
]

def generate_thumbnail(
    client: genai.Client,
    prompt: str,
    ref_image_paths: list[Path],
    output_path: Path
) -> None:
    """Generates a thumbnail using Gemini and saves it."""

    print(f"Generating thumbnail for '{output_path.name}'...")
    print(f"  Prompt preview: {prompt[:100]}...")

    # Prepare contents list starting with the prompt
    contents = [prompt]

    # Load and append valid reference images
    loaded_refs = 0
    for p in ref_image_paths:
        if not p.exists():
            print(f"  Warning: Reference image not found: {p}")
            continue
        try:
            # Open with PIL to verify and pass to API
            img = Image.open(p)
            contents.append(img)
            loaded_refs += 1
        except Exception as e:
            print(f"  Warning: Failed to load image {p}: {e}")

    print(f"  Using {loaded_refs} reference images.")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="1K" # Using 1K as it's a good preview size, could be "2K" or "4K"
                ),
            )
        )

        if response.parts:
            for part in response.parts:
                img = part.as_image()
                if img:
                    img.save(str(output_path))
                    print(f"  Saved to: {output_path}")
                    return

        print(f"  Error: No image found in response for {output_path.name}")

    except Exception as e:
        print(f"  Error generating thumbnail: {e}")


def process_story_file(story_path: Path, client: genai.Client, channel_config: dict) -> None:
    if not story_path.exists():
        print(f"Error: Story file not found: {story_path}")
        return

    try:
        with open(story_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading YAML {story_path}: {e}")
        return

    concept = data.get("concept")
    if not concept:
        print(f"Skipping {story_path.name}: No 'concept' field found.")
        return

    # Extract video_id and story_name
    # Assuming structure: stories/{video_id}/{story_name}.yaml
    # If passed a file directly, we might need to be careful about parent directory
    video_id = story_path.parent.name
    story_name = story_path.stem

    # Determine output path: output/{video_id}/{story_name}_thumbnail.png
    output_dir = Path("output") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{story_name}_thumbnail.png"

    if output_path.exists():
        print(f"Thumbnail already exists: {output_path}, skipping.")
        return

    # Gather reference images
    # 1. Base character refs
    ref_images = list(CHAR_REF_PATHS)

    # 2. Story specific refs: assets/ref/{video_id}/{story_name}/*
    # Note: Using video_id/story_name matches the structure you implied
    story_ref_dir = Path("assets/ref") / video_id / story_name
    if story_ref_dir.exists():
        for item in story_ref_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                ref_images.append(item)

    # Construct prompt
    prompt = prompt_builder.build_thumbnail_prompt(channel_config, concept)

    generate_thumbnail(client, prompt, ref_images, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YouTube thumbnails for stories.")
    parser.add_argument("path", help="Path to a story YAML file or a directory of stories.")
    parser.add_argument("--channel", default="pup-pop-pup", help="Channel configuration to use.")
    args = parser.parse_args()

    try:
        channel_config = prompt_builder.load_channel_config(args.channel)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        return

    if path.is_file():
        if path.suffix.lower() in ['.yaml', '.yml']:
            process_story_file(path, client, channel_config)
        else:
            print("Error: Input file must be a YAML file.")
    elif path.is_dir():
        # Find all .yaml files in the directory (recursively)
        yaml_files = sorted(path.rglob("*.yaml"))
        if not yaml_files:
            print(f"No .yaml files found in {path}")
            return

        print(f"Found {len(yaml_files)} story files.")
        for yaml_file in yaml_files:
            process_story_file(yaml_file, client, channel_config)

if __name__ == "__main__":
    main()
