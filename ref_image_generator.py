import argparse
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

import prompt_builder

load_dotenv()

# We need the real Gemini API for image generation. Imagen is not supported via the Vertex alias.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
MODEL = "gemini-3-pro-image-preview"


def generate_reference_image(
    client: genai.Client,
    prompt: str,
    output_path: Path
) -> None:
    """Generates an image using Gemini and saves it to output_path."""
    print(f"Generating image for prompt: '{prompt}'")

    # Generate the image
    result = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio="1:1"
            )
        )
    )

    if not result.parts:
        raise RuntimeError("Image generation failed (No images returned).")

    for part in result.parts:
        img = part.as_image()
        if img:
            img.save(str(output_path))
            print(f"Saved image to {output_path}")
            return

    raise RuntimeError("Image generation failed (No image part found in response).")


def run_story_mode(story_path: Path, channel_override: str | None) -> None:
    """Generates dynamic ref images defined inside a story YAML."""
    if not story_path.exists():
        print(f"Error: story file not found: {story_path}", file=sys.stderr)
        sys.exit(1)

    with open(story_path, "r") as f:
        story = yaml.safe_load(f)

    metadata = story.get("metadata", {})
    channel_name = channel_override or metadata.get("channel", "pup-pop-pup")

    try:
        channel_config = prompt_builder.load_channel_config(channel_name)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    new_refs = story.get("new_reference_images", [])
    if not new_refs:
        print("No 'new_reference_images' found in this story. Nothing to do.")
        return

    # E.g. stories/test_refactor_01/story1.yaml -> assets/ref/test_refactor_01/story1/
    video_id = story_path.parent.name
    story_name = story_path.stem
    out_dir = Path("assets") / "ref" / video_id / story_name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()

    print(f"Found {len(new_refs)} new reference images to generate for '{video_id}/{story_name}'.")
    for ref in new_refs:
        ref_id = ref["id"]
        description = ref["description"]
        out_path = out_dir / f"{ref_id}.png"

        if out_path.exists():
            print(f"Skipping '{ref_id}' - image already exists at {out_path}")
            continue

        full_prompt = prompt_builder.build_ref_image_prompt(channel_config, description)

        try:
            generate_reference_image(client, full_prompt, out_path)
        except Exception as e:
            print(f"Error generating '{ref_id}': {e}", file=sys.stderr)

    print("Done.")


def run_channel_mode(channel_name: str) -> None:
    """Generates character ref images for every character in a channel config."""
    try:
        channel_config = prompt_builder.load_channel_config(channel_name)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    characters = channel_config.get("characters", [])
    if not characters:
        print(f"No characters found in channel '{channel_name}'. Nothing to do.")
        return

    out_dir = Path("assets") / "ref" / channel_name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client()

    print(f"Found {len(characters)} character(s) to generate for channel '{channel_name}'.")
    for char in characters:
        char_id = char.get("id")
        visual_description = char.get("visual_description")
        if not char_id or not visual_description:
            print(f"Skipping character with missing 'id' or 'visual_description': {char}")
            continue

        out_path = out_dir / f"{char_id}.png"
        if out_path.exists():
            print(f"Skipping '{char_id}' - image already exists at {out_path}")
            continue

        # Characters are always isolated on a plain white background
        description_with_bg = f"{visual_description}, isolated entirely on a plain white background."
        full_prompt = prompt_builder.build_ref_image_prompt(channel_config, description_with_bg)

        try:
            generate_reference_image(client, full_prompt, out_path)
        except Exception as e:
            print(f"Error generating '{char_id}': {e}", file=sys.stderr)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reference images for a story or a channel.",
        epilog=(
            "Modes:\n"
            "  --story PATH         Generate dynamic ref images from a story YAML.\n"
            "  --channel NAME       Generate character ref images from a channel config.\n"
            "\n"
            "When --story is provided, story mode runs (use --channel to override the\n"
            "channel embedded in the story metadata). When only --channel is provided,\n"
            "channel mode runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--story", default=None, help="Path to the story YAML file (story mode).")
    parser.add_argument("--channel", default=None, help="Channel name: overrides story metadata in story mode, or selects the channel in channel mode.")
    args = parser.parse_args()

    if args.story:
        run_story_mode(Path(args.story), args.channel)
    elif args.channel:
        run_channel_mode(args.channel)
    else:
        parser.error("You must provide at least --story or --channel.")

if __name__ == "__main__":
    main()
