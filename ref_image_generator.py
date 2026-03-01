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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate new dynamic reference images for a story.")
    parser.add_argument("--story", required=True, help="Path to the story YAML file.")
    parser.add_argument("--channel", default="pup-pop-pup", help="Channel configuration to use.")
    args = parser.parse_args()

    try:
        channel_config = prompt_builder.load_channel_config(args.channel)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    story_path = Path(args.story)
    if not story_path.exists():
        print(f"Error: story file not found: {args.story}", file=sys.stderr)
        sys.exit(1)

    with open(story_path, "r") as f:
        story = yaml.safe_load(f)

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

if __name__ == "__main__":
    main()
