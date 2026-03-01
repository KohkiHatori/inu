"""Generate a story YAML file for the Pup-Pop-Pup video pipeline.

Uses Gemini 3.1 Pro to produce 15 shot descriptions from an optional idea,
then writes the result to stories/{video_id}.yaml.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import subprocess

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# gemini-3.1-pro-preview is only available on the Gemini API, not Vertex AI.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

import prompt_builder






def extract_yaml_block(text: str) -> str:
    """Pull the first fenced YAML block out of the LLM response."""
    match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _quote_description_values(raw: str) -> str:
    """Fallback: wrap bare description values in double quotes.

    Handles the common case where Gemini emits:
        description: Some text: with colons
    and converts it to:
        description: "Some text: with colons"
    Skips lines that already use block style (|, >) or are already quoted.
    """
    lines = raw.splitlines()
    out = []
    for line in lines:
        m = re.match(r'^(\s*description:\s*)(.+)$', line)
        if m:
            indent, value = m.group(1), m.group(2).strip()
            # Leave alone if already block scalar or quoted
            if value[0] not in ('"', "'", '|', '>'):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                line = f'{indent}"{escaped}"'
        out.append(line)
    return '\n'.join(out)


def parse_story_yaml(raw: str) -> dict:
    """Parse story YAML, auto-quoting description values as a fallback."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        fixed = _quote_description_values(raw)
        return yaml.safe_load(fixed)


def validate_story(data: dict, num_shots: int = 15) -> None:
    required_keys = {"title", "concept", "shots"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")

    if "new_reference_images" in data:
        for i, ref in enumerate(data["new_reference_images"]):
            if "id" not in ref or "description" not in ref:
                raise ValueError(f"New reference image {i} must have 'id' and 'description'")

    shots = data["shots"]
    if not isinstance(shots, list) or len(shots) != num_shots:
        raise ValueError(f"Expected exactly {num_shots} shots, got {len(shots) if isinstance(shots, list) else type(shots)}")

    for i, shot in enumerate(shots):
        for key in ("id", "description"):
            if key not in shot:
                raise ValueError(f"Shot {i + 1} missing '{key}'")
        if "mode" not in shot:
            raise ValueError(f"Shot {i + 1} missing 'mode' (required in I2V mode)")
        if shot["mode"] not in ("t2v", "i2v"):
            raise ValueError(f"Shot {i + 1} 'mode' must be 't2v' or 'i2v', got '{shot['mode']}'")

        if shot["mode"] == "t2v":
            if "reference_images" not in shot or not isinstance(shot["reference_images"], list):
                raise ValueError(f"Shot {i + 1} uses mode 't2v' but 'reference_images' is missing or not a list")
            if len(shot["reference_images"]) > 3:
                raise ValueError(f"Shot {i + 1} uses mode 't2v' but has {len(shot['reference_images'])} reference images (max 3 allowed)")



    shots[0]["mode"] = "t2v"


def generate_story(aspect_ratio: str, idea: str | None, channel: dict, num_shots: int = 15, shot_duration: int = 8) -> dict:
    client = genai.Client()
    idea_line = f"Story idea: {idea}" if idea else "Generate an original story idea. Be creative and engaging."

    user_prompt = prompt_builder.build_story_user_prompt(
        channel=channel,
        aspect_ratio=aspect_ratio,
        shot_dur=shot_duration,
        total_dur=shot_duration * num_shots,
        idea_line=idea_line,
        num_shots=num_shots
    )

    system_prompt = prompt_builder.build_story_system_prompt(channel)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
        ),
    )

    raw_yaml = extract_yaml_block(response.text)
    data = parse_story_yaml(raw_yaml)
    validate_story(data, num_shots)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Pup-Pop-Pup story YAML.")
    parser.add_argument("--channel", default="pup-pop-pup", help="Channel configuration to use.")
    parser.add_argument("--aspect_ratio", default="16:9", help="Video generation aspect ratio (e.g. 16:9, 9:16).")
    parser.add_argument("--idea", default=None, help="Optional story seed idea.")
    parser.add_argument("--video_id", required=True, help="Video identifier (used as output filename).")
    parser.add_argument("--num_shots", type=int, default=15, help="Number of shots to generate.")
    parser.add_argument("--shot_duration", type=int, choices=[4, 6, 8], default=8, help="Duration per shot in seconds.")
    args = parser.parse_args()

    try:
        channel_config = prompt_builder.load_channel_config(args.channel)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Generating story for {args.video_id} ({args.aspect_ratio} format  · I2V mode)...")
    if args.idea:
        print(f"  Idea: {args.idea}")
    else:
        print("  No idea provided — AI will generate one from scratch.")

    story = generate_story(args.aspect_ratio, args.idea, channel_config, args.num_shots, args.shot_duration)

    metadata = {
        "channel": args.channel,
        "aspect_ratio": args.aspect_ratio,
        "shot_duration": args.shot_duration,
        "num_shots": args.num_shots,
    }

    # Prepend metadata to the generated story output
    final_output = {"metadata": metadata}
    final_output.update(story)

    out_dir = Path("stories") / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in out_dir.glob("story*.yaml")]
    next_n = max((int(p.stem[5:]) for p in existing if p.stem[5:].isdigit()), default=0) + 1
    out_path = out_dir / f"story{next_n}.yaml"

    with open(out_path, "w") as f:
        yaml.dump(final_output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved to {out_path}")

    # Automatically trigger reference image generation
    print("\n--- Automatically generating reference images ---")
    try:
        subprocess.run(
            [sys.executable, "ref_image_generator.py", "--story", str(out_path)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during reference image generation: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
