"""Generate a story YAML file for the Pup-Pop-Pup video pipeline.

Uses Gemini 3.1 Pro to produce 15 shot descriptions from an optional idea,
then writes the result to stories/{video_id}.yaml.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# gemini-3.1-pro-preview is only available on the Gemini API, not Vertex AI.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

CHANNEL_SYSTEM_PROMPT = """\
You are a creative director for a YouTube Kids channel called Pup-Pop-Pup.
The channel stars Pop (a patient, loving father dog) and his two energetic pups.
Stories are dialogue-free, safe for children, and designed to be adorable and heartwarming.

CREATIVE DIRECTION:
- The dogs look photorealistic (real fur, real eyes, real anatomy) but they do human activities: \
cooking, cleaning, shopping, gardening, DIY projects, office work, yoga, camping, etc.
- Settings should be realistic, ordinary human environments — a cozy suburban house, a modern kitchen, \
a supermarket aisle, a sunny park, a messy garage, a laundry room, a beach, a library.
- Props and objects should be realistic everyday items suitable for the setting (e.g., real pots and pans, \
actual gardening tools, realistic groceries) — consistent with the photorealistic world.
- Prioritize adorableness, situational humor, and "slice of life" charm.
- Plots should focus on the dogs attempting to navigate everyday human life tasks, often with funny or cute results. For example, Pop trying to assemble furniture while pups steal screws, or the dogs running a lemonade stand.
- Natural sounds only (no human speech, no scary content).

CHARACTERS:
- Pop: calm, gentle adult golden retriever father dog with warm amber eyes and a broad snout.
- Pup-A: small fluffy golden puppy with oversized paws and floppy ears.
- Pup-B: slightly smaller golden puppy with a white chest patch and a constantly wagging tail.\
"""

FORMAT_PARAMS = {
    "normal": {"shot_duration": 8, "label": "landscape"},
    "short":  {"shot_duration": 4, "label": "portrait"},
}


def build_user_prompt(fmt: str, idea: str | None) -> str:
    params = FORMAT_PARAMS[fmt]
    shot_dur = params["shot_duration"]
    total_dur = shot_dur * 15

    idea_line = f"Story idea: {idea}" if idea else "Generate an original story idea. Be creative and engaging."

    description_rules = """\
    IMPORTANT: This story uses a hybrid generation strategy. Some shots are generated with \
    text-to-video (T2V) using character reference images, and others use image-to-video (I2V) where \
    the last frame of the previous shot becomes the first frame of the next. Write descriptions accordingly:

    SCENE STRUCTURE: Organise the story into short scenes of 2–3 shots each.
    - Shot 1 (T2V): Establish a new scene/angle/location. Use mode: "t2v".
    - Shot 2 (I2V): Continue the action from Shot 1. Use mode: "i2v".
    - Shot 3 (I2V/T2V): Optionally continue (I2V) or start a new scene (T2V).
    - Shot 4 (T2V): MUST start a new scene/angle (if Shot 3 was I2V).
    Goal: Change the visual angle or scene completely every 2–3 shots to maintain visual freshness \
    and re-anchor character consistency. Avoid long chains of I2V.

    - Do NOT use character names (Pop, Pup-A, Pup-B). Instead use generic roles: \
    "the adult dog", "the larger puppy", "the smaller puppy", "the two puppies", "all three dogs".
    
    - EXPLICIT COUNT & IDENTITY: Always state the exact number of dogs present to prevent hallucinations. \
    e.g., "The two puppies are playing while the adult dog watches (three dogs total)." \
    Reinforce the identity relationship: "The adult dog remains still while the smaller puppy moves."

    - MOTION & STABILITY: Describe gentle, grounded movements. Avoid chaotic action, fast pacing, or complex physics. \
    Use phrases like "slowly lifts head", "gently wagging tail", "sitting calmly". \
    Keep the camera motion smooth and steady.

    - T2V shots (mode: t2v) — these are the first shot of each new scene. Fully describe the setting, \
    lighting, environment, props, and the position and appearance of all three dogs. Write the description \
    as if the video model is seeing these characters for the first time.
    - I2V shots (mode: i2v) — these continue a scene from the previous shot. Open with a brief \
    continuity line referencing what each dog was doing at the end of the previous shot (e.g. \
    "Continuing from the previous frame where the adult dog was stirring batter at the counter and \
    the two puppies were watching from the doorway — the adult dog now lifts the bowl..."). \
    Do NOT re-describe the dogs' physical appearance; the video model already has them from the \
    starting frame.
    - Every description (both T2V and I2V) must fully re-state the setting, environment details, \
    lighting, color palette, and any relevant props.
    - Do not include dialogue, text overlays, or narration.\
    """
    mode_rules = """
    - mode: either "t2v" or "i2v".
      - "t2v" = text-to-video with character reference images. Use for the FIRST shot of each scene/angle.
      - "i2v" = image-to-video, continuing from the last frame of the previous shot. Use for \
continuation shots within a scene.
      - Shot 1 MUST be "t2v".
      - Start a new "t2v" scene every 2–3 shots.
      - Do NOT use "i2v" for more than 2 consecutive shots (drift risk)."""

    return f"""\
Format: {fmt} video ({params['label']}, {shot_dur}s per shot, 15 shots total, {total_dur}s total).
{idea_line}

Output ONLY a YAML block (fenced with ```yaml ... ```) with these fields:
- title: string
- concept: one-paragraph summary of the story
- shots: a list of exactly 15 items, each with:
    - id: integer 1-15{mode_rules}
    - description: (use YAML literal block style "|" to avoid parsing issues with colons and special \
characters) a self-contained visual scene description suitable as an AI video generation prompt. \
{description_rules}

Be creative with plots. The dogs should be doing relatable human activities in a realistic world. \
Props and objects should look realistic and detailed, matching the environment. \
The dogs themselves must always be described as photorealistic.
Make sure the 15 shots tell a coherent story arc with a beginning, middle, and satisfying end.

OBJECT CONTINUITY RULE:
- Within a scene (I2V shots): Any object/prop used in shot N must be visible in shot N-1. Introduce props early in the background before they are used.
- SCENE SWITCH (T2V shots): When a new scene starts (mode: t2v), NO objects or props should continue from the previous scene/shot. Start fresh. Do not mention or include objects from the previous scene. A T2V shot resets the environment entirely."""


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


def validate_story(data: dict) -> None:
    required_keys = {"title", "concept", "shots"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing top-level keys: {missing}")

    shots = data["shots"]
    if not isinstance(shots, list) or len(shots) != 15:
        raise ValueError(f"Expected exactly 15 shots, got {len(shots) if isinstance(shots, list) else type(shots)}")

    for i, shot in enumerate(shots):
        for key in ("id", "description"):
            if key not in shot:
                raise ValueError(f"Shot {i + 1} missing '{key}'")
        if "mode" not in shot:
            raise ValueError(f"Shot {i + 1} missing 'mode' (required in I2V mode)")
        if shot["mode"] not in ("t2v", "i2v"):
            raise ValueError(f"Shot {i + 1} 'mode' must be 't2v' or 'i2v', got '{shot['mode']}'")

    shots[0]["mode"] = "t2v"


def generate_story(fmt: str, idea: str | None) -> dict:
    client = genai.Client()

    user_prompt = build_user_prompt(fmt, idea)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=CHANNEL_SYSTEM_PROMPT,
            temperature=1.0,
        ),
    )

    raw_yaml = extract_yaml_block(response.text)
    data = parse_story_yaml(raw_yaml)
    validate_story(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Pup-Pop-Pup story YAML.")
    parser.add_argument("--format", default="normal", choices=["normal", "short"], dest="fmt")
    parser.add_argument("--idea", default=None, help="Optional story seed idea.")
    parser.add_argument("--video_id", required=True, help="Video identifier (used as output filename).")
    args = parser.parse_args()

    if args.fmt not in FORMAT_PARAMS:
        print(f"Error: unknown format '{args.fmt}'", file=sys.stderr)
        sys.exit(1)

    print(f"Generating story for {args.video_id} ({args.fmt} format  · I2V mode)...")
    if args.idea:
        print(f"  Idea: {args.idea}")
    else:
        print("  No idea provided — AI will generate one from scratch.")

    story = generate_story(args.fmt, args.idea)

    out_dir = Path("stories") / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in out_dir.glob("story*.yaml")]
    next_n = max((int(p.stem[5:]) for p in existing if p.stem[5:].isdigit()), default=0) + 1
    out_path = out_dir / f"story{next_n}.yaml"

    with open(out_path, "w") as f:
        yaml.dump(story, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
