import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Set up Jinja2 environment
TEMPLATE_DIR = Path(__file__).parent / "prompts"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)

def load_channel_config(channel_name: str) -> dict:
    """Loads the YAML configuration for a specific channel."""
    config_path = Path(__file__).parent / "channels" / f"{channel_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Channel config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_story_system_prompt(channel: dict) -> str:
    """Renders the system prompt for story generation."""
    template = env.get_template("story_system_prompt.jinja")
    return template.render(channel=channel)

def build_video_hero_prompt(channel: dict, description: str) -> str:
    """Renders the T2V prompt for video generation."""
    template = env.get_template("video_hero_prompt.jinja")
    return template.render(channel=channel, description=description)

def build_video_continuation_prompt(channel: dict, description: str) -> str:
    """Renders the I2V prompt for video generation."""
    template = env.get_template("video_continuation_prompt.jinja")
    return template.render(channel=channel, description=description)

def build_story_user_prompt(channel: dict, aspect_ratio: str, shot_dur: int, total_dur: int, idea_line: str, num_shots: int) -> str:
    """Renders the user prompt for story generation."""
    template = env.get_template("story_user_prompt.jinja")
    # We can generate a generic pool of up to 5 dynamic items per story.
    dynamic_slots = 5
    return template.render(
        channel=channel,
        aspect_ratio=aspect_ratio,
        shot_dur=shot_dur,
        total_dur=total_dur,
        idea_line=idea_line,
        dynamic_slots=dynamic_slots,
        num_shots=num_shots
    )

def build_ref_image_prompt(channel: dict, object_description: str) -> str:
    """Renders the prompt for generating a dynamic reference image."""
    template = env.get_template("ref_image_prompt.jinja")
    return template.render(channel=channel, object_description=object_description)

def build_thumbnail_prompt(channel: dict, concept: str) -> str:
    """Renders the prompt for generating a YouTube thumbnail."""
    template = env.get_template("thumbnail_prompt.jinja")
    return template.render(channel=channel, concept=concept)
