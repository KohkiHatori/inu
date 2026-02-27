"""Assemble raw video clips into a final video with looping background music.

Reads clips from output/{video_id}/raw_clips/{story_name}/ in ID order,
concatenates them, layers looping background music under the original audio,
and exports the result.
"""

import argparse
import sys
from pathlib import Path

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import AudioFadeOut
from moviepy.video.fx import FadeOut
import yaml

MUSIC_VOLUME = 0.6  # background music relative to original audio
TOTAL_VOLUME = 0.85  # Master volume for the final video
FPS = 24
I2V_TAIL_TRIM_SECONDS = 1.0  # keep in sync with video_generator.FRAME_OFFSET_SECONDS
MIN_CLIP_DURATION_AFTER_TRIM = 0.1



def _infer_story_path(clips_dir: Path) -> Path | None:
    """Infer stories/<video_id>/<story_name>.yaml from clips directory."""
    try:
        video_id = clips_dir.parent.parent.name
    except IndexError:
        return None
    story_name = clips_dir.name
    return Path("stories") / video_id / f"{story_name}.yaml"


def _load_shot_modes(clips_dir: Path) -> tuple[dict[int, str], Path | None]:
    """Return {shot_id: mode} plus the story path (if any)."""
    story_path = _infer_story_path(clips_dir)
    if story_path is None:
        print("  Warning: could not infer story path from clips directory; skipping adaptive trimming.")
        return {}, None

    if not story_path.exists():
        print(f"  Warning: story file not found at {story_path}; skipping adaptive trimming.")
        return {}, story_path

    try:
        data = yaml.safe_load(story_path.read_text())
    except Exception as exc:  # yaml errors or IO errors
        print(f"  Warning: failed to read story metadata from {story_path}: {exc}")
        return {}, story_path

    shots = data.get("shots", [])
    if not isinstance(shots, list):
        print(f"  Warning: story file {story_path} is missing a 'shots' list; skipping adaptive trimming.")
        return {}, story_path

    modes: dict[int, str] = {}
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("id")
        if shot_id is None:
            continue
        try:
            shot_id_int = int(shot_id)
        except (TypeError, ValueError):
            continue
        mode = shot.get("mode", "t2v")
        if isinstance(mode, str):
            mode = mode.lower()
        else:
            mode = "t2v"
        modes[shot_id_int] = mode

    if modes:
        print(f"  Loaded story metadata from {story_path}")
    else:
        print(f"  Warning: story file {story_path} did not specify shot modes; skipping adaptive trimming.")
    return modes, story_path


def assemble(
    clips_dir: str,
    music_path: str | None,
    output_path: str,
) -> None:
    clips_dir = Path(clips_dir)

    clip_files = sorted(clips_dir.glob("*.mp4"), key=lambda p: int(p.stem))
    if not clip_files:
        print(f"No .mp4 files found in {clips_dir}", file=sys.stderr)
        sys.exit(1)

    shot_modes, _ = _load_shot_modes(clips_dir)

    print(f"Loading {len(clip_files)} clips from {clips_dir}/")
    clips: list[VideoFileClip] = []

    for idx, clip_path in enumerate(clip_files):
        clip = VideoFileClip(str(clip_path))
        shot_id = int(clip_path.stem)
        trim_seconds = 0.0
        if idx < len(clip_files) - 1 and shot_modes:
            next_id = int(clip_files[idx + 1].stem)
            next_mode = str(shot_modes.get(next_id, "t2v")).lower()
            if next_mode == "i2v":
                trim_seconds = I2V_TAIL_TRIM_SECONDS

        if trim_seconds > 0:
            max_trim = max(0.0, clip.duration - MIN_CLIP_DURATION_AFTER_TRIM)
            actual_trim = min(trim_seconds, max_trim)
            if actual_trim <= 0:
                print(f"  Warning: clip {clip_path.name} is too short to trim; leaving unmodified.")
            else:
                new_end = clip.duration - actual_trim
                clip = clip.subclipped(0, new_end)
                print(f"  Shot {shot_id}: trimmed tail by {actual_trim:.2f}s (next shot is I2V)")

        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    print(f"  Total duration: {video.duration:.1f}s")

    if music_path:
        music = AudioFileClip(music_path)
        # Play music only once (no looping). Trim if longer than video.
        if music.duration > video.duration:
            music = music.subclipped(0, video.duration)

        bg_music = music.with_volume_scaled(MUSIC_VOLUME)

        if video.audio is not None:
            mixed = CompositeAudioClip([video.audio, bg_music])
        else:
            mixed = bg_music

        # Apply master volume
        mixed = mixed.with_volume_scaled(TOTAL_VOLUME)
        video = video.with_audio(mixed)
        print(f"  Music: {music_path} (once, volume {MUSIC_VOLUME}, master {TOTAL_VOLUME})")
    else:
        print("  No background music — using original audio only")
        if video.audio is not None:
            video = video.with_audio(video.audio.with_volume_scaled(TOTAL_VOLUME))

    # Add fade out and black screen at the end
    FADE_DURATION = 2.0
    BLACK_DURATION = 1.0

    print(f"  Adding {FADE_DURATION}s fade-out and {BLACK_DURATION}s black screen...")

    # Fade out video and audio
    video = video.with_effects([FadeOut(duration=FADE_DURATION)])
    if video.audio is not None:
        video = video.with_audio(video.audio.with_effects([AudioFadeOut(duration=FADE_DURATION)]))

    # Append black clip
    black_clip = ColorClip(size=video.size, color=(0, 0, 0), duration=BLACK_DURATION).with_fps(FPS)
    video = concatenate_videoclips([video, black_clip])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Exporting to {out}...")
    video.write_videofile(
        str(out),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger="bar",
    )

    for clip in clips:
        clip.close()
    video.close()

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble raw clips into a final video.")
    parser.add_argument("--clips_dir", required=True,
                        help="Path to the raw clips folder (e.g. output/1/raw_clips/story1).")
    parser.add_argument("--music", default="assets/music/bg.mp3",
                        help="Optional path to background music file.")
    parser.add_argument("--output", default=None,
                        help="Output file path. Defaults to output/{video_id}/{video_id}.mp4.")
    args = parser.parse_args()

    clips_path = Path(args.clips_dir)
    if not clips_path.exists():
        print(f"Error: clips directory not found: {clips_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        video_id = clips_path.parent.parent.name
        story_name = clips_path.name
        output_path = str(Path("output") / video_id / f"{story_name}.mp4")

    assemble(args.clips_dir, args.music, output_path)


if __name__ == "__main__":
    main()
