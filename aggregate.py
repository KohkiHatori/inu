"""Aggregate multiple story videos into a single final video file.

Combines videos from command line arguments in order,
inserting a fixed subscribe clip between stories.
"""

import argparse
import sys
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips

# Fixed path for subscribe clip
SUBSCRIBE_CLIP_PATH = Path("output/subscribe.mp4")

def aggregate(
    video_paths: list[Path],
    output_path: Path
) -> None:
    """Concatenates videos with subscribe clips in between."""

    if not video_paths:
        print("Error: No input videos provided.")
        return

    clips = []
    
    # Load subscribe clip if it exists
    subscribe_clip = None
    if SUBSCRIBE_CLIP_PATH.exists():
        print(f"Loading subscribe clip: {SUBSCRIBE_CLIP_PATH}")
        try:
            subscribe_clip = VideoFileClip(str(SUBSCRIBE_CLIP_PATH))
        except Exception as e:
             print(f"Error loading subscribe clip: {e}")
             sys.exit(1)
    else:
        print(f"Warning: Subscribe clip not found at {SUBSCRIBE_CLIP_PATH}. Proceeding without it.")

    loaded_clips = [] # Keep track to close them later

    # Load all story videos
    for i, vid_path in enumerate(video_paths):
        if not vid_path.exists():
            print(f"Error: Video file not found: {vid_path}")
            sys.exit(1)
        
        print(f"Loading video: {vid_path}")
        try:
            clip = VideoFileClip(str(vid_path))
            loaded_clips.append(clip)
            clips.append(clip)
        except Exception as e:
            print(f"Error loading video {vid_path}: {e}")
            sys.exit(1)

        # Insert subscribe clip after each story
        if subscribe_clip:
            # Reusing the same clip object works for concatenation
            clips.append(subscribe_clip)

    if not clips:
        print("No clips to aggregate.")
        return

    print(f"Concatenating {len(clips)} clips...")
    try:
        final_video = concatenate_videoclips(clips, method="compose")
        
        print(f"Exporting to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger="bar"
        )
        
        final_video.close()
        
    except Exception as e:
        print(f"Error during concatenation/export: {e}")
        
    finally:
        # Cleanup
        for clip in loaded_clips:
            clip.close()
        
        if subscribe_clip:
            subscribe_clip.close()

    print("Done.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate story videos into a single file.")
    parser.add_argument("videos", nargs="+", help="List of video files to combine in order.")
    parser.add_argument("--output", required=True, help="Path to the final output video file.")

    args = parser.parse_args()

    video_paths = [Path(p) for p in args.videos]
    output_path = Path(args.output)

    aggregate(video_paths, output_path)

if __name__ == "__main__":
    main()
