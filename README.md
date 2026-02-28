## 1. Project Overview
- **Goal:** Create a fully automated Python application that generates 4 separate short films (2 minutes each) for a YouTube Kids channel.
- **Core Tech:** Google Gemini API (Veo/Video capabilities), Python, MoviePy.
- **Content Strategy:** 
- For landscape, normal YouTube videos, 4 stories × 15 shots each x 8-second shots = 60 total shots, 8 minutes in total. 
- For portrait, Youtube shorts, Instagram Reels, Tiktok videos, 1 story x 15 shots x 4-second shots = 15 total shots, 60 seconds in total 
- **Audio:** AI-generated sound effects (diegetic) + Looping background music track.
**Channel Description:** 
Welcome to Pup-Pop-Pup! 🐾
Join the heartwarming adventures of Pop (the patient dad) and his two energetic Pups as they explore the world, one paw print at a time!
We deliver realistic animal stories designed to be safe, relaxing, and entertaining for kids and dog lovers of all ages. From backyard discoveries to gentle life lessons, our videos capture the pure joy of growing up—without any dialogue, just the sounds of nature and happy tails.
What you'll find here:
🐶 Funny & Cute Moments: Puppies being clumsy, playful, and curious.
🌳 Nature Adventures: Beautiful, high-quality 4K scenes of the outdoors.
❤️ Family Bonds: The special connection between a father dog and his little ones.
New videos every week! Subscribe to join our furry family. 🔔
Safe viewing for children. No scary content, just good vibes.

## 2. System Architecture

### **A. Directory Structure**
The agent should set up the project file structure as follows:
```text
/project_root
  ├── assets/
  │   ├── music/          # Background music files (e.g., loop.mp3)
  │   └── refs/           # Character reference images (e.g., cat_ref.jpg)
  ├── output/
  │   ├── {video_id}/     # Output folder per video project
  │   │   ├── raw_clips/  # Clips organized by story_id
  │   │   └── final.mp4   # The final assembled video
  ├── config.yaml         # Global settings (resolutions, fps, paths)
  ├── stories/
  │   └── {video_id}/     # One folder per video
  │       ├── story1.yaml # Stories in generation order
  │       └── story2.yaml
  ├── main.py             # Orchestrator
  ├── generator.py        # API interaction logic
  └── assembler.py        # MoviePy editing logic
```

### **B. Input Data Schema (`story*.yaml`)**
This file is the output of `story_generator.py` and the input to `video_generator.py`.
**Schema:**
```yaml
title: "The Pups Discover the River"
concept: "Pop takes the two pups on their first trip to the river. The pups are scared at first but gain courage and splash in the shallows."
shots:
  - id: 1
    description: "Wide shot of Pop leading the two pups along a forest trail toward the sound of water. Morning light filters through the trees."
  - id: 2
    description: "Close-up of a pup's paws stopping at the muddy riverbank, hesitating."
  # ... up to id: 15
```

## 3. Module Specifications

### **Module 1: Story Generator (`story_generator.py`)**
**Library:** `google-genai`
**Model:** Gemini 3.1 Pro (text generation)

**Responsibilities:** Accept an optional user-supplied idea and a target format, then produce a fully-populated `story*.yaml` file with 15 shot descriptions aligned with the channel concept.

### **Module 2: Video Generator (`video_generator.py`)**
**Library:** `google-genai`
**Model:** Veo 3.1 (`veo-3.1-generate-fast-preview`)


### **Module 3: Assembler (`assembler.py`)**
**Library:** `moviepy`


***

## 4. Getting Started

### **Prerequisites**
- Python 3.10+
- Google Cloud Project with Vertex AI API enabled (for Veo) or Gemini API Key.
- `ffmpeg` installed (required by MoviePy and for frame extraction).

### **Installation**
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd inu
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_GENAI_USE_VERTEXAI=false # Set to true if using Vertex AI
   GOOGLE_CLOUD_PROJECT=your_project_id # Required if using Vertex AI
   GOOGLE_CLOUD_LOCATION=us-central1   # Required if using Vertex AI
   ```

---

## 5. How to Create a Video (Step-by-Step)

### **Step 1: Generate a Story**
Generate a 15-shot story plan based on an idea (or let AI invent one).
This creates a YAML file in `stories/{video_id}/`.

```bash
# Usage: python story_generator.py --video_id <ID> --idea "Optional idea"
python story_generator.py --video_id 1 --idea "The dogs go camping in the backyard"
```
*Output:* `stories/1/story1.yaml`

### **Step 2: Generate Video Clips**
Turn the story descriptions into actual video clips using Veo.
This saves raw clips to `output/{video_id}/raw_clips/{story_name}/`.

```bash
# Usage: python video_generator.py --story <path_to_yaml>
python video_generator.py --story stories/1/story1.yaml
```
*Output:* `output/1/raw_clips/story1/1.mp4`, `2.mp4`, ...

### **Step 3: Assemble Clips into a Story Video**
Stitch the raw clips together, add background music, and apply master volume.
This creates a single video file for that specific story.

```bash
# Usage: python assembler.py --clips_dir <path_to_raw_clips>
python assembler.py --clips_dir output/1/raw_clips/story1
```
*Output:* `output/1/story1.mp4`

*(Repeat Steps 1-3 to create multiple stories, e.g., story2, story3)*

### **Step 4: Create a Thumbnail (Optional)**
Generate a YouTube thumbnail for the story.

```bash
# Usage: python thumbnail.py <path_to_story_yaml>
python thumbnail.py stories/1/story1.yaml
```
*Output:* `output/1/story1_thumbnail.png`

### **Step 5: Aggregate Stories into Final Video**
Combine multiple story videos into one long-form video (e.g., 8 mins), inserting a "Subscribe" clip between them.

1. Ensure you have the subscribe clip generated (run once):
   ```bash
   python subscribe.py
   ```
   *Output:* `output/subscribe.mp4`

2. Run the aggregator:
   ```bash
   # Usage: python aggregate.py <video1> <video2> ... --output <final_path>
   python aggregate.py output/1/story1.mp4 output/1/story2.mp4 --output output/1/final_video.mp4
   ```
   *Output:* `output/1/final_video.mp4`

