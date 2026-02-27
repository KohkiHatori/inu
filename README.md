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
