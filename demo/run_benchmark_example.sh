#!/bin/bash

# Example script showing how to use the VideoLLM-online benchmark tool
# Make sure to activate the virtual environment first

echo "VideoLLM-online Benchmark Examples"
echo "=================================="

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "1. Basic benchmark with cooking video:"
echo "--------------------------------------"
python -m demo.benchmark \
    --video_path demo/assets/cooking.mp4 \
    --question "Please narrate the video in real time." \
    --max_frames 20 \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --output_path benchmark_cooking.json

echo ""
echo "2. Performance test with bicycle video:"
echo "---------------------------------------"
python -m demo.benchmark \
    --video_path demo/assets/bicycle.mp4 \
    --question "What activities do you see in this video?" \
    --max_frames 30 \
    --frame_fps 4 \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --output_path benchmark_bicycle.json

echo ""
echo "3. Quick test with custom parameters:"
echo "-------------------------------------"
python -m demo.benchmark \
    --video_path demo/assets/cooking.mp4 \
    --question "Describe what's happening step by step." \
    --max_frames 10 \
    --frame_resolution 384 \
    --streaming_threshold 0.8 \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --output_path benchmark_quick.json \
    --quiet

echo ""
echo "Benchmark examples completed!"
echo "Check the generated JSON files for detailed results."