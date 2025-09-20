# VideoLLM-online Benchmark Tool

This directory contains a non-interactive benchmark tool for VideoLLM-online that allows you to run configured videos and questions while measuring processing time and performance metrics.

## Files

- `benchmark.py` - Main benchmark script with command-line interface
- `benchmark_example.py` - Example script showing how to use the benchmark functionality
- `BENCHMARK_README.md` - This documentation file

## Quick Start

### Command Line Usage

```bash
# Basic usage (with checkpoint)
python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "Please narrate the video in real time." --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus

# With custom output file
python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "What is happening?" --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus --output_path results.json

# With custom parameters
python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "Describe the activities" --frame_fps 4 --max_frames 100 --streaming_threshold 0.8 --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus
```

### Python API Usage

```python
from demo.benchmark import benchmark_video_processing

# Run benchmark
metrics = benchmark_video_processing(
    video_path="demo/assets/cooking.mp4",
    question="Please narrate the video in real time.",
    output_path="results.json",
    max_frames=100,
    frame_fps=2,
    frame_resolution=384,
    streaming_threshold=0.725,
    verbose=True
)

# Access metrics
summary = metrics.get_summary()
print(f"Average FPS: {summary['average_fps']:.2f}")
print(f"Total processing time: {summary['total_processing_time']:.3f}s")
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video_path` | str | Required | Path to the input video file |
| `--question` | str | Required | Question to ask about the video |
| `--output_path` | str | None | Path to save benchmark results (JSON format) |
| `--max_frames` | int | None | Maximum number of frames to process |
| `--frame_fps` | int | 2 | Frames per second for processing |
| `--frame_resolution` | int | 384 | Resolution for frame processing |
| `--streaming_threshold` | float | 0.725 | Threshold for streaming responses |
| `--resume_from_checkpoint` | str | None | Path or HuggingFace model ID to load checkpoint from |
| `--quiet` | flag | False | Suppress verbose output |

## Output Format

The benchmark tool generates a JSON file with the following structure:

```json
{
  "video_path": "demo/assets/cooking.mp4",
  "question": "Please narrate the video in real time.",
  "configuration": {
    "frame_fps": 2,
    "frame_resolution": 384,
    "streaming_threshold": 0.725,
    "max_frames": 100
  },
  "metrics": {
    "video_loading_time": 1.234,
    "total_processing_time": 45.678,
    "total_frames_processed": 100,
    "total_responses_generated": 15,
    "average_frame_processing_time": 0.456,
    "average_response_generation_time": 0.123,
    "average_fps": 2.19,
    "min_fps": 1.85,
    "max_fps": 2.45,
    "total_conversation_entries": 15
  },
  "conversation_history": [
    {
      "query": "(Video Time = 0.0s) User: Please narrate the video in real time.",
      "response": "(Video Time = 0.0s) Assistant: I can see someone starting to cook...",
      "video_time": 0.0,
      "processing_time": 0.456,
      "fps": 2.19,
      "timestamp": 1234567890.123
    }
  ],
  "frame_processing_times": [0.456, 0.423, 0.489, ...],
  "response_generation_times": [0.123, 0.145, 0.134, ...],
  "fps_measurements": [2.19, 2.15, 2.23, ...]
}
```

## Performance Metrics

The benchmark tool measures several key performance indicators:

### Basic Metrics
- **Video Loading Time**: Time to load and preprocess the video
- **Frame Processing Time**: Time to process each individual frame
- **Response Generation Time**: Time to generate responses
- **FPS (Frames Per Second)**: Real-time processing speed (min/max/average)
- **Total Processing Time**: Overall time to process the entire video
- **Conversation History**: Complete log of queries and responses with timing

### Detailed Frame Processing Metrics
- **Visual Embedding Time**: Time to encode video frames into embeddings
- **Model Forward Time**: Time for the language model forward pass
- **Generation Time**: Time for text generation (prefill + decode)
- **Prefill Time**: Time for initial context processing
- **Decode Time**: Time for token-by-token generation
- **Per-Frame Breakdown**: Detailed timing for each individual frame

### Advanced Analysis
- **Timing Breakdown**: Total time spent in each processing stage
- **Performance Trends**: FPS progression over time
- **Bottleneck Identification**: Which processing stage takes the most time
- **Response Frequency**: How often responses are generated vs. frame-only processing

## Examples

### Example 1: Basic Benchmark
```bash
python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "Please narrate the video in real time."
```

### Example 2: Performance Testing
```bash
python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "What is happening?" --max_frames 200 --frame_fps 4 --output_path performance_test.json
```

### Example 3: Batch Processing
```python
# Run multiple benchmarks programmatically
videos = ["demo/assets/cooking.mp4", "demo/assets/bicycle.mp4"]
questions = ["Please narrate the video.", "What activities do you see?"]

for i, video in enumerate(videos):
    for j, question in enumerate(questions):
        output_path = f"benchmark_{i}_{j}.json"
        benchmark_video_processing(video, question, output_path, max_frames=50)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- VideoLLM-online dependencies
- FFmpeg (for video preprocessing)

## Notes

- The tool automatically preprocesses videos using FFmpeg if needed
- Cached preprocessed videos are stored in `demo/assets/cache/`
- Processing time may vary significantly depending on hardware (GPU vs CPU)
- For accurate benchmarking, ensure consistent hardware and software conditions
- The tool supports both interactive and non-interactive modes

## Troubleshooting

1. **Video not found**: Ensure the video path is correct and the file exists
2. **CUDA errors**: The tool will fall back to CPU if CUDA is not available
3. **Memory issues**: Reduce `max_frames` or `frame_resolution` for large videos
4. **Slow processing**: Check if you're running on CPU instead of GPU