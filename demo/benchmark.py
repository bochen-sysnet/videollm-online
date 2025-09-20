#!/usr/bin/env python3
"""
Non-interactive benchmark script for VideoLLM-online demo.
Runs a configured video and question while measuring processing time.
"""

import os
import time
import json
import argparse
import torch
import torchvision
from dataclasses import asdict
from typing import Optional, Dict, List, Tuple
import torch.multiprocessing as mp

from data.utils import ffmpeg_once
from .inference import LiveInfer
from models import parse_args
import transformers

logger = transformers.logging.get_logger('liveinfer')

def create_liveinfer_without_args(resume_from_checkpoint: str = None):
    """Create a LiveInfer instance without parsing command line arguments."""
    import sys
    from models import build_model_and_tokenizer
    from dataclasses import asdict
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Temporarily clear sys.argv to avoid argument parsing conflicts
        sys.argv = ['benchmark.py']
        
        # Parse args manually with default values
        args = parse_args()
        
        # Override checkpoint if provided
        if resume_from_checkpoint:
            args.resume_from_checkpoint = resume_from_checkpoint
        
        # Build model and tokenizer
        if not torch.cuda.is_available() and args.attn_implementation == 'flash_attention_2':
            logger.warning("Flash attention not available without CUDA. Falling back to SDPA implementation.")
            args.attn_implementation = 'sdpa'
        model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            logger.warning("CUDA not available. Running inference on CPU, which will be extremely slow.")
        model.to(device)
        
        # Create LiveInfer instance manually
        liveinfer = LiveInfer.__new__(LiveInfer)
        liveinfer.model = model
        liveinfer.tokenizer = tokenizer
        liveinfer.device = device
        
        # Set up visual parameters
        liveinfer.hidden_size = model.config.hidden_size
        liveinfer.frame_fps = args.frame_fps
        liveinfer.frame_interval = 1 / liveinfer.frame_fps
        liveinfer.frame_resolution = model.config.frame_resolution
        liveinfer.frame_num_tokens = model.config.frame_num_tokens
        liveinfer.frame_v_placeholder = model.config.v_placeholder * liveinfer.frame_num_tokens
        liveinfer.frame_token_interval_id = model.config.frame_token_interval_id
        liveinfer.frame_placeholder_ids = torch.tensor(model.config.v_placeholder_id, device=device).repeat(model.config.frame_num_tokens).reshape(1,-1)
        
        # Set up generation parameters
        liveinfer.system_prompt = args.system_prompt
        liveinfer.inplace_output_ids = torch.zeros(1, 100, device=device, dtype=torch.long)
        liveinfer.frame_token_interval_threshold = 0.725
        liveinfer.eos_token_id = model.config.eos_token_id
        liveinfer._start_ids = tokenizer.apply_chat_template([{'role': 'system', 'content': liveinfer.system_prompt}], add_stream_prompt=True, return_tensors='pt').to(device)
        liveinfer._added_stream_prompt_ids = tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to(device)
        liveinfer._added_stream_generation_ids = tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to(device)
        
        # Initialize app state
        liveinfer.reset()
        
        return liveinfer
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

class InstrumentedLiveInfer:
    """Wrapper around LiveInfer that adds detailed timing instrumentation."""
    
    def __init__(self, liveinfer):
        self.liveinfer = liveinfer
        self.timing_data = {}
        
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LiveInfer instance."""
        return getattr(self.liveinfer, name)
    
    def input_video_stream(self, video_time):
        """Instrumented version of input_video_stream with timing.
        
        This measures the visual embedding process:
        Raw RGB frames → Vision encoder (SigLIP/CLIP) → Spatial pooling → Connector network → Visual token embeddings
        """
        start_time = time.time()
        
        # Measure visual embedding time: RGB frames → Visual token embeddings
        visual_start = time.time()
        frame_idx = int(video_time * self.liveinfer.frame_fps)
        if frame_idx > self.liveinfer.last_frame_idx:
            ranger = range(self.liveinfer.last_frame_idx + 1, frame_idx + 1)
            # This calls: vision_encoder → spatial_pooling → connector_network
            frames_embeds = self.liveinfer.model.visual_embed(self.liveinfer.video_tensor[ranger]).split(self.liveinfer.frame_num_tokens)
            self.liveinfer.frame_embeds_queue.extend([(r / self.liveinfer.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
        visual_embedding_time = time.time() - visual_start
        
        self.liveinfer.last_frame_idx = frame_idx
        self.liveinfer.video_time = video_time
        
        # Store timing data
        self.timing_data['visual_embedding_time'] = visual_embedding_time  # RGB → Visual embeddings
        self.timing_data['input_video_stream_time'] = time.time() - start_time
        
    def __call__(self):
        """Instrumented version of the call method with detailed timing.
        
        This measures two main processes:
        1. Streaming: Visual tokens + text context → LLM forward pass → logits
        2. Generation: Logits → token selection → text response (when occurs)
        """
        start_time = time.time()
        
        # Measure streaming processing time: Visual tokens + context → LLM → logits
        streaming_start = time.time()
        while not self.liveinfer.frame_embeds_queue:
            continue
        video_time, query = self.liveinfer._call_for_streaming()
        streaming_time = time.time() - streaming_start
        
        # Measure response generation time: Logits → token selection → text
        generation_start = time.time()
        response = None
        if video_time is not None:
            query, response = self.liveinfer._call_for_response(video_time, query)
        generation_time = time.time() - generation_start
        
        # Store timing data
        self.timing_data['streaming_time'] = streaming_time      # Visual tokens → Logits
        self.timing_data['generation_time'] = generation_time    # Logits → Text (when occurs)
        self.timing_data['total_call_time'] = time.time() - start_time
        
        return query, response
    
    def get_timing_data(self):
        """Get the timing data for the last operation."""
        return self.timing_data.copy()

class BenchmarkMetrics:
    """Class to track and store benchmark metrics."""
    
    def __init__(self):
        self.video_loading_time = 0.0
        self.total_processing_time = 0.0
        self.frame_processing_times = []
        self.response_generation_times = []
        self.total_frames_processed = 0
        self.total_responses_generated = 0
        self.fps_measurements = []
        self.conversation_history = []
        
        # Detailed frame processing metrics
        self.frame_metrics = []  # List of detailed metrics per frame
        self.visual_embedding_times = []
        self.model_forward_times = []
        self.generation_times = []
        
    def add_frame_processing_time(self, processing_time: float, fps: float):
        """Add a frame processing time measurement."""
        self.frame_processing_times.append(processing_time)
        self.fps_measurements.append(fps)
        self.total_frames_processed += 1
        
    def add_response_generation_time(self, generation_time: float):
        """Add a response generation time measurement."""
        self.response_generation_times.append(generation_time)
        self.total_responses_generated += 1
        
    def add_detailed_frame_metrics(self, frame_idx: int, video_time: float, 
                                 visual_embedding_time: float, model_forward_time: float,
                                 generation_time: float, total_time: float, fps: float):
        """Add detailed metrics for a single frame.
        
        Args:
            frame_idx: Frame number being processed
            video_time: Time in video (seconds)
            visual_embedding_time: Time to convert RGB frames → visual token embeddings
            model_forward_time: Time for LLM forward pass (visual tokens → logits)
            generation_time: Time for text generation (logits → text, when occurs)
            total_time: Total processing time for this frame
            fps: Frames per second at this point
        """
        frame_metric = {
            'frame_idx': frame_idx,
            'video_time': video_time,
            'visual_embedding_time': visual_embedding_time,      # RGB → Visual embeddings
            'model_forward_time': model_forward_time,            # Visual tokens → Logits
            'generation_time': generation_time,                  # Logits → Text (when occurs)
            'total_time': total_time,                            # Total frame processing
            'fps': fps                                           # Real-time processing speed
        }
        self.frame_metrics.append(frame_metric)
        
        # Add to individual timing lists
        self.visual_embedding_times.append(visual_embedding_time)
        self.model_forward_times.append(model_forward_time)
        self.generation_times.append(generation_time)
        
    def add_conversation_entry(self, query: str, response: str, video_time: float, 
                             processing_time: float, fps: float):
        """Add a conversation entry with timing information."""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'video_time': video_time,
            'processing_time': processing_time,
            'fps': fps,
            'timestamp': time.time()
        })
        
    def get_summary(self) -> Dict:
        """Get a summary of all metrics."""
        avg_frame_processing_time = sum(self.frame_processing_times) / len(self.frame_processing_times) if self.frame_processing_times else 0
        avg_response_generation_time = sum(self.response_generation_times) / len(self.response_generation_times) if self.response_generation_times else 0
        
        # Filter out zero FPS values for min calculation
        non_zero_fps = [fps for fps in self.fps_measurements if fps > 0]
        avg_fps = sum(self.fps_measurements) / len(self.fps_measurements) if self.fps_measurements else 0
        min_fps = min(non_zero_fps) if non_zero_fps else 0
        max_fps = max(self.fps_measurements) if self.fps_measurements else 0
        
        # Simple timing averages
        avg_visual_embedding_time = sum(self.visual_embedding_times) / len(self.visual_embedding_times) if self.visual_embedding_times else 0
        avg_model_forward_time = sum(self.model_forward_times) / len(self.model_forward_times) if self.model_forward_times else 0
        
        # Generation time should only be averaged over frames that actually generated responses
        # This should match average_response_generation_time
        avg_generation_time = sum(self.response_generation_times) / len(self.response_generation_times) if self.response_generation_times else 0
        
        return {
            'video_loading_time': self.video_loading_time,
            'total_processing_time': self.total_processing_time,
            'total_frames_processed': self.total_frames_processed,
            'total_responses_generated': self.total_responses_generated,
            'average_frame_processing_time': avg_frame_processing_time,
            'average_response_generation_time': avg_response_generation_time,
            'average_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'total_conversation_entries': len(self.conversation_history),
            # Simple timing metrics
            'average_visual_embedding_time': avg_visual_embedding_time,
            'average_model_forward_time': avg_model_forward_time,
            'average_generation_time': avg_generation_time,
            'total_visual_embedding_time': sum(self.visual_embedding_times),
            'total_model_forward_time': sum(self.model_forward_times),
            'total_generation_time': sum(self.generation_times)
        }

def benchmark_video_processing(
    video_path: str,
    question: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_fps: int = 2,
    frame_resolution: int = 384,
    streaming_threshold: float = 0.725,
    resume_from_checkpoint: str = None,
    verbose: bool = True
) -> BenchmarkMetrics:
    """
    Benchmark video processing with a given question.
    
    Args:
        video_path: Path to the input video file
        question: Question to ask about the video
        output_path: Path to save the benchmark results (optional)
        max_frames: Maximum number of frames to process (optional)
        frame_fps: Frames per second for processing
        frame_resolution: Resolution for frame processing
        streaming_threshold: Threshold for streaming responses
        verbose: Whether to print progress information
        
    Returns:
        BenchmarkMetrics object with all timing measurements
    """
    
    # Initialize metrics
    metrics = BenchmarkMetrics()
    
    # Initialize the model
    if verbose:
        print("Initializing VideoLLM-online model...")
    
    # Create a custom LiveInfer that doesn't parse command line arguments
    base_liveinfer = create_liveinfer_without_args(resume_from_checkpoint)
    base_liveinfer.frame_fps = frame_fps
    base_liveinfer.frame_resolution = frame_resolution
    base_liveinfer.frame_token_interval_threshold = streaming_threshold
    
    # Wrap with instrumentation
    liveinfer = InstrumentedLiveInfer(base_liveinfer)
    
    # Prepare video path
    name, ext = os.path.splitext(video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', 
                                    name + f'_{frame_fps}fps_{frame_resolution}' + ext)
    
    # Load and preprocess video
    if verbose:
        print(f"Loading video: {video_path}")
    
    start_time = time.time()
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(video_path, ffmpeg_video_path, fps=frame_fps, resolution=frame_resolution)
        if verbose:
            print(f"Video preprocessed: {video_path} -> {ffmpeg_video_path}")
    
    liveinfer.load_video(ffmpeg_video_path)
    metrics.video_loading_time = time.time() - start_time
    
    if verbose:
        print(f"Video loaded: {liveinfer.num_video_frames} frames, {liveinfer.video_duration:.2f}s duration")
        print(f"Video loading time: {metrics.video_loading_time:.3f}s")
    
    # Add the question
    liveinfer.input_query_stream(question, video_time=0.0)
    
    # Process video frames
    if verbose:
        print(f"Processing video frames...")
        print(f"Question: {question}")
        print("-" * 50)
    
    start_processing_time = time.time()
    total_frames = min(liveinfer.num_video_frames, max_frames) if max_frames else liveinfer.num_video_frames
    
    for i in range(total_frames):
        frame_start_time = time.time()
        video_time = i / frame_fps
        
        try:
            # Process the frame with detailed timing
            liveinfer.input_video_stream(video_time)
            query, response = liveinfer()
            
            frame_processing_time = time.time() - frame_start_time
            
            # Calculate current FPS
            current_fps = (i + 1) / (time.time() - start_processing_time) if i > 0 else 0
            
            # Get detailed timing data
            timing_data = liveinfer.get_timing_data()
            
            # Extract detailed metrics
            visual_embedding_time = timing_data.get('visual_embedding_time', 0.0)
            streaming_time = timing_data.get('streaming_time', 0.0)
            generation_time = timing_data.get('generation_time', 0.0)
            
            # Simple timing components - no complex prefill/decode concepts
            # Just show the actual measured times and their sum
            
            # Record detailed metrics
            metrics.add_detailed_frame_metrics(
                frame_idx=i,
                video_time=video_time,
                visual_embedding_time=visual_embedding_time,
                model_forward_time=streaming_time,
                generation_time=generation_time,
                total_time=frame_processing_time,
                fps=current_fps
            )
            
            # Record basic metrics
            metrics.add_frame_processing_time(frame_processing_time, current_fps)
            
            # Track response generation separately
            if response:
                metrics.add_response_generation_time(generation_time)
            
            # Record conversation if there's a query or response
            if query or response:
                metrics.add_conversation_entry(query, response, video_time, frame_processing_time, current_fps)
                
                if verbose:
                    if query:
                        print(f"[{video_time:.2f}s] {query}")
                    if response:
                        print(f"[{video_time:.2f}s] {response}")
                        if verbose:
                            print(f"  └─ Generation time: {generation_time:.3f}s")
            
            # Print progress every 10 frames
            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_frames} frames (FPS: {current_fps:.2f})")
                
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing frame {i}: {e}")
            # Continue processing other frames
            frame_processing_time = time.time() - frame_start_time
            current_fps = (i + 1) / (time.time() - start_processing_time) if i > 0 else 0
            metrics.add_frame_processing_time(frame_processing_time, current_fps)
    
    metrics.total_processing_time = time.time() - start_processing_time
    
    # Print summary
    if verbose:
        print("-" * 50)
        print("BENCHMARK SUMMARY")
        print("-" * 50)
        summary = metrics.get_summary()
        
        # Basic metrics
        print("BASIC METRICS:")
        basic_keys = ['video_loading_time', 'total_processing_time', 'total_frames_processed', 
                     'total_responses_generated', 'average_fps', 'min_fps', 'max_fps']
        for key in basic_keys:
            if key in summary:
                value = summary[key]
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\nTIMING COMPONENTS:")
        print(f"  average_frame_processing_time: {summary.get('average_frame_processing_time', 0):.3f}s")
        print(f"  average_response_generation_time: {summary.get('average_response_generation_time', 0):.3f}s")
        print(f"  average_visual_embedding_time: {summary.get('average_visual_embedding_time', 0):.3f}s")
        print(f"  average_model_forward_time: {summary.get('average_model_forward_time', 0):.3f}s")
        print(f"  average_generation_time: {summary.get('average_generation_time', 0):.3f}s")
        
        print(f"\nTOTAL TIMING BREAKDOWN:")
        print(f"  Total visual embedding time: {summary.get('total_visual_embedding_time', 0):.3f}s")
        print(f"  Total model forward time: {summary.get('total_model_forward_time', 0):.3f}s")
        print(f"  Total generation time: {summary.get('total_generation_time', 0):.3f}s")
    
    # Save results if output path is provided
    if output_path:
        results = {
            'video_path': video_path,
            'question': question,
            'configuration': {
                'frame_fps': frame_fps,
                'frame_resolution': frame_resolution,
                'streaming_threshold': streaming_threshold,
                'max_frames': max_frames,
                'resume_from_checkpoint': resume_from_checkpoint
            },
            'metrics': metrics.get_summary(),
            'conversation_history': metrics.conversation_history,
            'frame_processing_times': metrics.frame_processing_times,
            'response_generation_times': metrics.response_generation_times,
            'fps_measurements': metrics.fps_measurements,
            'detailed_frame_metrics': metrics.frame_metrics,
            'visual_embedding_times': metrics.visual_embedding_times,
            'model_forward_times': metrics.model_forward_times,
            'generation_times': metrics.generation_times
        }
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"Results saved to: {output_path}")
    
    return metrics

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Benchmark VideoLLM-online processing')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to the input video file')
    parser.add_argument('--question', type=str, required=True,
                       help='Question to ask about the video')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save benchmark results (JSON format)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--frame_fps', type=int, default=2,
                       help='Frames per second for processing (default: 2)')
    parser.add_argument('--frame_resolution', type=int, default=384,
                       help='Resolution for frame processing (default: 384)')
    parser.add_argument('--streaming_threshold', type=float, default=0.725,
                       help='Threshold for streaming responses (default: 0.725)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path or HuggingFace model ID to load checkpoint from (e.g., chenjoya/videollm-online-8b-v1plus)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    # Parse only our arguments, ignore unknown ones
    args, unknown = parser.parse_known_args()
    
    # Validate video path
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Run benchmark
    try:
        metrics = benchmark_video_processing(
            video_path=args.video_path,
            question=args.question,
            output_path=args.output_path,
            max_frames=args.max_frames,
            frame_fps=args.frame_fps,
            frame_resolution=args.frame_resolution,
            streaming_threshold=args.streaming_threshold,
            resume_from_checkpoint=args.resume_from_checkpoint,
            verbose=not args.quiet
        )
        
        return 0
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return 1

if __name__ == '__main__':
    exit(main())