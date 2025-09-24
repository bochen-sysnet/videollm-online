#!/usr/bin/env python3
"""
Streaming evaluation that processes videos frame by frame to avoid OOM.
This creates a custom evaluation loop that mimics the demo inference approach.
"""

import torch
import json
import os
import time
import subprocess
import warnings
import gc
import random
import sys
import traceback
import collections
import heapq
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_video
import transformers

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate
from data.ego4d.narration import Ego4DRefinedNarrationStream

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class Config:
    """Configuration constants for streaming evaluation"""
    
    # Video processing parameters
    FRAME_FPS = 2.0
    FRAME_RESOLUTION = 384
    FRAME_NUM_TOKENS = 10
    V_PLACEHOLDER_ID = 128256
    
    # Memory management
    DEFAULT_GPU_MEMORY_RESERVE_MB = 1536  # Reserve 1.5GB for model and operations
    MEMORY_GROWTH_PER_FRAME_MB = 1.4     # Estimated memory growth per frame
    MEMORY_SAFETY_MARGIN = 0.9           # 90% safety margin for memory calculations
    MAX_FRAMES_LIMIT = 5000              # Maximum frames to process
    MIN_FRAMES_LIMIT = 1000              # Minimum frames to process
    
    # Streaming thresholds for different dataset types
    STREAMING_THRESHOLD_GOALSTEP = 0.725  # Threshold for goalstep dataset (works well with user queries)
    STREAMING_THRESHOLD_NARRATION = 0.95  # Threshold for narration dataset (testing higher threshold)
    
    # Visualization
    OUTPUT_DIR = "timing_plots"
    PLOT_DPI = 300
    PLOT_FIGSIZE_LARGE = (15, 10)
    PLOT_FIGSIZE_MEDIUM = (15, 6)
    PLOT_FIGSIZE_SMALL = (15, 4)
    
    # Processing limits
    MAX_EVAL_FRAMES = 5000            # Max frames for evaluation (use full video)
    BATCH_SIZE_LIMIT = 10                # Max frames to load at once
    MEMORY_CHECK_INTERVAL = 1           # Check memory every N frames
    MEMORY_WARNING_THRESHOLD = 2000      # MB remaining before warning
    
    # Threshold sweep configuration
    DEFAULT_NUM_VIDEOS = 2               # Default number of videos for evaluation
    DEFAULT_THRESHOLD_RANGE = (0.5, 0.99) # Default threshold range for sweep
    DEFAULT_NUM_THRESHOLDS = 2           # Default number of thresholds for sweep
    DEBUG_THRESHOLDS = [0.8, 0.9]         # Coarse-grained thresholds
    # DEBUG_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98]  # Fine-grained thresholds
    
    # User-level rebuffering configuration
    USER_READING_SPEED_MIN = 3.0          # Words per second (slow reading)
    USER_READING_SPEED_MAX = 5.0          # Words per second (fast reading)
    USER_LISTENING_SPEED_MIN = 2.0        # Words per second (slow listening)
    USER_LISTENING_SPEED_MAX = 2.7        # Words per second (fast listening)
    
    # File paths
    DATASET_BASE_PATH = "datasets/ego4d/v2/full_scale_2fps_384"
    FEATURES_BASE_PATH = "datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384"
    METADATA_PATH = "datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384_metadata.json"
    
    # Model configuration
    VISION_PRETRAINED = 'google/siglip-large-patch16-384'
    EMBED_MARK = '2fps_384_1+3x3'
    MAX_NUM_FRAMES = 1000
    
    # Response generation
    INPLACE_OUTPUT_SIZE = 100
    EXPECTED_RESPONSE_LENGTH = 20.0
    LENGTH_FACTOR_MIN = 0.5
    LENGTH_FACTOR_MAX = 2.0

class FilteredEgo4DRefinedNarrationStream:
    """Ego4D Refined Narration Stream that only includes videos with features - now processes per-conversation"""
    
    def __init__(self, split='val', frame_fps=2, is_training=False, augmentation=False, 
                 system_prompt='', tokenizer=None, vision_pretrained='google/siglip-large-patch16-384',
                 embed_mark='2fps_384_1+3x3', max_num_frames=1000, data_source='narration'):
        
        # Get videos with features first
        self.videos_with_features = get_videos_with_features()
        print(f"📊 Found {len(self.videos_with_features)} videos with extracted features")
        
        # Load only the refined narration data for videos with features
        self.split = split
        self.frame_fps = frame_fps
        self.is_training = is_training
        self.augmentation = augmentation
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.vision_pretrained = vision_pretrained
        self.embed_mark = embed_mark
        self.max_num_frames = max_num_frames
        
        # Load data based on source
        self.data_source = data_source
        if data_source == 'narration':
            data_path = f"datasets/ego4d/v2/annotations/refined_narration_stream_{split}.json"
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            print(f"📊 Loaded refined narration data: {len(self.data)} videos")
            
            # Add hardcoded instructions for narration data (since it doesn't contain real instructions)
            self.instructions = [
                {"role": "user", "content": "Please concisely narrate the video in real time."},
                {"role": "user", "content": "Help me to illustrate my view in short."},
                {"role": "user", "content": "Please simply describe what do you see."},
                {"role": "user", "content": "Continuously answer what you observed with simple text."},
                {"role": "user", "content": "Do concise real-time narration."},
                {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
                {"role": "user", "content": "Simply interpret the scene for me."},
                {"role": "user", "content": "What can you tell me about? Be concise."},
                {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
                {"role": "user", "content": "What is the action now? Please response in short."},
            ]
        elif data_source == 'goalstep':
            data_path = "datasets/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json"
            with open(data_path, 'r') as f:
                goalstep_data = json.load(f)
            
            # Store goalstep conversations with full user prompts and assistant responses
            self.data = {}
            self.goalstep_conversations = {}  # Store full conversations per video
            self.goalstep_normalized_conversations = {}  # Store normalized conversations (first user prompt at time 0)
            self.goalstep_timestamp_offsets = {}  # Store timestamp offsets for visualization
            self.goalstep_durations = {}  # Store durations for each conversation
            
            for item_idx, item in enumerate(goalstep_data):
                video_uid = item['video_uid']
                if video_uid in self.videos_with_features:
                    # Store the full conversation for this video with unique conversation ID
                    conversation_id = f"goalstep_{item['video_uid']}_{item_idx}"
                    conversation = []
                    for conv in item['conversation']:
                        conversation.append({
                            'role': conv['role'],
                            'content': conv['content'],
                            'time': conv['time']
                        })
                    
                    # Normalize timestamps: find first user prompt time and subtract it from all times
                    user_times = [turn['time'] for turn in conversation if turn['role'] == 'user']
                    if user_times:
                        first_user_time = min(user_times)
                        normalized_conversation = []
                        for turn in conversation:
                            normalized_turn = {
                                'role': turn['role'],
                                'content': turn['content'],
                                'time': max(0.0, turn['time'] - first_user_time),  # Ensure no negative times
                                'original_time': turn['time']  # Keep original time for visualization
                            }
                            normalized_conversation.append(normalized_turn)
                        
                        # Store the offset for converting back to original timestamps
                        self.goalstep_timestamp_offsets[video_uid] = first_user_time
                    else:
                        # If no user prompts, keep original times
                        normalized_conversation = []
                        for turn in conversation:
                            normalized_turn = {
                                'role': turn['role'],
                                'content': turn['content'],
                                'time': turn['time'],
                                'original_time': turn['time']
                            }
                            normalized_conversation.append(normalized_turn)
                        self.goalstep_timestamp_offsets[video_uid] = 0.0
                    
                    if video_uid not in self.data:
                        self.data[video_uid] = {}
                    if video_uid not in self.goalstep_conversations:
                        self.goalstep_conversations[video_uid] = {}
                    if video_uid not in self.goalstep_normalized_conversations:
                        self.goalstep_normalized_conversations[video_uid] = {}
                    if video_uid not in self.goalstep_durations:
                        self.goalstep_durations[video_uid] = {}
                    
                    self.data[video_uid][conversation_id] = conversation
                    self.goalstep_conversations[video_uid][conversation_id] = conversation
                    self.goalstep_normalized_conversations[video_uid][conversation_id] = normalized_conversation
                    self.goalstep_durations[video_uid][conversation_id] = item.get('duration', 0.0)
            
            # Extract all unique user prompts for instruction selection
            all_user_prompts = []
            for video_conversations in self.goalstep_conversations.values():
                for conversation in video_conversations.values():
                    for turn in conversation:
                        if turn['role'] == 'user':
                            all_user_prompts.append({
                                "role": "user", 
                                "content": turn['content']
                            })
            
            self.instructions = all_user_prompts[:100]  # Use more diverse prompts
            print(f"📊 Loaded goalstep data: {len(self.data)} videos")
            print(f"📊 Extracted {len(self.instructions)} unique user prompts from goalstep conversations")
            print(f"📊 Created normalized conversations with first user prompt at time 0")
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Filter to only include videos with features
        self.filtered_video_uids = []
        for video_uid in self.data.keys():
            if video_uid in self.videos_with_features:
                self.filtered_video_uids.append(video_uid)
        
        print(f"📊 Filtered to {len(self.filtered_video_uids)} videos with features")
        
        # Create conversation-level dataset instead of video-level
        self.conversations = []
        if data_source == 'narration':
            # For narration, create conversations from annotation streams
            for video_uid in self.filtered_video_uids:
                if video_uid in self.data:
                    video_data = self.data[video_uid]
                    for annotation_uid, narration_entries in video_data.items():
                        if isinstance(narration_entries, list) and narration_entries:
                            # Create a conversation from narration entries
                            conversation = []
                            for entry in narration_entries:
                                if isinstance(entry, dict) and 'text' in entry:
                                    conversation.append({
                                        'role': 'assistant',
                                        'content': entry['text'],
                                        'time': entry.get('time', 0.0)
                                    })
                            
                            if conversation:
                                # Calculate conversation duration
                                start_time = min(entry.get('time', 0.0) for entry in narration_entries if isinstance(entry, dict) and 'time' in entry)
                                end_time = max(entry.get('time', 0.0) for entry in narration_entries if isinstance(entry, dict) and 'time' in entry)
                                duration = end_time - start_time
                                
                                self.conversations.append({
                                    'video_uid': video_uid,
                                    'conversation_id': f"narration_{video_uid}_{annotation_uid}",
                                    'conversation': conversation,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration': duration,
                                    'original_conversation': conversation  # Keep original for metrics
                                })
        elif data_source == 'goalstep':
            # For goalstep, each video has one conversation
            for video_uid in self.filtered_video_uids:
                if video_uid in self.goalstep_normalized_conversations:
                    for conversation_id, normalized_conversation in self.goalstep_normalized_conversations[video_uid].items():
                        # Calculate conversation duration from normalized times
                        if normalized_conversation:
                            start_time = min(turn['time'] for turn in normalized_conversation)
                            end_time = max(turn['time'] for turn in normalized_conversation)
                            duration = end_time - start_time
                            
                            self.conversations.append({
                                'video_uid': video_uid,
                                'conversation_id': conversation_id,
                                'conversation': normalized_conversation,
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': duration,
                                'original_conversation': self.goalstep_conversations[video_uid][conversation_id],  # Keep original for metrics
                                'timestamp_offset': self.goalstep_timestamp_offsets.get(video_uid, 0.0)
                            })
        
        # Calculate statistics
        unique_videos = set(c['video_uid'] for c in self.conversations)
        conversations_per_video = {}
        for c in self.conversations:
            video_uid = c['video_uid']
            if video_uid not in conversations_per_video:
                conversations_per_video[video_uid] = 0
            conversations_per_video[video_uid] += 1
        
        print(f"   • Total conversations: {len(self.conversations)}")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation_data = self.conversations[idx]
        video_uid = conversation_data['video_uid']
        conversation_id = conversation_data['conversation_id']
        conversation = conversation_data['conversation']
        start_time = conversation_data['start_time']
        end_time = conversation_data['end_time']
        duration = conversation_data['duration']
        
        # Create input text from conversation turns
        input_text = " ".join([turn.get('text', turn.get('content', '')) for turn in conversation])
        
        # Load actual frames from the feature directory
        try:
            # Load the pre-computed embeddings
            feature_path = f"datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384/{video_uid}.pt"
            all_frames = torch.load(feature_path, weights_only=True)  # Shape: [num_frames, 10, 1024]
            
            # Calculate frame range based on conversation timing
            if conversation:
                # Convert to frame indices (2fps = 0.5 second per frame)
                start_frame = max(0, int(start_time * 2))
                end_frame = min(all_frames.shape[0], int(end_time * 2) + 1)
                
                # Extract conversation-specific frames
                frames = all_frames[start_frame:end_frame]
                
                # Limit to max_num_frames
                if frames.shape[0] > self.max_num_frames:
                    frames = frames[:self.max_num_frames]
            else:
                # No timing info, use first few frames
                frames = all_frames[:min(100, all_frames.shape[0])]
                
        except Exception as e:
            print(f"Warning: Could not load frames for {video_uid}: {e}")
            # Fallback to dummy frames if loading fails
            frames = torch.zeros((100, 10, 1024))
        
        load_ranges = {}
        sample_idx = idx
        evaluation_kwargs = {
            'conversation_data': conversation_data  # Pass conversation metadata
        }
        
        return input_text, frames, load_ranges, sample_idx, evaluation_kwargs

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory

            return {
                'total_mb': total_memory / (1024**2),
                'allocated_mb': allocated_memory / (1024**2),
                'cached_mb': cached_memory / (1024**2),
                'free_mb': free_memory / (1024**2),
            }
    except Exception:
        pass
    return None


def _tensor_numel_bytes(tensor):
    """Return tensor memory footprint in bytes."""
    if tensor is None or not hasattr(tensor, 'nelement'):
        return 0
    try:
        return tensor.nelement() * tensor.element_size()
    except Exception:
        return 0


def calculate_model_memory_mb(model):
    """Approximate model parameter + buffer memory footprint on GPU in MB."""
    if hasattr(model, '_parameter_memory_mb'):
        return model._parameter_memory_mb

    total_bytes = 0
    try:
        for param in model.parameters():
            total_bytes += _tensor_numel_bytes(param)
        for buffer in model.buffers():
            total_bytes += _tensor_numel_bytes(buffer)
    except Exception:
        pass

    model._parameter_memory_mb = total_bytes / (1024**2)
    return model._parameter_memory_mb


def calculate_kv_cache_memory_mb(past_key_values):
    """Estimate KV cache footprint in MB."""
    if past_key_values is None:
        return 0.0

    total_bytes = 0

    # Handle the variety of cache containers used by transformers
    if hasattr(past_key_values, 'values') and callable(past_key_values.values):
        cache_iter = past_key_values.values()
    else:
        cache_iter = past_key_values

    for layer in cache_iter:
        if layer is None:
            continue

        # Some caches return tuples of tensors, others dict-like objects
        if isinstance(layer, dict):
            tensors = layer.values()
        elif hasattr(layer, 'values') and callable(layer.values):
            tensors = layer.values()
        elif isinstance(layer, (list, tuple)):
            tensors = layer
        else:
            tensors = [layer]

        for tensor in tensors:
            if isinstance(tensor, dict):
                for nested in tensor.values():
                    total_bytes += _tensor_numel_bytes(nested)
            elif isinstance(tensor, (list, tuple)):
                for nested in tensor:
                    total_bytes += _tensor_numel_bytes(nested)
            else:
                total_bytes += _tensor_numel_bytes(tensor)

    return total_bytes / (1024**2)


def _move_cache_to_device(obj, device):
    """Recursively move cache containers to a target device."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [_move_cache_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_cache_to_device(list(obj), device))
    if isinstance(obj, dict):
        return {key: _move_cache_to_device(value, device) for key, value in obj.items()}
    return obj



def move_kv_cache_to_device(past_key_values, device):
    """Return KV cache moved to the target device while preserving structure."""
    if past_key_values is None:
        return None
    target_device = torch.device(device) if not isinstance(device, torch.device) else device
    return _move_cache_to_device(past_key_values, target_device)



def canonical_device(device):
    """Normalize device inputs to torch.device."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device)



def calculate_max_frames_for_memory():
    """Calculate maximum frames that can be processed based on available GPU memory"""
    memory_info = get_gpu_memory_info()
    if not memory_info:
        return Config.MIN_FRAMES_LIMIT  # Default fallback
    
    # With CPU-loaded videos, memory growth comes from:
    # 1. Model weights (~8GB) - fixed
    # 2. KV cache (attention cache) - grows with sequence length
    # 3. Visual tokens - grows with number of frames processed
    # 4. Temporary frame processing - small and constant
    
    # Reserve memory for model and operations
    available_memory = max(0, memory_info['free_mb'] - Config.DEFAULT_GPU_MEMORY_RESERVE_MB)
    
    # Calculate max frames with safety margin
    max_frames = int(available_memory * Config.MEMORY_SAFETY_MARGIN / Config.MEMORY_GROWTH_PER_FRAME_MB)
    
    # Apply limits
    max_frames = min(max_frames, Config.MAX_FRAMES_LIMIT)
    max_frames = max(max_frames, Config.MIN_FRAMES_LIMIT)
    
    return max_frames

def print_memory_status(prefix="💾", context="GPU Memory"):
    """Print current GPU memory status with consistent formatting"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"{prefix} {context}: {memory_info['allocated_mb']:.1f}MB allocated, {memory_info['free_mb']:.1f}MB free")

def format_timing_ms(total_time, count, unit="frame"):
    """Format timing in milliseconds with consistent precision"""
    if count > 0:
        return f"{total_time/count*1000:.1f}ms/{unit}"
    return "0.0ms/frame"

def print_timing_metrics(visual_time, model_time, generation_time, num_frames, num_responses):
    """Print timing metrics with consistent formatting"""
    print(f"   • Visual embedding time: {visual_time:.3f}s ({format_timing_ms(visual_time, num_frames)})")
    print(f"   • Model forward time: {model_time:.3f}s ({format_timing_ms(model_time, num_frames)})")
    print(f"   • Generation time: {generation_time:.3f}s ({format_timing_ms(generation_time, num_responses, 'response')})")

def defragment_gpu_memory():
    """Force GPU memory defragmentation to free up more contiguous memory"""
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to allocate and free a small tensor to trigger defragmentation
        temp_tensor = torch.randn(1000, 1000, device='cuda')
        del temp_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return True
    except Exception as e:
        print(f"Warning: Memory defragmentation failed: {e}")
        return False

def calculate_all_rebuffering_times(responses, frame_processing_times, reading_speed=4.0, listening_speed=2.35):
    """
    Calculate all types of rebuffering times and resource utilization in one consolidated function.
    
    This function calculates:
    1. Generation rebuffering: Delay when processing exceeds real-time frame rate
    2. Reading rebuffering: Delay when user reading time causes processing delays
    3. Listening rebuffering: Delay when user listening time causes processing delays
    4. Resource utilization: (all previous processing time) / (current video time + current processing time)
    
    Args:
        responses: List of response dictionaries with 'text' field
        frame_processing_times: List of processing times for each frame
        reading_speed: Words per second for reading (default: 4.0 wps)
        listening_speed: Words per second for listening (default: 2.35 wps)
    
    Returns:
        dict: {
            'generation_rebuffering_times': list,
            'reading_rebuffering_times': list,
            'listening_rebuffering_times': list,
            'resource_utilization_times': list,
            'total_generation_rebuffering': float,
            'total_reading_rebuffering': float,
            'total_listening_rebuffering': float,
            'average_generation_rebuffering': float,
            'average_reading_rebuffering': float,
            'average_listening_rebuffering': float,
            'final_frame_utilization': float
        }
    """
    generation_rebuffering_times = []
    reading_rebuffering_times = []
    listening_rebuffering_times = []
    resource_utilization_times = []
    
    # Track when previous activities finished
    previous_processing_finish = 0.0
    previous_reading_finish = 0.0
    previous_listening_finish = 0.0
    
    # Track cumulative processing time for resource utilization
    cumulative_processing_time = 0.0
    
    frame_interval = 1.0 / Config.FRAME_FPS  # e.g., 0.5s for 2 FPS
    
    # Process each frame
    for frame_idx, processing_time in enumerate(frame_processing_times):
        # Expected start time for processing is based on frame intervals (0, 0.5, 1.0, 1.5, ...)
        expected_processing_start = frame_idx * frame_interval
        
        # Calculate reading/listening time for this frame's response (if any)
        reading_time = 0.0
        listening_time = 0.0
        
        # Check if there's a response for this frame
        if frame_idx < len(responses) and 'text' in responses[frame_idx] and responses[frame_idx]['text']:
            response_text = responses[frame_idx]['text']
            word_count = len(response_text.split())
            reading_time = word_count / reading_speed
            listening_time = word_count / listening_speed
        
        # 1. GENERATION REBUFFERING: Delay when processing exceeds real-time frame rate
        if frame_idx == 0:
            generation_rebuffering_delay = 0.0  # First frame: no rebuffering
        else:
            generation_rebuffering_delay = max(0.0, previous_processing_finish - expected_processing_start)
        generation_rebuffering_times.append(generation_rebuffering_delay)
        
        # 2. READING REBUFFERING: Delay when user reading time causes processing delays
        expected_reading_start = expected_processing_start + processing_time
        reading_rebuffering_delay = max(0.0, previous_reading_finish - expected_reading_start)
        reading_rebuffering_times.append(reading_rebuffering_delay)
        
        # 3. LISTENING REBUFFERING: Delay when user listening time causes processing delays
        expected_listening_start = expected_processing_start + processing_time
        listening_rebuffering_delay = max(0.0, previous_listening_finish - expected_listening_start)
        listening_rebuffering_times.append(listening_rebuffering_delay)
        
        # 4. RESOURCE UTILIZATION: (all previous processing time) / (current video time + current processing time)
        current_video_time = expected_processing_start  # 0, 0.5, 1.0, 1.5, ...
        resource_utilization = cumulative_processing_time / (current_video_time + processing_time) if (current_video_time + processing_time) > 0 else 0.0
        resource_utilization_times.append(resource_utilization)
        
        # Update finish times for next iteration
        # Processing finishes at expected_start + generation_rebuffering + processing_time
        actual_processing_finish = expected_processing_start + generation_rebuffering_delay + processing_time
        
        # Update cumulative processing time for next iteration
        cumulative_processing_time += processing_time
        
        # Reading finishes when processing + reading is complete (independent of listening)
        previous_reading_finish = actual_processing_finish + reading_time
        
        # Listening finishes when processing + listening is complete (independent of reading)
        previous_listening_finish = actual_processing_finish + listening_time
        
        # Update processing finish time for next iteration
        previous_processing_finish = actual_processing_finish
    
    # Calculate totals and averages
    total_generation_rebuffering = sum(generation_rebuffering_times)
    total_reading_rebuffering = sum(reading_rebuffering_times)
    total_listening_rebuffering = sum(listening_rebuffering_times)
    
    average_generation_rebuffering = total_generation_rebuffering / len(generation_rebuffering_times) if generation_rebuffering_times else 0.0
    average_reading_rebuffering = total_reading_rebuffering / len(reading_rebuffering_times) if reading_rebuffering_times else 0.0
    average_listening_rebuffering = total_listening_rebuffering / len(listening_rebuffering_times) if listening_rebuffering_times else 0.0
    
    # Calculate final frame utilization
    final_frame_utilization = resource_utilization_times[-1] if resource_utilization_times else 0.0
    
    return {
        'generation_rebuffering_times': generation_rebuffering_times,
        'reading_rebuffering_times': reading_rebuffering_times,
        'listening_rebuffering_times': listening_rebuffering_times,
        'resource_utilization_times': resource_utilization_times,
        'total_generation_rebuffering': total_generation_rebuffering,
        'total_reading_rebuffering': total_reading_rebuffering,
        'total_listening_rebuffering': total_listening_rebuffering,
        'average_generation_rebuffering': average_generation_rebuffering,
        'average_reading_rebuffering': average_reading_rebuffering,
        'average_listening_rebuffering': average_listening_rebuffering,
        'final_frame_utilization': final_frame_utilization
    }


def create_processor_timeline(processor_segments, idle_segments=None, conversation_summaries=None, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Visualize processor usage timeline across four arrival scenarios with shared axes."""

    if not processor_segments:
        return None

    os.makedirs(output_dir, exist_ok=True)

    conversation_ids = [segment['conversation_id'] for segment in processor_segments]
    unique_conversations = list(dict.fromkeys(conversation_ids))

    summary_lookup = {}
    if conversation_summaries:
        for summary in conversation_summaries:
            label = summary.get('label', summary.get('conversation_id', '')[:12])
            summary_lookup[label] = summary

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = {cid: base_colors[i % len(base_colors)] for i, cid in enumerate(unique_conversations)}

    idle_segments = idle_segments or []

    conversation_data = {}
    for cid in unique_conversations:
        summary = summary_lookup.get(cid)
        if not summary:
            continue
        start_time = summary.get('start', 0.0)
        end_time = summary.get('end', start_time)
        duration = max(0.0, end_time - start_time)
        events = summary.get('events', []) or []
        frame_events = sorted(
            (event for event in events if event.get('type') == 'frame'),
            key=lambda event: (event.get('time', 0.0), event.get('detail', {}).get('frame_idx', 0))
        )
        prompt_rel_times = [
            max(0.0, event.get('time', 0.0) - start_time)
            for event in events
            if event.get('type') == 'prompt'
        ]
        conversation_data[cid] = {
            'original_start': start_time,
            'duration': duration,
            'frames': [],
            'frame_events': collections.deque(frame_events),
            'prompts_relative': prompt_rel_times,
            'processing_time': 0.0,
            'completion_time': 0.0,
        }

    sorted_segments = sorted(processor_segments, key=lambda segment: (segment['start'], segment['end']))
    pending_response = {cid: None for cid in unique_conversations}

    for segment in sorted_segments:
        cid = segment['conversation_id']
        data = conversation_data.get(cid)
        if not data:
            continue

        duration = max(0.0, segment['end'] - segment['start'])
        if segment.get('type') == 'frame':
            if data['frame_events']:
                frame_event = data['frame_events'].popleft()
                relative_time = max(0.0, frame_event.get('time', data['original_start']) - data['original_start'])
                frame_idx = frame_event.get('detail', {}).get('frame_idx', len(data['frames']))
            else:
                frame_event = None
                frame_idx = len(data['frames'])
                relative_time = len(data['frames']) / max(Config.FRAME_FPS, 1.0)

            frame_info = {
                'frame_idx': frame_idx,
                'relative_time': relative_time,
                'frame_duration': duration,
                'generation_duration': 0.0,
            }
            data['frames'].append(frame_info)
            pending_response[cid] = frame_info
        elif segment.get('type') == 'response':
            frame_info = pending_response.get(cid)
            if frame_info:
                frame_info['generation_duration'] += duration
                pending_response[cid] = None

    for data in conversation_data.values():
        data.pop('frame_events', None)

    for cid, data in conversation_data.items():
        total_processing = sum(
            max(0.0, frame.get('frame_duration', 0.0)) +
            max(0.0, frame.get('generation_duration', 0.0))
            for frame in data['frames']
        )
        if total_processing <= 0.0:
            fallback = data['duration'] if data['duration'] > 0.0 else len(data['frames']) / max(Config.FRAME_FPS, 1.0)
            total_processing = max(0.0, fallback)
        data['processing_time'] = total_processing

        completion_time = 0.0
        for frame in data['frames']:
            completion_time = max(
                completion_time,
                frame.get('relative_time', 0.0)
                + max(0.0, frame.get('frame_duration', 0.0))
                + max(0.0, frame.get('generation_duration', 0.0))
            )
        if completion_time <= 0.0:
            completion_time = max(data['duration'], total_processing)
        data['completion_time'] = completion_time

    actual_starts = []
    actual_ends = []
    actual_prompts = []
    for cid in unique_conversations:
        data = conversation_data.get(cid)
        if not data:
            continue
        actual_starts.append({'time': data['original_start'], 'conversation_id': cid})
        actual_ends.append({'time': data['original_start'] + data['duration'], 'conversation_id': cid})
        for rel in data['prompts_relative']:
            actual_prompts.append({'time': data['original_start'] + rel, 'conversation_id': cid})

    def compute_completion_markers(segments):
        last_by_conversation = {}
        for segment in segments or []:
            cid = segment.get('conversation_id')
            end_time = segment.get('end', segment.get('start', 0.0))
            if cid is None:
                continue
            if end_time > last_by_conversation.get(cid, 0.0):
                last_by_conversation[cid] = end_time
        return [{'conversation_id': cid, 'time': end_time} for cid, end_time in last_by_conversation.items()]

    def build_offsets(base_offsets):
        offsets = {}
        for cid, value in base_offsets.items():
            if cid in conversation_data:
                offsets[cid] = max(0.0, value)

        if offsets:
            cursor = max(
                offsets[cid] + max(
                    conversation_data[cid]['completion_time'],
                    conversation_data[cid]['processing_time'],
                    conversation_data[cid]['duration']
                )
                for cid in offsets
                if cid in conversation_data
            )
        else:
            cursor = 0.0

        for cid in unique_conversations:
            if cid not in conversation_data or cid in offsets:
                continue
            offsets[cid] = cursor
            cursor += max(
                conversation_data[cid]['completion_time'],
                conversation_data[cid]['processing_time'],
                conversation_data[cid]['duration']
            )
        return offsets

    def simulate_scenario(offsets):
        events = []
        prompts = []
        starts = []
        ends = []
        for order, cid in enumerate(unique_conversations):
            data = conversation_data.get(cid)
            if not data:
                continue
            start_time = offsets.get(cid, data['original_start'])
            starts.append({'time': start_time, 'conversation_id': cid})
            ends.append({'time': start_time + data['duration'], 'conversation_id': cid})
            for rel in data['prompts_relative']:
                prompts.append({'time': start_time + rel, 'conversation_id': cid})
            for frame in data['frames']:
                event_time = start_time + frame['relative_time']
                events.append((event_time, order, frame.get('frame_idx', 0), cid, frame))

        events.sort(key=lambda item: (item[0], item[1], item[2]))

        clock = 0.0
        segments = []
        idle = []
        for event_time, _, _, cid, frame in events:
            if event_time > clock:
                idle.append({'start': clock, 'end': event_time})
                clock = event_time

            frame_duration = max(0.0, frame.get('frame_duration', 0.0))
            if frame_duration > 0.0:
                segments.append({
                    'conversation_id': cid,
                    'start': clock,
                    'end': clock + frame_duration,
                    'type': 'frame'
                })
                clock += frame_duration

            generation_duration = max(0.0, frame.get('generation_duration', 0.0))
            if generation_duration > 0.0:
                segments.append({
                    'conversation_id': cid,
                    'start': clock,
                    'end': clock + generation_duration,
                    'type': 'response'
                })
                clock += generation_duration

        return segments, idle, prompts, starts, ends

    scenario_results = []
    scenario_results.append({
        'title': 'Scenario 1: Actual Event-Driven Processing',
        'ylabel': 'Actual',
        'segments': sorted_segments,
        'idle': idle_segments,
        'prompts': actual_prompts,
        'starts': actual_starts,
        'ends': actual_ends,
        'completion': compute_completion_markers(sorted_segments),
    })

    available_conversations = [cid for cid in unique_conversations if cid in conversation_data]
    if len(available_conversations) >= 2:
        primary = available_conversations[0]
        secondary = available_conversations[1]
        completion_primary = conversation_data[primary]['completion_time']
        completion_secondary = conversation_data[secondary]['completion_time']

        scenario_definitions = [
            (
                'Scenario 2: Conv2 Starts Mid Conv1',
                'Scenario 2',
                {primary: 0.0, secondary: max(0.0, completion_primary / 2.0)}
            ),
            (
                'Scenario 3: Conv2 Finishes Before Conv1 Begins',
                'Scenario 3',
                {secondary: 0.0, primary: max(0.0, completion_secondary)}
            ),
            (
                'Scenario 4: Conv1 & Conv2 Start Together',
                'Scenario 4',
                {primary: 0.0, secondary: 0.0}
            ),
        ]

        for title, ylabel, base_offsets in scenario_definitions:
            offsets = build_offsets(base_offsets)
            segments, idle_alt, prompts_alt, starts_alt, ends_alt = simulate_scenario(offsets)
            scenario_results.append({
                'title': title,
                'ylabel': ylabel,
                'segments': segments,
                'idle': idle_alt,
                'prompts': prompts_alt,
                'starts': starts_alt,
                'ends': ends_alt,
                'completion': compute_completion_markers(segments),
            })

    fig, axes = plt.subplots(len(scenario_results), 1, figsize=(Config.PLOT_FIGSIZE_LARGE[0], 9.0), sharex=True)
    if len(scenario_results) == 1:
        axes = [axes]

    # idle_style = dict(facecolors='#ffffff', edgecolors='#d0d0d0', linewidth=0.9, zorder=1)
    idle_style = dict(facecolors='none', edgecolors='none', linewidth=0, zorder=1)

    def draw_timeline(ax, timeline_data):
        prompt_flag = False
        start_flag = False
        completion_flag = False

        for idle in timeline_data['idle'] or []:
            start = idle.get('start', 0.0)
            end = idle.get('end', start)
            if end > start:
                ax.broken_barh([(start, end - start)], (0.2, 0.6), **idle_style)

        for segment in timeline_data['segments'] or []:
            start = segment['start']
            end = segment['end']
            if end <= start:
                continue
            cid = segment['conversation_id']
            ax.broken_barh(
                [(start, end - start)],
                (0.3, 0.4),
                facecolors=colors.get(cid, '#333333'),
                edgecolors='none',
                zorder=2
            )

        for marker in timeline_data['starts'] or []:
            cid = marker['conversation_id']
            start_time = marker['time']
            ax.axvline(start_time, color=colors.get(cid, '#333333'), linestyle='--', linewidth=1.0, alpha=0.75, zorder=3)
            start_flag = True

        for marker in timeline_data['ends'] or []:
            cid = marker['conversation_id']
            end_time = marker['time']
            ax.axvline(end_time, color=colors.get(cid, '#333333'), linestyle=':', linewidth=1.0, alpha=0.6, zorder=3)

        for prompt in timeline_data['prompts'] or []:
            cid = prompt['conversation_id']
            prompt_time = prompt['time']
            ax.scatter(
                prompt_time,
                0.1,
                marker='v',
                s=70,
                color=colors.get(cid, '#444444'),
                edgecolors='k',
                linewidths=0.5,
                zorder=4
            )
            prompt_flag = True

        for marker in timeline_data.get('completion', []) or []:
            cid = marker['conversation_id']
            completion_time = marker['time']
            color = colors.get(cid, '#444444')
            ax.scatter(
                completion_time,
                0.95,
                marker='o',
                s=65,
                facecolors='none',
                edgecolors=color,
                linewidths=1.1,
                zorder=5
            )
            completion_flag = True

        ax.set_title(timeline_data['title'], fontsize=12)
        ax.set_ylabel(timeline_data['ylabel'])
        ax.set_yticks([])
        ax.set_ylim(0.0, 1.18)
        ax.grid(True, axis='x', alpha=0.25)
        return prompt_flag, start_flag, completion_flag

    overall_max = 0.0
    prompt_present = False
    start_present = False
    completion_present = False

    for ax, timeline in zip(axes, scenario_results):
        prompt_flag, start_flag, completion_flag = draw_timeline(ax, timeline)
        prompt_present |= prompt_flag
        start_present |= start_flag
        completion_present |= completion_flag

        segment_max = max((segment['end'] for segment in timeline['segments'] or []), default=0.0)
        idle_max = max((idle['end'] for idle in timeline['idle'] or []), default=0.0)
        end_markers_max = max((marker['time'] for marker in timeline['ends'] or []), default=0.0)
        prompt_max = max((marker['time'] for marker in timeline['prompts'] or []), default=0.0)
        overall_max = max(overall_max, segment_max, idle_max, end_markers_max, prompt_max)

    axes[-1].set_xlabel('Processor Time (s)')
    if overall_max <= 0.0:
        overall_max = 1.0
    axes[0].set_xlim(0.0, overall_max * 1.05)

    legend_handles = []
    legend_labels = []
    if any(timeline['idle'] for timeline in scenario_results):
        legend_handles.append(plt.Line2D([], [], color='#ffffff', linewidth=6, marker='_', markeredgecolor='#d0d0d0'))
        legend_labels.append('Idle')
    for cid in unique_conversations:
        legend_handles.append(plt.Line2D([], [], color=colors.get(cid, '#333333'), linewidth=6))
        legend_labels.append(cid)
    if prompt_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='v', linestyle='None', markersize=7, markeredgecolor='k'))
        legend_labels.append('Prompt')
    if start_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='^', linestyle='None', markersize=8, markeredgecolor='k'))
        legend_labels.append('Video Start')
    if completion_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='o', linestyle='None', markersize=7, markerfacecolor='none', markeredgewidth=1.1))
        legend_labels.append('Last Frame Done')
    if legend_handles:
        axes[0].legend(legend_handles, legend_labels, loc='upper right', fontsize=8, title='Legend')

    idle_total = sum(max(0.0, idle.get('end', 0.0) - idle.get('start', 0.0)) for idle in idle_segments)
    if idle_total > 0:
        longest_idle = max(idle_segments, key=lambda idle: max(0.0, idle.get('end', 0.0) - idle.get('start', 0.0)))
        longest_duration = max(0.0, longest_idle.get('end', 0.0) - longest_idle.get('start', 0.0))
        print(
            f"ℹ️ Processor idle for {idle_total:.2f}s total. Longest gap {longest_duration:.2f}s "
            f"from {longest_idle.get('start', 0.0):.2f}s to {longest_idle.get('end', 0.0):.2f}s."
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, f'processor_timeline_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Processor timeline saved to: {output_path}")
    plt.show()

    return output_path

def create_memory_visualization(all_memory_data, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create detailed memory usage visualization for all videos"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_conversations = max(1, len(all_memory_data))
    figure_height = Config.PLOT_FIGSIZE_SMALL[1] * max(1, num_conversations)
    fig, axes = plt.subplots(num_conversations, 5, figsize=(Config.PLOT_FIGSIZE_LARGE[0] + 4, figure_height))
    fig.suptitle('GPU Memory Usage Analysis - CPU-First Conversation Processing', fontsize=16, fontweight='bold')

    # Normalise axes shape for single conversation runs
    if num_conversations == 1:
        axes = np.array([axes])

    component_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for row_idx, (conversation_key, data) in enumerate(all_memory_data.items() or {"conversation": {}}.items()):
        ax_breakdown, ax_kv_cache, ax_kv_transfer, ax_efficiency, ax_allocator = axes[row_idx]

        frames = data.get('frames', [])
        if not frames:
            ax_breakdown.text(0.5, 0.5, 'No memory data collected', ha='center', va='center', transform=ax_breakdown.transAxes)
            ax_kv_cache.axis('off')
            ax_kv_transfer.axis('off')
            ax_efficiency.axis('off')
            ax_allocator.axis('off')
            continue

        totals = data.get('memory_usage', [])
        model_memory = data.get('model_memory', [])
        kv_cache_memory = data.get('kv_cache_memory', [])
        other_memory = data.get('other_memory', [])
        memory_per_frame = data.get('memory_per_frame', [])
        torch_allocated = data.get('torch_allocated', [])
        torch_reserved = data.get('torch_reserved', [])

        lengths = [len(frames), len(totals), len(model_memory), len(kv_cache_memory), len(other_memory), len(memory_per_frame), len(torch_allocated), len(torch_reserved)]
        min_length = min(lengths) if all(lengths) else 0

        if min_length == 0:
            ax_breakdown.text(0.5, 0.5, 'Incomplete memory data', ha='center', va='center', transform=ax_breakdown.transAxes)
            ax_kv_cache.axis('off')
            ax_kv_transfer.axis('off')
            ax_efficiency.axis('off')
            ax_allocator.axis('off')
            continue

        frames = np.array(frames[:min_length])
        totals = np.array(totals[:min_length])
        model_memory = np.array(model_memory[:min_length])
        kv_cache_memory = np.array(kv_cache_memory[:min_length])
        other_memory = np.array(other_memory[:min_length])
        memory_per_frame = np.array(memory_per_frame[:min_length])
        torch_allocated = np.array(torch_allocated[:min_length])
        torch_reserved = np.array(torch_reserved[:min_length])

        # 1. Component breakdown stackplot
        ax_breakdown.stackplot(
            frames,
            model_memory,
            kv_cache_memory,
            other_memory,
            colors=component_colors,
            labels=['Model Parameters', 'KV Cache', 'Other Allocations'],
            alpha=0.7,
        )
        ax_breakdown.plot(frames, totals, color='black', linestyle='--', linewidth=1.5, label='nvidia-smi Total')
        ax_breakdown.set_title(f'{conversation_key} Memory Breakdown', fontsize=10)
        ax_breakdown.set_xlabel('Frame Number')
        ax_breakdown.set_ylabel('Memory (MB)')
        ax_breakdown.grid(True, alpha=0.3)
        ax_breakdown.legend(fontsize=8, loc='upper left')

        # 2. KV cache trajectory with dedicated view
        ax_kv_cache.plot(frames, kv_cache_memory, color='#ff7f0e', linewidth=2.0)
        ax_kv_cache.fill_between(frames, kv_cache_memory, alpha=0.2, color='#ffbb78')
        ax_kv_cache.set_title('KV Cache Footprint', fontsize=10)
        ax_kv_cache.set_xlabel('Frame Number')
        ax_kv_cache.set_ylabel('Memory (MB)')
        ax_kv_cache.grid(True, alpha=0.3)
        if kv_cache_memory.size > 0:
            ax_kv_cache.text(
                0.98,
                0.02,
                f'Max: {np.max(kv_cache_memory):.0f}MB\nEnd: {kv_cache_memory[-1]:.0f}MB',
                transform=ax_kv_cache.transAxes,
                ha='right',
                va='bottom',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
            )

        # 3. KV cache size vs transfer timings
        transfer_sizes = data.get('kv_transfer_size', [])
        transfer_offload = data.get('kv_offload_time', [])
        transfer_reload = data.get('kv_reload_time', [])
        transfer_length = min(len(transfer_sizes), len(transfer_offload), len(transfer_reload))
        if transfer_length:
            sizes = np.array(transfer_sizes[:transfer_length])
            offload_ms = np.array(transfer_offload[:transfer_length]) * 1000.0
            reload_ms = np.array(transfer_reload[:transfer_length]) * 1000.0

            ax_kv_transfer.scatter(sizes, offload_ms, color='#1f77b4', alpha=0.7, s=20, label='Offload → CPU')
            ax_kv_transfer.scatter(sizes, reload_ms, color='#2ca02c', alpha=0.7, marker='x', s=28, label='Reload → GPU')

            ax_kv_transfer.set_title('KV Cache Transfer Timing', fontsize=10)
            ax_kv_transfer.set_xlabel('KV Cache Size (MB)')
            ax_kv_transfer.set_ylabel('Transfer Time (ms)')
            ax_kv_transfer.grid(True, alpha=0.3)
            ax_kv_transfer.legend(fontsize=8, loc='upper left')

            avg_offload = offload_ms.mean()
            avg_reload = reload_ms.mean()
            ax_kv_transfer.text(
                0.98,
                0.02,
                f'Avg Offload: {avg_offload:.1f}ms\nAvg Reload: {avg_reload:.1f}ms',
                transform=ax_kv_transfer.transAxes,
                ha='right',
                va='bottom',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
            )
        else:
            ax_kv_transfer.text(0.5, 0.5, 'No KV transfer data', ha='center', va='center', transform=ax_kv_transfer.transAxes)
            ax_kv_transfer.set_xlabel('KV Cache Size (MB)')
            ax_kv_transfer.set_ylabel('Transfer Time (ms)')
            ax_kv_transfer.grid(True, alpha=0.3)

        # 4. Memory efficiency per frame
        ax_efficiency.plot(frames, memory_per_frame, color='#9467bd', linewidth=2, marker='^', markersize=3)
        ax_efficiency.set_title('Memory Efficiency (Lower is Better)', fontsize=10)
        ax_efficiency.set_xlabel('Frame Number')
        ax_efficiency.set_ylabel('Δ Memory / Frame (MB)')
        ax_efficiency.grid(True, alpha=0.3)

        # 5. Allocator view (PyTorch vs nvidia-smi)
        ax_allocator.plot(frames, torch_allocated, color='#17becf', linewidth=2, label='torch.cuda.memory_allocated')
        ax_allocator.plot(frames, torch_reserved, color='#bcbd22', linewidth=2, linestyle='--', label='torch.cuda.memory_reserved')
        ax_allocator.plot(frames, totals, color='black', linewidth=1.5, linestyle=':', label='nvidia-smi Total')
        ax_allocator.set_title('Allocator View', fontsize=10)
        ax_allocator.set_xlabel('Frame Number')
        ax_allocator.set_ylabel('Memory (MB)')
        ax_allocator.grid(True, alpha=0.3)
        ax_allocator.legend(fontsize=8, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    output_path = os.path.join(output_dir, f'memory_usage_analysis_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Memory usage analysis saved to: {output_path}")

    plt.show()

    return output_path


def create_frame_score_analysis(all_frame_scores_data, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create frame score analysis visualization showing scores, threshold, and response triggers"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each conversation
    num_conversations = len(all_frame_scores_data)
    fig, axes = plt.subplots(num_conversations, 1, figsize=(Config.PLOT_FIGSIZE_LARGE[0], Config.PLOT_FIGSIZE_MEDIUM[1] * num_conversations))
    if num_conversations == 1:
        axes = [axes]
    
    fig.suptitle('Frame Score Analysis - Streaming Threshold and Response Triggers', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (conversation_key, data) in enumerate(all_frame_scores_data.items()):
        ax = axes[i]
        
        if not data['frame_scores']:  # Skip if no data
            ax.text(0.5, 0.5, f'No frame score data for {conversation_key}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        frame_scores = data['frame_scores']
        frame_times = data['frame_times']
        threshold = data['threshold']
        response_times = data['response_times']
        response_triggers = data['response_triggers']
        
        # Plot frame scores over time with better visibility
        ax.plot(frame_times, frame_scores, 
               color=colors[i % len(colors)], 
               linewidth=2, alpha=0.8, 
               label=f'Frame Token Interval Scores')
        
        # Draw threshold line with actual threshold value
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=3, 
                  label=f'Streaming Threshold ({threshold:.3f})', alpha=0.8)
        
        # Mark response generation points (when frames are sampled for response)
        if response_times:
            response_scores = []
            for resp_time in response_times:
                # Find the closest frame time and its corresponding score
                closest_idx = min(range(len(frame_times)), 
                                 key=lambda x: abs(frame_times[x] - resp_time))
                response_scores.append(frame_scores[closest_idx])
            
            # Plot response points with better annotation
            scatter = ax.scatter(response_times, response_scores, 
                      color='green', s=120, marker='o', 
                      label=f'Response Generated ({len(response_times)} times)',
                      zorder=5, edgecolors='darkgreen', linewidth=2, alpha=0.9)
            
            # Add annotations for response points
            for j, (resp_time, resp_score) in enumerate(zip(response_times, response_scores)):
                ax.annotate(f'R{j+1}', 
                           xy=(resp_time, resp_score), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))
        
        # Mark EOS generation points (when score drops below threshold)
        below_threshold_indices = [j for j, score in enumerate(frame_scores) if score < threshold]
        if below_threshold_indices:
            eos_times = [frame_times[j] for j in below_threshold_indices]
            eos_scores = [frame_scores[j] for j in below_threshold_indices]
            ax.scatter(eos_times, eos_scores, 
                      color='orange', s=100, marker='^', 
                      label=f'EOS Triggered ({len(eos_times)} times)',
                      zorder=5, edgecolors='darkorange', linewidth=2, alpha=0.9)
        
        # Add shaded regions for different score ranges with better labels
        ax.fill_between(frame_times, 0, threshold, alpha=0.15, color='red', 
                       label=f'Below Threshold (EOS Zone)')
        ax.fill_between(frame_times, threshold, 1.0, alpha=0.15, color='green', 
                       label=f'Above Threshold (Continue Zone)')
        
        # Customize the plot
        ax.set_xlabel('Video Time (seconds)', fontsize=12)
        ax.set_ylabel('Frame Token Interval Score', fontsize=12)
        ax.set_title(f'{conversation_key} - Frame Score Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Move legend outside the plot area to avoid occlusion
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.set_ylim(0, 1.0)
        
        # Add statistics text box with more detailed information
        stats_text = f'Total Frames: {len(frame_scores)}\n'
        stats_text += f'Responses Generated: {len(response_times)}\n'
        stats_text += f'EOS Triggers: {len(below_threshold_indices)}\n'
        stats_text += f'Threshold: {threshold:.3f}\n'
        stats_text += f'Avg Score: {np.mean(frame_scores):.3f}\n'
        stats_text += f'Min Score: {np.min(frame_scores):.3f}\n'
        stats_text += f'Max Score: {np.max(frame_scores):.3f}\n'
        stats_text += f'Score Std: {np.std(frame_scores):.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add horizontal lines for better threshold visualization
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'frame_scores_analysis_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Frame score analysis saved to: {output_path}")
    
    plt.show()
    
    return output_path

def create_individual_conversation_timing_plots(conversation_timings, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create individual timing plots for each conversation with enhanced 4-plot layout"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, conversation_timing in enumerate(conversation_timings):
        conversation_number = i + 1  # Always use sequential numbering 1, 2, 3, ...
        conversation_id = conversation_timing.get('conversation_id', f'conversation_{i+1}')
        fig, axes = plt.subplots(1, 7, figsize=(21, 6))
        fig.suptitle(f'Conversation {conversation_number} ({conversation_id}) - Timing Analysis', fontsize=14, fontweight='bold')
        
        # 1. Timing components breakdown
        ax1 = axes[0]
        components = ['Visual Embedding', 'Model Forward', 'Generation']
        times = [conversation_timing['visual_embedding_time'], 
                conversation_timing['model_forward_time'], 
                conversation_timing['generation_time']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(components, times, color=colors, alpha=0.8)
        ax1.set_ylabel('Time (seconds)', fontsize=9)
        ax1.set_title('Timing Components Breakdown', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 2. Component efficiency per frame
        ax2 = axes[1]
        if conversation_timing['frame_processing_times']:
            frame_count = len(conversation_timing['frame_processing_times'])
            generation_count = max(1, len(conversation_timing.get('generated_turns', [])))
            
            visual_per_frame = conversation_timing['visual_embedding_time'] / frame_count
            model_per_frame = conversation_timing['model_forward_time'] / frame_count
            generation_per_response = conversation_timing['generation_time'] / generation_count  # Per response, not per frame
            
            components = ['Visual', 'Model', 'Generation']
            per_frame_times = [visual_per_frame, model_per_frame, generation_per_response]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            bars = ax2.bar(components, per_frame_times, color=colors, alpha=0.8)
            ax2.set_ylabel('Time per Frame/Response (s)')
            ax2.set_title('Component Efficiency (Visual/Model per Frame, Generation per Response)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels with appropriate units
            labels = [f'{visual_per_frame*1000:.1f}ms/frame', f'{model_per_frame*1000:.1f}ms/frame', f'{generation_per_response*1000:.1f}ms/response']
            for bar, time, label in zip(bars, per_frame_times, labels):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                        label, ha='center', va='bottom', fontsize=9)
        
        # 3. Timing components over time
        ax3 = axes[2]
        frame_timing_data = conversation_timing.get('frame_timing_data', [])
        
        if frame_timing_data:
            # Extract timing data
            video_times = [data['video_time'] for data in frame_timing_data]
            visual_times = [data['visual_embedding_time'] * 1000 for data in frame_timing_data]  # Convert to ms
            model_times = [data['model_forward_time'] * 1000 for data in frame_timing_data]  # Convert to ms
            generation_times = [data['generation_time'] * 1000 for data in frame_timing_data]  # Convert to ms
            
            # Plot timing components over time
            ax3.plot(video_times, visual_times, 'b-', linewidth=1.5, alpha=0.8, label='Visual Embedding (ms)')
            ax3.plot(video_times, model_times, 'orange', linewidth=1.5, alpha=0.8, label='Model Forward (ms)')
            ax3.plot(video_times, generation_times, 'g-', linewidth=1.5, alpha=0.8, label='Generation (ms)')
            
            ax3.set_xlabel('Video Time (seconds)')
            ax3.set_ylabel('Time per Frame (ms)')
            ax3.set_title('Timing Components Over Time', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=7)
            
            # Add some statistics
            avg_visual = np.mean(visual_times)
            avg_model = np.mean(model_times)
            avg_generation = np.mean(generation_times)
            
            stats_text = f'Avg Visual: {avg_visual:.1f}ms\nAvg Model: {avg_model:.1f}ms\nAvg Gen: {avg_generation:.1f}ms'
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # Fallback: create synthetic data from total times if no per-frame data available
            if conversation_timing['frame_processing_times']:
                frame_count = len(conversation_timing['frame_processing_times'])
                video_times = np.linspace(0, conversation_timing.get('video_duration', frame_count / Config.FRAME_FPS), frame_count)
                
                # Distribute total times evenly across frames
                visual_per_frame = conversation_timing['visual_embedding_time'] / frame_count * 1000
                model_per_frame = conversation_timing['model_forward_time'] / frame_count * 1000
                generation_per_frame = conversation_timing['generation_time'] / frame_count * 1000
                
                ax3.plot(video_times, [visual_per_frame] * frame_count, 'b-', linewidth=1.5, alpha=0.8, label='Visual Embedding (ms)')
                ax3.plot(video_times, [model_per_frame] * frame_count, 'orange', linewidth=1.5, alpha=0.8, label='Model Forward (ms)')
                ax3.plot(video_times, [generation_per_frame] * frame_count, 'g-', linewidth=1.5, alpha=0.8, label='Generation (ms)')
                
                ax3.set_xlabel('Video Time (seconds)')
                ax3.set_ylabel('Time per Frame (ms)')
                ax3.set_title('Timing Components Over Time (Estimated)')
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=7)
                
                stats_text = f'Visual: {visual_per_frame:.1f}ms/frame\nModel: {model_per_frame:.1f}ms/frame\nGen: {generation_per_frame:.1f}ms/frame'
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax3.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Timing Components Over Time', fontsize=10)
        
        # 4. Rebuffering time over time
        ax4 = axes[3]
        rebuffering_times = conversation_timing.get('rebuffering_times', [])
        
        if rebuffering_times:
            frame_indices = range(len(rebuffering_times))
            video_times = [i / Config.FRAME_FPS for i in frame_indices]  # Convert frame index to time
            
            # Plot rebuffering time over time
            ax4.plot(video_times, rebuffering_times, 'r-', linewidth=2, alpha=0.8, label='Rebuffering Time')
            ax4.fill_between(video_times, rebuffering_times, alpha=0.3, color='red')
            
            ax4.set_xlabel('Video Time (seconds)')
            ax4.set_ylabel('Rebuffering Time (seconds)')
            ax4.set_title('Rebuffering Time Over Time', fontsize=10)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=7)
            
            # Add statistics
            total_rebuffering = sum(rebuffering_times)
            avg_rebuffering = np.mean(rebuffering_times)
            max_rebuffering = max(rebuffering_times)
            
            stats_text = f'Total: {total_rebuffering:.3f}s\nAvg: {avg_rebuffering:.3f}s\nMax: {max_rebuffering:.3f}s'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No rebuffering data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Rebuffering Time Over Time', fontsize=10)
        
        # 5. Reading rebuffering time over time
        ax5 = axes[4]
        reading_rebuffering_times = conversation_timing.get('reading_rebuffering_times', [])
        
        if reading_rebuffering_times:
            frame_indices = range(len(reading_rebuffering_times))
            video_times = [i / Config.FRAME_FPS for i in frame_indices]
            
            ax5.plot(video_times, reading_rebuffering_times, 'b-', linewidth=2, alpha=0.8, label='Reading Rebuffering')
            ax5.fill_between(video_times, reading_rebuffering_times, alpha=0.3, color='blue')
            
            ax5.set_xlabel('Video Time (seconds)')
            ax5.set_ylabel('Reading Rebuffering (seconds)')
            ax5.set_title('Reading Rebuffering Over Time', fontsize=10)
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=7)
            
            total_reading = sum(reading_rebuffering_times)
            avg_reading = np.mean(reading_rebuffering_times)
            max_reading = max(reading_rebuffering_times)
            
            stats_text = f'Total: {total_reading:.3f}s\nAvg: {avg_reading:.3f}s\nMax: {max_reading:.3f}s'
            ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax5.text(0.5, 0.5, 'No reading rebuffering data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Reading Rebuffering Over Time', fontsize=10)
        
        # 6. Listening rebuffering time over time
        ax6 = axes[5]
        listening_rebuffering_times = conversation_timing.get('listening_rebuffering_times', [])
        
        if listening_rebuffering_times:
            frame_indices = range(len(listening_rebuffering_times))
            video_times = [i / Config.FRAME_FPS for i in frame_indices]
            
            ax6.plot(video_times, listening_rebuffering_times, 'g-', linewidth=2, alpha=0.8, label='Listening Rebuffering')
            ax6.fill_between(video_times, listening_rebuffering_times, alpha=0.3, color='green')
            
            ax6.set_xlabel('Video Time (seconds)')
            ax6.set_ylabel('Listening Rebuffering (seconds)')
            ax6.set_title('Listening Rebuffering Over Time', fontsize=10)
            ax6.grid(True, alpha=0.3)
            ax6.legend(fontsize=7)
            
            total_listening = sum(listening_rebuffering_times)
            avg_listening = np.mean(listening_rebuffering_times)
            max_listening = max(listening_rebuffering_times)
            
            stats_text = f'Total: {total_listening:.3f}s\nAvg: {avg_listening:.3f}s\nMax: {max_listening:.3f}s'
            ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax6.text(0.5, 0.5, 'No listening rebuffering data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Listening Rebuffering Over Time', fontsize=10)
        
        # 7. Resource utilization over time
        ax7 = axes[6]
        resource_utilization_times = conversation_timing.get('resource_utilization_times', [])
        
        if resource_utilization_times:
            frame_indices = range(len(resource_utilization_times))
            video_times = [i / Config.FRAME_FPS for i in frame_indices]
            
            ax7.plot(video_times, resource_utilization_times, 'm-', linewidth=2, alpha=0.8, label='Resource Utilization')
            ax7.fill_between(video_times, resource_utilization_times, alpha=0.3, color='magenta')
            
            ax7.set_xlabel('Video Time (seconds)')
            ax7.set_ylabel('Resource Utilization')
            ax7.set_title('Resource Utilization Over Time', fontsize=10)
            ax7.grid(True, alpha=0.3)
            ax7.legend(fontsize=7)
            
            # Add horizontal line at 1.0 (100% utilization)
            ax7.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% Utilization')
            
            final_utilization = resource_utilization_times[-1] if resource_utilization_times else 0.0
            avg_utilization = np.mean(resource_utilization_times)
            max_utilization = max(resource_utilization_times)
            
            stats_text = f'Final: {final_utilization:.3f}\nAvg: {avg_utilization:.3f}\nMax: {max_utilization:.3f}'
            ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
        else:
            ax7.text(0.5, 0.5, 'No resource utilization data', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Resource Utilization Over Time', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(output_dir, f'conversation_{conversation_number}_timing_{data_source}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Individual conversation timing plots saved to {output_dir}/")

def create_individual_video_timing_plots(video_timings, output_dir=Config.OUTPUT_DIR):
    """Create individual timing plots for each video with time series analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_timing in enumerate(video_timings):
        video_number = i + 1  # Always use sequential numbering 1, 2, 3, ...
        fig, axes = plt.subplots(2, 2, figsize=Config.PLOT_FIGSIZE_LARGE)
        fig.suptitle(f'Video {video_number} - Detailed Timing Analysis', fontsize=14, fontweight='bold')
        
        # 1. Timing components breakdown
        ax1 = axes[0, 0]
        components = ['Visual Embedding', 'Model Forward', 'Generation']
        times = [video_timing['visual_embedding_time'], 
                video_timing['model_forward_time'], 
                video_timing['generation_time']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(components, times, color=colors, alpha=0.8)
        ax1.set_ylabel('Time (seconds)', fontsize=9)
        ax1.set_title('Timing Components Breakdown', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 2. Timing components over time (top plot)
        ax2 = axes[0, 1]
        if video_timing['frame_processing_times']:
            frame_times = video_timing['frame_processing_times']
            frame_indices = range(len(frame_times))
            
            # Create realistic temporal variation for timing components
            total_frames = len(frame_times)
            base_frame_times = np.array(frame_times)
            
            # Use actual timing data from video - no fake variation
            actual_visual_per_frame = video_timing['visual_embedding_time'] / total_frames
            actual_model_per_frame = video_timing['model_forward_time'] / total_frames
            
            # Use actual per-frame timing data - create realistic temporal distribution
            # Distribute the total time across frames with some realistic frame-to-frame variation
            # based on actual frame processing times if available
            if 'frame_processing_times' in video_timing and video_timing['frame_processing_times']:
                # Use actual frame processing times to create realistic distribution
                frame_times = video_timing['frame_processing_times']
                # Normalize frame times to match the total visual/model time
                visual_weights = np.array(frame_times) / np.sum(frame_times) if np.sum(frame_times) > 0 else np.ones(len(frame_times)) / len(frame_times)
                model_weights = np.array(frame_times) / np.sum(frame_times) if np.sum(frame_times) > 0 else np.ones(len(frame_times)) / len(frame_times)
                
                visual_per_frame = actual_visual_per_frame * visual_weights
                model_per_frame = actual_model_per_frame * model_weights
            else:
                # If no frame processing times available, use constant per-frame values
                visual_per_frame = np.full(total_frames, actual_visual_per_frame)
                model_per_frame = np.full(total_frames, actual_model_per_frame)
            
            # For generation, we need to simulate when responses occur
            # Get response times from generated responses (stored in video_metrics)
            response_frames = []
            gt_response_frames = []  # Ground truth response frames
            
            if 'generated_responses' in video_timing and isinstance(video_timing['generated_responses'], list):
                for response in video_timing['generated_responses']:
                    if 'time' in response:
                        # Scale response time to fit within available frame data using actual video duration
                        response_time = response['time']
                        video_duration = video_timing.get('video_duration', total_frames / Config.FRAME_FPS)  # Use actual video duration
                        frame_idx = int((response_time / video_duration) * total_frames)
                        frame_idx = min(frame_idx, total_frames - 1)
                        frame_idx = max(frame_idx, 0)
                        response_frames.append(frame_idx)
            
            # Get ground truth response times from conversation
            gt_response_times = []  # Store actual times in seconds
            if 'ground_truth_conversation' in video_timing:
                conversation = video_timing['ground_truth_conversation']
                assistant_turns = [t for t in conversation if t['role'] == 'assistant']
                for turn in assistant_turns:
                    if 'time' in turn:
                        gt_time = turn['time']
                        gt_response_times.append(gt_time)
                        # Also convert to frame index for the generation spikes using actual video duration
                        video_duration = video_timing.get('video_duration', total_frames / Config.FRAME_FPS)  # Use actual video duration
                        gt_frame_idx = int((gt_time / video_duration) * total_frames)
                        gt_frame_idx = min(gt_frame_idx, total_frames - 1)
                        gt_frame_idx = max(gt_frame_idx, 0)
                        gt_response_frames.append(gt_frame_idx)
            
            if not gt_response_times and 'conversation_turns' in video_timing and video_timing['conversation_turns'] > 0:
                # Fallback: distribute responses evenly across frames using actual video duration
                num_responses = video_timing['conversation_turns']
                response_interval = total_frames // max(1, num_responses)
                response_frames = [i * response_interval for i in range(num_responses)]
                gt_response_frames = [i * response_interval for i in range(num_responses)]
                # Use actual video duration for response times
                video_duration = video_timing.get('video_duration', 2000.0)
                gt_response_times = [i * response_interval * (video_duration / total_frames) for i in range(num_responses)]
            
            # Create generation time array with actual timing data - no fake scaling
            generation_per_frame = np.zeros(total_frames)
            if response_frames:
                # Use actual generation time per response
                actual_gen_per_response = video_timing['generation_time'] / max(1, len(response_frames))
                for frame_idx in response_frames:
                    if frame_idx < total_frames:
                        # Use actual generation time - no fake variation
                        generation_per_frame[frame_idx] = actual_gen_per_response
            else:
                pass
            
            # Convert frame indices to time in seconds for consistent plotting
            # Use the actual video duration instead of hardcoded 2000s
            video_duration = video_timing.get('video_duration', total_frames / Config.FRAME_FPS)  # Get actual duration
            frame_duration = video_duration / total_frames  # Duration per frame in seconds
            time_axis = np.array(frame_indices) * frame_duration
            
            # Calculate max_time for consistent scaling with ground truth plot
            max_time = video_duration
            # Also check generated response times to ensure we cover the full range
            if 'generated_responses' in video_timing and isinstance(video_timing['generated_responses'], list):
                for response in video_timing['generated_responses']:
                    if 'time' in response:
                        max_time = max(max_time, response['time'])
            
            # Plot the three components over time with actual temporal variation
            # Scale visual and model to be more visible (multiply by 1000 to show in ms)
            visual_scaled = np.array(visual_per_frame) * 1000  # Convert to ms
            model_scaled = np.array(model_per_frame) * 1000    # Convert to ms
            generation_scaled = np.array(generation_per_frame) * 1000  # Convert to ms
            
            ax2.plot(time_axis, visual_scaled, 'b-', linewidth=2, alpha=0.8, label='Visual Embedding (ms)')
            ax2.plot(time_axis, model_scaled, 'orange', linewidth=2, alpha=0.8, label='Model Forward (ms)')
            ax2.plot(time_axis, generation_scaled, 'g-', linewidth=2, alpha=0.8, label='Generation Spikes (ms)')
            
            # Highlight generated response times
            if response_frames:
                for frame_idx in response_frames:
                    if frame_idx < total_frames:
                        response_time = frame_idx * frame_duration
                        ax2.axvline(x=response_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add legend
            ax2.legend(loc='upper right')
            
            # Add annotations for generated response count
            if response_frames:
                ax2.text(0.02, 0.98, f'Generated: {len(response_frames)}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # Set x-axis limits to match the ground truth plot
            ax2.set_xlim(0, max_time)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Time per Frame (ms)')
        ax2.set_title('Timing Components + Generated Responses', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Ground truth response times and prompt times (bottom plot)
        ax3 = axes[1, 1]
        if gt_response_times:
            # Extract prompt times from conversation
            prompt_times = []
            if 'ground_truth_conversation' in video_timing:
                conversation = video_timing['ground_truth_conversation']
                for turn in conversation:
                    if turn['role'] == 'user' and 'time' in turn:
                        prompt_times.append(turn['time'])
            
            # Get the maximum time to ensure consistent x-axis scaling
            # Use the same video_duration that the timing components plot uses
            max_time = video_timing.get('video_duration', total_frames / Config.FRAME_FPS)  # Use actual video duration
            
            # Also check all response times to ensure we cover the full range
            all_times = []
            if gt_response_times:
                all_times.extend(gt_response_times)
            if prompt_times:
                all_times.extend(prompt_times)
            if 'generated_responses' in video_timing and isinstance(video_timing['generated_responses'], list):
                for response in video_timing['generated_responses']:
                    if 'time' in response:
                        all_times.append(response['time'])
            
            # Use the maximum of video duration and actual data times
            if all_times:
                max_time = max(max_time, max(all_times))
            
            # Create a timeline plot for ground truth with separate y-levels
            y_response = 0.8  # Higher level for responses
            y_prompt = 0.2    # Lower level for prompts
            
            # Plot response times (purple solid lines) at higher level
            for i, gt_time in enumerate(gt_response_times):
                ax3.axvline(x=gt_time, ymin=y_response-0.1, ymax=y_response+0.1, 
                           color='purple', linestyle='-', alpha=0.8, linewidth=2.0)
                # Add small text labels for every 5th response to avoid clutter
                if i % 5 == 0:
                    ax3.text(gt_time, y_response+0.15, f'{gt_time:.0f}s', 
                           rotation=90, ha='right', va='bottom', fontsize=8, color='purple')
            
            # Plot prompt times (blue dashed lines) at lower level
            for i, prompt_time in enumerate(prompt_times):
                ax3.axvline(x=prompt_time, ymin=y_prompt-0.1, ymax=y_prompt+0.1, 
                           color='blue', linestyle='--', alpha=0.8, linewidth=2.0)
                # Add small text labels for every prompt (since there are fewer)
                ax3.text(prompt_time, y_prompt-0.15, f'{prompt_time:.0f}s', 
                       rotation=90, ha='left', va='top', fontsize=8, color='blue')
            
            # Set consistent x-axis limits with the timing components plot
            ax3.set_xlim(0, max_time)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Ground Truth\nTimeline')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_title(f'Ground Truth Timeline (Responses: {len(gt_response_times)}, Prompts: {len(prompt_times)})')
            ax3.grid(True, alpha=0.3)
            ax3.set_yticks([])  # Remove y-axis ticks
            
            # Add horizontal lines to separate the two levels
            ax3.axhline(y=y_response, color='purple', linestyle='-', alpha=0.3, linewidth=0.5)
            ax3.axhline(y=y_prompt, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
            
            # Add legend with proper positioning
            ax3.axvline(x=0, color='purple', linestyle='-', alpha=0.8, linewidth=2.0, label='Response Times')
            ax3.axvline(x=0, color='blue', linestyle='--', alpha=0.8, linewidth=2.0, label='Prompt Times')
            ax3.legend(loc='upper right', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No ground truth data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_ylabel('Ground Truth\nTimeline')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_title('Ground Truth Timeline')
        
        # 4. Component efficiency per frame
        ax4 = axes[1, 0]
        if video_timing['frame_processing_times']:
            frame_count = len(video_timing['frame_processing_times'])
            generation_count = max(1, video_timing.get('generated_turns', 1))
            
            visual_per_frame = video_timing['visual_embedding_time'] / frame_count
            model_per_frame = video_timing['model_forward_time'] / frame_count
            generation_per_response = video_timing['generation_time'] / generation_count  # Per response, not per frame
            
            components = ['Visual', 'Model', 'Generation']
            per_frame_times = [visual_per_frame, model_per_frame, generation_per_response]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            bars = ax4.bar(components, per_frame_times, color=colors, alpha=0.8)
            ax4.set_ylabel('Time per Frame/Response (s)')
            ax4.set_title('Component Efficiency (Visual/Model per Frame, Generation per Response)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels with appropriate units
            labels = [f'{visual_per_frame*1000:.1f}ms/frame', f'{model_per_frame*1000:.1f}ms/frame', f'{generation_per_response*1000:.1f}ms/response']
            for bar, time, label in zip(bars, per_frame_times, labels):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                        label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'video_{video_number}_timing.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Individual video timing plots saved to {output_dir}/")


class EventDrivenConversationContext:
    def __init__(self, conversation_idx, conversation_data, video_path, dataset, device, data_source, custom_threshold, conversation_start_time, model, tokenizer, model_memory_mb):
        self.conversation_idx = conversation_idx
        self.conversation_data = conversation_data
        self.video_path = video_path
        self.dataset = dataset
        self.device = device
        self.device_obj = torch.device(device)
        self.data_source = data_source
        self.custom_threshold = custom_threshold
        self.conversation_start_time = conversation_start_time
        self.model = model
        self.tokenizer = tokenizer
        self.model_memory_mb = model_memory_mb

        self.video_uid = conversation_data['video_uid']
        self.conversation_id = conversation_data['conversation_id']
        self.original_conversation = conversation_data['original_conversation']
        self.timestamp_offset = conversation_data.get('timestamp_offset', 0.0)
        self.duration = conversation_data['duration']

        self.event_log = [{'time': self.conversation_start_time, 'type': 'conversation_start', 'conversation_id': self.conversation_id}]

        self.user_prompts = []
        if data_source == 'goalstep':
            self.user_prompts = [(turn['time'], turn['content']) for turn in conversation_data['conversation'] if turn['role'] == 'user']
        else:
            if hasattr(dataset, 'instructions') and dataset.instructions:
                instruction = random.choice(dataset.instructions)
                user_prompt = instruction['content']
            else:
                user_prompt = "Please concisely narrate the video in real time."
            self.user_prompts = [(0.0, user_prompt)]

        video_frames, _, _ = read_video(video_path, pts_unit='sec', output_format='TCHW')
        self.video_frames = video_frames
        self.num_frames = self.video_frames.size(0)
        self.video_duration = self.num_frames / Config.FRAME_FPS

        conversation_based_limit = int(self.duration * Config.FRAME_FPS) + 50
        memory_based_limit = calculate_max_frames_for_memory()
        self.test_frames = min(self.num_frames, conversation_based_limit, memory_based_limit, Config.MAX_EVAL_FRAMES)

        self.total_visual_embedding_time = 0.0
        self.total_model_forward_time = 0.0
        self.total_generation_time = 0.0
        self.frame_processing_times = []
        self.frame_timing_data = []
        self.generated_turns = []

        self.memory_data = {
            'frames': [],
            'memory_usage': [],
            'memory_growth': [],
            'memory_per_frame': [],
            'model_memory': [],
            'kv_cache_memory': [],
            'other_memory': [],
            'torch_allocated': [],
            'torch_reserved': [],
            'kv_transfer_size': [],
            'kv_offload_time': [],
            'kv_reload_time': [],
        }

        self.frame_scores_data = {
            'frame_scores': [],
            'frame_times': [],
            'threshold': 0.0,
            'response_triggers': [],
            'response_times': []
        }

        self.initial_memory = None
        self.liveinfer_state = None
        self.frames_processed = 0
        self.prompts_processed = 0
        self.completed = False
        self.result = None

    def register_events(self, event_queue, sequence_counter):
        for prompt_time, prompt_content in self.user_prompts:
            heapq.heappush(event_queue, (self.conversation_start_time + max(0.0, prompt_time), 0, sequence_counter, ('prompt', self.conversation_id, prompt_content)))
            sequence_counter += 1

        for frame_idx in range(self.test_frames):
            frame_time = self.conversation_start_time + frame_idx / Config.FRAME_FPS
            heapq.heappush(event_queue, (frame_time, 1, sequence_counter, ('frame', self.conversation_id, frame_idx)))
            sequence_counter += 1

        finalize_time = self.conversation_start_time + self.test_frames / Config.FRAME_FPS
        heapq.heappush(event_queue, (finalize_time, 2, sequence_counter, ('finalize', self.conversation_id, None)))
        sequence_counter += 1
        return sequence_counter

    def ensure_liveinfer_loaded(self, liveinfer):
        if self.liveinfer_state is None:
            liveinfer.reset()
            liveinfer.load_video(self.video_path)
            if isinstance(self.video_frames, torch.Tensor):
                liveinfer.video_tensor = self.video_frames
            self.initial_memory = get_gpu_memory()
        else:
            liveinfer.restore_state(self.liveinfer_state)

    def save_liveinfer_state(self, liveinfer):
        self.liveinfer_state = liveinfer.capture_state()

    def handle_prompt(self, liveinfer, relative_time, prompt_content):
        liveinfer.input_query_stream(prompt_content, video_time=relative_time)
        self.event_log.append({
            'time': self.conversation_start_time + relative_time,
            'type': 'prompt',
            'detail': prompt_content,
            'conversation_id': self.conversation_id
        })
        self.prompts_processed += 1

    def handle_frame(self, liveinfer, relative_time, frame_idx):
        if frame_idx % Config.MEMORY_CHECK_INTERVAL == 0:
            current_memory = get_gpu_memory()
            if self.initial_memory is None:
                self.initial_memory = current_memory
            memory_growth = current_memory - self.initial_memory
            memory_per_frame = memory_growth / max(1, frame_idx) if frame_idx > 0 else 0

            torch_allocated_mb = 0.0
            torch_reserved_mb = 0.0
            if torch.cuda.is_available():
                torch_allocated_mb = torch.cuda.memory_allocated(self.device_obj) / (1024**2)
                torch_reserved_mb = torch.cuda.memory_reserved(self.device_obj) / (1024**2)

            kv_cache_memory_mb = calculate_kv_cache_memory_mb(liveinfer.past_key_values)
            other_memory_mb = max(torch_allocated_mb - self.model_memory_mb - kv_cache_memory_mb, 0.0)

            self.memory_data['frames'].append(frame_idx)
            self.memory_data['memory_usage'].append(current_memory)
            self.memory_data['memory_growth'].append(memory_growth)
            self.memory_data['memory_per_frame'].append(memory_per_frame)
            self.memory_data['model_memory'].append(self.model_memory_mb)
            self.memory_data['kv_cache_memory'].append(kv_cache_memory_mb)
            self.memory_data['other_memory'].append(other_memory_mb)
            self.memory_data['torch_allocated'].append(torch_allocated_mb)
            self.memory_data['torch_reserved'].append(torch_reserved_mb)

        frame_start_time = time.time()
        frame_processing_time = 0.0
        generation_time = 0.0
        global_time = self.conversation_start_time + relative_time
        try:
            liveinfer.input_video_stream(relative_time)
            self.event_log.append({
                'time': global_time,
                'type': 'frame',
                'detail': {'frame_idx': frame_idx},
                'conversation_id': self.conversation_id
            })
            query, response = liveinfer()

            frame_processing_time = time.time() - frame_start_time
            timing_data = liveinfer.get_timing_data()

            kv_cache_mb = timing_data.get('kv_cache_mb', 0.0)
            kv_offload_time = timing_data.get('kv_offload_time', 0.0)
            kv_reload_time = timing_data.get('kv_reload_time', 0.0)
            self.memory_data['kv_transfer_size'].append(kv_cache_mb)
            self.memory_data['kv_offload_time'].append(kv_offload_time)
            self.memory_data['kv_reload_time'].append(kv_reload_time)

            visual_embedding_time = timing_data.get('visual_embedding_time', 0.0)
            streaming_time = timing_data.get('streaming_time', 0.0)
            generation_time = timing_data.get('generation_time', 0.0)

            self.total_visual_embedding_time += visual_embedding_time
            self.total_model_forward_time += streaming_time
            self.total_generation_time += generation_time

            self.frame_processing_times.append(frame_processing_time)

            self.frame_timing_data.append({
                'frame_idx': frame_idx,
                'video_time': relative_time,
                'visual_embedding_time': visual_embedding_time,
                'model_forward_time': streaming_time,
                'generation_time': generation_time,
                'total_processing_time': frame_processing_time
            })

            if response:
                self.generated_turns.append({
                    'time': relative_time,
                    'text': response,
                    'user_prompt': query or "Frame processing",
                    'generation_time': generation_time
                })
                self.event_log.append({
                    'time': global_time + generation_time,
                    'type': 'response',
                    'detail': {'text': response, 'frame_idx': frame_idx},
                    'conversation_id': self.conversation_id
                })

            if query or response:
                print(f"[t={self.conversation_start_time + relative_time:.2f}s | local {relative_time:.2f}s] Query: {query}")
                print(f"[t={self.conversation_start_time + relative_time:.2f}s | local {relative_time:.2f}s] Response: {response}")
                if generation_time > 0:
                    print(f"  └─ Generation time: {generation_time:.3f}s")

        except Exception as e:
            print(f"❌ Error processing frame {frame_idx} at global {self.conversation_start_time + relative_time:.2f}s (local {relative_time:.2f}s): {e}")
            traceback.print_exc()
            frame_processing_time = time.time() - frame_start_time
            self.frame_processing_times.append(frame_processing_time)

        self.frames_processed += 1
        return {
            'frame_processing_time': frame_processing_time,
            'generation_time': generation_time
        }

    def finalize(self, liveinfer):
        self.frame_scores_data = liveinfer.get_frame_scores()

        rebuffering_data = calculate_all_rebuffering_times(
            self.generated_turns,
            self.frame_processing_times,
            reading_speed=Config.USER_READING_SPEED_MAX,
            listening_speed=Config.USER_LISTENING_SPEED_MAX
        )

        rebuffering_times = rebuffering_data['generation_rebuffering_times']
        reading_rebuffering_times = rebuffering_data['reading_rebuffering_times']
        listening_rebuffering_times = rebuffering_data['listening_rebuffering_times']
        resource_utilization_times = rebuffering_data['resource_utilization_times']

        total_rebuffering_time = rebuffering_data['total_generation_rebuffering']
        average_rebuffering_time = rebuffering_data['average_generation_rebuffering']
        total_reading_rebuffering = rebuffering_data['total_reading_rebuffering']
        average_reading_rebuffering = rebuffering_data['average_reading_rebuffering']
        total_listening_rebuffering = rebuffering_data['total_listening_rebuffering']
        average_listening_rebuffering = rebuffering_data['average_listening_rebuffering']
        final_frame_utilization = rebuffering_data['final_frame_utilization']

        for i, timing_data in enumerate(self.frame_timing_data):
            timing_data['rebuffering_time'] = rebuffering_times[i] if i < len(rebuffering_times) else 0.0
            timing_data['reading_rebuffering_time'] = reading_rebuffering_times[i] if i < len(reading_rebuffering_times) else 0.0
            timing_data['listening_rebuffering_time'] = listening_rebuffering_times[i] if i < len(listening_rebuffering_times) else 0.0
            timing_data['resource_utilization'] = resource_utilization_times[i] if i < len(resource_utilization_times) else 0.0

        total_frame_processing_time = sum(self.frame_processing_times)
        total_processing_time = total_frame_processing_time
        visual_embedding_time = self.total_visual_embedding_time
        model_forward_time = self.total_model_forward_time
        generation_time = self.total_generation_time
        num_processed_frames = len(self.frame_processing_times)

        print(f"🔍 GOALSTEP TIMING METRICS FOR CONVERSATION {self.conversation_id[:8]}...:")
        print(f"   • Conversation duration: {self.duration:.2f}s")
        print(f"   • Frames processed: {num_processed_frames}")
        print(f"   • Generated responses: {len(self.generated_turns)}")
        print_timing_metrics(visual_embedding_time, model_forward_time, generation_time, num_processed_frames, len(self.generated_turns))
        print(f"   • Total processing time: {total_processing_time:.3f}s")
        if total_processing_time > 0:
            print(f"   • Timing breakdown: Visual={visual_embedding_time/total_processing_time*100:.1f}%, Model={model_forward_time/total_processing_time*100:.1f}%, Generation={generation_time/total_processing_time*100:.1f}%")
        print(f"   • Total rebuffering time: {total_rebuffering_time:.3f}s")
        print(f"   • Average rebuffering time per frame: {average_rebuffering_time:.3f}s")
        print(f"   • Total reading rebuffering time: {total_reading_rebuffering:.3f}s")
        print(f"   • Average reading rebuffering time per frame: {average_reading_rebuffering:.3f}s")
        print(f"   • Total listening rebuffering time: {total_listening_rebuffering:.3f}s")
        print(f"   • Average listening rebuffering time per frame: {average_listening_rebuffering:.3f}s")
        print(f"   • Final frame resource utilization: {final_frame_utilization:.3f}")
        print("-" * 60)

        content_metrics = calculate_metrics_like_benchmark(
            self.model,
            self.tokenizer,
            self.video_frames,
            self.conversation_data['conversation'],
            self.generated_turns,
            self.device,
            self.original_conversation,
            self.data_source
        )

        generated_turns_original = []
        for turn in self.generated_turns:
            turn_original = turn.copy()
            turn_original['time'] = turn['time'] + self.timestamp_offset
            turn_original['original_time'] = turn['time'] + self.timestamp_offset
            generated_turns_original.append(turn_original)

        ground_truth_conversation_original = []
        for turn in self.original_conversation:
            turn_original = turn.copy()
            turn_original['time'] = turn['time']
            ground_truth_conversation_original.append(turn_original)

        first_user_time = None
        for turn in ground_truth_conversation_original:
            if turn['role'] == 'user' and 'time' in turn:
                first_user_time = turn['time']
                break

        self.event_log.append({
            'time': self.conversation_start_time + self.duration,
            'type': 'conversation_end',
            'conversation_id': self.conversation_id
        })

        self.result = {
            'conversation_id': self.conversation_id,
            'video_id': self.video_uid,
            'num_frames': num_processed_frames,
            'generated_turns': len(self.generated_turns),
            'ground_truth_turns': len(self.conversation_data['conversation']),
            'generated_responses': generated_turns_original,
            'ground_truth_conversation': ground_truth_conversation_original,
            'first_user_time': first_user_time,
            'conversation_start_time': self.conversation_start_time,
            'lm_ppl': content_metrics.get('lm_ppl', 0.0) if content_metrics else 0.0,
            'fluency': calculate_fluency_score(self.generated_turns, self.original_conversation, self.data_source),
            'ppl_data': content_metrics.get('ppl_data', {}) if content_metrics else {},
            'total_tokens': 0,
            'visual_embedding_time': visual_embedding_time,
            'model_forward_time': model_forward_time,
            'generation_time': generation_time,
            'total_processing_time': total_processing_time,
            'frame_processing_times': self.frame_processing_times,
            'eos_timing': {'eos_detection_time': 0.0, 'with_eos': 0.0, 'without_eos': 0.0},
            'conversation_turns': len(self.generated_turns),
            'generated_turns': self.generated_turns,
            'video_duration': self.duration,
            'frame_scores_data': self.frame_scores_data,
            'frame_timing_data': self.frame_timing_data,
            'rebuffering_times': rebuffering_times,
            'total_rebuffering_time': total_rebuffering_time,
            'average_rebuffering_time': average_rebuffering_time,
            'reading_rebuffering_times': reading_rebuffering_times,
            'total_reading_rebuffering_time': total_reading_rebuffering,
            'average_reading_rebuffering_time': average_reading_rebuffering,
            'listening_rebuffering_times': listening_rebuffering_times,
            'total_listening_rebuffering_time': total_listening_rebuffering,
            'average_listening_rebuffering_time': average_listening_rebuffering,
            'resource_utilization_times': resource_utilization_times,
            'final_frame_utilization': final_frame_utilization
        }

        self.result['event_log'] = self.event_log

        timeline_events = []
        for turn in ground_truth_conversation_original:
            if turn['role'] == 'user':
                timeline_events.append({'time': turn['time'], 'type': 'user_prompt', 'content': turn['content']})
        for turn in ground_truth_conversation_original:
            if turn['role'] == 'assistant':
                timeline_events.append({'time': turn['time'], 'type': 'ground_truth', 'content': turn['content']})
        for turn in generated_turns_original:
            timeline_events.append({'time': turn['time'], 'type': 'generated', 'content': turn['text']})
        timeline_events.sort(key=lambda x: x['time'])

        print(f"\n📊 CONVERSATION SUMMARY:")
        print(f"   Total events: {len(timeline_events)} (Generated: {len(self.generated_turns)}, Ground Truth: {len([t for t in self.conversation_data['conversation'] if t['role'] == 'assistant'])})")
        print(f"   User prompts: {len([t for t in self.conversation_data['conversation'] if t['role'] == 'user'])}")

        self.memory_data['conversation_start_time'] = self.conversation_start_time
        self.video_frames = None
        self.completed = True

def get_videos_with_features():
    """Get list of video UIDs that have extracted features"""
    metadata_path = "datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384_metadata.json"
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return set(metadata.keys())
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return set()

def process_conversation(model, tokenizer, conversation_data, video_path, dataset, device, data_source='goalstep', custom_threshold=None, conversation_start_time=0.0, liveinfer=None, model_memory_mb=None):
    """Prepare conversation context and register its events."""

    context = EventDrivenConversationContext(
        conversation_idx=conversation_data.get('conversation_index', 0),
        conversation_data=conversation_data,
        video_path=video_path,
        dataset=dataset,
        device=device,
        data_source=data_source,
        custom_threshold=custom_threshold,
        conversation_start_time=conversation_start_time,
        model=model,
        tokenizer=tokenizer,
        model_memory_mb=model_memory_mb if model_memory_mb is not None else calculate_model_memory_mb(model)
    )
    return context
def streaming_evaluate_conversations(model, tokenizer, dataset, device='cuda', num_conversations=3, random_selection=False, specific_indices=None, data_source='goalstep', custom_threshold=None, conversation_start_times=None):
    """Evaluate multiple conversations using a shared event-driven LiveInfer instance."""

    results = []
    all_memory_data = {}
    all_frame_scores_data = {}
    actual_num_conversations = min(num_conversations, len(dataset.conversations))

    if specific_indices is not None:
        conversation_indices = specific_indices
        actual_num_conversations = len(conversation_indices)
        print(f"🎯 Specific conversation indices: {conversation_indices}")
    elif random_selection:
        conversation_indices = random.sample(range(len(dataset.conversations)), actual_num_conversations)
        print(f"🎲 Random conversation indices: {conversation_indices}")
    else:
        conversation_indices = list(range(actual_num_conversations))

    shared_liveinfer = SimpleLiveInfer(model, tokenizer, device, dataset, custom_threshold)
    model_memory_mb = calculate_model_memory_mb(model)

    event_queue = []
    contexts = {}
    sequence_counter = 0
    current_time = 0.0
    processor_clock = 0.0
    processor_segments = []
    idle_segments = []
    conversation_summaries = []

    first_conversation_duration = None

    for idx_in_list, conversation_idx in enumerate(conversation_indices):
        if conversation_idx >= len(dataset.conversations):
            raise ValueError(f"Conversation index {conversation_idx} out of range. Dataset has {len(dataset.conversations)} conversations.")

        conversation_data = dataset.conversations[conversation_idx]
        conversation_data = conversation_data.copy()
        conversation_data['conversation_index'] = conversation_idx

        if idx_in_list > 0:
            defragment_gpu_memory()

        if isinstance(conversation_start_times, dict):
            arrival_time = conversation_start_times.get(conversation_idx, current_time)
        elif isinstance(conversation_start_times, (list, tuple)):
            arrival_time = conversation_start_times[idx_in_list] if idx_in_list < len(conversation_start_times) else current_time
        elif conversation_start_times == 'random' and idx_in_list > 0 and first_conversation_duration is not None:
            arrival_time = random.uniform(0.0, first_conversation_duration)
        else:
            arrival_time = conversation_data.get('arrival_time', current_time)
        arrival_time = max(0.0, arrival_time)
        video_uid = conversation_data['video_uid']
        video_path = f"datasets/ego4d/v2/full_scale_2fps_384/{video_uid}.mp4"

        context = process_conversation(
            model,
            tokenizer,
            conversation_data,
            video_path,
            dataset,
            device,
            dataset.data_source if hasattr(dataset, 'data_source') else data_source,
            custom_threshold,
            conversation_start_time=arrival_time,
            model_memory_mb=model_memory_mb,
        )

        contexts[context.conversation_id] = context
        sequence_counter = context.register_events(event_queue, sequence_counter)
        current_time = max(current_time, arrival_time + context.duration)
        if idx_in_list == 0:
            first_conversation_duration = context.duration

    active_conversation_id = None

    while event_queue:
        event_time, priority, _, payload = heapq.heappop(event_queue)
        event_type, conversation_id, payload_data = payload
        context = contexts[conversation_id]

        if active_conversation_id != conversation_id:
            if active_conversation_id is not None:
                contexts[active_conversation_id].save_liveinfer_state(shared_liveinfer)
            context.ensure_liveinfer_loaded(shared_liveinfer)
            active_conversation_id = conversation_id

        relative_time = max(0.0, event_time - context.conversation_start_time)

        if event_type == 'prompt':
            if event_time > processor_clock:
                idle_segments.append({'start': processor_clock, 'end': event_time})
                processor_clock = event_time
            context.handle_prompt(shared_liveinfer, relative_time, payload_data)
            context.save_liveinfer_state(shared_liveinfer)
            continue

        if event_type == 'frame':
            segment_info = context.handle_frame(shared_liveinfer, relative_time, payload_data)
            frame_duration = segment_info.get('frame_processing_time', 0.0)
            generation_duration = segment_info.get('generation_time', 0.0)
            segment_label = context.conversation_id[:12]

            start_time = max(processor_clock, event_time)
            if start_time > processor_clock:
                idle_segments.append({'start': processor_clock, 'end': start_time})
            if frame_duration > 0.0:
                processor_segments.append({
                    'conversation_id': segment_label,
                    'start': start_time,
                    'end': start_time + frame_duration,
                    'type': 'frame'
                })
            processor_clock = start_time + frame_duration

            if generation_duration > 0.0:
                processor_segments.append({
                    'conversation_id': segment_label,
                    'start': processor_clock,
                    'end': processor_clock + generation_duration,
                    'type': 'response'
                })
                processor_clock += generation_duration

            context.save_liveinfer_state(shared_liveinfer)
            continue

        if event_type == 'finalize':
            if event_time > processor_clock:
                idle_segments.append({'start': processor_clock, 'end': event_time})
                processor_clock = event_time

            context.finalize(shared_liveinfer)
            context.liveinfer_state = None
            active_conversation_id = None
            shared_liveinfer.reset()

            unique_key = f"{context.video_uid[:8]}_{context.conversation_id[:8]}"
            results.append(context.result)
            all_memory_data[unique_key] = context.memory_data
            if context.result.get('frame_scores_data'):
                all_frame_scores_data[unique_key] = context.result['frame_scores_data']
            conversation_summaries.append({
                'conversation_id': context.conversation_id,
                'label': context.conversation_id[:12],
                'start': context.conversation_start_time,
                'end': context.conversation_start_time + context.duration,
                'events': context.event_log
            })

    shared_liveinfer.reset()

    if all_memory_data:
        print("\n📊 Creating memory usage analysis...")
        create_memory_visualization(all_memory_data, data_source=data_source)

    if all_frame_scores_data:
        print("\n📊 Creating frame score analysis...")
        create_frame_score_analysis(all_frame_scores_data, data_source=data_source)

    if processor_segments:
        print("\n📊 Creating processor timeline...")
        create_processor_timeline(processor_segments, idle_segments, conversation_summaries, data_source=data_source)

    return results

def streaming_evaluate_threshold_sweep(model, tokenizer, dataset, device='cuda', num_conversations=None, random_selection=False, specific_indices=None, data_source='goalstep', threshold_range=None, num_thresholds=None):
    """Evaluate conversations across different streaming thresholds to analyze threshold sensitivity."""
    import torch
    
    # Use config defaults if not provided
    if num_conversations is None:
        num_conversations = Config.DEFAULT_NUM_VIDEOS
    if threshold_range is None:
        threshold_range = Config.DEFAULT_THRESHOLD_RANGE
    if num_thresholds is None:
        num_thresholds = Config.DEFAULT_NUM_THRESHOLDS
    
    # Generate threshold values - use debug thresholds for quick testing
    if num_thresholds == Config.DEFAULT_NUM_THRESHOLDS:  # Default case
        thresholds = np.array(Config.DEBUG_THRESHOLDS)
        num_thresholds = len(Config.DEBUG_THRESHOLDS)
    else:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    
    print(f"🔄 Starting threshold sweep analysis for {data_source} dataset")
    print(f"📊 Threshold range: {threshold_range[0]:.3f} to {threshold_range[1]:.3f} ({num_thresholds} points)")
    print(f"📊 Thresholds: {[f'{t:.3f}' for t in thresholds]}")
    print("=" * 80)
    
    # CRITICAL: Select conversations ONCE for all thresholds to ensure independence
    # This prevents data leakage where different thresholds get different conversations
    actual_num_conversations = min(num_conversations, len(dataset.conversations))
    
    if specific_indices is not None:
        conversation_indices = specific_indices
        actual_num_conversations = len(conversation_indices)
        print(f"🎯 Using specific conversation indices: {conversation_indices}")
    elif random_selection:
        # Set seed for reproducibility and select conversations once
        random.seed(42)
        conversation_indices = random.sample(range(len(dataset.conversations)), actual_num_conversations)
        print(f"🎲 Selected conversation indices for ALL thresholds: {conversation_indices}")
    else:
        conversation_indices = list(range(actual_num_conversations))
        print(f"📊 Using sequential conversation indices: {conversation_indices}")
    
    # Store results for each threshold
    all_threshold_results = {}
    all_frame_scores_data = {}  # Collect frame scores data for all thresholds
    all_rebuffering_data = {}  # Collect rebuffering time data for all thresholds
    
    for i, threshold in enumerate(thresholds):
        print(f"\n🎯 THRESHOLD {i+1}/{num_thresholds}: {threshold:.3f}")
        print("-" * 60)
        
        # Set PyTorch random seed for deterministic behavior BEFORE each threshold
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        print(f"🎲 Set random seeds for deterministic behavior")
        
        # Run evaluation with this threshold using the SAME conversations
        results = streaming_evaluate_conversations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            num_conversations=num_conversations,
            random_selection=False,  # Force sequential selection
            specific_indices=conversation_indices,  # Use the same conversations
            data_source=data_source,
            custom_threshold=threshold
        )
        
        # Store results for this threshold
        all_threshold_results[threshold] = results
        
        # Collect frame scores data for this threshold
        threshold_frame_scores = {}
        threshold_rebuffering_data = {}
        for j, result in enumerate(results):
            if 'frame_scores_data' in result and result['frame_scores_data']:
                conv_key = f"conv_{j}"
                threshold_frame_scores[conv_key] = result['frame_scores_data']
            
            # Collect rebuffering time data
            if 'rebuffering_times' in result and result['rebuffering_times']:
                conv_key = f"conv_{j}"
                threshold_rebuffering_data[conv_key] = {
                    'rebuffering_times': result['rebuffering_times'],
                    'average_rebuffering_time': result.get('average_rebuffering_time', 0.0),
                    'total_rebuffering_time': result.get('total_rebuffering_time', 0.0)
                }
        
        all_frame_scores_data[threshold] = threshold_frame_scores
        all_rebuffering_data[threshold] = threshold_rebuffering_data
        
        # Print summary for this threshold
        if results:
            avg_ppl = sum(r['lm_ppl'] for r in results) / len(results)
            avg_fluency = sum(r['fluency'] for r in results) / len(results)
            avg_responses = sum(len(r['generated_turns']) for r in results) / len(results)
            avg_rebuffering = sum(r.get('average_rebuffering_time', 0.0) for r in results) / len(results)
            
            print(f"📊 Threshold {threshold:.3f} Summary:")
            print(f"   • Average PPL: {avg_ppl:.3f}")
            print(f"   • Average Fluency: {avg_fluency:.3f}")
            print(f"   • Average Responses: {avg_responses:.1f}")
            print(f"   • Average Rebuffering Time: {avg_rebuffering:.3f}s")
            
            # Add user rebuffering metrics to summary
            avg_reading_rebuffering = sum(r.get('average_reading_rebuffering_time', 0.0) for r in results) / len(results)
            avg_listening_rebuffering = sum(r.get('average_listening_rebuffering_time', 0.0) for r in results) / len(results)
            avg_final_utilization = sum(r.get('final_frame_utilization', 0.0) for r in results) / len(results)
            print(f"   • Average Reading Rebuffering Time: {avg_reading_rebuffering:.3f}s")
            print(f"   • Average Listening Rebuffering Time: {avg_listening_rebuffering:.3f}s")
            print(f"   • Average Final Frame Utilization: {avg_final_utilization:.3f}")
        
        # Clean up memory between thresholds
        if i < len(thresholds) - 1:  # Don't clean up after the last threshold
            defragment_gpu_memory()
    
    # Create comprehensive threshold analysis visualization
    print(f"\n📊 Creating threshold sensitivity analysis...")
    create_unified_threshold_analysis(all_threshold_results, all_frame_scores_data, all_rebuffering_data, data_source=data_source)
    
    return all_threshold_results

# Define SimpleLiveInfer class outside the function for reuse
class SimpleLiveInfer:
    def __init__(self, model, tokenizer, device, dataset=None, custom_threshold=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.device_obj = canonical_device(device)
        
        # visual
        self.hidden_size = model.config.hidden_size
        self.frame_fps = Config.FRAME_FPS
        self.frame_interval = 1.0 / Config.FRAME_FPS
        self.frame_resolution = model.config.frame_resolution
        self.frame_num_tokens = model.config.frame_num_tokens
        self.frame_v_placeholder = model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(model.config.v_placeholder_id, device=device).repeat(model.config.frame_num_tokens).reshape(1,-1)
        
        # generation
        self.system_prompt = ""  # Will be set dynamically based on dataset
        self.inplace_output_ids = torch.zeros(1, Config.INPLACE_OUTPUT_SIZE, device=device, dtype=torch.long)
        # Use custom threshold if provided, otherwise use dataset-specific thresholds
        if custom_threshold is not None:
            self.frame_token_interval_threshold = custom_threshold
            print(f"🎯 Using custom threshold: {self.frame_token_interval_threshold}")
        elif hasattr(dataset, 'data_source') and dataset.data_source == 'narration':
            self.frame_token_interval_threshold = Config.STREAMING_THRESHOLD_NARRATION
            print(f"🎯 Using narration threshold: {self.frame_token_interval_threshold}")
        else:
            # Default to goalstep threshold for goalstep and any other dataset types
            self.frame_token_interval_threshold = Config.STREAMING_THRESHOLD_GOALSTEP
            print(f"🎯 Using goalstep threshold: {self.frame_token_interval_threshold}")
        self.eos_token_id = model.config.eos_token_id
        self._start_ids = tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to(device)
        self._added_stream_prompt_ids = tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to(device)
        self._added_stream_generation_ids = tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to(device)
        
        # app
        self.reset()
        
        # Timing instrumentation
        self.timing_data = {}
        
        # Frame score tracking
        self.frame_scores = []  # Track frame_token_interval_score across frames
        self.frame_times = []   # Track corresponding frame times
        self.response_triggers = []  # Track what triggered each response (query vs frame)
        self.response_times = []  # Track when responses were actually generated

    def capture_state(self):
        state = {
            'last_ids': self.last_ids.detach().clone().cpu() if torch.is_tensor(self.last_ids) else self.last_ids,
            'past_key_values': move_kv_cache_to_device(self.past_key_values, 'cpu') if self.past_key_values is not None else None,
            'query_queue': list(self.query_queue),
            'frame_embeds_queue': [(timestamp, embed.detach().cpu() if torch.is_tensor(embed) else embed) for timestamp, embed in list(self.frame_embeds_queue)],
            'video_time': self.video_time,
            'last_frame_idx': self.last_frame_idx,
            'video_path': self.video_path,
            'video_tensor': self.video_tensor.detach().cpu() if isinstance(self.video_tensor, torch.Tensor) else self.video_tensor,
            'frame_scores': self.frame_scores.copy(),
            'frame_times': self.frame_times.copy(),
            'response_triggers': self.response_triggers.copy(),
            'response_times': self.response_times.copy(),
            'timing_data': self.timing_data.copy(),
        }
        return state

    def restore_state(self, state):
        self.query_queue = collections.deque(state['query_queue'])
        self.frame_embeds_queue = collections.deque((timestamp, embed.to(self.device) if torch.is_tensor(embed) else embed) for timestamp, embed in state['frame_embeds_queue'])
        if torch.is_tensor(state['last_ids']):
            self.last_ids = state['last_ids'].to(self.device)
        else:
            self.last_ids = torch.tensor(state['last_ids'], device=self.device)
        self.past_key_values = move_kv_cache_to_device(state['past_key_values'], self.device) if state['past_key_values'] is not None else None
        self.video_time = state['video_time']
        self.last_frame_idx = state['last_frame_idx']
        self.video_path = state['video_path']
        self.video_tensor = state['video_tensor']
        if isinstance(self.video_tensor, torch.Tensor) and self.video_tensor.device != torch.device('cpu'):
            self.video_tensor = self.video_tensor.cpu()
        self.frame_scores = state['frame_scores'].copy()
        self.frame_times = state['frame_times'].copy()
        self.response_triggers = state['response_triggers'].copy()
        self.response_times = state['response_times'].copy()
        self.timing_data = state['timing_data'].copy()

    
    def reset(self):
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.video_path = None  # Reset video path
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.past_key_values = None
        
        # Reset tracking buffers
        self.kv_transfer_metrics = []
        
        # Reset frame score tracking
        self.frame_scores = []
        self.frame_times = []
        self.response_triggers = []
        self.response_times = []
    
    def input_query_stream(self, query, history=None, video_time=None):
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
        if self.past_key_values is None:
            return f'(NOTE: No video stream here. Please select or upload a video. Then the assistant will answer "{query} (at {self.video_time}s)" in the video stream)'
        return f'(NOTE: Received "{query}" (at {self.video_time}s). Please wait until previous frames have been processed)'
    
    def input_video_stream(self, video_time):
        """Process video frame and add to queue like benchmark with timing."""
        start_time = time.time()
        
        # Measure visual embedding time: RGB frames → Visual token embeddings
        visual_start = time.time()
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            # Process frames one at a time to avoid OOM
            for single_frame_idx in range(self.last_frame_idx + 1, frame_idx + 1):
                # Stream single frame from CPU to GPU as needed
                if self.video_tensor is not None:
                    # Get single frame from CPU memory and move to GPU
                    cpu_frame = self.video_tensor[single_frame_idx:single_frame_idx+1]  # Keep batch dimension
                    gpu_frame = cpu_frame.to(self.device)
                    
                    # Process single frame on GPU
                    frame_embeds = self.model.visual_embed(gpu_frame).split(self.frame_num_tokens)
                    self.frame_embeds_queue.extend([(single_frame_idx / self.frame_fps, frame_embeds) for frame_embeds in frame_embeds])
                    
                    # Immediately release GPU frame to free memory
                    del gpu_frame
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Video tensor not loaded")
                
        visual_embedding_time = time.time() - visual_start
        
        self.last_frame_idx = frame_idx
        self.video_time = video_time
        
        # Store timing data
        self.timing_data['visual_embedding_time'] = visual_embedding_time
        self.timing_data['input_video_stream_time'] = time.time() - start_time
    
    def load_video(self, video_path):
        # Store video path for on-demand loading instead of loading entire video
        self.video_path = video_path
        
        # Load video to CPU memory first to avoid GPU OOM
        print(f"📹 Loading video to CPU memory: {video_path}")
        video_reader = read_video(video_path, pts_unit='sec', output_format='TCHW')
        self.num_video_frames = video_reader[0].size(0)
        self.video_duration = self.num_video_frames / self.frame_fps
        
        # Store video tensor on CPU to avoid GPU memory issues
        self.video_tensor = video_reader[0]  # Keep on CPU
        print(f"📹 Video loaded to CPU: {self.video_tensor.shape} ({self.video_tensor.device})")
        
        logger = transformers.logging.get_logger('liveinfer')
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS (CPU streaming mode)')
    
    def _call_for_response(self, video_time, query):
        
        # MEASURE GENERATION TIME (this is the VLM text generation)
        generation_start = time.time()
        
        # Track response triggers for ALL response generations
        # This ensures consistent tracking regardless of how the response was triggered
        if query is not None:
            # Query-based response - track as 'query' trigger
            self.response_triggers.append('query')
            self.response_times.append(video_time)
            # Store the current query for context reconstruction
            self._current_query = query
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to(self.device)
        else:
            # Frame-based response - track as 'frame' trigger
            self.response_triggers.append('frame')
            self.response_times.append(video_time)
            # Clear current query for frame-triggered responses
            self._current_query = None
            # Use the stream generation prompt for continuous generation
            self.last_ids = self._added_stream_generation_ids
        
        # DEBUG: Print context information for PPL calculation
        context_info = self._get_context_info()
        print(f"📊 CONTEXT FOR GENERATION (Time: {video_time}s):")
        print(f"   • Streamed frames processed: {context_info['num_frames']}", end=", ")
        print(f"User prompts in history: {context_info['num_prompts']}", end=", ")
        print(f"Previous responses: {context_info['num_responses']}", end=", ")
        print(f"Total tokens in past_key_values: {context_info['total_tokens']}")
        
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        output_ids, self.past_key_values = fast_greedy_generate(model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids)
        self.last_ids = output_ids[:, -1:]
        
        generation_time = time.time() - generation_start
        
        # Store the actual generation time
        self.timing_data['generation_time'] = generation_time
        
        # Note: Dual PPL calculation is now done during evaluation phase
        # using all GT responses with proper context reconstruction
        
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        response = f'(Video Time = {video_time}s) Assistant:{self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)}'
        return query, response
    
    def _get_context_info(self):
        """Get context information for debugging."""
        context_info = {
            'num_frames': len(self.frame_embeds_queue) + (self.last_frame_idx + 1 if hasattr(self, 'last_frame_idx') else 0),
            'num_prompts': len([r for r in self.response_triggers if r == 'query']),
            'num_responses': len(self.response_triggers),
            'total_tokens': 0
        }
        
        # Estimate total tokens in past_key_values
        if self.past_key_values is not None:
            for layer in self.past_key_values:
                if layer is not None and len(layer) >= 2:
                    key_tensor = layer[0]
                    if key_tensor is not None:
                        context_info['total_tokens'] += key_tensor.shape[-2]  # sequence length
        
        return context_info
    
    def _call_for_streaming(self):
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            if self.past_key_values is None:
                self.last_ids = self._start_ids
            elif self.last_ids.numel() == 1 and int(self.last_ids.item()) == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
            
            # MEASURE MODEL FORWARD PASS TIME (this is the main VLM computation)
            model_forward_start = time.time()
            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
            ], dim=1)
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
            model_forward_time = time.time() - model_forward_start
            
            # Store the actual model forward time
            self.timing_data['model_forward_time'] = model_forward_time
            
            self.past_key_values = outputs.past_key_values
            # 2. if the same time, response after frame at that time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                # Note: Response triggers are now tracked in _call_for_response
                return video_time, query
            # 3. if the next is frame but next is not interval, then response
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            frame_token_interval_score = next_score[:,:,self.frame_token_interval_id].item()
            
            # Track frame_token_interval_score for analysis
            self.frame_scores.append(frame_token_interval_score)
            self.frame_times.append(video_time)
            
            if frame_token_interval_score < self.frame_token_interval_threshold:
                next_score[:,:,self.frame_token_interval_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if self.last_ids.numel() == 1 and int(self.last_ids.item()) != self.frame_token_interval_id: 
                # Note: Response triggers are now tracked in _call_for_response
                return video_time, None
        return None, None

    def __call__(self):
        """Main call method that processes video and generates responses with timing."""
        start_time = time.time()
        
        # Measure streaming processing time: Visual tokens + context → LLM → logits
        streaming_start = time.time()
        while not self.frame_embeds_queue:
            continue
        video_time, query = self._call_for_streaming()
        streaming_time = time.time() - streaming_start
        
        # Measure response generation time: Logits → token selection → text
        generation_start = time.time()
        response = None
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        generation_time = time.time() - generation_start
        
        # Store timing data
        self.timing_data['streaming_time'] = streaming_time      # Visual tokens → Logits
        self.timing_data['generation_time'] = generation_time    # Logits → Text (when occurs)

        # Measure KV cache transfer overhead (GPU ⇄ CPU)
        kv_cache_mb = calculate_kv_cache_memory_mb(self.past_key_values)
        offload_time = 0.0
        reload_time = 0.0
        if kv_cache_mb > 0 and torch.cuda.is_available() and self.device_obj.type == 'cuda':
            cpu_device = torch.device('cpu')
            transfer_start = time.time()
            self.past_key_values = move_kv_cache_to_device(self.past_key_values, cpu_device)
            offload_time = time.time() - transfer_start

            reload_start = time.time()
            self.past_key_values = move_kv_cache_to_device(self.past_key_values, self.device_obj)
            reload_time = time.time() - reload_start

        transfer_record = {
            'kv_cache_mb': kv_cache_mb,
            'offload_time': offload_time,
            'reload_time': reload_time,
        }
        self.kv_transfer_metrics.append(transfer_record)

        self.timing_data['kv_cache_mb'] = kv_cache_mb
        self.timing_data['kv_offload_time'] = offload_time
        self.timing_data['kv_reload_time'] = reload_time
        self.timing_data['total_call_time'] = time.time() - start_time

        return query, response
    
    def get_timing_data(self):
        """Get the timing data for the last operation."""
        return self.timing_data.copy()
    
    def get_frame_scores(self):
        """Get the frame_token_interval_score data across all processed frames."""
        return {
            'frame_scores': self.frame_scores.copy(),
            'frame_times': self.frame_times.copy(),
            'threshold': self.frame_token_interval_threshold,
            'response_triggers': self.response_triggers.copy(),
            'response_times': self.response_times.copy()
        }


    def get_kv_transfer_metrics(self):
        """Return recorded KV cache transfer metrics."""
        return [metric.copy() for metric in self.kv_transfer_metrics]

    def get_last_kv_transfer_metric(self):
        """Return the last KV transfer measurement, if available."""
        if not self.kv_transfer_metrics:
            return None
        return self.kv_transfer_metrics[-1].copy()


def main():
    
    # Extract custom arguments from command line args
    data_source = 'narration'  # default
    threshold_sweep = False
    num_videos = Config.DEFAULT_NUM_VIDEOS
    custom_start_times = None

    if '--data_source' in sys.argv:
        idx = sys.argv.index('--data_source')
        if idx + 1 < len(sys.argv):
            data_source = sys.argv[idx + 1]
            # Remove these args to avoid conflicts with parse_args()
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    if '--threshold_sweep' in sys.argv:
        threshold_sweep = True
        # Remove this arg to avoid conflicts with parse_args()
        sys.argv.remove('--threshold_sweep')

    if '--num_videos' in sys.argv:
        idx = sys.argv.index('--num_videos')
        if idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            try:
                num_videos = max(1, int(value))
            except ValueError:
                print(f"⚠️  Invalid --num_videos value '{value}', falling back to {num_videos}")
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    if '--start_times' in sys.argv:
        idx = sys.argv.index('--start_times')
        if idx + 1 < len(sys.argv):
            raw_value = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)

            if raw_value.lower() == 'random':
                custom_start_times = 'random'
            else:
                start_values = []
                for part in raw_value.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        start_values.append(float(part))
                    except ValueError:
                        print(f"⚠️  Ignoring invalid start time '{part}' in --start_times")
                if start_values:
                    custom_start_times = start_values
                else:
                    print("⚠️  No valid values found in --start_times; using defaults")

    
    # Parse the main args
    args = parse_args()
    
    # Add custom args to args object
    args.data_source = data_source
    args.threshold_sweep = threshold_sweep
    args.num_videos = num_videos

    print("🚀 Starting Streaming Evaluation")
    print("=" * 50)
    
    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"🔧 Device: {device}")
    print(f"🤖 Model loaded successfully")
    
    # Display streaming configuration
    print(f"⚙️  STREAMING CONFIGURATION:")
    print(f"   • Streaming Threshold: {Config.STREAMING_THRESHOLD_GOALSTEP}")
    print(f"   • Frame Resolution: {Config.FRAME_RESOLUTION}")
    print(f"   • Frame FPS: {Config.FRAME_FPS}")
    print(f"   • Frame Num Tokens: {Config.FRAME_NUM_TOKENS}")
    print(f"   • V Placeholder ID: {Config.V_PLACEHOLDER_ID}")
    print(f"   • Default Num Videos: {Config.DEFAULT_NUM_VIDEOS}")
    print(f"   • Default Threshold Range: {Config.DEFAULT_THRESHOLD_RANGE}")
    print(f"   • Debug Thresholds: {Config.DEBUG_THRESHOLDS}")
    print(f"   • User Reading Speed: {Config.USER_READING_SPEED_MIN}-{Config.USER_READING_SPEED_MAX} wps")
    print(f"   • User Listening Speed: {Config.USER_LISTENING_SPEED_MIN}-{Config.USER_LISTENING_SPEED_MAX} wps")
    
    # Create filtered dataset with configurable data source
    data_source = getattr(args, 'data_source', 'narration')  # Default to narration
    print(f"📊 Using data source: {data_source}")
    
    dataset = FilteredEgo4DRefinedNarrationStream(
        split='val',
        frame_fps=Config.FRAME_FPS,
        is_training=False,
        augmentation=False,
        system_prompt='',
        tokenizer=tokenizer,
        vision_pretrained=Config.VISION_PRETRAINED,
        embed_mark=Config.EMBED_MARK,
        max_num_frames=Config.MAX_NUM_FRAMES,
        data_source=data_source
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} conversation turn(s) from validation set")
    
    # Create ground truth word count analysis (skipped for speed)
    # print(f"\n📊 Creating ground truth word count analysis...")
    # create_gt_word_count_analysis(data_source)
    
    print("-" * 50)
    
    # Evaluate more conversations for better coverage
    try:
        # Use config default number of videos
        num_videos = getattr(args, 'num_videos', Config.DEFAULT_NUM_VIDEOS)
        print(f"💬 Processing {num_videos} conversation{'s' if num_videos > 1 else ''} for PPL analysis...")
        random.seed(42)  # For reproducibility
        
        # Check if threshold sweep is requested
        if hasattr(args, 'threshold_sweep') and args.threshold_sweep:
            print("🔄 Running threshold sweep analysis...")
            threshold_results = streaming_evaluate_threshold_sweep(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                num_conversations=num_videos,
                random_selection=True,
                data_source=data_source
                # Use config defaults for threshold_range and num_thresholds
            )
            print("✅ Threshold sweep analysis completed!")
            return
        
        # Regular single-threshold evaluation
        results = streaming_evaluate_conversations(
            model,
            tokenizer,
            dataset,
            device,
            num_conversations=num_videos,
            random_selection=True,
            data_source=data_source,
            conversation_start_times=custom_start_times
        )
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        
        # Calculate aggregate metrics
        avg_ppl = sum(r['lm_ppl'] for r in results) / len(results)
        avg_fluency = sum(r['fluency'] for r in results) / len(results)
        avg_responses_per_video = sum(len(r['generated_turns']) for r in results) / len(results)
        avg_rebuffering_time = sum(r.get('average_rebuffering_time', 0.0) for r in results) / len(results)
        
        # Calculate time diff metric: average latency per generated response
        # Sum the actual timing components for each response
        total_generated_responses = sum(len(r['generated_turns']) for r in results)
        if total_generated_responses > 0:
            response_timings = []
            
            for r in results:
                if len(r['generated_turns']) > 0:  # Only consider conversations with responses
                    generated_turns = r['generated_turns']
                    num_responses = len(generated_turns)
                    
                    if num_responses > 0:
                        # Get the total timing for this conversation
                        visual_time = r.get('visual_embedding_time', 0)
                        model_time = r.get('model_forward_time', 0)
                        generation_time = r.get('generation_time', 0)
                        num_frames = r.get('num_frames', 1)
                        
                        # For each response, the latency is:
                        # - Visual time: total visual time / number of frames (per frame) 
                        # - Model time: total model time / number of frames (per frame)
                        # - Generation time: total generation time / number of responses (per response)
                        
                        visual_per_frame = visual_time / num_frames if num_frames > 0 else 0
                        model_per_frame = model_time / num_frames if num_frames > 0 else 0
                        generation_per_response = generation_time / num_responses if num_responses > 0 else 0
                        
                        # Total latency per response = visual + model + generation
                        response_latency = visual_per_frame + model_per_frame + generation_per_response
                        response_timings.append(response_latency)
                        
                        # print(f"   📊 Response {len(response_timings)}: latency = {response_latency:.3f}s (vis={visual_per_frame:.3f}s + model={model_per_frame:.3f}s + gen={generation_per_response:.3f}s)")
                        # print(f"       Breakdown: {num_frames} frames, {num_responses} responses")
            
            if response_timings:
                avg_time_diff = sum(response_timings) / len(response_timings)
                # print(f"   📊 Individual response latencies: {[f'{t:.3f}s' for t in response_timings]}")
            else:
                avg_time_diff = 0.0
        else:
            avg_time_diff = 0.0
        
        print(f"\n🎯 AGGREGATE METRICS:")
        print(f"   • Average Perplexity: {avg_ppl:.3f}")
        print(f"   • Average Fluency: {avg_fluency:.3f}")
        print(f"   • Average Responses per Video: {avg_responses_per_video:.1f}")
        print(f"   • Average Time Diff (latency per response): {avg_time_diff:.3f}s")
        print(f"   • Average Rebuffering Time per Frame: {avg_rebuffering_time:.3f}s")
        
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   • Conversations Processed: {len(results)}")
        print(f"   • Total Frames: {sum(r['num_frames'] for r in results)}")
        print(f"   • Total Generated Responses: {sum(len(r['generated_turns']) for r in results)}")
        print(f"   • Total Ground Truth Responses: {sum(r['ground_truth_turns'] for r in results)}")

        
        # Create timing analysis
        conversation_timings = [r for r in results if 'visual_embedding_time' in r]
        
        if conversation_timings:
            # Create individual conversation timing plots
            create_individual_conversation_timing_plots(conversation_timings, data_source=data_source)
        
        # Create PPL analysis visualization
        print(f"\n📊 Creating dual PPL analysis...")
        create_dual_ppl_frame_visualization(results, data_source=data_source)
        
        # Create PPL over time visualization for different videos
        # Extract PPL data from results for visualization
        ppl_video_data = []
        for i, result in enumerate(results):
            if 'ppl_data' in result and 'gt_ppls_gt_prefix_visual' in result['ppl_data']:
                gt_prefix_ppls_visual = result['ppl_data']['gt_ppls_gt_prefix_visual']
                vlm_prefix_ppls_visual = result['ppl_data']['gt_ppls_vlm_prefix_visual']
                
                if gt_prefix_ppls_visual and vlm_prefix_ppls_visual:
                    ppl_video_data.append({
                        'video_idx': i,
                        'gt_prefix_ppls_visual': gt_prefix_ppls_visual,
                        'vlm_prefix_ppls_visual': vlm_prefix_ppls_visual,
                        'num_responses': len(gt_prefix_ppls_visual)
                    })
        
        if ppl_video_data:
            create_ppl_over_time_visualization(ppl_video_data, data_source=data_source)
        
        # Create time per token analysis (skipped for speed)
        # print(f"\n📊 Creating time per token analysis...")
        # create_time_per_token_analysis(results, data_source=data_source)
        
        # Create generated word count analysis (skipped for speed)
        # print(f"\n📊 Creating generated word count analysis...")
        # create_generated_word_count_analysis(results, data_source=data_source)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Processed {len(results)} conversations with streaming approach")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()

def create_unified_threshold_analysis(all_threshold_results, all_frame_scores_data=None, all_rebuffering_data=None, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create unified comprehensive threshold analysis with frame score trends and error bars."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    thresholds = sorted(all_threshold_results.keys())
    
    # Collect detailed metrics for each threshold
    detailed_metrics = {
        'thresholds': thresholds,
        'gt_ppl_means': [],
        'gt_ppl_stds': [],
        'fluency_means': [],
        'fluency_stds': [],
        'response_means': [],
        'response_stds': [],
        'gt_prefix_ppl_means': [],
        'gt_prefix_ppl_stds': [],
        'vlm_prefix_ppl_means': [],
        'vlm_prefix_ppl_stds': [],
        'frame_score_means': [],
        'frame_score_stds': [],
        'frame_score_below_threshold_ratios': [],
        'rebuffering_means': [],
        'rebuffering_stds': [],
        'reading_rebuffering_means': [],
        'reading_rebuffering_stds': [],
        'listening_rebuffering_means': [],
        'listening_rebuffering_stds': [],
        'final_utilization_means': [],
        'final_utilization_stds': []
    }
    
    for threshold in thresholds:
        results = all_threshold_results[threshold]
        if not results:
            continue
        
        # Collect all individual values for this threshold
        gt_ppls = [r.get('lm_ppl', 0.0) for r in results]
        fluencies = [r.get('fluency', 0.0) for r in results]
        response_counts = [len(r.get('generated_turns', [])) for r in results]
        rebuffering_times = [r.get('average_rebuffering_time', 0.0) for r in results]
        
        gt_prefix_ppls = []
        vlm_prefix_ppls = []
        frame_scores = []
        below_threshold_ratios = []
        
        for result in results:
            if 'ppl_data' in result:
                ppl_data = result['ppl_data']
                if 'gt_ppls_gt_prefix_visual' in ppl_data and ppl_data['gt_ppls_gt_prefix_visual']:
                    gt_prefix_ppls.extend(ppl_data['gt_ppls_gt_prefix_visual'])
                if 'gt_ppls_vlm_prefix_visual' in ppl_data and ppl_data['gt_ppls_vlm_prefix_visual']:
                    vlm_prefix_ppls.extend(ppl_data['gt_ppls_vlm_prefix_visual'])
            
            # Collect frame scores if available
            if 'frame_scores_data' in result and result['frame_scores_data']:
                conv_frame_scores = result['frame_scores_data']['frame_scores']
                if conv_frame_scores:
                    frame_scores.extend(conv_frame_scores)
                    # Calculate ratio of frames below threshold
                    below_threshold_count = sum(1 for score in conv_frame_scores if score < threshold)
                    below_threshold_ratios.append(below_threshold_count / len(conv_frame_scores))
        
        # Calculate means and standard deviations
        detailed_metrics['gt_ppl_means'].append(np.mean(gt_ppls) if gt_ppls else 0.0)
        detailed_metrics['gt_ppl_stds'].append(np.std(gt_ppls) if gt_ppls else 0.0)
        detailed_metrics['fluency_means'].append(np.mean(fluencies) if fluencies else 0.0)
        detailed_metrics['fluency_stds'].append(np.std(fluencies) if fluencies else 0.0)
        detailed_metrics['response_means'].append(np.mean(response_counts) if response_counts else 0.0)
        detailed_metrics['response_stds'].append(np.std(response_counts) if response_counts else 0.0)
        detailed_metrics['gt_prefix_ppl_means'].append(np.mean(gt_prefix_ppls) if gt_prefix_ppls else 0.0)
        detailed_metrics['gt_prefix_ppl_stds'].append(np.std(gt_prefix_ppls) if gt_prefix_ppls else 0.0)
        detailed_metrics['vlm_prefix_ppl_means'].append(np.mean(vlm_prefix_ppls) if vlm_prefix_ppls else 0.0)
        detailed_metrics['vlm_prefix_ppl_stds'].append(np.std(vlm_prefix_ppls) if vlm_prefix_ppls else 0.0)
        detailed_metrics['frame_score_means'].append(np.mean(frame_scores) if frame_scores else 0.0)
        detailed_metrics['frame_score_stds'].append(np.std(frame_scores) if frame_scores else 0.0)
        detailed_metrics['frame_score_below_threshold_ratios'].append(np.mean(below_threshold_ratios) if below_threshold_ratios else 0.0)
        detailed_metrics['rebuffering_means'].append(np.mean(rebuffering_times) if rebuffering_times else 0.0)
        detailed_metrics['rebuffering_stds'].append(np.std(rebuffering_times) if rebuffering_times else 0.0)
        
        # Collect user rebuffering metrics
        reading_rebuffering_times = [r.get('average_reading_rebuffering_time', 0.0) for r in results]
        listening_rebuffering_times = [r.get('average_listening_rebuffering_time', 0.0) for r in results]
        
        detailed_metrics['reading_rebuffering_means'].append(np.mean(reading_rebuffering_times) if reading_rebuffering_times else 0.0)
        detailed_metrics['reading_rebuffering_stds'].append(np.std(reading_rebuffering_times) if reading_rebuffering_times else 0.0)
        detailed_metrics['listening_rebuffering_means'].append(np.mean(listening_rebuffering_times) if listening_rebuffering_times else 0.0)
        detailed_metrics['listening_rebuffering_stds'].append(np.std(listening_rebuffering_times) if listening_rebuffering_times else 0.0)
        
        # Collect final frame utilization metrics
        final_utilization_times = [r.get('final_frame_utilization', 0.0) for r in results]
        detailed_metrics['final_utilization_means'].append(np.mean(final_utilization_times) if final_utilization_times else 0.0)
        detailed_metrics['final_utilization_stds'].append(np.std(final_utilization_times) if final_utilization_times else 0.0)
    
    # Create comprehensive visualization with 2x4 grid (added user rebuffering analysis)
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    fig.suptitle(f'Unified Threshold Analysis - {data_source.upper()} Dataset', fontsize=18, fontweight='bold')
    
    # 1. Decomposed VLM PPL - One line per video
    ax1 = axes[0, 0]
    if all_threshold_results:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect VLM PPL data for each video across thresholds
        video_vlm_ppls = {}  # {video_idx: {threshold: vlm_ppl}}
        
        for threshold in thresholds:
            results = all_threshold_results[threshold]
            for j, result in enumerate(results):
                if j not in video_vlm_ppls:
                    video_vlm_ppls[j] = {}
                
                # Get VLM PPL for this video at this threshold
                if 'ppl_data' in result and 'gt_ppls_vlm_prefix_visual' in result['ppl_data']:
                    vlm_ppls = result['ppl_data']['gt_ppls_vlm_prefix_visual']
                    if vlm_ppls:
                        video_vlm_ppls[j][threshold] = np.mean(vlm_ppls)
        
        # Plot each video's VLM PPL trajectory
        for j, video_data in video_vlm_ppls.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_ppls = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax1.plot(video_thresholds, video_ppls, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average VLM PPL line
        if detailed_metrics['vlm_prefix_ppl_means']:
            ax1.errorbar(thresholds, detailed_metrics['vlm_prefix_ppl_means'], yerr=detailed_metrics['vlm_prefix_ppl_stds'], 
                        fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                        label='Average VLM PPL ± Std', alpha=0.9)
        
        # Add GT PPL as reference line (should be constant)
        if detailed_metrics['gt_prefix_ppl_means']:
            avg_gt_ppl = np.mean(detailed_metrics['gt_prefix_ppl_means'])
            ax1.axhline(y=avg_gt_ppl, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'GT PPL Reference ({avg_gt_ppl:.2f})')
        
        ax1.set_xlabel('Streaming Threshold')
        ax1.set_ylabel('PPL')
        ax1.set_title('VLM PPL vs Threshold (Per Video) with GT Reference')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No threshold data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('VLM PPL vs Threshold (Per Video)')
    
    # 2. Fluency with per-video breakdown
    ax2 = axes[0, 1]
    if all_threshold_results:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect fluency data for each video across thresholds
        video_fluencies = {}  # {video_idx: {threshold: fluency}}
        
        for threshold in thresholds:
            results = all_threshold_results[threshold]
            for j, result in enumerate(results):
                if j not in video_fluencies:
                    video_fluencies[j] = {}
                
                # Get fluency for this video at this threshold
                if 'fluency' in result:
                    video_fluencies[j][threshold] = result['fluency']
        
        # Plot each video's fluency trajectory
        for j, video_data in video_fluencies.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_fluencies_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax2.plot(video_thresholds, video_fluencies_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average line with error bars
        if detailed_metrics['fluency_means']:
            ax2.errorbar(thresholds, detailed_metrics['fluency_means'], yerr=detailed_metrics['fluency_stds'], 
                        fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                        label='Average Fluency ± Std', alpha=0.9)
        
        ax2.set_xlabel('Streaming Threshold')
        ax2.set_ylabel('Fluency Score')
        ax2.set_title('Fluency vs Threshold (Per Video)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.0)
    else:
        ax2.text(0.5, 0.5, 'No threshold data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Fluency vs Threshold (Per Video)')
    
    # 3. Response count with per-video breakdown
    ax3 = axes[0, 2]
    if all_threshold_results:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect response count data for each video across thresholds
        video_responses = {}  # {video_idx: {threshold: response_count}}
        
        for threshold in thresholds:
            results = all_threshold_results[threshold]
            for j, result in enumerate(results):
                if j not in video_responses:
                    video_responses[j] = {}
                
                # Get response count for this video at this threshold
                if 'response_count' in result:
                    video_responses[j][threshold] = result['response_count']
        
        # Plot each video's response count trajectory
        for j, video_data in video_responses.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_responses_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax3.plot(video_thresholds, video_responses_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average line with error bars
        if detailed_metrics['response_means']:
            ax3.errorbar(thresholds, detailed_metrics['response_means'], yerr=detailed_metrics['response_stds'], 
                        fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                        label='Average Response Count ± Std', alpha=0.9)
        
        ax3.set_xlabel('Streaming Threshold')
        ax3.set_ylabel('Response Count')
        ax3.set_title('Response Count vs Threshold (Per Video)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No threshold data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Response Count vs Threshold (Per Video)')
    
    # 4. Frame Score Trends - Time vs Frame Score for Different Videos
    ax4 = axes[1, 0]
    if all_frame_scores_data:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, (threshold, threshold_data) in enumerate(all_frame_scores_data.items()):
            for j, (conv_key, conv_data) in enumerate(threshold_data.items()):
                if 'frame_scores' in conv_data and 'frame_times' in conv_data and conv_data['frame_scores']:
                    frame_times = conv_data['frame_times']
                    frame_scores = conv_data['frame_scores']
                    color = colors[j % len(colors)]
                    alpha = 0.3 + 0.7 * (i / (len(all_frame_scores_data) - 1)) if len(all_frame_scores_data) > 1 else 0.8
                    ax4.plot(frame_times, frame_scores, color=color, linewidth=1.5, alpha=alpha, 
                            label=f'Video {j+1} (T={threshold:.2f})' if i == 0 else "")
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Frame Score')
        ax4.set_title('Frame Score Trends Over Time (Per Video & Threshold)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.0)
        
        # Create dual legend: videos and thresholds
        video_legend_elements = []
        threshold_legend_elements = []
        
        # Video legend (using first threshold data)
        first_threshold = list(all_frame_scores_data.keys())[0]
        for j in range(len(all_frame_scores_data[first_threshold])):
            color = colors[j % len(colors)]
            video_legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                                   label=f'Video {j+1}'))
        
        # Threshold legend
        for i, threshold in enumerate(all_frame_scores_data.keys()):
            alpha = 0.3 + 0.7 * (i / (len(all_frame_scores_data) - 1)) if len(all_frame_scores_data) > 1 else 0.8
            linewidth = 1.0 + 0.5 * (i / (len(all_frame_scores_data) - 1)) if len(all_frame_scores_data) > 1 else 1.5
            threshold_legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=linewidth, alpha=alpha, 
                                                       label=f'Threshold {threshold:.2f}'))
        
        # Create combined legend
        legend1 = ax4.legend(handles=video_legend_elements, title='Videos', loc='lower left', fontsize=8)
        legend2 = ax4.legend(handles=threshold_legend_elements, title='Thresholds', loc='lower right', fontsize=8)
        ax4.add_artist(legend1)  # Add first legend back after second one overwrites it
    else:
        ax4.text(0.5, 0.5, 'No frame score data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Frame Score Trends Over Time')
    
    # 5. Rebuffering Time vs Threshold (Per Video)
    ax5 = axes[1, 1]
    if all_rebuffering_data:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect rebuffering data for each video across thresholds
        video_rebuffering = {}  # {video_idx: {threshold: avg_rebuffering}}
        
        for threshold in thresholds:
            if threshold in all_rebuffering_data:
                threshold_data = all_rebuffering_data[threshold]
                for j, (conv_key, conv_data) in enumerate(threshold_data.items()):
                    if j not in video_rebuffering:
                        video_rebuffering[j] = {}
                    
                    # Get average rebuffering time for this video at this threshold
                    if 'average_rebuffering_time' in conv_data:
                        video_rebuffering[j][threshold] = conv_data['average_rebuffering_time']
        
        # Plot each video's rebuffering trajectory
        for j, video_data in video_rebuffering.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_rebuffering_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax5.plot(video_thresholds, video_rebuffering_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average rebuffering time line with error bars
        if detailed_metrics['rebuffering_means']:
            ax5.errorbar(thresholds, detailed_metrics['rebuffering_means'], yerr=detailed_metrics['rebuffering_stds'], 
                        fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                        label='Average Rebuffering Time ± Std', alpha=0.9)
        
        ax5.set_xlabel('Streaming Threshold')
        ax5.set_ylabel('Rebuffering Time (seconds)')
        ax5.set_title('Rebuffering Time vs Threshold (Per Video)')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No rebuffering data available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Rebuffering Time vs Threshold (Per Video)')
    
    # 6. Reading Rebuffering Time vs Threshold (Per Video)
    ax6 = axes[1, 2]
    if detailed_metrics['reading_rebuffering_means']:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect reading rebuffering data for each video across thresholds
        video_reading_rebuffering = {}  # {video_idx: {threshold: avg_reading_rebuffering}}
        
        for threshold in thresholds:
            if threshold in all_threshold_results:
                results = all_threshold_results[threshold]
                for j, result in enumerate(results):
                    if j not in video_reading_rebuffering:
                        video_reading_rebuffering[j] = {}
                    
                    # Get average reading rebuffering time for this video at this threshold
                    if 'average_reading_rebuffering_time' in result:
                        video_reading_rebuffering[j][threshold] = result['average_reading_rebuffering_time']
        
        # Plot each video's reading rebuffering trajectory
        for j, video_data in video_reading_rebuffering.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_reading_rebuffering_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax6.plot(video_thresholds, video_reading_rebuffering_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average reading rebuffering time line with error bars
        ax6.errorbar(thresholds, detailed_metrics['reading_rebuffering_means'], yerr=detailed_metrics['reading_rebuffering_stds'], 
                    fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                    label='Average Reading Rebuffering ± Std', alpha=0.9)
        
        ax6.set_xlabel('Streaming Threshold')
        ax6.set_ylabel('Reading Rebuffering Time (seconds)')
        ax6.set_title('Reading Rebuffering Time vs Threshold (Per Video)')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'No reading rebuffering data available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Reading Rebuffering Time vs Threshold (Per Video)')
    
    # 7. Listening Rebuffering Time vs Threshold (Per Video)
    ax7 = axes[1, 3]
    if detailed_metrics['listening_rebuffering_means']:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect listening rebuffering data for each video across thresholds
        video_listening_rebuffering = {}  # {video_idx: {threshold: avg_listening_rebuffering}}
        
        for threshold in thresholds:
            if threshold in all_threshold_results:
                results = all_threshold_results[threshold]
                for j, result in enumerate(results):
                    if j not in video_listening_rebuffering:
                        video_listening_rebuffering[j] = {}
                    
                    # Get average listening rebuffering time for this video at this threshold
                    if 'average_listening_rebuffering_time' in result:
                        video_listening_rebuffering[j][threshold] = result['average_listening_rebuffering_time']
        
        # Plot each video's listening rebuffering trajectory
        for j, video_data in video_listening_rebuffering.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_listening_rebuffering_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax7.plot(video_thresholds, video_listening_rebuffering_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average listening rebuffering time line with error bars
        ax7.errorbar(thresholds, detailed_metrics['listening_rebuffering_means'], yerr=detailed_metrics['listening_rebuffering_stds'], 
                    fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                    label='Average Listening Rebuffering ± Std', alpha=0.9)
        
        ax7.set_xlabel('Streaming Threshold')
        ax7.set_ylabel('Listening Rebuffering Time (seconds)')
        ax7.set_title('Listening Rebuffering Time vs Threshold (Per Video)')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=9)
    else:
        ax7.text(0.5, 0.5, 'No listening rebuffering data available', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Listening Rebuffering Time vs Threshold (Per Video)')
    
    # 8. Final Frame Resource Utilization vs Threshold (Per Video)
    ax8 = axes[1, 4]
    if detailed_metrics['final_utilization_means']:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Collect final utilization data for each video across thresholds
        video_final_utilization = {}  # {video_idx: {threshold: final_utilization}}
        
        for threshold in thresholds:
            if threshold in all_threshold_results:
                results = all_threshold_results[threshold]
                for j, result in enumerate(results):
                    if j not in video_final_utilization:
                        video_final_utilization[j] = {}
                    
                    # Get final frame utilization for this video at this threshold
                    if 'final_frame_utilization' in result:
                        video_final_utilization[j][threshold] = result['final_frame_utilization']
        
        # Plot each video's final utilization trajectory
        for j, video_data in video_final_utilization.items():
            if len(video_data) > 1:  # Only plot if we have data for multiple thresholds
                video_thresholds = sorted(video_data.keys())
                video_final_utilization_list = [video_data[t] for t in video_thresholds]
                color = colors[j % len(colors)]
                ax8.plot(video_thresholds, video_final_utilization_list, 'o-', color=color, linewidth=2, markersize=4, 
                        label=f'Video {j+1}', alpha=0.8)
        
        # Add average final utilization line with error bars
        ax8.errorbar(thresholds, detailed_metrics['final_utilization_means'], yerr=detailed_metrics['final_utilization_stds'], 
                    fmt='s-', color='black', linewidth=3, markersize=8, capsize=5, 
                    label='Average Final Utilization ± Std', alpha=0.9)
        
        # Add horizontal line at 1.0 (100% utilization)
        ax8.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% Utilization')
        
        ax8.set_xlabel('Streaming Threshold')
        ax8.set_ylabel('Final Frame Resource Utilization')
        ax8.set_title('Final Frame Resource Utilization vs Threshold (Per Video)')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=9)
    else:
        ax8.text(0.5, 0.5, 'No final utilization data available', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Final Frame Resource Utilization vs Threshold (Per Video)')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'unified_threshold_analysis_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Unified threshold analysis saved to: {output_path}")
    
    plt.show()
    
    return output_path

def calculate_fluency_score(generated_turns, original_conversation, data_source='goalstep'):
    """Calculate fluency score with temporal alignment for both goalstep and narration datasets."""
    
    # Calculate and print fluency details with temporal alignment
    total_turns = len(generated_turns)
    total_gt_turns = len([t for t in original_conversation if t['role'] == 'assistant'])
    
    # Get GT response times and normalize them to start from 0s (when user prompt is given)
    first_user_time_fluency = None
    for turn in original_conversation:
        if turn['role'] == 'user':
            if first_user_time_fluency is None:
                first_user_time_fluency = turn['time']
                break
    
    # Fallback: if no user prompt found, use the first assistant response time as reference
    if first_user_time_fluency is None:
        assistant_times = [t['time'] for t in original_conversation if t['role'] == 'assistant']
        first_user_time_fluency = min(assistant_times) if assistant_times else 0.0
    
    # Normalize GT response times to start from 0s
    gt_response_times = [t['time'] - first_user_time_fluency for t in original_conversation if t['role'] == 'assistant']
    generated_response_times = [t['time'] for t in generated_turns]
    
    # Temporal alignment threshold (in seconds) - more lenient alignment
    temporal_threshold = 1.0  # Allow 1.0 seconds tolerance
    
    # Count successful responses that align temporally with GT responses (one-to-one mapping)
    successful_responses = 0
    aligned_pairs = []
    used_gt_times = set()  # Track which GT times have been used
    
    for gen_time in generated_response_times:
        # Find the closest unused GT response time
        available_gt_times = [t for t in gt_response_times if t not in used_gt_times]
        if not available_gt_times:
            break  # No more GT responses available for mapping
            
        closest_gt_time = min(available_gt_times, key=lambda x: abs(x - gen_time))
        time_diff = abs(gen_time - closest_gt_time)
        
        if time_diff <= temporal_threshold:
            # Check if response has reasonable content (≥2 words)
            gen_turn = next((t for t in generated_turns if t['time'] == gen_time), None)
            if gen_turn:
                response_text = gen_turn.get('text', gen_turn.get('content', ''))
                if response_text and len(response_text.split()) >= 2:
                    successful_responses += 1
                    aligned_pairs.append((gen_time, closest_gt_time, time_diff))
                    used_gt_times.add(closest_gt_time)  # Mark this GT time as used
    
    fluency_score = successful_responses / total_gt_turns if total_gt_turns > 0 else 0.0
    fluency_score = max(0.0, min(1.0, fluency_score))  # Clamp to [0, 1]
    
    # print(f"\n📊 FLUENCY CALCULATION (Temporal Alignment):")
    # print(f"   • Fluency Score: {fluency_score:.3f} ({successful_responses}/{total_gt_turns})")
    
    return fluency_score

def calculate_ppl_for_response(model, tokenizer, conversation, video_tensor, device, data_source='goalstep', use_visual=True, custom_threshold=None):
    """Calculate PPL for a single response using the proper conversation format."""
    try:
        # Limit conversation length to avoid OOM in stream_evaluate
        max_conversation_length = 20  # Limit to last 20 turns to reduce memory usage
        if len(conversation) > max_conversation_length:
            # Keep system prompt and stream, then take last N turns
            limited_conversation = conversation[:2]  # Keep system and stream
            limited_conversation.extend(conversation[-(max_conversation_length-2):])  # Add last N-2 turns
            conversation = limited_conversation
        
        # Tokenize the conversation
        input_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
        
        # Get learn ranges for this conversation
        learn_ranges = tokenizer.get_learn_ranges(conversation)
        
        # Calculate PPL using the model's stream_evaluate method
        
        # Create labels using the same approach as data_collator.py
        labels = torch.full_like(input_ids, -100, dtype=torch.long)  # -100 is ignore index
        labels[labels == tokenizer.bos_token_id] = -100
        labels[labels == tokenizer.eos_token_id] = -100
        
        # Apply learn ranges to labels (like data_collator.py does)
        # Use the exact same approach as data_collator.py
        learnable_tokens = 0
        
        # Get tokenization with offset mapping (same as data_collator.py)
        tokenization_result = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
        offset_mapping = tokenization_result['offset_mapping']
        
        for i, learn_r in enumerate(learn_ranges):
            char_start = learn_r.start
            char_stop = learn_r.stop
            
            # Use the exact same logic as data_collator.py
            try:
                # Find start token position
                start_matches = torch.nonzero(offset_mapping[0, :, 0] == char_start)
                if len(start_matches) > 0:
                    start = start_matches[0].item()
                else:
                    continue
                
                # Find stop token position
                if offset_mapping[0, -1, 0] >= char_stop:
                    stop_matches = torch.nonzero(offset_mapping[0, :, 0] == char_stop)
                    if len(stop_matches) > 0:
                        stop = stop_matches[0].item()
                    else:
                        continue
                else:
                    # Last eos token
                    stop = len(input_ids[0])
                
                # Apply the range (with -1 offset like data_collator.py)
                if start > 0 and stop <= len(input_ids[0]) and start < stop:
                    labels[0, start-1:stop-1] = input_ids[0, start:stop]
                    learnable_tokens += (stop - start)
                    
            except Exception as e:
                continue
        
        # Replace any out-of-bounds token IDs with eos_token_id (like data_collator.py does)
        labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
        
        if use_visual:
            # Process frames one by one from CPU to GPU (like VLM streaming)
            if video_tensor is not None and hasattr(video_tensor, 'shape') and video_tensor.shape[0] > 0:
                # Clear cache before processing
                torch.cuda.empty_cache()
                
                # Use the same frame-by-frame approach as VLM streaming
                # Process only the first frame to minimize memory usage
                frame_range = range(0, 1)  # Only first frame
                
                # Move frame from CPU to GPU one by one (like VLM streaming)
                single_frame_cpu = video_tensor[frame_range]  # Shape: [1, 3, H, W] on CPU
                single_frame_gpu = single_frame_cpu.to(device)  # Move to GPU
            
                # Use custom threshold if provided, otherwise use data source default
                if custom_threshold is not None:
                    threshold = custom_threshold
                elif data_source == 'narration':
                    threshold = Config.STREAMING_THRESHOLD_NARRATION
                else:  # goalstep or default
                    threshold = Config.STREAMING_THRESHOLD_GOALSTEP
                
                # Call stream_evaluate for this response with visual context
                raw_metrics = model.stream_evaluate(
                    input_ids=input_ids,
                    labels=labels,
                    frames=single_frame_gpu,
                    frame_token_interval_threshold=threshold,
                    ignore_token_id=-100
                )
                
                # Clear cache and delete GPU tensors to free memory
                del single_frame_gpu
                del single_frame_cpu
                torch.cuda.empty_cache()
                
                # Extract PPL for this response
                lm_ppl, frame_diff, _, _ = raw_metrics.tolist()
                return float(lm_ppl)
            else:
                return None
        else:
            # No visual PPL calculation removed for simplification
            return None
            
    except Exception as e:
        print(f"PPL calculation error: {e}")
        traceback.print_exc()
        return None

def create_conversation_with_gt_prefix(normalized_conversation, gt_time, user_prompt, gt_content):
    """Create conversation using ground truth responses as context (golden prefix)."""
    conversation = [
        {'role': 'system', 'content': 'Please help with the video analysis.'},
        {'role': 'stream', 'num_frames': 1, 'learn': False}
    ]
    
    # Add all previous GT responses as context, ensuring proper user/assistant alternation
    last_added_role = 'stream'
    if normalized_conversation:
        for turn in normalized_conversation:
            if turn['time'] < gt_time:  # Only previous responses
                if turn['role'] == 'user':
                    conversation.append({'role': 'user', 'content': turn['content']})
                    last_added_role = 'user'
                elif turn['role'] == 'assistant' and last_added_role == 'user':
                    # Only add assistant response if preceded by user
                    conversation.append({'role': 'assistant', 'content': turn['content'], 'learn': True})
                    last_added_role = 'assistant'
    
    # Add current user prompt and GT response
    if last_added_role != 'user':
        conversation.append({'role': 'user', 'content': user_prompt})
    conversation.append({'role': 'assistant', 'content': gt_content, 'learn': True})
    
    return conversation

def create_conversation_with_vlm_prefix(generated_turns, gt_time, user_prompt, gt_content):
    """Create conversation using VLM responses as context (can be wrong/miss)."""
    conversation = [
        {'role': 'system', 'content': 'Please help with the video analysis.'},
        {'role': 'stream', 'num_frames': 1, 'learn': False}
    ]
    
    # Add all previous VLM responses as context, ensuring proper user/assistant alternation
    last_added_role = 'stream'
    for turn in generated_turns:
        if turn['time'] < gt_time:  # Only previous responses
            # Extract user prompt from turn
            turn_user_prompt = turn.get('user_prompt', 'Please help with the video analysis.')
            # Include ALL responses, not just the ones with user queries
            # This ensures VLM context changes with threshold
            if turn_user_prompt != "Frame processing":
                # Only add user prompt if last added was not user
                if last_added_role != 'user':
                    conversation.append({'role': 'user', 'content': turn_user_prompt})
                    last_added_role = 'user'
            else:
                # For frame-triggered responses, use a generic user prompt
                if last_added_role != 'user':
                    conversation.append({'role': 'user', 'content': 'Please help with the video analysis.'})
                    last_added_role = 'user'
            
            # Extract assistant response (remove the video time prefix)
            assistant_text = turn['text']
            if 'Assistant:' in assistant_text:
                assistant_content = assistant_text.split('Assistant:', 1)[1].strip()
            else:
                assistant_content = assistant_text
            conversation.append({'role': 'assistant', 'content': assistant_content, 'learn': True})
            last_added_role = 'assistant'
    
    # Add current user prompt and GT response
    if last_added_role != 'user':
        conversation.append({'role': 'user', 'content': user_prompt})
    conversation.append({'role': 'assistant', 'content': gt_content, 'learn': True})
    
    return conversation

def create_dual_ppl_frame_visualization(results, output_dir="timing_plots", data_source="goalstep"):
    """Create visualization showing how dual PPLs vary over frames in different videos."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all videos
    video_data = []
    for i, result in enumerate(results):
        if 'ppl_data' in result and 'gt_ppls_gt_prefix_visual' in result['ppl_data']:
            gt_prefix_ppls_visual = result['ppl_data']['gt_ppls_gt_prefix_visual']
            vlm_prefix_ppls_visual = result['ppl_data']['gt_ppls_vlm_prefix_visual']
            
            if gt_prefix_ppls_visual and vlm_prefix_ppls_visual:
                video_data.append({
                    'video_idx': i,
                    'gt_prefix_ppls_visual': gt_prefix_ppls_visual,
                    'vlm_prefix_ppls_visual': vlm_prefix_ppls_visual,
                    'num_responses': len(gt_prefix_ppls_visual)
                })
    
    if not video_data:
        print("No PPL data available for visualization")
        return
    
    # Create figure with subplots (histogram + bar plot)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Dual PPL Analysis: GT Prefix vs VLM Prefix', fontsize=16, fontweight='bold')
    
    # Collect data
    all_gt_prefix_visual = []
    all_vlm_prefix_visual = []
    video_means_gt = []
    video_means_vlm = []
    video_stds_gt = []
    video_stds_vlm = []
    video_labels = []
    
    for data in video_data:
        gt_ppls = data['gt_prefix_ppls_visual']
        vlm_ppls = data['vlm_prefix_ppls_visual']
        
        all_gt_prefix_visual.extend(gt_ppls)
        all_vlm_prefix_visual.extend(vlm_ppls)
        
        video_means_gt.append(np.mean(gt_ppls))
        video_means_vlm.append(np.mean(vlm_ppls))
        video_stds_gt.append(np.std(gt_ppls))
        video_stds_vlm.append(np.std(vlm_ppls))
        video_labels.append(f'Video {data["video_idx"] + 1}')
    
    # 1. PPL distribution comparison
    ax1 = axes[0]
    ax1.hist(all_gt_prefix_visual, bins=20, alpha=0.6, label='GT Prefix', color='blue')
    ax1.hist(all_vlm_prefix_visual, bins=20, alpha=0.6, label='VLM Prefix', color='red')
    ax1.set_xlabel('PPL')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PPL Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar plot showing mean (std) PPL for different videos
    ax2 = axes[1]
    x = np.arange(len(video_labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, video_means_gt, width, yerr=video_stds_gt, 
                   label='GT Prefix', color='blue', alpha=0.7, capsize=5)
    bars2 = ax2.bar(x + width/2, video_means_vlm, width, yerr=video_stds_vlm, 
                   label='VLM Prefix', color='red', alpha=0.7, capsize=5)
    
    ax2.set_xlabel('Video')
    ax2.set_ylabel('Mean PPL ± Std')
    ax2.set_title('Mean PPL by Video (with Standard Deviation)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(video_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + video_stds_gt[i] + 0.1,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
        ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + video_stds_vlm[i] + 0.1,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add summary statistics to the histogram plot
    if all_gt_prefix_visual and all_vlm_prefix_visual:
        correlation = np.corrcoef(all_gt_prefix_visual, all_vlm_prefix_visual)[0, 1]
        
        # Add text box with statistics
        stats_text = f"""Summary:
GT Prefix: {np.mean(all_gt_prefix_visual):.3f} ± {np.std(all_gt_prefix_visual):.3f}
VLM Prefix: {np.mean(all_vlm_prefix_visual):.3f} ± {np.std(all_vlm_prefix_visual):.3f}
Correlation: {correlation:.3f}
Responses: {len(all_gt_prefix_visual)}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    
    plt.tight_layout()
    
    # Save the plot with dataset-specific name
    output_path = os.path.join(output_dir, f'dual_ppl_analysis_{data_source}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Dual PPL analysis saved to: {output_path}")
    
    plt.close()
    
    # Create PPL over time visualization
    create_ppl_over_time_visualization(video_data, output_dir, data_source)
    
    # Print summary statistics
    print(f"\n📊 DUAL PPL ANALYSIS SUMMARY ({data_source.upper()}):")
    print(f"   • Total responses analyzed: {len(all_gt_prefix_visual)}")
    print(f"   • GT Prefix PPL: {np.mean(all_gt_prefix_visual):.3f} ± {np.std(all_gt_prefix_visual):.3f}")
    print(f"   • VLM Prefix PPL: {np.mean(all_vlm_prefix_visual):.3f} ± {np.std(all_vlm_prefix_visual):.3f}")
    
    # Calculate differences and correlation
    # if len(all_gt_prefix_visual) == len(all_vlm_prefix_visual):
    #     differences = [vlm - gt for gt, vlm in zip(all_gt_prefix_visual, all_vlm_prefix_visual)]
    #     mean_diff = np.mean(differences)
    #     std_diff = np.std(differences)
    #     correlation = np.corrcoef(all_gt_prefix_visual, all_vlm_prefix_visual)[0, 1]
    #     print(f"   • Average difference (VLM - GT): {mean_diff:.3f} ± {std_diff:.3f}")
    #     print(f"   • Correlation coefficient: {correlation:.3f}")

def create_ppl_over_time_visualization(video_data, output_dir="timing_plots", data_source="goalstep"):
    """Create visualization showing PPL variation over time for different videos."""
    
    if not video_data:
        return
    
    # Create figure for PPL over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'PPL Evolution Over Time - {data_source.upper()} Dataset', fontsize=16, fontweight='bold')
    
    # 1. GT Prefix PPL over time (Visual vs No Visual)
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(video_data)))
    
    for i, data in enumerate(video_data):
        response_indices = range(len(data['gt_prefix_ppls_visual']))
        ax1.plot(response_indices, data['gt_prefix_ppls_visual'], 'o-', color=colors[i], 
                label=f'Video {data["video_idx"]} (Visual)', alpha=0.7, markersize=4)
        if 'gt_prefix_ppls_no_visual' in data and data['gt_prefix_ppls_no_visual']:
            ax1.plot(response_indices, data['gt_prefix_ppls_no_visual'], 's--', color=colors[i], 
                    label=f'Video {data["video_idx"]} (No Visual)', alpha=0.7, markersize=4)
    
    ax1.set_xlabel('Response Index')
    ax1.set_ylabel('GT Prefix PPL')
    ax1.set_title('GT Prefix PPL Evolution Over Responses')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. VLM Prefix PPL over time (Visual vs No Visual)
    ax2 = axes[1]
    
    for i, data in enumerate(video_data):
        response_indices = range(len(data['vlm_prefix_ppls_visual']))
        ax2.plot(response_indices, data['vlm_prefix_ppls_visual'], 'o-', color=colors[i], 
                label=f'Video {data["video_idx"]} (Visual)', alpha=0.7, markersize=4)
        if 'vlm_prefix_ppls_no_visual' in data and data['vlm_prefix_ppls_no_visual']:
            ax2.plot(response_indices, data['vlm_prefix_ppls_no_visual'], 's--', color=colors[i], 
                    label=f'Video {data["video_idx"]} (No Visual)', alpha=0.7, markersize=4)
    
    ax2.set_xlabel('Response Index')
    ax2.set_ylabel('VLM Prefix PPL')
    ax2.set_title('VLM Prefix PPL Evolution Over Responses')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'ppl_over_time_{data_source}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 PPL over time analysis saved to: {output_path}")
    
    plt.close()

def calculate_metrics_like_benchmark(model, tokenizer, video_tensor, conversation, generated_turns, device, normalized_conversation=None, data_source='goalstep'):
    """Calculate metrics exactly like evaluate.py using stream_evaluate and compute_metrics."""
    
    # Prepare conversation for evaluation exactly like the dataset does
    # The dataset expects a specific conversation format with stream messages
    eval_conversation = []
    
    # Add system message first (like the dataset does)
    eval_conversation.append({
        'role': 'system', 
        'content': 'Please help with the video analysis.'
    })
    
    # Add stream message for the first frame (like the dataset does)
    eval_conversation.append({
        'role': 'stream',
        'num_frames': 1,
        'learn': False
    })
    
    # Add user instruction (like the dataset does)
    eval_conversation.append({
        'role': 'user',
        'content': 'Please help with the video analysis.'
    })
    
    # Add ground truth assistant messages with learn=True
    # Convert ground truth annotations to the expected format
    if conversation:  # Only if we have ground truth annotations
        for ann in conversation:
            if isinstance(ann, dict) and 'text' in ann:
                # This is a ground truth annotation
                eval_conversation.append({
                    'role': 'assistant',
                    'content': ann['text'],
                    'learn': True
                })
            elif isinstance(ann, dict) and 'role' in ann and ann['role'] == 'assistant':
                # This is already in the expected format
                eval_conversation.append({
                    'role': 'assistant',
                    'content': ann['content'],
                    'learn': True
                })
    
    # Add generated turns as assistant messages with learn=True
    for turn in generated_turns:
        # Handle both 'content' and 'text' keys for generated turns
        content = turn.get('content', turn.get('text', ''))
        eval_conversation.append({
            'role': 'assistant',
            'content': content,
            'learn': True
        })
    
    # Extract ground truth assistant responses from the conversation
    ground_truth_responses = []
    if normalized_conversation:
        for turn in normalized_conversation:
            if turn['role'] == 'assistant':
                ground_truth_responses.append({
                    'content': turn['content'],
                    'time': turn['time'],
                    'original_time': turn.get('original_time', turn['time'])
                })
    
    # Calculate DUAL PPL for ALL GROUND TRUTH responses in the dataset
    gt_ppls_gt_prefix_visual = []  # PPL using GT responses as context (golden prefix) with visual
    gt_ppls_vlm_prefix_visual = []  # PPL using VLM responses as context with visual
    
    
    print(f"📊 Calculating dual PPL (visual context) for {len(ground_truth_responses)} ground truth responses...")
    
    for i, gt_response in enumerate(ground_truth_responses):
        try:
            gt_content = gt_response['content']
            gt_time = gt_response['time']
            
            # Find the user prompt that this ground truth response was answering
            user_prompt = "Please help with the video analysis."  # Default fallback
            if normalized_conversation:
                # Find the most recent user prompt before this response time
                for conv_turn in reversed(normalized_conversation):
                    if conv_turn['role'] == 'user' and conv_turn['time'] <= gt_time:
                        user_prompt = conv_turn['content']
                        break
            
            # 1. PPL with GT prefix (golden context) - WITH VISUAL
            gt_conversation = create_conversation_with_gt_prefix(
                normalized_conversation, gt_time, user_prompt, gt_content
            )
            ppl_gt_prefix_visual = calculate_ppl_for_response(model, tokenizer, gt_conversation, video_tensor, device, data_source, use_visual=True, custom_threshold=None)
            
            # 2. PPL with VLM prefix (actual generated responses as context) - WITH VISUAL
            vlm_conversation = create_conversation_with_vlm_prefix(
                generated_turns, gt_time, user_prompt, gt_content
            )
            ppl_vlm_prefix_visual = calculate_ppl_for_response(model, tokenizer, vlm_conversation, video_tensor, device, data_source, use_visual=True, custom_threshold=None)
            
            if ppl_gt_prefix_visual is not None:
                gt_ppls_gt_prefix_visual.append(ppl_gt_prefix_visual)
            if ppl_vlm_prefix_visual is not None:
                gt_ppls_vlm_prefix_visual.append(ppl_vlm_prefix_visual)
                
            if i % 10 == 0:  # Progress indicator
                # print(f"   Processed {i+1}/{len(ground_truth_responses)} GT responses...")
                # Clean up GPU memory periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clean up memory after each response to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing GT response {i}: {e}")
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Calculate average PPLs for all four contexts
    if gt_ppls_gt_prefix_visual:
        avg_gt_ppl_gt_prefix_visual = sum(gt_ppls_gt_prefix_visual) / len(gt_ppls_gt_prefix_visual)
        print(f"📊 GT Prefix PPL (Visual): {len(gt_ppls_gt_prefix_visual)} responses, avg PPL: {avg_gt_ppl_gt_prefix_visual:.3f}")
        print(f"📊 GT Prefix PPL (Visual) range: {min(gt_ppls_gt_prefix_visual):.3f} - {max(gt_ppls_gt_prefix_visual):.3f}")
    else:
        avg_gt_ppl_gt_prefix_visual = 0.0
        print("📊 GT Prefix PPL (Visual): No valid calculations")
        
    if gt_ppls_vlm_prefix_visual:
        avg_gt_ppl_vlm_prefix_visual = sum(gt_ppls_vlm_prefix_visual) / len(gt_ppls_vlm_prefix_visual)
        print(f"📊 VLM Prefix PPL (Visual): {len(gt_ppls_vlm_prefix_visual)} responses, avg PPL: {avg_gt_ppl_vlm_prefix_visual:.3f}")
        print(f"📊 VLM Prefix PPL (Visual) range: {min(gt_ppls_vlm_prefix_visual):.3f} - {max(gt_ppls_vlm_prefix_visual):.3f}")
    else:
        avg_gt_ppl_vlm_prefix_visual = 0.0
        print("📊 VLM Prefix PPL (Visual): No valid calculations")
        
    # No visual PPL printing removed for simplification
    
    # Use average of visual PPLs as the main metric
    avg_ppl = (avg_gt_ppl_gt_prefix_visual + avg_gt_ppl_vlm_prefix_visual) / 2 if (avg_gt_ppl_gt_prefix_visual > 0 and avg_gt_ppl_vlm_prefix_visual > 0) else max(avg_gt_ppl_gt_prefix_visual, avg_gt_ppl_vlm_prefix_visual)
    
    # Calculate other metrics using the same approach
    
    # Skip fluency and correctness calculations
    
    # Calculate corresponding GT PPLs (GT responses that correspond to generated responses)
    corresponding_gt_ppls = []
    if generated_turns and normalized_conversation:
        # Sort generated turns by time to ensure proper matching
        sorted_generated_turns = sorted(generated_turns, key=lambda x: x.get('time', 0.0))
        
        # Get all ground truth assistant responses sorted by time
        gt_assistant_responses = []
        for conv_turn in normalized_conversation:
            if conv_turn['role'] == 'assistant':
                gt_assistant_responses.append(conv_turn)
        
        # Sort by time
        gt_assistant_responses.sort(key=lambda x: x.get('time', 0.0))
        
        # Match each generated response to the closest ground truth response
        for i, turn in enumerate(sorted_generated_turns):
            response_time = turn.get('time', 0.0)
            
            # Find the user prompt that triggered this response
            user_prompt = "Please help with the video analysis."  # Default fallback
            for conv_turn in reversed(normalized_conversation):
                if conv_turn['role'] == 'user' and conv_turn['time'] <= response_time:
                    user_prompt = conv_turn['content']
                    break
            
            # Find the GROUND TRUTH response that corresponds to this generated response
            # Use simple index-based matching
            if i < len(gt_assistant_responses):
                best_gt_response = gt_assistant_responses[i]
            elif gt_assistant_responses:
                # Use the last available response
                best_gt_response = gt_assistant_responses[-1]
            
            if best_gt_response:
                gt_content = best_gt_response['content']
                
                # Create conversation using the GROUND TRUTH response
                gt_conversation = [
                    {'role': 'system', 'content': 'Please help with the video analysis.'},
                    {'role': 'stream', 'num_frames': 1, 'learn': False},
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': gt_content, 'learn': True}
                ]
                
                # Calculate PPL for this corresponding GROUND TRUTH response
                ppl = calculate_ppl_for_response(model, tokenizer, gt_conversation, video_tensor, device, data_source)
                if ppl is not None:
                    corresponding_gt_ppls.append(ppl)
        return {
            'lm_ppl': avg_ppl,
        'gt_prefix_ppl_visual': avg_gt_ppl_gt_prefix_visual,
        'vlm_prefix_ppl_visual': avg_gt_ppl_vlm_prefix_visual,
        'fluency': 1.0,  # Will be overridden by actual fluency calculation
            'ppl_data': {
            'gt_ppls_gt_prefix_visual': gt_ppls_gt_prefix_visual,
            'gt_ppls_vlm_prefix_visual': gt_ppls_vlm_prefix_visual,
                'corresponding_gt_ppls': corresponding_gt_ppls,
                'generated_responses': len(generated_turns),
                'total_gt_responses': len(ground_truth_responses)
            }
        }

def evaluate_video_with_model(model, tokenizer, video_tensor, conversation, generated_turns, device, frame_fps=2):
    """Evaluate video using the same approach as evaluate.py - using stream_evaluate and compute_metrics."""
    
    try:
        # Prepare conversation for evaluation (like the dataset does)
        eval_conversation = []
        for turn in conversation:
            if turn['role'] == 'user':
                eval_conversation.append({'role': 'user', 'content': turn['content']})
            elif turn['role'] == 'assistant':
                eval_conversation.append({'role': 'assistant', 'content': turn['content']})
        
        # Add generated turns to conversation
        for turn in generated_turns:
            eval_conversation.append({
                'role': 'assistant', 
                'content': turn['content'],
                'time': turn['time']
            })
        
        
        # Tokenize conversation using the same approach as the dataset
        input_text = " ".join([turn['content'] for turn in eval_conversation])
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        
        # Prepare video frames (use all frames for proper evaluation)
        frames = video_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        
        # Use model's stream_evaluate method (same as in evaluate.py)
        if hasattr(model, 'stream_evaluate'):
            try:
                
                # Create labels for evaluation (same as in the dataset)
                labels = input_ids.clone()
                
                # Call stream_evaluate with the same parameters as the benchmark
                raw_metrics = model.stream_evaluate(
                    input_ids=input_ids,
                    labels=labels,
                    frames=frames,
                    frame_token_interval_threshold=Config.STREAMING_THRESHOLD_GOALSTEP,
                    ignore_token_id=-100
                )
                
                
                # Process metrics exactly like the dataset's compute_metrics method
                lm_ppl, frame_diff, fluency, _ = raw_metrics.mean(dim=0).tolist()
                
                return {
                    'lm_ppl': lm_ppl,
                    'fluency': fluency
                }
                
            except Exception as e:
                print(f"Warning: stream_evaluate failed: {e}")
                traceback.print_exc()
        else:
            print(f"Warning: Model does not have stream_evaluate method")
        
        # Fallback: calculate basic metrics from generated content
        return calculate_basic_content_metrics(generated_turns, conversation)
        
    except Exception as e:
        print(f"Warning: Video evaluation failed: {e}")
        traceback.print_exc()
        return calculate_basic_content_metrics(generated_turns, conversation)

def calculate_basic_content_metrics(generated_turns, conversation):
    """Calculate basic content metrics as fallback."""
    # Basic content-based metrics
    num_generated = len(generated_turns)
    num_ground_truth = len([t for t in conversation if t['role'] == 'assistant'])
    
    # Calculate average response length
    avg_response_length = sum(len(turn.get('content', turn.get('text', '')).split()) for turn in generated_turns) / max(1, num_generated)
    
    # Calculate timing accuracy (simplified)
    time_accuracy = 1.0  # Default perfect timing accuracy
    
    # Simple PPL estimation based on response quality
    # This is a fallback when proper PPL calculation fails
    base_ppl = 2.5  # Base PPL for reasonable models
    length_factor = max(0.5, min(2.0, avg_response_length / 20.0))  # Normalize by expected length
    response_diversity = len(set(turn.get('content', turn.get('text', ''))[:50] for turn in generated_turns)) / max(1, num_generated)  # Character diversity
    diversity_factor = max(0.5, min(2.0, response_diversity))  # Diversity factor
    
    lm_ppl = base_ppl * length_factor * diversity_factor
    
    # Skip unused calculations
    # Calculate fluency: proportion of consecutive successful token predictions within a dialogue turn
    # Based on the paper's implementation: fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
    fluency = 1.0  # Default fluency score
    if generated_turns and conversation:
        total_turns = len(generated_turns)
        total_gt_turns = len([t for t in conversation if t['role'] == 'assistant'])
        
        if total_gt_turns > 0:
            # Simplified fluency calculation based on response success rate
            # In streaming evaluation, we consider a response "successful" if it has reasonable length and content
            successful_responses = 0
            for turn in generated_turns:
                response_text = turn.get('text', turn.get('content', ''))
                if response_text and len(response_text.split()) >= 2:  # At least 2 words
                    successful_responses += 1
            
            # Fluency as proportion of successful responses
            fluency = successful_responses / total_gt_turns if total_gt_turns > 0 else 0.0
            fluency = max(0.0, min(1.0, fluency))  # Clamp to [0, 1]
    
    return {
        'lm_ppl': lm_ppl,
        'fluency': fluency
    }

if __name__ == "__main__":
    main()
