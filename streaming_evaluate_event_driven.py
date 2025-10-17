#!/usr/bin/env python3
"""
Streaming evaluation that processes videos frame by frame to avoid OOM.
This creates a custom evaluation loop that mimics the demo inference approach.
"""

import torch
import json
import os
import re
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for live updates
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_video
import transformers
from scipy.stats import spearmanr
from collections import defaultdict
import cv2
from transformers.cache_utils import DynamicCache
import pandas as pd
import av

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

from models import build_model_and_tokenizer, parse_args, fast_greedy_generate

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
    
    # Asynchronous memory transfers
    # set to False as it is working better for now
    ASYNC_KV_OFFLOAD = True              # Use asynchronous KV cache offloading
    
    # Visualization
    OUTPUT_DIR = "figures"
    PLOT_DPI = 300
    PLOT_FIGSIZE_LARGE = (15, 10)
    PLOT_FIGSIZE_MEDIUM = (15, 6)
    PLOT_FIGSIZE_SMALL = (15, 4)
    
    # Live visualization settings
    LIVE_VIZ_UPDATE_INTERVAL = 10  # Update plot every N events
    LIVE_VIZ_ENABLED = True         # Enable live visualization
    
    # Processing limits
    BATCH_SIZE_LIMIT = 10                # Max frames to load at once
    MEMORY_CHECK_INTERVAL = 1           # Check memory every N frames
    MEMORY_WARNING_THRESHOLD = 2000      # MB remaining before warning
    
    # Threshold sweep configuration
    DEBUG_THRESHOLDS = [0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]         # Coarse-grained thresholds
    # DEBUG_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92]  # Fine-grained thresholds
    
    # File paths
    DATASET_BASE_PATH = "datasets/ego4d/v2/full_scale_2fps_384"
    
    # Model configuration
    VISION_PRETRAINED = 'google/siglip-large-patch16-384'
    EMBED_MARK = '2fps_384_1+3x3'
    MAX_NUM_FRAMES = 1000
    
    # Response generation
    INPLACE_OUTPUT_SIZE = 100
    EXPECTED_RESPONSE_LENGTH = 20.0
    LENGTH_FACTOR_MIN = 0.5
    LENGTH_FACTOR_MAX = 2.0
    
    # Streaming thresholds for different dataset types
    STREAMING_THRESHOLD = 0.725  # Threshold for goalstep dataset (works well with user queries)
    
    # network config
    RTT = 0.04

    # configurable parameters
    DATA_SOURCE = 'goalstep' # 'goalstep' or 'narration'
    RL_WEIGHT = 0 # discount factor for remaining length compared to the age
    EWMA_FACTOR = 0.9
    AGE_WEIGHT = 1 # discount factor for age compared to the remaining length
    SCHEDULING_METHOD = 'lowest_buffer' # 'round_robin' or 'random' or 'lowest_buffer' 
    SCORE_IMPACT = 1 # 0 means disable score and it becomes the same as lowest_buffer
    GENERATION_CHUNK_SIZE = 2      # chunk size for generation
    BUFFER_URGENT_FACTOR = 0.2          # the fraction of the chunk size to determine whether the buffer is urgent
    MAX_EVAL_FRAMES = 10            # Max frames for evaluation (use full video)
    DEFAULT_NUM_VIDEOS = 3             # Default number of videos for evaluation
    USER_CONSUMPTION_SPEED = 2.7        # Words per second (fast listening)
    
    BUFFER_URGENT_VALUE = GENERATION_CHUNK_SIZE * BUFFER_URGENT_FACTOR # higher means easier to select gen events

# =============================================================================
# IMAGE DIFFERENCE FEATURE CALCULATION
# =============================================================================

def calculate_frame_diff_features(current_frame, prev_frame):
    """
    Calculate lightweight image features for both frame differences and single-frame properties.
    All features are computationally efficient for real-time processing.
    
    Args:
        current_frame: torch.Tensor of shape (C, H, W) in range [0, 255]
        prev_frame: torch.Tensor of shape (C, H, W) in range [0, 255] or None
    
    Returns:
        dict with 12 lightweight features:
        - Frame difference (if prev exists): pixel_diff, edge_diff, corner_diff, histogram_diff, 
                                             optical_flow_mag, motion_energy
        - Single frame: brightness, contrast, edge_density, corner_count, blur_score, color_variance
        Returns None for diff features if prev_frame is None
    """
    # Convert from torch tensor (C, H, W) to numpy (H, W, C) and ensure uint8
    curr_np = current_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)
    
    # === SINGLE-FRAME FEATURES (always computed) ===
    
    # 1. Brightness (mean intensity)
    brightness = float(curr_gray.mean())
    
    # 2. Contrast (standard deviation of intensity)
    contrast = float(curr_gray.std())
    
    # 3. Edge Density (proportion of edge pixels)
    curr_edges = cv2.Canny(curr_gray, threshold1=50, threshold2=150)
    edge_density = float(curr_edges.sum() / (curr_edges.size * 255.0))  # Normalize to [0, 1]
    
    # 4. Corner Count (number of detected corners)
    curr_corners = cv2.cornerHarris(curr_gray, blockSize=2, ksize=3, k=0.04)
    corner_threshold = curr_corners.max() * 0.01 if curr_corners.max() > 0 else 0
    corner_count = float(np.sum(curr_corners > corner_threshold))
    
    # 5. Blur Score (Laplacian variance - higher = sharper)
    laplacian = cv2.Laplacian(curr_gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    
    # 6. Color Variance (variance across RGB channels)
    # Measures color diversity in the frame
    color_variance = float(np.mean([curr_np[:, :, i].std() for i in range(3)]))
    
    # === FRAME DIFFERENCE FEATURES (only if prev_frame exists) ===
    
    if prev_frame is None:
        return {
            # Single-frame features
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'corner_count': corner_count,
            'blur_score': blur_score,
            'color_variance': color_variance,
            # Difference features (None for first frame)
            'pixel_diff': None,
            'edge_diff': None,
            'corner_diff': None,
            'histogram_diff': None,
            'optical_flow_mag': None,
            'motion_energy': None,
        }
    
    # Convert previous frame
    prev_np = prev_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
    
    # 7. Pixel Difference (mean absolute difference)
    pixel_diff = float(np.abs(curr_gray.astype(float) - prev_gray.astype(float)).mean())
    
    # 8. Edge Difference (change in edge maps)
    prev_edges = cv2.Canny(prev_gray, threshold1=50, threshold2=150)
    edge_diff = float(np.abs(curr_edges.astype(float) - prev_edges.astype(float)).mean())
    
    # 9. Corner Difference (change in corner response)
    prev_corners = cv2.cornerHarris(prev_gray, blockSize=2, ksize=3, k=0.04)
    corner_diff = float(np.abs(curr_corners - prev_corners).mean())
    
    # 10. Histogram Difference (intensity distribution change)
    curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
    hist_corr = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CORREL)
    histogram_diff = float(1.0 - hist_corr)
    
    # 11. Optical Flow Magnitude (pixel-level motion)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    optical_flow_mag = float(flow_mag.mean())
    
    # 12. Motion Energy (total change magnitude)
    motion_energy = float(np.sum((curr_gray.astype(float) - prev_gray.astype(float)) ** 2))
    
    return {
        # Single-frame features
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'corner_count': corner_count,
        'blur_score': blur_score,
        'color_variance': color_variance,
        # Difference features
        'pixel_diff': pixel_diff,
        'edge_diff': edge_diff,
        'corner_diff': corner_diff,
        'histogram_diff': histogram_diff,
        'optical_flow_mag': optical_flow_mag,
        'motion_energy': motion_energy,
    }

def get_frame_count_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return int(out.strip())

def create_frame_features_vs_response_length_visualization(frame_features_data, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """
    Create comprehensive visualization showing:
    1. Correlation between 12 frame features and response length
    2. Correlation between prior response lengths and current response length
    
    Args:
        frame_features_data: List of dicts with keys:
            - frame_idx, video_time, conversation_id
            - 12 features: brightness, contrast, edge_density, corner_count, blur_score, color_variance,
                          pixel_diff, edge_diff, corner_diff, histogram_diff, optical_flow_mag, motion_energy
            - response_length: int (number of words in response)
    """
    if not frame_features_data:
        print("⚠️ No frame features data to visualize")
        return
    
    # Filter out frames without responses
    all_data_with_responses = [d for d in frame_features_data if d['response_length'] > 0]
    
    if not all_data_with_responses:
        print("⚠️ No frames with responses found")
        return
    
    # Filter for frames with previous frame features (for frame feature analysis)
    valid_data = [d for d in all_data_with_responses if d['pixel_diff'] is not None]
    
    if not valid_data:
        print("⚠️ No valid frame features data")
        return
    
    print(f"📊 Creating comprehensive frame features visualization with {len(valid_data)} frames")
    
    # Extract response lengths
    response_lengths = [d['response_length'] for d in valid_data]
    
    # === ANALYZE PRIOR RESPONSE LENGTH CORRELATIONS ===
    # Group responses by conversation to analyze temporal patterns
    conversation_responses = defaultdict(list)
    for d in all_data_with_responses:
        conversation_responses[d['conversation_id']].append({
            'frame_idx': d['frame_idx'],
            'length': d['response_length']
        })
    
    # Sort by frame index within each conversation
    for conv_id in conversation_responses:
        conversation_responses[conv_id].sort(key=lambda x: x['frame_idx'])
    
    # Compute prior response features for each response
    prior_1_lengths = []  # Length of immediately previous response
    prior_2_lengths = []  # Length of 2 responses ago
    prior_3_lengths = []  # Length of 3 responses ago
    prior_avg_lengths = []  # Average of all prior responses
    current_lengths_for_prior = []  # Current response lengths (for correlation)
    
    for conv_id, responses in conversation_responses.items():
        for idx, resp in enumerate(responses):
            if idx > 0:  # Has at least 1 prior response
                current_lengths_for_prior.append(resp['length'])
                prior_1_lengths.append(responses[idx - 1]['length'])
                prior_2_lengths.append(responses[idx - 2]['length'] if idx > 1 else None)
                prior_3_lengths.append(responses[idx - 3]['length'] if idx > 2 else None)
                prior_avg_lengths.append(np.mean([r['length'] for r in responses[:idx]]))
    
    # Remove None values for prior-2 and prior-3
    prior_2_data = [(prior_2_lengths[i], current_lengths_for_prior[i]) 
                    for i in range(len(prior_2_lengths)) if prior_2_lengths[i] is not None]
    prior_3_data = [(prior_3_lengths[i], current_lengths_for_prior[i]) 
                    for i in range(len(prior_3_lengths)) if prior_3_lengths[i] is not None]
    
    # Define all 12 lightweight features with display names and categories
    feature_specs = [
        # Single-Frame Features (always available)
        ('Brightness', 'brightness', 'Single-Frame'),
        ('Contrast', 'contrast', 'Single-Frame'),
        ('Edge Density', 'edge_density', 'Single-Frame'),
        ('Corner Count', 'corner_count', 'Single-Frame'),
        ('Blur Score', 'blur_score', 'Single-Frame'),
        ('Color Variance', 'color_variance', 'Single-Frame'),
        # Frame Difference Features (when prev frame exists)
        ('Pixel Diff', 'pixel_diff', 'Difference'),
        ('Edge Diff', 'edge_diff', 'Difference'),
        ('Corner Diff', 'corner_diff', 'Difference'),
        ('Histogram Diff', 'histogram_diff', 'Difference'),
        ('Optical Flow Mag', 'optical_flow_mag', 'Motion'),
        ('Motion Energy', 'motion_energy', 'Motion'),
    ]
    
    # Add prior response length features
    prior_features = [
        ('Prior-1 Response', prior_1_lengths, current_lengths_for_prior, 'Temporal'),
        ('Prior-2 Response', [p[0] for p in prior_2_data], [p[1] for p in prior_2_data], 'Temporal'),
        ('Prior-3 Response', [p[0] for p in prior_3_data], [p[1] for p in prior_3_data], 'Temporal'),
        ('Prior Avg Response', prior_avg_lengths, current_lengths_for_prior, 'Temporal'),
    ]
    
    # Create figure with 4x4 grid (12 frame features + 4 temporal features)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'Frame Features & Prior Response Lengths vs Response Length ({data_source})\n'
                 f'{len(valid_data)} frames with responses from {len(set(d["conversation_id"] for d in valid_data))} conversations',
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Calculate and plot correlations for each feature
    correlation_results = []
    
    for idx, (display_name, feature_key, category) in enumerate(feature_specs):
        ax = axes_flat[idx]
        
        # Extract feature values
        feature_vals = [d[feature_key] for d in valid_data]
        
        # Scatter plot
        ax.scatter(feature_vals, response_lengths, alpha=0.4, s=15, c='steelblue', edgecolors='none')
        
        # Calculate correlations
        if len(feature_vals) > 1 and len(set(feature_vals)) > 1 and len(set(response_lengths)) > 1:
            try:
                pearson_r = np.corrcoef(feature_vals, response_lengths)[0, 1]
                spearman_rho, spearman_p = spearmanr(feature_vals, response_lengths)
                
                # Store results
                correlation_results.append({
                    'feature': display_name,
                    'category': category,
                    'pearson_r': pearson_r,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p
                })
                
                # Title with correlation values
                significance = "**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else "")
                ax.set_title(f'{display_name} ({category})\n'
                           f'Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}{significance}',
                           fontsize=10, fontweight='bold' if abs(spearman_rho) > 0.3 else 'normal')
            except:
                ax.set_title(f'{display_name}\n(Calculation error)', fontsize=10)
        else:
            ax.set_title(f'{display_name}\n(Insufficient variance)', fontsize=10)
        
        ax.set_xlabel(f'{display_name}', fontsize=9)
        ax.set_ylabel('Response Length (words)', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(labelsize=8)
    
    # Plot prior response length correlations (last 4 plots)
    for idx, (display_name, prior_vals, current_vals, category) in enumerate(prior_features):
        ax = axes_flat[len(feature_specs) + idx]
        
        if len(prior_vals) > 1 and len(current_vals) > 1:
            # Scatter plot
            ax.scatter(prior_vals, current_vals, alpha=0.4, s=15, c='coral', edgecolors='none')
            
            # Calculate correlations
            if len(set(prior_vals)) > 1 and len(set(current_vals)) > 1:
                try:
                    pearson_r = np.corrcoef(prior_vals, current_vals)[0, 1]
                    spearman_rho, spearman_p = spearmanr(prior_vals, current_vals)
                    
                    # Store results
                    correlation_results.append({
                        'feature': display_name,
                        'category': category,
                        'pearson_r': pearson_r,
                        'spearman_rho': spearman_rho,
                        'spearman_p': spearman_p
                    })
                    
                    # Title with correlation values
                    significance = "**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else "")
                    ax.set_title(f'{display_name} ({category})\n'
                               f'Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}{significance}',
                               fontsize=10, fontweight='bold' if abs(spearman_rho) > 0.3 else 'normal')
                except:
                    ax.set_title(f'{display_name}\n(Calculation error)', fontsize=10)
            else:
                ax.set_title(f'{display_name}\n(Insufficient variance)', fontsize=10)
            
            ax.set_xlabel(f'{display_name} Length (words)', fontsize=9)
            ax.set_ylabel('Current Response Length (words)', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, f'{display_name}\n(Insufficient data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save main plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'frame_features_vs_response_length_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive frame features visualization to {output_path}")
    
    # Sort by absolute Spearman correlation
    correlation_results.sort(key=lambda x: abs(x['spearman_rho']), reverse=True)
    
    # Create a summary bar chart of correlations with category colors
    fig_summary, ax_summary = plt.subplots(figsize=(14, 8))
    
    features_sorted = [r['feature'] for r in correlation_results]
    correlations_sorted = [r['spearman_rho'] for r in correlation_results]
    categories_sorted = [r['category'] for r in correlation_results]
    
    # Color by category with intensity based on correlation strength
    category_colors = {
        'Single-Frame': 'steelblue',
        'Difference': 'orange',
        'Motion': 'green',
        'Temporal': 'purple'
    }
    colors = [category_colors.get(cat, 'gray') for cat in categories_sorted]
    
    bars = ax_summary.barh(features_sorted, correlations_sorted, color=colors, alpha=0.7)
    ax_summary.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax_summary.axvline(x=0.3, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Strong threshold (|ρ|=0.3)')
    ax_summary.axvline(x=-0.3, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Create legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=category_colors['Single-Frame'], alpha=0.7, label='Single-Frame'),
        Patch(facecolor=category_colors['Difference'], alpha=0.7, label='Difference'),
        Patch(facecolor=category_colors['Motion'], alpha=0.7, label='Motion'),
        Patch(facecolor=category_colors['Temporal'], alpha=0.7, label='Temporal (Prior Responses)'),
    ]
    ax_summary.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax_summary.set_xlabel('Spearman Correlation Coefficient (ρ)', fontsize=12, fontweight='bold')
    ax_summary.set_title(f'Feature & Temporal Correlations with Response Length (sorted by |ρ|) - {data_source}',
                        fontsize=14, fontweight='bold')
    ax_summary.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f'frame_features_correlation_summary_{data_source}.png')
    plt.savefig(summary_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved correlation summary bar chart to {summary_path}")


def trace2bwlist(N, trace_type='fcc'):
    """
    Convert network trace to bandwidth list.
    
    Args:
        N (int): Number of consecutive bandwidth samples to extract (default: 4000)
        trace_type (str): Type of trace file - 'fcc' for FCC CSV format with 'bytes_sec' column,
                         'hsdpa' for HSDPA log format with 6 columns
    
    Returns:
        list: List of bandwidth values in bytes per second
    """
    
    if trace_type == 'fcc':
        file_path = '../network_traces/curr_httpgetmt.csv'
        
        # Read the entire file into a DataFrame
        df = pd.read_csv(file_path)

        # Filter out rows where 'bytes_sec' is less than 6Mbps and higher than 0.2Mbps
        df_filtered = df[(df['bytes_sec'] <= 6000000) & (df['bytes_sec'] >= 200000)]

        # Get the total number of rows in the filtered DataFrame
        total_rows = df_filtered.shape[0]

        # Ensure there are enough rows to sample N consecutive rows
        if total_rows < N:
            raise ValueError("The filtered DataFrame does not contain enough rows to sample 4000 consecutive rows.")

        # Randomly select a starting index such that you can get N consecutive rows
        start_index = np.random.randint(0, total_rows - N + 1)

        # Extract N consecutive rows starting from the selected index
        consecutive_bw = df_filtered['bytes_sec'].iloc[start_index:start_index + N]

        # to list
        consecutive_bw = consecutive_bw.tolist()
        
    elif trace_type == 'hsdpa':
        # List of all available HSDPA trace files
        import glob
        import os
        
        hsdpa_traces_dir = '../network_traces/2013/pathbandwidth'
        trace_pattern = os.path.join(hsdpa_traces_dir, '**/*.log')
        available_traces = glob.glob(trace_pattern, recursive=True)
        
        if not available_traces:
            raise FileNotFoundError(f"No HSDPA trace files found in {hsdpa_traces_dir}")
        
        # Randomly select one trace file
        file_path = np.random.choice(available_traces)
        print(f"Selected HSDPA trace: {os.path.basename(file_path)}")
        
        # Read HSDPA trace file (space-separated format)
        # Format: unix_timestamp monotonic_timestamp lat lon bytes_received time_elapsed_ms
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if len(parts) >= 6:
                        bytes_received = int(parts[4])
                        time_elapsed_ms = int(parts[5])
                        
                        # Calculate bandwidth in bytes per second
                        if time_elapsed_ms > 0:
                            bandwidth_bps = (bytes_received * 1000) / time_elapsed_ms
                            data.append(bandwidth_bps)
        
        # Get the total number of rows
        total_rows = len(data)
        
        # Randomly select a starting index such that you can get N consecutive rows
        start_index = np.random.randint(0, total_rows - N + 1)
        
        # Extract N consecutive bandwidth values
        consecutive_bw = data[start_index:start_index + N]
        
    else:
        raise ValueError(f"Unsupported trace_type: {trace_type}. Must be 'fcc' or 'hsdpa'")

    return consecutive_bw

class FilteredEgo4DRefinedNarrationStream:
    """Ego4D Refined Narration Stream - processes videos directly without pre-computed features"""
    
    def __init__(self, split='val', frame_fps=2, is_training=False, augmentation=False, 
                 system_prompt='', tokenizer=None, vision_pretrained='google/siglip-large-patch16-384',
                 embed_mark='2fps_384_1+3x3', max_num_frames=1000, data_source='narration'):
        
        # Basic configuration
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
                user_times = [turn['time'] for turn in conversation]
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
                self.goalstep_timestamp_offsets[conversation_id] = first_user_time
                
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
            
            # print(f"📊 Loaded goalstep data: {len(self.data)} videos")
            # print(f"📊 Extracted {len(all_user_prompts)} unique user prompts from goalstep conversations")
            # print(f"📊 Created normalized conversations with first user prompt at time 0")
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Filter to only include videos that have corresponding video files
        self.filtered_video_uids = []
        missing_videos = []
        total_videos = len(self.data.keys())
        
        for video_uid in self.data.keys():
            video_path = f"{Config.DATASET_BASE_PATH}/{video_uid}.mp4"
            if os.path.exists(video_path):
                self.filtered_video_uids.append(video_uid)
            else:
                missing_videos.append(video_uid)
        
        # print(f"📊 Dataset filtering: {len(self.filtered_video_uids)}/{total_videos} videos have available files")
        print(f"📊 Using {len(self.filtered_video_uids)} videos with available video files")
        
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

                            # Normalize timestamps: find first user prompt time and subtract it from all times
                            user_times = [turn['time'] for turn in conversation]
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
                            
                            if conversation:
                                # Calculate conversation duration
                                start_time = min(turn['time'] for turn in normalized_conversation)
                                end_time = max(turn['time'] for turn in normalized_conversation)
                                duration = end_time - start_time
                                
                                self.conversations.append({
                                    'video_uid': video_uid,
                                    'conversation_id': annotation_uid,
                                    'conversation': normalized_conversation,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration': duration,
                                    'original_conversation': conversation,  # Keep original for metrics
                                    'timestamp_offset': first_user_time
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
                                'timestamp_offset': self.goalstep_timestamp_offsets.get(conversation_id, 0.0)
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
        
        # Since we process videos directly, we don't need to load pre-computed features
        # The frames will be processed on-the-fly during evaluation
        frames = None  # No pre-computed frames needed
        
        load_ranges = {}
        sample_idx = idx
        evaluation_kwargs = {
            'conversation_data': conversation_data  # Pass conversation metadata
        }
        
        return input_text, frames, load_ranges, sample_idx, evaluation_kwargs

def get_cpu_memory():
    """Get current process CPU memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        # RSS (Resident Set Size) - actual physical memory used
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
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


def calculate_inputs_embeds_memory_mb(inputs_embeds):
    """Estimate input embeddings footprint in MB."""
    if inputs_embeds is None:
        return 0.0
    return _tensor_numel_bytes(inputs_embeds) / (1024**2)

def move_dynamic_cache_to(cache: DynamicCache, device: str = "cpu", non_blocking: bool = False):
    # Generic: traverse known structure (Cache -> layers -> attributes)
    # print("move_dynamic_cache_to", device, "non_blocking:", non_blocking)
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            # Move any tensor attributes on the layer
            for name, val in vars(layer).items():
                if torch.is_tensor(val):
                    setattr(layer, name, val.to(device, non_blocking=non_blocking))
                elif isinstance(val, (list, tuple)):
                    seq = []
                    changed = False
                    for x in val:
                        if torch.is_tensor(x):
                            seq.append(x.to(device, non_blocking=non_blocking)); changed = True
                        else:
                            seq.append(x)
                    if changed:
                        setattr(layer, name, type(val)(seq))
    return cache

def _move_cache_to_device(obj, device, non_blocking: bool = False):
    """Recursively move cache containers to a target device."""
    if obj is None:
        return None
    if isinstance(obj, DynamicCache):
        return move_dynamic_cache_to(obj, device, non_blocking=non_blocking)
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return [_move_cache_to_device(item, device, non_blocking=non_blocking) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_cache_to_device(list(obj), device, non_blocking=non_blocking))
    if isinstance(obj, dict):
        return {key: _move_cache_to_device(value, device, non_blocking=non_blocking) for key, value in obj.items()}
    # Handle objects with .to() method (like DynamicCache from transformers)
    if hasattr(obj, 'to') and callable(obj.to):
        # Try to pass non_blocking parameter if supported
        try:
            return obj.to(device, non_blocking=non_blocking)
        except TypeError:
            return obj.to(device)
    return obj


def move_kv_cache_to_device(past_key_values, device, non_blocking: bool = False):
    """Return KV cache moved to the target device while preserving structure."""
    if past_key_values is None:
        return None
    return _move_cache_to_device(past_key_values, device, non_blocking=non_blocking)

def canonical_device(device):
    """Normalize device inputs to torch.device."""
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

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
        temp_tensor = torch.randn(1000, 1000, device='cuda:0')
        del temp_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return True
    except Exception as e:
        print(f"Warning: Memory defragmentation failed: {e}")
        return False

# simulate_text_buffer_trajectories function removed - using on-the-fly buffer tracking instead


def create_serializable_timeline_data(processor_segments, contexts=None):
    """Create serializable data structure for processor timeline visualization."""
    timeline_data = {
        'processor_segments': processor_segments,
        'prompt_data': {}
    }
    
    if contexts:
        for cid, context in contexts.items():
            if hasattr(context, 'user_prompts') and context.user_prompts:
                timeline_data['prompt_data'][cid] = context.user_prompts
    
    return timeline_data

def save_timeline_data(timeline_data, filepath):
    """Save timeline data to JSON file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(timeline_data, f, indent=2)
    print(f"💾 Timeline data saved to: {filepath}")

def load_timeline_data(filepath):
    """Load timeline data from JSON file."""
    import json
    with open(filepath, 'r') as f:
        timeline_data = json.load(f)
    print(f"📂 Timeline data loaded from: {filepath}")
    return timeline_data

def create_processor_timeline(overall_summary, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Visualize processor usage timeline showing when the processor was busy processing frames."""

    processor_segments, prompt_data = overall_summary['timeline_data']['processor_segments'], overall_summary['timeline_data']['prompt_data']

    if not processor_segments:
        return None, None

    os.makedirs(output_dir, exist_ok=True)

    conversation_ids = [segment['conversation_id'] for segment in processor_segments]
    unique_conversations = list(dict.fromkeys(conversation_ids))
    conversation_levels = {cid: idx for idx, cid in enumerate(unique_conversations)}

    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = {cid: base_colors[i % len(base_colors)] for i, cid in enumerate(unique_conversations)}

    # Extract timing information directly from processor segments
    conversation_data = {}
    for cid in unique_conversations:
        cid_segments = [seg for seg in processor_segments if seg['conversation_id'] == cid]
        if not cid_segments:
            continue
            
        start_time = min(seg['start'] for seg in cid_segments)
        end_time = max(seg['end'] for seg in cid_segments)
        duration = max(0.0, end_time - start_time)
        
        # Calculate total processing time
        total_processing = sum(max(0.0, seg['end'] - seg['start']) for seg in cid_segments)
        
        conversation_data[cid] = {
            'original_start': start_time,
            'duration': duration,
            'processing_time': total_processing,
            'completion_time': end_time,
        }

    # Create timeline data directly from processor segments and contexts
    actual_starts = []
    actual_ends = []
    actual_prompts = []
    actual_last_arrivals = []
    
    for cid in unique_conversations:
        data = conversation_data.get(cid)
        if not data:
            continue
            
        # Video start time
        actual_starts.append({'time': data['original_start'], 'conversation_id': cid})
        
        # Video end time (when processing completed)
        actual_ends.append({'time': data['completion_time'], 'conversation_id': cid})
        
        # Last frame arrival time (when the last frame was processed)
        cid_segments = [seg for seg in processor_segments if seg['conversation_id'] == cid]
        if cid_segments:
            last_segment = max(cid_segments, key=lambda seg: seg['start'])
            actual_last_arrivals.append({'time': last_segment['start'], 'conversation_id': cid})
        
        # Extract prompt timing from serializable prompt_data if available
        if prompt_data and cid in prompt_data:
            user_prompts = prompt_data[cid]
            for prompt_time, prompt_content in user_prompts:
                # Convert relative time to absolute time
                absolute_time = data['original_start'] + prompt_time
                actual_prompts.append({'time': absolute_time, 'conversation_id': cid})

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

    sorted_segments = sorted(processor_segments, key=lambda segment: (segment['start'], segment['end']))
    
    scenario_results = [{
        'title': 'Processor Utilization: Concurrent Conversations',
        'ylabel': 'Actual',
        'segments': sorted_segments,
        'prompts': actual_prompts,
        'starts': actual_starts,
        'ends': actual_ends,
        'completion': compute_completion_markers(sorted_segments),
        'last_arrivals': actual_last_arrivals,
    }]

    fig, ax = plt.subplots(1, 1, figsize=(Config.PLOT_FIGSIZE_LARGE[0], 4.5), sharex=True)
    axes = [ax]

    def draw_timeline(ax, timeline_data):
        prompt_flag = False
        start_flag = False
        completion_flag = False
        last_arrival_flag = False

        # Draw processor utilization bars
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
        
        # Draw prompt markers (triangles)
        for prompt in timeline_data.get('prompts', []) or []:
            cid = prompt['conversation_id']
            prompt_time = prompt['time']
            level = conversation_levels.get(cid, 0)
            prompt_y = max(0.05, 0.1 + level * 0.08)
            ax.scatter(
                prompt_time,
                prompt_y,
                marker='v',
                s=70,
                color=colors.get(cid, '#444444'),
                edgecolors='k',
                linewidths=0.5,
                zorder=4
            )
            prompt_flag = True
        
        # Draw video start markers (dashed vertical lines)
        for marker in timeline_data['starts'] or []:
            cid = marker['conversation_id']
            start_time = marker['time']
            ax.axvline(start_time, color=colors.get(cid, '#333333'), linestyle='--', linewidth=1.0, alpha=0.75, zorder=3)
            start_flag = True

        # Draw video end markers (dotted vertical lines)
        for marker in timeline_data['ends'] or []:
            cid = marker['conversation_id']
            end_time = marker['time']
            ax.axvline(end_time, color=colors.get(cid, '#333333'), linestyle=':', linewidth=1.0, alpha=0.6, zorder=3)

        # Draw completion markers (empty circles)
        for marker in timeline_data.get('completion', []) or []:
            cid = marker['conversation_id']
            completion_time = marker['time']
            color = colors.get(cid, '#444444')
            level = conversation_levels.get(cid, 0)
            completion_y = min(1.1, 1.02 - level * 0.08)
            ax.scatter(
                completion_time,
                completion_y,
                marker='o',
                s=65,
                facecolors='none',
                edgecolors=color,
                linewidths=1.1,
                zorder=5
            )
            completion_flag = True

        # Draw last frame arrival markers (stars)
        for marker in timeline_data.get('last_arrivals', []) or []:
            cid = marker['conversation_id']
            arrival_time = marker['time']
            color = colors.get(cid, '#333333')
            level = conversation_levels.get(cid, 0)
            arrival_y = max(0.4, 0.75 - level * 0.08)
            ax.scatter(
                arrival_time,
                arrival_y,
                marker='*',
                s=80,
                color=color,
                edgecolors='k',
                linewidths=0.5,
                zorder=5
            )
            last_arrival_flag = True

        ax.set_title(timeline_data['title'], fontsize=12)
        ax.set_ylabel(timeline_data['ylabel'])
        ax.set_yticks([])
        ax.set_ylim(0.0, 1.18)
        ax.grid(True, axis='x', alpha=0.25)
        return prompt_flag, start_flag, completion_flag, last_arrival_flag

    overall_max = 0.0
    prompt_present = False
    start_present = False
    completion_present = False
    arrival_present = False

    for ax, timeline in zip(axes, scenario_results):
        prompt_flag, start_flag, completion_flag, arrival_flag = draw_timeline(ax, timeline)
        prompt_present |= prompt_flag
        start_present |= start_flag
        completion_present |= completion_flag
        arrival_present |= arrival_flag

        segment_max = max((segment['end'] for segment in timeline['segments'] or []), default=0.0)
        end_markers_max = max((marker['time'] for marker in timeline['ends'] or []), default=0.0)
        prompt_max = max((marker['time'] for marker in timeline.get('prompts', []) or []), default=0.0)
        overall_max = max(overall_max, segment_max, end_markers_max, prompt_max)

    axes[-1].set_xlabel('Processor Time (s)')
    if overall_max <= 0.0:
        overall_max = 1.0
    axes[0].set_xlim(0.0, overall_max * 1.05)

    legend_handles = []
    legend_labels = []
    for cid in unique_conversations:
        legend_handles.append(plt.Line2D([], [], color=colors.get(cid, '#333333'), linewidth=6))
        legend_labels.append(cid)
    if prompt_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='v', linestyle='None', markersize=7, markeredgecolor='k'))
        legend_labels.append('Prompt')
    if start_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', linestyle='--', linewidth=1.0))
        legend_labels.append('Video Start')
    if completion_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='o', linestyle='None', markersize=7, markerfacecolor='none', markeredgewidth=1.1))
        legend_labels.append('Processing Complete')
    if arrival_present:
        legend_handles.append(plt.Line2D([], [], color='#444444', marker='*', linestyle='None', markersize=8, markeredgecolor='k'))
        legend_labels.append('Last Frame Processed')
    if legend_handles:
        axes[0].legend(legend_handles, legend_labels, loc='upper right', fontsize=8, title='Legend')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, f'processor_timeline_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Processor timeline saved to: {output_path}")

    return output_path

# create_buffer_comparison_visualization function removed - using on-the-fly buffer tracking only

def create_memory_visualization(overall_summary, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create detailed memory usage visualization for all videos"""

    all_memory_data = overall_summary['all_memory_data']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_conversations = max(1, len(all_memory_data))
    figure_height = Config.PLOT_FIGSIZE_SMALL[1] * max(1, num_conversations)
    fig, axes = plt.subplots(num_conversations, 5, figsize=(Config.PLOT_FIGSIZE_LARGE[0] * 1.25, figure_height))
    fig.suptitle('Memory Usage Analysis - CPU-First Conversation Processing (GPU + CPU)', fontsize=16, fontweight='bold')

    # Normalise axes shape for single conversation runs
    if num_conversations == 1:
        axes = np.array([axes])

    component_colors = ['#1f77b4', '#ff7f0e', '#8c564b', '#7f7f7f']  # Blue, Orange, Brown, Gray

    for row_idx, (conversation_key, data) in enumerate(all_memory_data.items() or {"conversation": {}}.items()):
        ax_breakdown, ax_components, ax_kv_transfer, ax_peak, ax_cpu = axes[row_idx]

        frames = data.get('frames', [])
        if not frames:
            ax_breakdown.text(0.5, 0.5, 'No memory data collected', ha='center', va='center', transform=ax_breakdown.transAxes)
            ax_components.axis('off')
            ax_kv_transfer.axis('off')
            ax_peak.axis('off')
            ax_cpu.axis('off')
            continue

        totals = data.get('memory_usage', [])
        model_memory = data.get('model_memory', [])
        kv_cache_memory = data.get('kv_cache_memory', [])
        activation_memory = data.get('activation_memory', [])
        other_memory = data.get('other_memory', [])
        memory_per_frame = data.get('memory_per_frame', [])
        torch_allocated = data.get('torch_allocated', [])
        torch_reserved = data.get('torch_reserved', [])

        lengths = [len(frames), len(totals), len(model_memory), len(kv_cache_memory), 
                   len(activation_memory), len(other_memory), len(memory_per_frame), 
                   len(torch_allocated), len(torch_reserved)]
        min_length = min(lengths) if all(lengths) else 0

        if min_length == 0:
            ax_breakdown.text(0.5, 0.5, 'Incomplete memory data', ha='center', va='center', transform=ax_breakdown.transAxes)
            ax_components.axis('off')
            ax_kv_transfer.axis('off')
            ax_peak.axis('off')
            ax_cpu.axis('off')
            continue

        frames = np.array(frames[:min_length])
        totals = np.array(totals[:min_length])
        model_memory = np.array(model_memory[:min_length])
        kv_cache_memory = np.array(kv_cache_memory[:min_length])
        activation_memory = np.array(activation_memory[:min_length])
        other_memory = np.array(other_memory[:min_length])
        memory_per_frame = np.array(memory_per_frame[:min_length])
        torch_allocated = np.array(torch_allocated[:min_length])
        torch_reserved = np.array(torch_reserved[:min_length])

        # 1. Component breakdown stackplot with 4 components
        ax_breakdown.stackplot(
            frames,
            model_memory,
            kv_cache_memory,
            activation_memory,
            other_memory,
            colors=component_colors,
            labels=['Model Params', 'KV Cache', 'Activations', 'Other (CUDA Context + Misc)'],
            alpha=0.7,
        )
        ax_breakdown.plot(frames, totals, color='black', linestyle='--', linewidth=1.5, label='nvidia-smi Total')
        ax_breakdown.set_title(f'{conversation_key[:12]} Memory Breakdown', fontsize=10)
        ax_breakdown.set_xlabel('Frame Number')
        ax_breakdown.set_ylabel('Memory (MB)')
        ax_breakdown.grid(True, alpha=0.3)
        ax_breakdown.legend(fontsize=8, loc='upper left')

        # 2. Combined component trajectories (KV Cache, Activations, Other)
        ax_components.plot(frames, kv_cache_memory, color='#ff7f0e', linewidth=2.0, label='KV Cache', alpha=0.9)
        ax_components.plot(frames, activation_memory, color='#8c564b', linewidth=2.0, label='Activations', alpha=0.9)
        ax_components.plot(frames, other_memory, color='#7f7f7f', linewidth=2.0, label='Other (CUDA + Pool)', alpha=0.9)
        
        ax_components.set_title('Memory Component Trajectories', fontsize=10)
        ax_components.set_xlabel('Frame Number')
        ax_components.set_ylabel('Memory (MB)')
        ax_components.grid(True, alpha=0.3)
        ax_components.legend(fontsize=8, loc='upper left')
        
        # Add summary stats in text box
        if kv_cache_memory.size > 0 and activation_memory.size > 0 and other_memory.size > 0:
            kv_growth = kv_cache_memory[-1] - kv_cache_memory[0] if len(kv_cache_memory) > 0 else 0
            act_growth = activation_memory[-1] - activation_memory[0] if len(activation_memory) > 0 else 0
            other_growth = other_memory[-1] - other_memory[0] if len(other_memory) > 0 else 0
            ax_components.text(
                0.98,
                0.02,
                f'Growth:\nKV: {kv_growth:+.0f}MB\nAct: {act_growth:+.0f}MB\nOther: {other_growth:+.0f}MB',
                transform=ax_components.transAxes,
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

        # 4. Peak Memory Consumption Bar Plot
        component_names = ['Model\nParams', 'KV\nCache', 'Activations', 'Other']
        peak_values = [
            max(model_memory) if len(model_memory) > 0 else 0,
            max(kv_cache_memory) if len(kv_cache_memory) > 0 else 0,
            max(activation_memory) if len(activation_memory) > 0 else 0,
            max(other_memory) if len(other_memory) > 0 else 0,
        ]
        
        bars = ax_peak.bar(component_names, peak_values, color=component_colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, peak_values):
            height = bar.get_height()
            ax_peak.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax_peak.set_title(f'Peak Memory by Component (Total: {max(totals):.0f} MB)', fontsize=10)
        ax_peak.set_ylabel('Memory (MB)')
        ax_peak.grid(True, alpha=0.3, axis='y')
        ax_peak.tick_params(axis='x', labelsize=9)

        # 5. CPU Memory Usage Over Time
        cpu_memory = data.get('cpu_memory', [])
        cpu_memory_growth = data.get('cpu_memory_growth', [])
        
        if cpu_memory and len(cpu_memory) >= min_length:
            cpu_memory = np.array(cpu_memory[:min_length])
            cpu_memory_growth = np.array(cpu_memory_growth[:min_length]) if cpu_memory_growth else np.zeros(min_length)
            
            # Plot absolute CPU memory and growth
            ax_cpu_twin = ax_cpu.twinx()
            
            line1 = ax_cpu.plot(frames, cpu_memory, color='#2ca02c', linewidth=2.0, label='CPU Memory (MB)', alpha=0.9)
            line2 = ax_cpu_twin.plot(frames, cpu_memory_growth, color='#d62728', linewidth=2.0, label='Growth (MB)', alpha=0.9, linestyle='--')
            
            ax_cpu.set_title('CPU Memory Usage (Process RSS)', fontsize=10)
            ax_cpu.set_xlabel('Frame Number')
            ax_cpu.set_ylabel('Total CPU Memory (MB)', color='#2ca02c')
            ax_cpu_twin.set_ylabel('Growth (MB)', color='#d62728')
            ax_cpu.grid(True, alpha=0.3)
            ax_cpu.tick_params(axis='y', labelcolor='#2ca02c')
            ax_cpu_twin.tick_params(axis='y', labelcolor='#d62728')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_cpu.legend(lines, labels, fontsize=8, loc='upper left')
            
            # Add summary stats
            if cpu_memory.size > 0 and cpu_memory_growth.size > 0:
                cpu_initial = cpu_memory[0]
                cpu_final = cpu_memory[-1]
                cpu_max_growth = cpu_memory_growth.max() if len(cpu_memory_growth) > 0 else 0
                ax_cpu.text(
                    0.98,
                    0.02,
                    f'Initial: {cpu_initial:.0f}MB\nFinal: {cpu_final:.0f}MB\nMax Growth: {cpu_max_growth:.0f}MB',
                    transform=ax_cpu.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
                )
        else:
            ax_cpu.text(0.5, 0.5, 'No CPU memory data', ha='center', va='center', transform=ax_cpu.transAxes)
            ax_cpu.set_xlabel('Frame Number')
            ax_cpu.set_ylabel('CPU Memory (MB)')
            ax_cpu.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    output_path = os.path.join(output_dir, f'memory_usage_analysis_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Memory usage analysis saved to: {output_path}")

    # Create additional comparison bar chart
    create_memory_comparison_chart(all_memory_data, output_dir, data_source)

    return output_path


def create_memory_comparison_chart(all_memory_data, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """Create comparison bar chart for final memory usage and KV cache transfer times across videos"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_memory_data:
        print("⚠️ No memory data available for comparison chart")
        return None
    
    # Extract data for comparison
    conversation_keys = []
    final_memory_usage = []
    total_kv_offload_time = []
    total_kv_reload_time = []
    max_kv_cache_memory = []
    
    for conversation_key, data in all_memory_data.items():
        conversation_keys.append(conversation_key)
        
        # Final memory usage (last frame)
        memory_usage = data.get('memory_usage', [])
        final_memory = memory_usage[-1] if memory_usage else 0.0
        final_memory_usage.append(final_memory)
        
        # Total KV cache transfer times
        kv_offload_times = data.get('kv_offload_time', [])
        kv_reload_times = data.get('kv_reload_time', [])
        total_offload = sum(kv_offload_times) * 1000.0  # Convert to ms
        total_reload = sum(kv_reload_times) * 1000.0    # Convert to ms
        total_kv_offload_time.append(total_offload)
        total_kv_reload_time.append(total_reload)
        
        # Max KV cache memory usage
        kv_cache_memory = data.get('kv_cache_memory', [])
        max_kv_cache = max(kv_cache_memory) if kv_cache_memory else 0.0
        max_kv_cache_memory.append(max_kv_cache)
    
    # Create the comparison chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Memory Usage Comparison Across Videos - {data_source.upper()} Dataset', fontsize=16, fontweight='bold')
    
    # Convert conversation keys to shorter labels for better display
    video_labels = [f'Video {i+1}' for i in range(len(conversation_keys))]
    
    # 1. Final Memory Usage Comparison
    bars1 = ax1.bar(video_labels, final_memory_usage, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_title('Final Memory Usage (nvidia-smi)', fontweight='bold')
    ax1.set_ylabel('Memory (MB)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, final_memory_usage):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_memory_usage)*0.01,
                f'{value:.0f}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. KV Cache Transfer Times Comparison
    x_pos = np.arange(len(video_labels))
    width = 0.35
    
    bars2_offload = ax2.bar(x_pos - width/2, total_kv_offload_time, width, 
                           label='Offload → CPU', color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2_reload = ax2.bar(x_pos + width/2, total_kv_reload_time, width,
                          label='Reload → GPU', color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Total KV Cache Transfer Times', fontweight='bold')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(video_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars, values in [(bars2_offload, total_kv_offload_time), (bars2_reload, total_kv_reload_time)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(max(total_kv_offload_time), max(total_kv_reload_time))*0.01,
                    f'{value:.1f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Max KV Cache Memory Usage
    bars3 = ax3.bar(video_labels, max_kv_cache_memory, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_title('Peak KV Cache Memory Usage', fontweight='bold')
    ax3.set_ylabel('Memory (MB)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars3, max_kv_cache_memory):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(max_kv_cache_memory)*0.01,
                f'{value:.0f}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the comparison chart
    comparison_output_path = os.path.join(output_dir, f'memory_comparison_{data_source}.png')
    plt.savefig(comparison_output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    print(f"📊 Memory comparison chart saved to: {comparison_output_path}")
    
    plt.show()
    
    return comparison_output_path


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
        ax.set_title(f'Frame Score Analysis', fontsize=14, fontweight='bold')
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
    """Create unified timing plot grid: rows=conversations, columns=metric types"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_conversations = len(conversation_timings)
    if num_conversations == 0:
        print("⚠️ No conversation timing data available")
        return
    
    # Create figure with grid: rows=conversations, columns=3 metric types
    fig, axes = plt.subplots(num_conversations, 3, figsize=(21, 6 * num_conversations))
    fig.suptitle(f'Conversation Timing Analysis - {data_source.upper()} Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Ensure axes is always 2D
    if num_conversations == 1:
        axes = axes.reshape(1, -1)
    
    for i, conversation_timing in enumerate(conversation_timings):
        conversation_number = i + 1
        conversation_id = conversation_timing.get('conversation_id', f'conversation_{i+1}')
        
        # Get axes for this conversation (row i)
        ax1 = axes[i, 0]  # Timing components breakdown
        ax2 = axes[i, 1]  # Component efficiency
        ax3 = axes[i, 2]  # Timing over time
        
        # Add row label
        row_label = f"Conv {conversation_number}\n{conversation_id[:15]}"
        ax1.text(-0.15, 0.5, row_label, transform=ax1.transAxes, 
                fontsize=10, fontweight='bold', va='center', ha='right', rotation=0)
        
        # === Column 1: Timing Components Breakdown ===
        components = ['Visual\nEmbedding', 'Model\nForward', 'Generation']
        times = [conversation_timing['visual_embedding_time'], 
                conversation_timing['model_forward_time'], 
                conversation_timing['generation_time']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax1.bar(components, times, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Time (seconds)', fontsize=9)
        if i == 0:
            ax1.set_title('Timing Components Breakdown', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', labelsize=8)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # === Column 2: Component Efficiency per Frame ===
        if conversation_timing['frame_processing_times']:
            frame_count = len(conversation_timing['frame_processing_times'])
            
            visual_per_frame = conversation_timing['visual_embedding_time'] / frame_count
            model_per_frame = conversation_timing['model_forward_time'] / frame_count
            generation_per_response = conversation_timing['generation_time'] / frame_count
            
            components_short = ['Visual', 'Model', 'Gen']
            per_frame_times = [visual_per_frame, model_per_frame, generation_per_response]
            
            bars = ax2.bar(components_short, per_frame_times, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=0.5)
            ax2.set_ylabel('Time per Frame (s)', fontsize=9)
            if i == 0:
                ax2.set_title('Component Efficiency', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='x', labelsize=8)
            
            # Add value labels with units
            labels = [f'{visual_per_frame*1000:.1f}ms', 
                     f'{model_per_frame*1000:.1f}ms', 
                     f'{generation_per_response*1000:.1f}ms']
            for bar, time, label in zip(bars, per_frame_times, labels):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(per_frame_times)*0.02,
                        label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # === Column 3: Timing Components Over Time ===
        frame_timing_data = conversation_timing.get('frame_timing_data', [])
        
        if frame_timing_data:
            # Extract timing data
            video_times = [data['video_time'] for data in frame_timing_data]
            visual_times = [data['visual_embedding_time'] * 1000 for data in frame_timing_data]
            model_times = [data['model_forward_time'] * 1000 for data in frame_timing_data]
            generation_times = [data['generation_time'] * 1000 for data in frame_timing_data]
            
            # Plot timing components over time
            ax3.plot(video_times, visual_times, color='#1f77b4', linewidth=1.5, 
                    alpha=0.8, label='Visual Emb')
            ax3.plot(video_times, model_times, color='#ff7f0e', linewidth=1.5, 
                    alpha=0.8, label='Model Fwd')
            ax3.plot(video_times, generation_times, color='#2ca02c', linewidth=1.5, 
                    alpha=0.8, label='Generation')
            
            ax3.set_xlabel('Video Time (seconds)', fontsize=9)
            ax3.set_ylabel('Time per Frame (ms)', fontsize=9)
            if i == 0:
                ax3.set_title('Timing Components Over Time', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=7, loc='upper right')
            
            # Add statistics in a compact format
            avg_visual = np.mean(visual_times)
            avg_model = np.mean(model_times)
            avg_generation = np.mean(generation_times)
            
            stats_text = f'Avg: V={avg_visual:.1f} M={avg_model:.1f} G={avg_generation:.1f}ms'
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, 'No timing data', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=10)
            if i == 0:
                ax3.set_title('Timing Components Over Time', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.98])
    
    # Save unified plot
    output_path = os.path.join(output_dir, f'all_conversations_timing_{data_source}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Unified conversation timing plot saved to: {output_path}")

class EventDrivenConversationContext:
    def __init__(self, conversation_idx, conversation_data, video_path, dataset, device, data_source, custom_threshold, conversation_start_time, model, tokenizer, model_memory_mb):
        self.conversation_idx = conversation_idx
        self.conversation_data = conversation_data
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
        self.actual_end_time = self.conversation_start_time + self.duration
        self.processing_span = 0.0

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

        # Calculate how many frames we actually need for testing
        conversation_based_limit = int(self.duration * Config.FRAME_FPS)
        self.test_frames = min(conversation_based_limit, Config.MAX_EVAL_FRAMES-1)
        test_duration = self.test_frames / Config.FRAME_FPS
        self.num_frames = int(self.duration * Config.FRAME_FPS)
        
        # Read only the frames we need to save CPU memory
        start_time = conversation_data['start_time']
        end_time = test_duration + start_time
        video_frames, _, _ = read_video(video_path, pts_unit='sec', output_format='TCHW', start_pts=start_time, end_pts=end_time)
        self.test_frames = video_frames.size(0)
        
        # Only keep the frames we need for testing
        self.video_frames = video_frames
        # print(f"📹 Loaded all {self.test_frames} frames for {self.conversation_id[0:12]} to CPU: {self.video_frames.shape} ({self.video_frames.device})")

        # Load bandwidth trace with the same length as the video frames
        self.bandwidth_trace = trace2bwlist(self.test_frames)

        # calculate the size of video file in bytes
        self.video_size = os.path.getsize(video_path)

        # probe the number of frames in the video path
        self.frame_count = get_frame_count_ffprobe(video_path)

        # calculate the size of each video frame in bytes
        self.video_frame_size = self.video_size / self.frame_count

        # calculate the transmission time of each video frame
        self.transmission_times = [self.video_frame_size / self.bandwidth_trace[i] + Config.RTT for i in range(self.test_frames)]
        
        self.total_visual_embedding_time = 0.0
        self.total_model_forward_time = 0.0
        self.total_generation_time = 0.0
        self.total_kv_offload_time = 0.0
        self.total_kv_reload_time = 0.0
        self.total_sending_time = sum(self.transmission_times)
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
            'activation_memory': [],
            'other_memory': [],
            'torch_allocated': [],
            'torch_reserved': [],
            'kv_transfer_size': [],
            'kv_offload_time': [],
            'kv_reload_time': [],
            # CPU memory tracking
            'cpu_memory': [],
            'cpu_memory_growth': [],
            'time': []
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
        self.response_generated = 0
        self.response_expected = 0
        self.received_prompt_cnt = 0
        self.processed_prompt_cnt = 0
        
        # OOM handling
        self.oom_occurred = False
        self.oom_frame_idx = None
        self.oom_time = None
    
    def handle_oom(self, frame_idx, current_time):
        """Handle OOM event by marking conversation as truncated."""
        self.oom_occurred = True
        self.oom_frame_idx = frame_idx
        self.oom_time = current_time
        
        # Log the OOM event
        self.event_log.append({
            'time': current_time,
            'type': 'oom_event',
            'detail': {'frame_idx': frame_idx, 'reason': 'out_of_memory'},
            'conversation_id': self.conversation_id
        })
        
        # print(f"🚨 OOM occurred for conversation {self.conversation_id} at frame {frame_idx}, truncating conversation")
    
    def track_memory_snapshot(self, liveinfer, frame_idx, timestamp):
        """Track memory usage snapshot at a given frame index."""
        # GPU memory tracking
        gpu_memory_info = get_gpu_memory_info()
        torch_allocated_mb = gpu_memory_info['allocated_mb']
        torch_reserved_mb = gpu_memory_info['cached_mb']
        if self.initial_memory is None:
            self.initial_memory = torch_allocated_mb
        memory_growth = torch_allocated_mb - self.initial_memory
        memory_per_frame = memory_growth / max(1, frame_idx) if frame_idx > 0 else 0

        # CPU memory tracking
        current_cpu_memory = get_cpu_memory()
        if not hasattr(self, 'initial_cpu_memory') or self.initial_cpu_memory is None:
            self.initial_cpu_memory = current_cpu_memory
        cpu_memory_growth = current_cpu_memory - self.initial_cpu_memory

        kv_cache_memory_mb = calculate_kv_cache_memory_mb(liveinfer.past_key_values)
        
        # === Complete Memory Breakdown ===
        # 1. Model parameters (static)
        model_memory_mb = self.model_memory_mb
        
        # 2. KV cache (grows with sequence length)
        # kv_cache_memory_mb already calculated above
        
        # 3. Other allocated tensors (intermediate activations, embeddings, buffers)
        activation_memory_mb = max(torch_allocated_mb - model_memory_mb - kv_cache_memory_mb, 0.0)
        
        # 4. PyTorch memory pool overhead (reserved - allocated)
        other_memory_mb = max(torch_reserved_mb - torch_allocated_mb, 0.0)

        self.memory_data['frames'].append(frame_idx)
        self.memory_data['memory_usage'].append(torch_allocated_mb)
        self.memory_data['memory_growth'].append(memory_growth)
        self.memory_data['memory_per_frame'].append(memory_per_frame)
        self.memory_data['model_memory'].append(model_memory_mb)
        self.memory_data['kv_cache_memory'].append(kv_cache_memory_mb)
        self.memory_data['activation_memory'].append(activation_memory_mb)
        self.memory_data['other_memory'].append(other_memory_mb)
        self.memory_data['torch_allocated'].append(torch_allocated_mb)
        self.memory_data['torch_reserved'].append(torch_reserved_mb)
        self.memory_data['time'].append(timestamp)
        
        # CPU memory tracking
        self.memory_data['cpu_memory'].append(current_cpu_memory)
        self.memory_data['cpu_memory_growth'].append(cpu_memory_growth)
        
        # print(f"\n📊 Memory Breakdown at Frame {frame_idx}:")
        # print(f"   nvidia-smi total:        {torch_allocated_mb:7.1f} MB")
        # print(f"   ├─ torch.reserved:       {torch_reserved_mb:7.1f} MB")
        # print(f"   │  ├─ torch.allocated:   {torch_allocated_mb:7.1f} MB")
        # print(f"   │  │  ├─ Model params:   {model_memory_mb:7.1f} MB")
        # print(f"   │  │  ├─ KV cache:       {kv_cache_memory_mb:7.1f} MB")
        # print(f"   │  │  └─ Activations:    {other_allocated_mb:7.1f} MB (model buffers, embeddings, etc.)")
        # print(f"   │  └─ PyTorch pool:      {pytorch_pool_mb:7.1f} MB (reserved - allocated)")
        # print(f"   └─ CUDA overhead:        {cuda_overhead_mb:7.1f} MB (nvidia-smi - reserved)")
            
        self.completed = False
        self.result = None
        self.generation_event_pending = False
        self.pending_frame_events = collections.deque()

    def register_events(self, event_queue, sequence_counter):
        prompt_cutoff = self.test_frames / Config.FRAME_FPS if Config.FRAME_FPS > 0 else float('inf')
        for prompt_time, prompt_content in self.user_prompts:
            if prompt_time <= prompt_cutoff:
                heapq.heappush(event_queue, (self.conversation_start_time + max(0.0, prompt_time), 0, sequence_counter, ('prompt', self.conversation_id, prompt_content)))
                sequence_counter += 1
            # else:
            #     print(f"🔕 Skipping prompt at {prompt_time:.2f}s for {self.conversation_id} (beyond frame budget)")

        for frame_idx in range(self.test_frames):
            frame_time = self.conversation_start_time + frame_idx / Config.FRAME_FPS
            heapq.heappush(event_queue, (frame_time, 1, sequence_counter, ('frame', self.conversation_id, frame_idx)))
            sequence_counter += 1

        # Don't schedule finalize event - we'll finalize all conversations at once after event loop
        return sequence_counter

    def ensure_liveinfer_loaded(self, liveinfer):
        if self.liveinfer_state is None:
            liveinfer.reset()
            liveinfer.set_conversation_context(self.conversation_id)  # Set conversation context for feature tracking
            assert isinstance(self.video_frames, torch.Tensor), f"video_frames is not a torch.Tensor: {type(self.video_frames)}"
            liveinfer.video_tensor = self.video_frames
            self.initial_memory = get_gpu_memory_info()['allocated_mb']
        else:
            liveinfer.restore_state(self.liveinfer_state)
            # liveinfer.set_conversation_context(self.conversation_id)  # Ensure context is set after restore
        self.generation_event_pending = getattr(liveinfer, 'generation_event_pending', False)

    def save_liveinfer_state(self, liveinfer):
        if self.oom_occurred:
            return
        self.liveinfer_state = liveinfer.capture_state()
        # print("saved pending_frame_events (2)", liveinfer.pending_frame_events, self.conversation_id[:12])

    def handle_prompt(self, liveinfer, relative_time, prompt_content):
        liveinfer.input_query_stream(prompt_content, video_time=relative_time)
        self.event_log.append({
            'time': self.conversation_start_time + relative_time,
            'type': 'prompt',
            'detail': prompt_content,
            'conversation_id': self.conversation_id
        })

    def schedule_generation_event(self, event_queue, event_time, sequence_counter):
        # print("schedule_generation_event", self.conversation_id, event_time)
        if self.generation_event_pending:
            return sequence_counter
        heapq.heappush(event_queue, (event_time, 1, sequence_counter, ('generation', self.conversation_id, None)))
        self.generation_event_pending = True
        return sequence_counter + 1

    def handle_frame(self, liveinfer, relative_time, frame_idx, start_time):
        
        # Skip processing if OOM has already occurred
        if self.oom_occurred:
            # print(f"⏭️ Skipping frame {frame_idx} for conversation {self.conversation_id} due to previous OOM")
            return {
                'frame_processing_time': 0.0,
                'generation_time': 0.0,
                'prompt_count': 0,
            }

        frame_start_time = time.time()
        frame_processing_time = 0.0
        generation_time = 0.0
        global_time = self.conversation_start_time + relative_time
        
        liveinfer.input_video_stream(relative_time)
        
        # Check for OOM after input_video_stream call
        if liveinfer.oom_occurred:
            self.handle_oom(frame_idx, global_time)
            liveinfer.oom_occurred = False
            return {
                'frame_processing_time': time.time() - frame_start_time,
                'generation_time': 0.0,
                'prompt_count': 0,
            }
        
        self.event_log.append({
            'time': global_time,
            'type': 'frame',
            'detail': {'frame_idx': frame_idx},
            'conversation_id': self.conversation_id
        })
        liveinfer.texts_generated_previous = ""
        query, response = liveinfer()
        frame_processing_time = time.time() - frame_start_time
        if frame_idx % Config.MEMORY_CHECK_INTERVAL == 0:
            self.track_memory_snapshot(liveinfer, frame_idx, start_time + time.time() - frame_start_time)

        # Check for OOM after liveinfer call
        if liveinfer.oom_occurred:
            self.handle_oom(frame_idx, global_time)
            liveinfer.oom_occurred = False
            return {
                'frame_processing_time': time.time() - frame_start_time,
                'generation_time': 0.0,
                'prompt_count': 1 if query else 0,
            }

        timing_data = liveinfer.get_timing_data()

        kv_cache_mb = timing_data.get('kv_cache_mb', 0.0)
        kv_offload_time = timing_data.get('kv_offload_time', 0.0)
        kv_reload_time = timing_data.get('kv_reload_time', 0.0)
        self.memory_data['kv_transfer_size'].append(kv_cache_mb)
        self.memory_data['kv_offload_time'].append(kv_offload_time)
        self.memory_data['kv_reload_time'].append(kv_reload_time)

        visual_embedding_time = timing_data.get('visual_embedding_time', 0.0)
        streaming_time = timing_data.get('streaming_time', 0.0)
        chunk_generation_time = timing_data.get('generation_chunk_time', timing_data.get('generation_time', 0.0))
        
        texts_generated_previous = liveinfer.texts_generated_previous
        self.total_visual_embedding_time += visual_embedding_time
        self.total_model_forward_time += streaming_time
        self.total_generation_time += chunk_generation_time - kv_reload_time
        self.total_kv_reload_time += kv_reload_time
        frame_processing_time += self.transmission_times[frame_idx]

        self.frame_processing_times.append(frame_processing_time)

        self.frame_timing_data.append({
            'frame_idx': frame_idx,
            'video_time': relative_time,
            'visual_embedding_time': visual_embedding_time, # from video (cpu) to embedding (gpu) (input_video_stream)
            'model_forward_time': streaming_time, # from embedding to KV cache (call for streaming); kv cache offload involved
            'generation_time': chunk_generation_time, # a chunk from KV cache to response (call for response) (gpu); kv cache offload involved
            'total_processing_time': frame_processing_time,
            'response_time': frame_processing_time if query else 0, # only count for frames that have prompts
            'prompt_count': 1 if query else 0,
        })
        
        if response:
            self.generated_turns.append({
                'time': relative_time,
                'text': response,
                'user_prompt': query or "Frame processing",
                'generation_time': frame_processing_time
            })

        assert frame_processing_time > 0, f"frame_processing_time: {frame_processing_time}, frame_idx: {frame_idx}, start_time: {start_time}, relative_time: {relative_time}"
        if frame_processing_time > 0:
            self.event_log.append({
                'time': start_time + frame_processing_time,
                'type': 'response',
                'detail': {'text': texts_generated_previous, 'frame_idx': frame_idx},
                'conversation_id': self.conversation_id,
                'response_idx': self.response_generated,
                'is_response': True if response else False,
                'is_last_chunk': True if response else False,
                'is_first_chunk': True,
                'trigger_method': liveinfer.trigger_method,
            })
            if len(texts_generated_previous) > 0:
                # Update frame features with response length
                word_count = len(re.findall(r"\b\w+\b", texts_generated_previous))
                liveinfer.update_frame_response_length(frame_idx, word_count, self.conversation_id)
            if response:
                self.response_generated += 1
            #     print(f"[t={self.conversation_start_time + relative_time:.2f}s] Response: {response}")
            # elif texts_generated_previous:
            #     print(f"[t={self.conversation_start_time + relative_time:.2f}s] Chunk: {texts_generated_previous}")
            # elif query:
            #     print(f"[t={self.conversation_start_time + relative_time:.2f}s] Query: {query}")
            # print(f"  └─ Generation time: {frame_processing_time:.3f}s\t start_time: {start_time:.3f}\t prompt_idx: {self.response_generated}")

        self.frames_processed += 1
        return {
            'frame_processing_time': frame_processing_time,
            'generation_time': generation_time
        }

    def handle_generation(self, liveinfer, start_time):
        video_time = self.frame_timing_data[-1]['video_time']
        
        # Track memory during generation chunks (use video_time as pseudo frame index for tracking)
        # Convert video time to frame index for consistency
        frame_idx = self.frame_timing_data[-1]['frame_idx']
            
        # Skip processing if OOM has already occurred
        if self.oom_occurred:
            # print(f"⏭️ Skipping generation for conversation {self.conversation_id} due to previous OOM")
            return {
                'frame_processing_time': 0.0,
                'generation_time': 0.0,
            }
        
        chunk_start = time.time()
        liveinfer.texts_generated_previous = ""
        query, response = liveinfer()
        chunk_duration = time.time() - chunk_start
        if frame_idx % Config.MEMORY_CHECK_INTERVAL == 0:
            self.track_memory_snapshot(liveinfer, frame_idx, start_time + time.time() - chunk_start)
        
        # Check for OOM after liveinfer call
        if liveinfer.oom_occurred:
            self.handle_oom(-1, start_time + chunk_duration)  # Use -1 for generation events
            liveinfer.oom_occurred = False
            return {
                'frame_processing_time': 0.0,
                'generation_time': chunk_duration,
            }
        
        texts_generated_previous = liveinfer.texts_generated_previous

        # update frame timing data for the same frame
        
        self.total_generation_time += chunk_duration - liveinfer.timing_data.get('kv_reload_time', 0.0)
        self.total_kv_reload_time += liveinfer.timing_data.get('kv_reload_time', 0.0)
        self.frame_processing_times[-1] += chunk_duration

        # Reset pending flag so the scheduler can decide whether to queue another chunk
        self.generation_event_pending = False
        liveinfer.generation_event_pending = False

        # print(f"========== chunk_duration: {chunk_duration}, total_generation_time: {self.total_generation_time}")

        if response:
            generation_total = liveinfer.timing_data.get('generation_time', chunk_duration)
            self.generated_turns.append({
                'time': video_time,
                'text': response,
                'user_prompt': query or "Frame processing",
                'generation_time': generation_total
            })
        assert chunk_duration > 0, f"chunk_duration: {chunk_duration}, start_time: {start_time}, video_time: {video_time}"
        if chunk_duration > 0:
            
            self.event_log.append({
                'time': chunk_duration + start_time,
                'type': 'response',
                'detail': {'text': texts_generated_previous, 'frame_idx': None},
                'conversation_id': self.conversation_id,
                'response_idx': self.response_generated,
                'is_response': True if response else False,
                'is_last_chunk': True if response else False,
                'is_first_chunk': False,
                'trigger_method': liveinfer.trigger_method,
            })
            if len(texts_generated_previous) > 0:
                word_count = len(re.findall(r"\b\w+\b", texts_generated_previous))
                liveinfer.update_frame_response_length(frame_idx, word_count, self.conversation_id)
            if response:
                self.response_generated += 1
            #     print(f"[t={video_time:.2f}s] Response: {response}")
            # elif texts_generated_previous:
            #     print(f"[t={video_time:.2f}s] Chunk: {texts_generated_previous}")
            # elif query:
            #     print(f"[t={video_time:.2f}s] Query: {query}")
            # print(f"  └─ Generation time: {chunk_duration:.3f}s\t start_time: {start_time:.3f}\t prompt_idx: {self.response_generated}")
        return {
            'frame_processing_time': 0.0,
            'generation_time': chunk_duration,
            'query': query,
            'response': response,
        }

    def finalize(self, liveinfer):
        self.frame_scores_data = liveinfer.get_frame_scores()
        response_time = sum(timing_data['response_time'] for timing_data in self.frame_timing_data)/sum(timing_data['prompt_count'] for timing_data in self.frame_timing_data)
        total_processing_time = sum(self.frame_processing_times)
        visual_embedding_time = self.total_visual_embedding_time
        model_forward_time = self.total_model_forward_time
        generation_time = self.total_generation_time
        kv_offload_time = self.total_kv_offload_time
        kv_reload_time = self.total_kv_reload_time
        total_sending_time = self.total_sending_time
        num_processed_frames = len(self.frame_processing_times)

        content_metrics = calculate_metrics(
            self.model,
            self.tokenizer,
            self.video_frames,
            self.conversation_data['conversation'],
            self.generated_turns,
            self.device,
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

        if self.frame_timing_data:
            last_frame = self.frame_timing_data[-1]
            last_frame_time = last_frame.get('video_time', 0.0)
            last_compute = last_frame.get('compute_time', last_frame.get('total_processing_time', 0.0))
            last_generation = last_frame.get('generation_time', 0.0)
            self.actual_end_time = self.conversation_start_time + last_frame_time + max(0.0, last_compute + last_generation)
        else:
            self.actual_end_time = self.conversation_start_time + self.duration

        self.processing_span = max(0.0, self.actual_end_time - self.conversation_start_time)

        self.event_log.append({
            'time': self.actual_end_time,
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
            'lm_ppl': content_metrics.get('lm_ppl', 0.0) if content_metrics else 0.0,
            'fluency': calculate_fluency_score(self.generated_turns, self.original_conversation, self.data_source),
            'ppl_data': content_metrics.get('ppl_data', {}) if content_metrics else {},
            'total_tokens': 0,
            'visual_embedding_time': visual_embedding_time,
            'model_forward_time': model_forward_time,
            'generation_time': generation_time,
            'kv_offload_time': kv_offload_time,
            'kv_reload_time': kv_reload_time,
            'total_sending_time': total_sending_time,
            'total_processing_time': total_processing_time,
            'response_time': response_time,
            'processing_span': self.processing_span,
            'frame_processing_times': self.frame_processing_times,
            'eos_timing': {'eos_detection_time': 0.0, 'with_eos': 0.0, 'without_eos': 0.0},
            'conversation_turns': len(self.generated_turns),
            'generated_turns': self.generated_turns,
            'video_duration': self.duration,
            'frame_scores_data': self.frame_scores_data,
            'frame_timing_data': self.frame_timing_data,
            'oom_occurred': self.oom_occurred,
            'oom_frame_idx': self.oom_frame_idx,
            'oom_time': self.oom_time
        }

        self.result['event_log'] = self.event_log

        self.generation_state = None
        self.generation_event_pending = False
        self.pending_frame_events = collections.deque()

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

        self.video_frames = None
        self.completed = True


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

class LiveBufferVisualizer:
    """Real-time visualization of buffer and rebuffering for all conversations"""
    
    def __init__(self, conversation_ids, output_dir=Config.OUTPUT_DIR, data_source='goalstep', enabled=True):
        self.conversation_ids = list(conversation_ids)
        self.output_dir = output_dir
        self.data_source = data_source
        self.enabled = enabled and Config.LIVE_VIZ_ENABLED
        self.event_count = 0
        
        if not self.enabled:
            return
            
        # Create color map for conversations using matplotlib colormap
        num_conversations = len(self.conversation_ids)
        self.colors = {}
        if num_conversations > 0:
            # Use a colormap that works well for multiple colors
            colormap = plt.cm.Set3  # Good for up to 12 distinct colors
            for i, cid in enumerate(self.conversation_ids):
                self.colors[cid] = colormap(i / max(1, num_conversations - 1))
        
        # Create figure with 7 subplots: 2 rows, 4 columns (last subplot spans 2 positions)
        self.fig = plt.figure(figsize=(32, 12))
        self.fig.suptitle(f'Live System Monitoring - {data_source.title()} (Listening Mode)', 
                         fontsize=14, fontweight='bold')
        
        # Create a 2x4 grid with the last subplot spanning 2 positions
        gs = self.fig.add_gridspec(2, 4)
        
        self.ax_buffer = self.fig.add_subplot(gs[0, 0])
        self.ax_rebuffer = self.fig.add_subplot(gs[0, 1])
        self.ax_memory = self.fig.add_subplot(gs[0, 2])  # Combined GPU and CPU memory
        self.ax_scheduling = self.fig.add_subplot(gs[0, 3])  # Combined scheduling metrics
        self.ax_age = self.fig.add_subplot(gs[1, 0])  # Age of conversations
        self.ax_erl_crl = self.fig.add_subplot(gs[1, 1])  # ERL and CRL of conversations
        self.ax_delays = self.fig.add_subplot(gs[1, 2])  # Processing and Queuing delays - NEW
        
        # Configure buffer axes
        self.ax_buffer.set_ylabel('Buffer Size (words)')
        self.ax_buffer.set_title('Buffer Level')
        self.ax_buffer.set_xlabel('Processor Time (s)')
        self.ax_buffer.grid(True, alpha=0.3)
        
        # Configure rebuffer axes
        self.ax_rebuffer.set_xlabel('Processor Time (s)')
        self.ax_rebuffer.set_ylabel('Cumulative Rebuffer (s)')
        self.ax_rebuffer.set_title('Cumulative Rebuffering Time')
        self.ax_rebuffer.grid(True, alpha=0.3)
        
        # Configure combined memory axes
        self.ax_memory.set_xlabel('Processor Time (s)')
        self.ax_memory.set_ylabel('Memory Usage (MB)')
        self.ax_memory.set_title('Memory Usage (GPU + CPU)')
        self.ax_memory.grid(True, alpha=0.3)
        
        # Configure combined scheduling axes
        self.ax_scheduling.set_xlabel('Processor Time (s)')
        self.ax_scheduling.set_ylabel('Scheduling Metrics')
        self.ax_scheduling.set_title('Scheduling Decisions & Scores')
        self.ax_scheduling.grid(True, alpha=0.3)
        
        # Configure age axes
        self.ax_age.set_xlabel('Processor Time (s)')
        self.ax_age.set_ylabel('Age (events)')
        self.ax_age.set_title('Conversation Age')
        self.ax_age.grid(True, alpha=0.3)
        
        # Configure ERL/CRL axes
        self.ax_erl_crl.set_xlabel('Processor Time (s)')
        self.ax_erl_crl.set_ylabel('Length (words)')
        self.ax_erl_crl.set_title('Expected Remaining Length')
        self.ax_erl_crl.grid(True, alpha=0.3)
        
        # Configure delays axes
        self.ax_delays.set_xlabel('Processor Time (s)')
        self.ax_delays.set_ylabel('Delay Time (s)')
        self.ax_delays.set_title('Processing and Queuing Delays (Averaged Across Conversations)')
        self.ax_delays.grid(True, alpha=0.3)
        
        # Store line objects for each conversation
        self.buffer_lines = {}
        self.rebuffer_lines = {}
        self.age_lines = {}  # Age lines per conversation
        self.erl_lines = {}  # ERL lines per conversation
        
        # Combined memory lines (both GPU and CPU on same plot)
        self.gpu_line, = self.ax_memory.plot([], [], color='#1f77b4', linewidth=2.5, 
                                          label='GPU Memory', alpha=0.9)
        self.cpu_line, = self.ax_memory.plot([], [], color='#2ca02c', linewidth=2.5, 
                                          label='CPU Memory', alpha=0.9)
        
        # Combined scheduling lines (both decisions and scores on same plot)
        self.selected_buffer_line, = self.ax_scheduling.plot([], [], color='#d62728', linewidth=2, 
                                                           label='Selected Lowest Buffer Chunk', alpha=0.9)
        self.lowest_buffer_line, = self.ax_scheduling.plot([], [], color='#ff7f0e', linewidth=2, 
                                                         label='Selected Nonzero Chunk', alpha=0.9)
        self.buffer_diff_line, = self.ax_scheduling.plot([], [], color='#9467bd', linewidth=2, 
                                                            label='Scheduling Scores', alpha=0.9)
        self.ending_line, = self.ax_scheduling.plot([], [], color='#8c564b', linewidth=2, 
                                                   label='Selected Ending Chunk', alpha=0.9)
        
        # Delay lines (averaged across conversations)
        self.processing_delay_line, = self.ax_delays.plot([], [], color='#e377c2', linewidth=2, 
                                                        label='Processing Delay', alpha=0.9)
        self.queuing_delay_line, = self.ax_delays.plot([], [], color='#17becf', linewidth=2, 
                                                     label='Queuing Delay', alpha=0.9)
        self.networking_delay_line, = self.ax_delays.plot([], [], color='#1f77b4', linewidth=2, 
                                                     label='Networking Delay', alpha=0.9)
        
        for cid in self.conversation_ids:
            color = self.colors[cid]
            self.buffer_lines[cid], = self.ax_buffer.plot([], [], color=color, linewidth=2, 
                                                          label=cid[:12], alpha=0.9)
            self.rebuffer_lines[cid], = self.ax_rebuffer.plot([], [], color=color, linewidth=2, 
                                                              alpha=0.9)
            self.age_lines[cid], = self.ax_age.plot([], [], color=color, linewidth=2, 
                                                   label=cid[:12], alpha=0.9)
            self.erl_lines[cid], = self.ax_erl_crl.plot([], [], color=color, linewidth=2, 
                                                       label=f'{cid[:12]} ERL', alpha=0.9, linestyle='-')
        
        # Create legends with limited entries (max 5 per subfigure)
        max_legend_entries = 5
        
        # Buffer legend - limit to 5 conversations
        buffer_handles, buffer_labels = self.ax_buffer.get_legend_handles_labels()
        if len(buffer_handles) > max_legend_entries:
            buffer_handles = buffer_handles[:max_legend_entries]
            buffer_labels = buffer_labels[:max_legend_entries]
            if len(self.conversation_ids) > max_legend_entries:
                buffer_labels.append(f'+{len(self.conversation_ids) - max_legend_entries} more')
                buffer_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5))
        self.ax_buffer.legend(buffer_handles, buffer_labels, loc='upper right', fontsize=12, title='Conversation')
        
        # Memory legend - keep as is (single lines)
        self.ax_memory.legend(loc='upper left', fontsize=14)
        
        # Scheduling legend - keep as is (single lines)
        self.ax_scheduling.legend(loc='upper left', fontsize=14)
        
        # Age legend - limit to 5 conversations
        age_handles, age_labels = self.ax_age.get_legend_handles_labels()
        if len(age_handles) > max_legend_entries:
            age_handles = age_handles[:max_legend_entries]
            age_labels = age_labels[:max_legend_entries]
            if len(self.conversation_ids) > max_legend_entries:
                age_labels.append(f'+{len(self.conversation_ids) - max_legend_entries} more')
                age_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5))
        self.ax_age.legend(age_handles, age_labels, loc='upper left', fontsize=12, title='Conversation')
        
        # ERL/CRL legend - limit to 5 conversations
        erl_handles, erl_labels = self.ax_erl_crl.get_legend_handles_labels()
        if len(erl_handles) > max_legend_entries:
            erl_handles = erl_handles[:max_legend_entries]
            erl_labels = erl_labels[:max_legend_entries]
            if len(self.conversation_ids) > max_legend_entries:
                erl_labels.append(f'+{len(self.conversation_ids) - max_legend_entries} more')
                erl_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5))
        self.ax_erl_crl.legend(erl_handles, erl_labels, loc='upper right', fontsize=12, title='Conversation')
        
        # Delays legend - keep as is (single lines)
        self.ax_delays.legend(loc='upper left', fontsize=14)
        
        # Initialize memory tracking lists
        self.gpu_memory_times = []
        self.gpu_memory_values = []
        self.cpu_memory_times = []
        self.cpu_memory_values = []
        
        # Initialize age, erl, crl tracking per conversation
        self.age_data = {}  # {conversation_id: {'times': [], 'ages': []}}
        self.erl_data = {}  # {conversation_id: {'times': [], 'erl': []}}
        
        for cid in self.conversation_ids:
            self.age_data[cid] = {'times': [], 'ages': []}
            self.erl_data[cid] = {'times': [], 'erl': []}
        
        # Initialize delay tracking (averaged across conversations)
        self.delay_times = []
        self.avg_processing_delays = []
        self.avg_queuing_delays = []
        self.avg_networking_delays = []

        plt.tight_layout()
        
        # Save initial empty plot
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f'live_buffer_evolution_{self.data_source}.png')
        self.fig.savefig(self.output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Live visualization initialized: {self.output_path}")
    
    def update(self, onthefly_buffer_data, current_time=None, scheduling_data=None, age_erl_crl_data=None, delay_data=None):
        """Update the visualization with current buffer data, memory usage, scheduling metrics, age/erl/crl data, and delay data"""
        if not self.enabled:
            return
        
        self.event_count += 1
        
        # Only update every N events to avoid performance issues
        if self.event_count % Config.LIVE_VIZ_UPDATE_INTERVAL != 0:
            return
        
        max_time = 0.0
        max_buffer = 0.0
        max_rebuffer = 0.0
        
        # Update each conversation's line data
        for cid in self.conversation_ids:
            if cid not in onthefly_buffer_data:
                continue
                
            state = onthefly_buffer_data[cid]
            times = state.get('times', [])
            values = state.get('values', [])
            rebuffer_times = state.get('rebuffer_times', [])
            rebuffer_values = state.get('rebuffer_values', [])
            
            if times and values:
                self.buffer_lines[cid].set_data(times, values)
                max_time = max(max_time, max(times))
                max_buffer = max(max_buffer, max(values))
            
            if rebuffer_times and rebuffer_values:
                self.rebuffer_lines[cid].set_data(rebuffer_times, rebuffer_values)
                max_time = max(max_time, max(rebuffer_times))
                max_rebuffer = max(max_rebuffer, max(rebuffer_values))
        
        # Sample current memory usage
        if current_time is not None:
            gpu_mem = get_gpu_memory_info()['allocated_mb']
            cpu_mem = get_cpu_memory()
            
            self.gpu_memory_times.append(current_time)
            self.gpu_memory_values.append(gpu_mem)
            self.cpu_memory_times.append(current_time)
            self.cpu_memory_values.append(cpu_mem)
            
            # Update memory plots (combined on same axis)
            self.gpu_line.set_data(self.gpu_memory_times, self.gpu_memory_values)
            self.cpu_line.set_data(self.cpu_memory_times, self.cpu_memory_values)
        
        # Update scheduling metrics if provided
        if scheduling_data is not None:
            self.scheduling_times = scheduling_data.get('times', [])
            self.selected_lowest = scheduling_data.get('selected_lowest', [])
            self.selected_increment = scheduling_data.get('selected_increment', [])
            self.selected_score = scheduling_data.get('selected_score', [])
            self.selected_ending = scheduling_data.get('selected_ending', [])
            
            # Update scheduling plots (combined on same axis)
            if self.scheduling_times:
                self.selected_buffer_line.set_data(self.scheduling_times, self.selected_lowest)
                self.lowest_buffer_line.set_data(self.scheduling_times, self.selected_increment)
                self.buffer_diff_line.set_data(self.scheduling_times, self.selected_score)
                self.ending_line.set_data(self.scheduling_times, self.selected_ending)
        
        # Update age, erl, crl plots if provided
        if age_erl_crl_data is not None:
            for cid in self.conversation_ids:
                if cid in age_erl_crl_data:
                    data = age_erl_crl_data[cid]
                    
                    # Update age data
                    if 'age_times' in data and 'ages' in data:
                        self.age_data[cid]['times'].extend(data['age_times'])
                        self.age_data[cid]['ages'].extend(data['ages'])
                        self.age_lines[cid].set_data(self.age_data[cid]['times'], self.age_data[cid]['ages'])
                    
                    # Update ERL data
                    if 'erl_times' in data and 'erl' in data:
                        self.erl_data[cid]['times'].extend(data['erl_times'])
                        self.erl_data[cid]['erl'].extend(data['erl'])
                        self.erl_lines[cid].set_data(self.erl_data[cid]['times'], self.erl_data[cid]['erl'])
        
        # Update delay plots if provided
        if delay_data is not None:
            self.delay_times.append(delay_data['times'])
            self.avg_processing_delays.append(delay_data['avg_processing_delays'])
            self.avg_queuing_delays.append(delay_data['avg_queuing_delays'])
            self.avg_networking_delays.append(delay_data['avg_networking_delays'])
            self.processing_delay_line.set_data(self.delay_times, self.avg_processing_delays)
            self.queuing_delay_line.set_data(self.delay_times, self.avg_queuing_delays)
            self.networking_delay_line.set_data(self.delay_times, self.avg_networking_delays)
            max_time = max(max_time, max(self.delay_times))
        
        # Update axis limits
        if max_time > 0:
            self.ax_buffer.set_xlim(0, max_time * 1.05)
            self.ax_rebuffer.set_xlim(0, max_time * 1.05)
            self.ax_memory.set_xlim(0, max_time * 1.05)
            self.ax_age.set_xlim(0, max_time * 1.05)
            self.ax_erl_crl.set_xlim(0, max_time * 1.05)
            self.ax_delays.set_xlim(0, max_time * 1.05)

        max_schedule_rounds = max(self.scheduling_times)
        if max_schedule_rounds > 0:
            self.ax_scheduling.set_xlim(0, max_schedule_rounds * 1.05)
        
        if max_buffer > 0:
            self.ax_buffer.set_ylim(0, max_buffer * 1.1)
        
        if max_rebuffer > 0:
            self.ax_rebuffer.set_ylim(0, max_rebuffer * 1.1)
        
        # Update combined memory axis limits
        max_memory = 0
        if self.gpu_memory_values:
            max_memory = max(max_memory, max(self.gpu_memory_values))
        if self.cpu_memory_values:
            max_memory = max(max_memory, max(self.cpu_memory_values))
        if max_memory > 0:
            self.ax_memory.set_ylim(0, max_memory * 1.1)
        
        # Update scheduling axis limits
        max_scheduling = 0
        if self.selected_score:
            max_scheduling = max(self.selected_score)
        if max_scheduling > 0:
            self.ax_scheduling.set_ylim(0, max_scheduling * 1.1)
        
        # Update age axis limits
        max_age = 0
        for cid in self.conversation_ids:
            if cid in self.age_data and self.age_data[cid]['ages']:
                max_age = max(max_age, max(self.age_data[cid]['ages']))
        if max_age > 0:
            self.ax_age.set_ylim(0, max_age * 1.1)
        
        # Update ERL/CRL axis limits
        max_length = 0
        for cid in self.conversation_ids:
            if cid in self.erl_data and self.erl_data[cid]['erl']:
                max_length = max(max_length, max(self.erl_data[cid]['erl']))
        if max_length > 0:
            self.ax_erl_crl.set_ylim(0, max_length * 1.1)
        
        # Update delays axis limits
        max_delay = 0
        if self.avg_processing_delays:
            max_delay = max(max_delay, max(self.avg_processing_delays))
        if self.avg_queuing_delays:
            max_delay = max(max_delay, max(self.avg_queuing_delays))
        if self.avg_networking_delays:
            max_delay = max(max_delay, max(self.avg_networking_delays))
        if max_delay > 0:
            self.ax_delays.set_ylim(0, max_delay * 1.1)
        
        # Redraw and save
        self.fig.canvas.draw()
        self.fig.savefig(self.output_path, dpi=150, bbox_inches='tight')
    
    def finalize(self, onthefly_buffer_data):
        """Create final high-quality visualization"""
        if not self.enabled:
            return
        
        # Do one final update with all data
        max_time = 0.0
        max_buffer = 0.0
        max_rebuffer = 0.0
        
        for cid in self.conversation_ids:
            if cid not in onthefly_buffer_data:
                continue
                
            state = onthefly_buffer_data[cid]
            times = state.get('times', [])
            values = state.get('values', [])
            rebuffer_times = state.get('rebuffer_times', [])
            rebuffer_values = state.get('rebuffer_values', [])
            
            if times and values:
                self.buffer_lines[cid].set_data(times, values)
                max_time = max(max_time, max(times))
                max_buffer = max(max_buffer, max(values))
            
            if rebuffer_times and rebuffer_values:
                self.rebuffer_lines[cid].set_data(rebuffer_times, rebuffer_values)
                max_time = max(max_time, max(rebuffer_times))
                max_rebuffer = max(max_rebuffer, max(rebuffer_values))
        
        # Update memory plots (already set during live updates)
        # Just ensure they're using the data we collected
        if self.gpu_memory_times and self.gpu_memory_values:
            self.gpu_line.set_data(self.gpu_memory_times, self.gpu_memory_values)
        if self.cpu_memory_times and self.cpu_memory_values:
            self.cpu_line.set_data(self.cpu_memory_times, self.cpu_memory_values)
        
        # Update axis limits
        if max_time > 0:
            self.ax_buffer.set_xlim(0, max_time * 1.05)
            self.ax_rebuffer.set_xlim(0, max_time * 1.05)
            self.ax_memory.set_xlim(0, max_time * 1.05)
            self.ax_age.set_xlim(0, max_time * 1.05)
            self.ax_erl_crl.set_xlim(0, max_time * 1.05)

        max_schedule_rounds = max(self.scheduling_times)
        if max_schedule_rounds > 0:
            self.ax_scheduling.set_xlim(0, max_schedule_rounds * 1.05)
        
        if max_buffer > 0:
            self.ax_buffer.set_ylim(0, max_buffer * 1.1)
        
        if max_rebuffer > 0:
            self.ax_rebuffer.set_ylim(0, max_rebuffer * 1.1)

        # Update scheduling axis limits
        max_scheduling = 0
        if self.selected_score:
            max_scheduling = max(self.selected_score)
        if max_scheduling > 0:
            self.ax_scheduling.set_ylim(0, max_scheduling * 1.1)
        
        # Update combined memory axis limits
        max_memory = 0
        min_memory = float('inf')
        memory_stats = []
        
        if self.gpu_memory_values:
            max_gpu = max(self.gpu_memory_values)
            min_gpu = min(self.gpu_memory_values)
            avg_gpu = sum(self.gpu_memory_values) / len(self.gpu_memory_values)
            max_memory = max(max_memory, max_gpu)
            min_memory = min(min_memory, min_gpu)
            memory_stats.append(f'GPU: {min_gpu:.0f}-{max_gpu:.0f} MB (avg: {avg_gpu:.0f})')
        
        if self.cpu_memory_values:
            max_cpu = max(self.cpu_memory_values)
            min_cpu = min(self.cpu_memory_values)
            avg_cpu = sum(self.cpu_memory_values) / len(self.cpu_memory_values)
            max_memory = max(max_memory, max_cpu)
            min_memory = min(min_memory, min_cpu)
            memory_stats.append(f'CPU: {min_cpu:.0f}-{max_cpu:.0f} MB (avg: {avg_cpu:.0f})')
        
        if max_memory > 0:
            self.ax_memory.set_ylim(min_memory * 0.95, max_memory * 1.05)
            
            # Add combined stats text
            stats_text = '\n'.join(memory_stats)
            self.ax_memory.text(0.02, 0.98, stats_text,
                transform=self.ax_memory.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=9)
        
        # Save final high-quality version
        self.fig.savefig(self.output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
        plt.close(self.fig)
        # print(f"✅ Final live visualization saved: {self.output_path}")
    
def streaming_evaluate_conversations(model, tokenizer, dataset, device='cuda:0', num_conversations=3, random_selection=False, specific_indices=None, data_source='goalstep', custom_threshold=None, conversation_start_times=None):
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
    conversation_summaries = []
    # defaultdict for age_of_conversations, erl_of_conversations, crl_of_conversations
    age_of_conversations = defaultdict(int)
    erl_of_conversations = defaultdict(int)
    crl_of_conversations = defaultdict(int)
    
    # On-the-fly buffer tracking for each conversation
    listening_speed = Config.USER_CONSUMPTION_SPEED  # Use listening speed as requested
    onthefly_buffer_data = {}  
    
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
        video_path = f"{Config.DATASET_BASE_PATH}/{video_uid}.mp4"

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
        
        # Initialize on-the-fly buffer tracking for this conversation
        onthefly_buffer_data[context.conversation_id] = {
            'buffer': 0.0,
            'last_update_time': arrival_time,
            'times': [arrival_time],
            'values': [0.0],
            'rebuffer_times': [arrival_time],
            'rebuffer_values': [0.0],
            'total_rebuffer': 0.0,
            'pending_responses': [],
            'unanswered_prompts': 0,
            'processing_delays': [0.0],
            'total_processing_delay': 0.0,
            'queuing_delays': [0.0],
            'total_queuing_delay': 0.0,
            'networking_delays': [0.0],
            'total_networking_delay': 0.0,
        }

    # Initialize live visualization
    live_viz = LiveBufferVisualizer(
        conversation_ids=[ctx.conversation_id for ctx in contexts.values()],
        output_dir=Config.OUTPUT_DIR,
        data_source=data_source,
        enabled=True
    )
    
    active_conversation_id = None
    
    # Helper functions for on-the-fly buffer tracking (following simulate_text_buffer_trajectories logic)
    def update_buffer_to_time(buffer_state, target_time, speed, conversation_id, oom_occurred=False, status=0):
        """Advance buffer simulation to target_time, consuming buffer at speed and accumulating rebuffering
        
        Args:
            buffer_state: Current buffer state dictionary
            target_time: Time to advance to
            speed: Consumption speed (words/second)
            conversation_id: ID of the conversation
            oom_occurred: Whether OOM has occurred for this conversation
            status: 0 for queuing, 1 for processing, 2 for networking
        """
        # Skip buffer updates if OOM has occurred
        if oom_occurred:
            return
        current_time = buffer_state['last_update_time']
        current_buffer = buffer_state['buffer']
        rebuffer_total = buffer_state['total_rebuffer']
        processing_delay_total = buffer_state['total_processing_delay']
        queuing_delay_total = buffer_state['total_queuing_delay']
        networking_delay_total = buffer_state['total_networking_delay']
        pending_responses = buffer_state['pending_responses']
        # print("update",conversation_id[:12],pending_responses,buffer_state['unanswered_prompts'])
        # if 'goalstep_4c6' in conversation_id:
        #     print("update", "current time", current_time, "target time", target_time, "current buffer", current_buffer, "pending responses", pending_responses, \
        #         buffer_state['unanswered_prompts'])
        
        # handle the case where prompt arrives before the finish of a chunk
        if target_time <= current_time + 1e-9:
            buffer_state['last_update_time'] = max(current_time, target_time)
            buffer_state['times'].append(current_time)
            buffer_state['values'].append(current_buffer)
            buffer_state['rebuffer_times'].append(current_time)
            if buffer_state.get('last_rebuffering_start', None) is not None:
                if buffer_state['unanswered_prompts'] > 0 and target_time < buffer_state['last_rebuffering_start']:
                    diff_rebuffering = buffer_state['last_rebuffering_start'] - target_time
                    rebuffer_total += diff_rebuffering
                    assert status == 0, f"status: {status}"
                    queuing_delay_total += diff_rebuffering
                    buffer_state['last_rebuffering_start'] = target_time
            buffer_state['rebuffer_values'].append(rebuffer_total)
            buffer_state['processing_delays'].append(processing_delay_total)
            buffer_state['queuing_delays'].append(queuing_delay_total)
            buffer_state['networking_delays'].append(networking_delay_total)
            return

        last_rebuffering_start = target_time # means no rebuffering
        if buffer_state['unanswered_prompts'] > 0:
            last_rebuffering_start = current_time
            
        
        while current_time + 1e-9 < target_time:
            remaining = target_time - current_time
            
            if current_buffer > 1e-9:
                time_to_empty = current_buffer / speed
                if time_to_empty <= remaining + 1e-9:
                    # Buffer empties before target time
                    empty_time = current_time + time_to_empty
                    current_buffer = 0.0
                    current_time = empty_time

                    if buffer_state['unanswered_prompts'] > 0:
                        rebuffer_total += time_to_empty
                        if status == 0:
                            queuing_delay_total += time_to_empty
                        elif status == 1:
                            processing_delay_total += time_to_empty
                        elif status == 2:
                            networking_delay_total += time_to_empty
                        else:
                            raise ValueError(f"Invalid status: {status}")
                    
                    buffer_state['times'].append(empty_time)
                    buffer_state['values'].append(current_buffer)
                    buffer_state['rebuffer_times'].append(empty_time)
                    buffer_state['rebuffer_values'].append(rebuffer_total)
                    buffer_state['processing_delays'].append(processing_delay_total)
                    buffer_state['queuing_delays'].append(queuing_delay_total)
                    buffer_state['networking_delays'].append(networking_delay_total)
                    continue
                
                # Buffer doesn't empty - consume and advance to target
                reduction = speed * remaining
                current_buffer = max(0.0, current_buffer - reduction)
                current_time = target_time

                if buffer_state['unanswered_prompts'] > 0:
                    rebuffer_total += remaining
                    if status == 0:
                        queuing_delay_total += remaining
                    elif status == 1:
                        processing_delay_total += remaining
                    elif status == 2:
                        networking_delay_total += remaining
                    else:
                        raise ValueError(f"Invalid status: {status}")

                buffer_state['times'].append(target_time)
                buffer_state['values'].append(current_buffer)
                buffer_state['rebuffer_times'].append(target_time)
                buffer_state['rebuffer_values'].append(rebuffer_total)
                buffer_state['processing_delays'].append(processing_delay_total)
                buffer_state['queuing_delays'].append(queuing_delay_total)
                buffer_state['networking_delays'].append(networking_delay_total)
                break
            else:
                if last_rebuffering_start is None:
                    last_rebuffering_start = current_time
                # Buffer is empty - accumulate rebuffering if there are pending prompts
                if len(pending_responses) > 0 and current_buffer <= 1e-9:
                    rebuffer_total += remaining
                    if status == 0:
                        queuing_delay_total += remaining
                    elif status == 1:
                        processing_delay_total += remaining
                    elif status == 2:
                        networking_delay_total += remaining
                    else:
                        raise ValueError(f"Invalid status: {status}")
                elif buffer_state['unanswered_prompts'] > 0:
                    rebuffer_total += remaining
                    if status == 0:
                        queuing_delay_total += remaining
                    elif status == 1:
                        processing_delay_total += remaining
                    elif status == 2:
                        networking_delay_total += remaining
                    else:
                        raise ValueError(f"Invalid status: {status}")
                current_time = target_time
                
                buffer_state['times'].append(target_time)
                buffer_state['values'].append(current_buffer)
                buffer_state['rebuffer_times'].append(target_time)
                buffer_state['rebuffer_values'].append(rebuffer_total)
                buffer_state['processing_delays'].append(processing_delay_total)
                buffer_state['queuing_delays'].append(queuing_delay_total)
                buffer_state['networking_delays'].append(networking_delay_total)
                break
        
        buffer_state['last_update_time'] = current_time
        buffer_state['buffer'] = current_buffer
        buffer_state['total_rebuffer'] = rebuffer_total
        buffer_state['total_processing_delay'] = processing_delay_total
        buffer_state['total_queuing_delay'] = queuing_delay_total
        buffer_state['last_rebuffering_start'] = last_rebuffering_start
        buffer_state['total_networking_delay'] = networking_delay_total

    scheduling_selected_lowest_buffer = [0]
    scheduling_selected_increment = [0]
    scheduling_selected_ending = [0]
    scheduling_selection_times = [0]
    
    while event_queue:
        # ===== UPDATE ALL CONVERSATIONS TO CURRENT TIME BEFORE NEXT EVENT =====
        
        # Update all conversations' buffer states to the next event time
        # This advances the buffer simulation for ALL conversations to the same point in time
        for cid, buffer_state in onthefly_buffer_data.items():
            context = contexts.get(cid)
            oom_occurred = context.oom_occurred if context else False
            update_buffer_to_time(buffer_state, processor_clock, listening_speed, cid, oom_occurred)
        
        # Update live visualization with all conversations synchronized at the same time
        # Prepare scheduling data for visualization
        scheduling_data = {
            'times': scheduling_selection_times,
            'selected_lowest': scheduling_selected_lowest_buffer,
            'selected_increment': scheduling_selected_increment,
            'selected_score': [increment + lowest + ending for lowest, increment, ending in zip(scheduling_selected_lowest_buffer, scheduling_selected_increment, scheduling_selected_ending)],
            'selected_ending': scheduling_selected_ending
        }
        # Prepare age, erl, crl data for live visualization
        age_erl_crl_data = {}
        for cid in contexts.keys():
            age_erl_crl_data[cid] = {
                'age_times': [processor_clock],  # Use current time for age tracking
                'ages': [age_of_conversations[cid]],
                'erl_times': [processor_clock],  # Use current time for erl tracking
                'erl': [erl_of_conversations[cid]],
            }
        
        # Collect delay data from all conversations
        all_processing_delays = []
        all_queuing_delays = []
        all_networking_delays = []

        for cid, buffer_state in onthefly_buffer_data.items():
            if 'total_processing_delay' in buffer_state and 'total_queuing_delay' in buffer_state:
                processing_delays = buffer_state['total_processing_delay']
                queuing_delays = buffer_state['total_queuing_delay']
                networking_delays = buffer_state['total_networking_delay']

                # Use rebuffer_times as the time base for delays
                all_processing_delays.append(processing_delays)
                all_queuing_delays.append(queuing_delays)
                all_networking_delays.append(networking_delays)

        avg_processing = sum(all_processing_delays) / len(all_processing_delays)
        avg_queuing = sum(all_queuing_delays) / len(all_queuing_delays)
        avg_networking = sum(all_networking_delays) / len(all_networking_delays)

        delay_data = {
            'times': processor_clock,
            'avg_processing_delays': avg_processing,
            'avg_queuing_delays': avg_queuing,
            'avg_networking_delays': avg_networking
        }
        
        live_viz.update(onthefly_buffer_data, current_time=processor_clock, scheduling_data=scheduling_data, age_erl_crl_data=age_erl_crl_data, delay_data=delay_data)
        
        # flush out any delayed events to process prompts first
        cached_events = []
        unprocessed_prompt_event = None
        cached_frame_events = []
        cached_generation_events = []
        cached_event_by_conversation = {}
        unschedulable_events = []
        total_events = len(event_queue)
        while event_queue and event_queue[0][0] < processor_clock:
            event_time, priority, sequence_counter, payload = heapq.heappop(event_queue)
            event_type, conversation_id, payload_data = payload

            if event_type == 'prompt':
                unprocessed_prompt_event = (event_time, priority, sequence_counter, payload)
                break
            else:
                # delay frames to the processor clock but keep the time of the event
                # if no prompt is found, the frames will be processed in order of the event queue as original
                buffer_state = onthefly_buffer_data[conversation_id]
                buffer_level = buffer_state['buffer']
                if Config.SCHEDULING_METHOD == 'round_robin' or Config.SCHEDULING_METHOD == 'random':
                    if conversation_id not in cached_event_by_conversation:
                        cached_event_by_conversation[conversation_id] = (event_time, priority, sequence_counter, payload, buffer_level)
                    else:
                        unschedulable_events.append((event_time, priority, sequence_counter, payload))
                elif Config.SCHEDULING_METHOD == 'lowest_buffer':
                    r = Config.RL_WEIGHT
                    a = Config.AGE_WEIGHT
                    remaining_length = erl_of_conversations[conversation_id] - crl_of_conversations[conversation_id]
                    score = (r * remaining_length - a * age_of_conversations[conversation_id]) * Config.SCORE_IMPACT
                    if conversation_id not in cached_event_by_conversation or event_type == 'generation':
                        if conversation_id in cached_event_by_conversation:
                            buffer_level0, score0, event_time0, priority0, sequence_counter0, payload0 = cached_event_by_conversation[conversation_id]
                            unschedulable_events.append((event_time0, priority0, sequence_counter0, payload0))
                        cached_event_by_conversation[conversation_id] = (buffer_level, score, event_time, priority, sequence_counter, payload)
                        # print("Add:", payload[0], event_time, buffer_level, payload[1][:12])
                    else:
                        # print("Skip:", payload[0], event_time, buffer_level, payload[1][:12])
                        unschedulable_events.append((event_time, priority, sequence_counter, payload))
                else:
                    raise ValueError(f"Invalid scheduling method: {Config.SCHEDULING_METHOD}")

        if cached_event_by_conversation:
            for conversation_id, event in cached_event_by_conversation.items():
                heapq.heappush(cached_events, event)
                if Config.SCHEDULING_METHOD == 'lowest_buffer':
                    buffer_level, score, event_time, priority, sequence_counter, payload = event
                    if buffer_level < Config.BUFFER_URGENT_VALUE:
                        # we need to sort by score instead of buffer level here
                        if event_type == 'frame' and not cached_generation_events:
                            heapq.heappush(cached_frame_events, (buffer_level, score, event_time, priority, sequence_counter, payload))
                        elif event_type == 'generation':
                            heapq.heappush(cached_generation_events, (buffer_level, score, event_time, priority, sequence_counter, payload))
        
        # print(f"Unschedulable events: {len(unschedulable_events)}, cached_events: {len(cached_events)}, cached_generation_events: {len(cached_generation_events)}, cached_frame_events: {len(cached_frame_events)}")
        if cached_generation_events:
            tmp_events = []
            for event in cached_events:
                buffer_level, score, event_time, priority, sequence_counter, payload = event
                if event not in cached_generation_events:
                    unschedulable_events.append((event_time, priority, sequence_counter, payload))
                else:
                    heapq.heappush(tmp_events, (0, score, event_time, priority, sequence_counter, payload))
            cached_events = tmp_events
        elif cached_frame_events:
            tmp_events = []
            for event in cached_events:
                buffer_level, score, event_time, priority, sequence_counter, payload = event
                if event not in cached_frame_events:
                    unschedulable_events.append((event_time, priority, sequence_counter, payload))
                else:
                    heapq.heappush(tmp_events, (0, score, event_time, priority, sequence_counter, payload))
            cached_events = tmp_events

        scheduled_event = False
        
        if unprocessed_prompt_event is not None:
            if Config.SCHEDULING_METHOD == 'round_robin' or Config.SCHEDULING_METHOD == 'random':
                for event_time, priority, sequence_counter, payload, _ in cached_events:
                    heapq.heappush(event_queue, (event_time, priority, sequence_counter, payload))
            elif Config.SCHEDULING_METHOD == 'lowest_buffer':
                for _, _, event_time, priority, sequence_counter, payload in cached_events:
                    heapq.heappush(event_queue, (event_time, priority, sequence_counter, payload))
            else:
                raise ValueError(f"Invalid scheduling method: {Config.SCHEDULING_METHOD}")
            event_time, priority, _, payload = unprocessed_prompt_event
        elif cached_events:
            selected_lowest = 1
            if Config.SCHEDULING_METHOD == 'round_robin' or Config.SCHEDULING_METHOD == 'random':
                if Config.SCHEDULING_METHOD == 'round_robin':
                    event_time, priority, sequence_counter, payload, buffer_level = heapq.heappop(cached_events)
                else:
                    # select a random event and remove it from the cached events
                    event_time, priority, sequence_counter, payload, buffer_level = random.choice(cached_events)
                    cached_events.remove((event_time, priority, sequence_counter, payload, buffer_level))
                    
                lowest_buffer_level = buffer_level
                # check the lowest buffer level of the cached events
                for cid, buffer_state in onthefly_buffer_data.items():
                    available_buffer_level = buffer_state['buffer']
                    in_cached_events = False
                    for et, pri, seq, pl, bf in cached_events:
                        if cid == pl[1]:
                            in_cached_events = True
                            break
                    if in_cached_events and available_buffer_level < lowest_buffer_level:
                        selected_lowest = 0
                        lowest_buffer_level = available_buffer_level
                # push back the cached events
                for et, pri, seq, pl, bf in cached_events:
                    heapq.heappush(event_queue, (et, pri, seq, pl))

            elif Config.SCHEDULING_METHOD == 'lowest_buffer':
                buffer_level, score, event_time, priority, _, payload = heapq.heappop(cached_events)
                lowest_buffer_level = buffer_level
                # check the lowest buffer level of the cached events
                for cid, buffer_state in onthefly_buffer_data.items():
                    available_buffer_level = buffer_state['buffer']
                    in_cached_events = False
                    for bf, sc, et, pri, seq, pl in cached_events:
                        if cid == pl[1]:
                            in_cached_events = True
                            break
                    if in_cached_events and available_buffer_level < lowest_buffer_level:
                        selected_lowest = 0
                        lowest_buffer_level = available_buffer_level
                assert selected_lowest == 1, f"selected_lowest: {selected_lowest}, lowest_buffer_level: {lowest_buffer_level}, buffer_level: {buffer_level}"
                # push back the cached events
                for bf, sc, et, pri, seq, pl in cached_events:
                    heapq.heappush(event_queue, (et, pri, seq, pl))

            else:
                raise ValueError(f"Invalid scheduling method: {Config.SCHEDULING_METHOD}")

            # only interested in scheduled events
            scheduled_event = True
        else:
            event_time, priority, _, payload = heapq.heappop(event_queue)
        
        # push back the unschedulable events
        for event in unschedulable_events:
            heapq.heappush(event_queue, event)

        assert len(event_queue) == total_events-1, f"event_queue: {len(event_queue)}, total_events: {total_events}"
            
        event_type, conversation_id, payload_data = payload
        context = contexts[conversation_id]

        # Skip event processing if OOM has occurred for this conversation
        if context.oom_occurred:
            continue

        # switch conversation
        if active_conversation_id != conversation_id:
            if active_conversation_id is not None:
                if contexts[active_conversation_id].frame_processing_times:
                    shared_liveinfer.offload_kv_cache()
                    kv_offload_time = shared_liveinfer.timing_data.get('kv_offload_time', 0.0)
                    contexts[active_conversation_id].total_kv_offload_time += kv_offload_time
                    contexts[active_conversation_id].frame_processing_times[-1] += kv_offload_time
                contexts[active_conversation_id].save_liveinfer_state(shared_liveinfer)
            context.ensure_liveinfer_loaded(shared_liveinfer)
            shared_liveinfer.generation_event_pending = context.generation_event_pending
            active_conversation_id = conversation_id

        # pend all late frames
        if event_type == 'frame':
            # this is necessary to handle frames that arrive before the generation event is finished
            if shared_liveinfer.generation_state is not None or getattr(shared_liveinfer, 'generation_event_pending', False):
                shared_liveinfer.pending_frame_events.append((event_time, priority, payload_data))
                # context.save_liveinfer_state(shared_liveinfer)
                continue

        # make sure frame event has correct relative time
        relative_time = max(0.0, event_time - context.conversation_start_time)
        # print("--------EVENT", event_type, event_time, processor_clock, shared_liveinfer.generation_state is not None, getattr(shared_liveinfer, 'generation_event_pending', False), active_conversation_id[:12], "--------")

        if event_type == 'prompt':
            processor_clock = max(processor_clock, event_time)
            # Update buffer to current processor_clock before handling prompt
            buffer_state = onthefly_buffer_data[conversation_id]
            update_buffer_to_time(buffer_state, processor_clock, listening_speed, conversation_id, context.oom_occurred)
            
            # Skip buffer updates if OOM has occurred
            if context.response_expected not in buffer_state['pending_responses']:
                buffer_state['pending_responses'].append(context.response_expected)
            context.response_expected += 1
            buffer_state['unanswered_prompts'] += 1
            context.received_prompt_cnt += 1

            # Record buffer state after adding words
            buffer_state['buffer'] = 0
            buffer_state['times'].append(processor_clock)
            buffer_state['values'].append(0)
            
            context.handle_prompt(shared_liveinfer, relative_time, payload_data)
            shared_liveinfer.generation_event_pending = context.generation_event_pending
            # context.save_liveinfer_state(shared_liveinfer)
            continue

        # now the event is at least frame or generation
        if scheduled_event:
            scheduling_selected_lowest_buffer.append(selected_lowest + scheduling_selected_lowest_buffer[-1])
            scheduling_selection_times.append(scheduling_selection_times[-1] + 1)

        if event_type == 'frame':

            if event_time > processor_clock:
                update_buffer_to_time(buffer_state, event_time, listening_speed, conversation_id, context.oom_occurred)
            start_time = max(processor_clock, event_time)
            
            segment_info = context.handle_frame(shared_liveinfer, relative_time, payload_data, start_time)
            
            generation_duration = segment_info.get('generation_time', 0.0)
            segment_label = context.conversation_id
            frame_idx = payload_data
            
            total_duration = segment_info.get('frame_processing_time', 0.0)
            if total_duration > 0.0:
                segment_end = start_time + total_duration
                processor_segments.append({
                    'conversation_id': segment_label,
                    'start': start_time,
                    'end': segment_end,
                    'type': 'frame',
                    'frame_idx': frame_idx,
                    'frame_duration': max(0.0, total_duration),
                    'generation_duration': max(0.0, total_duration)
                })
                # print(f"🔍 DEBUG: Created processor segment for {segment_label}: start={start_time:.2f}s, end={segment_end:.2f}s")
            transmission_complete_time = start_time + context.transmission_times[frame_idx]
            update_buffer_to_time(buffer_state, transmission_complete_time, listening_speed, conversation_id, context.oom_occurred, status=2)
            serve_completion_time = start_time + total_duration - Config.RTT/2
            processor_clock = start_time + total_duration

            if context.oom_occurred:
                continue
            
            word_count = 0
            # Check if text was generated and update buffer
            assert context.event_log, f"event_log: {context.event_log}"
            assert context.event_log[-1].get('type') == 'response', f"event_log: {context.event_log}"
            last_event = context.event_log[-1]
            text = last_event.get('detail', {}).get('text', '') if isinstance(last_event.get('detail'), dict) else last_event.get('detail', '')
            is_response = last_event.get('is_response', False)
            if text or is_response:
                tokens = re.findall(r"\b\w+\b", text)
                word_count = float(len(tokens))
                crl_of_conversations[conversation_id] += word_count
                
                buffer_state = onthefly_buffer_data[conversation_id]
                response_idx = last_event.get('response_idx', 0)
                
                update_buffer_to_time(buffer_state, processor_clock, listening_speed, conversation_id, context.oom_occurred, status=1)
                
                is_last_chunk = last_event.get('is_last_chunk', False)
                is_first_chunk = last_event.get('is_first_chunk', False)
                
                # update processed prompt cnt
                assert is_first_chunk, f"is_first_chunk: {is_first_chunk}"
                if last_event.get('trigger_method') == 'prompt':
                    context.processed_prompt_cnt += 1
                                    
                # Add words to buffer (happens at chunk completion time)
                if not context.oom_occurred:
                    buffer_state['buffer'] = 0
                    buffer_state['unanswered_prompts'] -= 1
                    if last_event.get('trigger_method') == 'score':
                        if context.response_expected not in buffer_state['pending_responses']:
                            buffer_state['pending_responses'].append(context.response_expected)
                        context.response_expected += 1

                if is_last_chunk:
                    if response_idx in buffer_state['pending_responses']:
                        buffer_state['pending_responses'].remove(response_idx)
                    assert shared_liveinfer.generation_state is None, f"generation_state: {shared_liveinfer.generation_state}"

                # update buffer only if the latest prompt has been processed
                if context.processed_prompt_cnt == context.received_prompt_cnt:
                    buffer_state['buffer'] += word_count
                        
                # Record buffer state after adding words
                buffer_state['times'].append(processor_clock)
                buffer_state['values'].append(buffer_state['buffer'])

                if not is_last_chunk:
                    # update age of conversation if not finished
                    age_of_conversations[conversation_id] += 1
                else:
                    # finished, reset age of conversation
                    age_of_conversations[conversation_id] = 0
                    # update erl of conversation
                    if erl_of_conversations[conversation_id] == 0:
                        erl_of_conversations[conversation_id] = crl_of_conversations[conversation_id]
                    else:
                        erl_of_conversations[conversation_id] = (1-Config.EWMA_FACTOR) * erl_of_conversations[conversation_id] + Config.EWMA_FACTOR * crl_of_conversations[conversation_id]
                    # reset crl of conversation
                    crl_of_conversations[conversation_id] = 0

            if shared_liveinfer.generation_state is not None:
                sequence_counter = context.schedule_generation_event(event_queue, serve_completion_time, sequence_counter)
                

            shared_liveinfer.generation_event_pending = context.generation_event_pending
            # context.save_liveinfer_state(shared_liveinfer)

            if scheduled_event:
                valid_scheduling = word_count > 0
                scheduling_selected_increment.append(valid_scheduling + scheduling_selected_increment[-1])
                scheduling_selected_ending.append(is_response + scheduling_selected_ending[-1])

            # print("----END----EVENT", event_type, shared_liveinfer.generation_state is not None, getattr(shared_liveinfer, 'generation_event_pending', False), active_conversation_id, "--------")
            continue

        if event_type == 'generation':
                
            if event_time > processor_clock:
                update_buffer_to_time(buffer_state, event_time, listening_speed, conversation_id, context.oom_occurred)
            start_time = max(processor_clock, event_time)
            
            segment_info = context.handle_generation(shared_liveinfer, start_time)
            generation_duration = segment_info.get('generation_time', 0.0)
            segment_label = context.conversation_id

            if generation_duration > 0.0:
                segment_end = start_time + generation_duration
                processor_segments.append({
                    'conversation_id': segment_label,
                    'start': start_time,
                    'end': segment_end,
                    'type': 'generation',
                    'frame_idx': None,
                    'frame_duration': generation_duration,
                    'generation_duration': generation_duration
                })
                # print(f"🔍 DEBUG: Created generation segment for {segment_label}: start={start_time:.2f}s, end={segment_end:.2f}s")
            processor_clock = start_time + max(0.0, generation_duration)

            if context.oom_occurred:
                continue
            
            word_count = 0
            # Check if text was generated and update buffer
            assert context.event_log, f"event_log: {context.event_log}"
            assert context.event_log[-1].get('type') == 'response', f"event_log: {context.event_log}"
            last_event = context.event_log[-1]
            text = last_event.get('detail', {}).get('text', '') if isinstance(last_event.get('detail'), dict) else last_event.get('detail', '')
            is_response = last_event.get('is_response', False)
            if text or is_response:
                tokens = re.findall(r"\b\w+\b", text)
                word_count = float(len(tokens))
                crl_of_conversations[conversation_id] += word_count
                
                buffer_state = onthefly_buffer_data[conversation_id]
                response_idx = last_event.get('response_idx', 0)
                
                update_buffer_to_time(buffer_state, processor_clock, listening_speed, conversation_id, context.oom_occurred, status=1)

            # Add words to buffer (happens at chunk completion time)
            if not context.oom_occurred:
                if context.processed_prompt_cnt == context.received_prompt_cnt:
                    buffer_state['buffer'] += word_count
                
                is_last_chunk = last_event.get('is_last_chunk', False)
                is_first_chunk = last_event.get('is_first_chunk', False)
                assert not is_first_chunk, f"is_first_chunk: {is_first_chunk}"
                
                if is_last_chunk:
                    if response_idx in buffer_state['pending_responses']:
                        buffer_state['pending_responses'].remove(response_idx)
                    assert shared_liveinfer.generation_state is None, f"generation_state: {shared_liveinfer.generation_state}"

                    # Record buffer state after adding words
                    buffer_state['times'].append(processor_clock)
                    buffer_state['values'].append(buffer_state['buffer'])

                if not is_last_chunk:
                    # update age of conversation if not finished
                    age_of_conversations[conversation_id] += 1
                else:
                    # if finished, reset age of conversation
                    while shared_liveinfer.pending_frame_events:
                        pending_time, pending_priority, pending_payload = shared_liveinfer.pending_frame_events.popleft()
                        heapq.heappush(event_queue, (pending_time, pending_priority, sequence_counter, ('frame', conversation_id, pending_payload)))
                    sequence_counter += 1
                    # reset age of conversation
                    age_of_conversations[conversation_id] = 0
                    # update erl of conversation
                    if erl_of_conversations[conversation_id] == 0:
                        erl_of_conversations[conversation_id] = crl_of_conversations[conversation_id]
                    else:
                        erl_of_conversations[conversation_id] = (1-Config.EWMA_FACTOR) * erl_of_conversations[conversation_id] + Config.EWMA_FACTOR * crl_of_conversations[conversation_id]
                    # reset crl of conversation
                    crl_of_conversations[conversation_id] = 0

            if shared_liveinfer.generation_state is not None:
                sequence_counter = context.schedule_generation_event(event_queue, processor_clock, sequence_counter)

            shared_liveinfer.generation_event_pending = context.generation_event_pending
            # context.save_liveinfer_state(shared_liveinfer)

            if scheduled_event:
                valid_scheduling = word_count > 0
                scheduling_selected_increment.append(valid_scheduling + scheduling_selected_increment[-1])
                scheduling_selected_ending.append(is_response + scheduling_selected_ending[-1])

            # print("----END----EVENT", event_type, shared_liveinfer.generation_state is not None, getattr(shared_liveinfer, 'generation_event_pending', False), active_conversation_id, "--------")
            continue

    # ===== FINALIZE ALL CONVERSATIONS AFTER EVENT LOOP COMPLETES =====
    print("\n🏁 All events processed. Finalizing all conversations...")
    
    # Finalize each conversation and collect results
    defragment_gpu_memory()

    for cid, context in contexts.items():
        context.finalize(shared_liveinfer)
        context.liveinfer_state = None
        
        unique_key = f"{context.video_uid}_{context.conversation_id}"
        results.append(context.result)
        all_memory_data[unique_key] = context.memory_data
        if context.result.get('frame_scores_data'):
            all_frame_scores_data[unique_key] = context.result['frame_scores_data']
    timeline_data = create_serializable_timeline_data(processor_segments, contexts)
    shared_liveinfer.reset()

    # if all_frame_scores_data:
    #     print("\n📊 Creating frame score analysis...")
    #     create_frame_score_analysis(all_frame_scores_data, data_source=data_source)

    # Finalize live visualization with final data
    live_viz.finalize(onthefly_buffer_data)
    
    # Collect frame features data from liveinfer
    # all_frame_features_data = shared_liveinfer.get_frame_features_data()
    # if all_frame_features_data:
    #     print(f"\n📊 Creating frame features vs response length visualization with {len(all_frame_features_data)} frames...")
    #     create_frame_features_vs_response_length_visualization(all_frame_features_data, data_source=data_source)
    
        # Optionally save timeline data for later use
        # timeline_file = os.path.join(Config.OUTPUT_DIR, f'processor_timeline_data_{data_source}.json')
        # save_timeline_data(timeline_data, timeline_file)

    scheduling_data = {
        'times': scheduling_selection_times,
        'selected_lowest': scheduling_selected_lowest_buffer,
        'selected_increment': scheduling_selected_increment,
        'selected_score': [increment + lowest + ending for lowest, increment, ending in zip(scheduling_selected_lowest_buffer, scheduling_selected_increment, scheduling_selected_ending)],
        'selected_ending': scheduling_selected_ending
    }

    # print(f"🎯 PERFORMANCE SUMMARY:")
    # print(f"   • Conversations Processed: {len(results)}")
    # print(f"   • Total Frames: {sum(r['num_frames'] for r in results)}")
    # print(f"   • Total Generated Responses: {sum(len(r['generated_turns']) for r in results)}")
    # print(f"   • Total Ground Truth Responses: {sum(r['ground_truth_turns'] for r in results)}")

    overall_summary = {
        'results': results,
        'onthefly_buffer_data': onthefly_buffer_data,
        'all_memory_data': all_memory_data,
        'scheduling_data': scheduling_data,
        'timeline_data': timeline_data
    }

    return overall_summary

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
        self.frame_token_interval_threshold = Config.STREAMING_THRESHOLD
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

        # Generation chunking configuration
        self.generation_chunk_size = max(0, Config.GENERATION_CHUNK_SIZE)
        self.generation_state = None
        self.kv_cache_location = 'cpu'
        self._kv_reload_time = 0.0
        self._kv_offload_time = 0.0
        self.generation_event_pending = False
        
        # Frame difference features tracking (per conversation)
        self.frame_features_data = {}  # Dict: conversation_id -> List of {frame_idx, video_time, corner_diff, ...}
        self.prev_frame_per_conversation = {}  # Dict: conversation_id -> previous frame tensor
        self.current_conversation_id = None
        self.oom_occurred = False  # Track which conversation is currently being processed

    def capture_state(self):
        state = {
            'last_ids': self.last_ids.detach().clone().cpu() if torch.is_tensor(self.last_ids) else self.last_ids,
            # 'past_key_values': move_kv_cache_to_device(self.past_key_values, 'cpu') if self.past_key_values is not None else None,
            'past_key_values': self.past_key_values,
            'query_queue': list(self.query_queue),
            'frame_embeds_queue': [(timestamp, embed.detach().cpu() if torch.is_tensor(embed) else embed) for timestamp, embed in list(self.frame_embeds_queue)],
            'video_time': self.video_time,
            'last_frame_idx': self.last_frame_idx,
            'video_tensor': self.video_tensor.detach().cpu() if isinstance(self.video_tensor, torch.Tensor) else self.video_tensor,
            'frame_scores': self.frame_scores.copy(),
            'frame_times': self.frame_times.copy(),
            'response_triggers': self.response_triggers.copy(),
            'response_times': self.response_times.copy(),
            'timing_data': self.timing_data.copy(),
            'generation_state': None,
            'generation_event_pending': self.generation_event_pending,
            'pending_frame_events': list(self.pending_frame_events),
        }
        if self.generation_state is not None:
            gen_state = self.generation_state
            state['generation_state'] = {
                'video_time': gen_state['video_time'],
                'formatted_query': gen_state['formatted_query'],
                'tokens': [t.clone().cpu() for t in gen_state['tokens']],
                'tokens_generated': gen_state['tokens_generated'],
                'total_generation_time': gen_state['total_generation_time'],
                'next_inputs_embeds_cpu': gen_state['next_inputs_embeds_cpu'].clone() if gen_state['next_inputs_embeds_cpu'] is not None else None,
                # 'past_key_values_cpu': move_kv_cache_to_device(gen_state['past_key_values_cpu'], 'cpu') if gen_state['past_key_values_cpu'] is not None else None,
                'past_key_values_cpu': gen_state['past_key_values_cpu'],
                'finished': gen_state['finished'],
                'chunk_invocations': gen_state['chunk_invocations'],
            }
        return state

    def restore_state(self, state):
        self.query_queue = collections.deque(state['query_queue'])
        self.frame_embeds_queue = collections.deque((timestamp, embed.to(self.device) if torch.is_tensor(embed) else embed) for timestamp, embed in state['frame_embeds_queue'])
        if torch.is_tensor(state['last_ids']):
            self.last_ids = state['last_ids'].to(self.device)
        else:
            self.last_ids = torch.tensor(state['last_ids'], device=self.device)
        # self.past_key_values = move_kv_cache_to_device(state['past_key_values'], self.device) if state['past_key_values'] is not None else None
        self.past_key_values = state['past_key_values']
        self.video_time = state['video_time']
        self.last_frame_idx = state['last_frame_idx']
        self.video_tensor = state['video_tensor']
        if isinstance(self.video_tensor, torch.Tensor) and self.video_tensor.device != torch.device('cpu'):
            self.video_tensor = self.video_tensor.cpu()
        self.frame_scores = state['frame_scores'].copy()
        self.frame_times = state['frame_times'].copy()
        self.response_triggers = state['response_triggers'].copy()
        self.response_times = state['response_times'].copy()
        self.timing_data = state['timing_data'].copy()
        self.generation_event_pending = state.get('generation_event_pending', False)
        self.pending_frame_events = collections.deque(state.get('pending_frame_events', []))
        stored_generation_state = state.get('generation_state')
        if stored_generation_state is None:
            self.generation_state = None
        else:
            self.generation_state = {
                'video_time': stored_generation_state['video_time'],
                'formatted_query': stored_generation_state['formatted_query'],
                'tokens': [t.clone() for t in stored_generation_state['tokens']],
                'tokens_generated': stored_generation_state['tokens_generated'],
                'total_generation_time': stored_generation_state['total_generation_time'],
                'next_inputs_embeds_cpu': stored_generation_state['next_inputs_embeds_cpu'].clone() if stored_generation_state['next_inputs_embeds_cpu'] is not None else None,
                'past_key_values_cpu': stored_generation_state['past_key_values_cpu'],
                'finished': stored_generation_state['finished'],
                'chunk_invocations': stored_generation_state['chunk_invocations'],
            }

    
    def reset(self):
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.last_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        self.past_key_values = None
        self.pending_frame_events = collections.deque()
        self.generation_event_pending = False
        # Reset tracking buffers
        self.kv_transfer_metrics = []
        
        # Reset frame score tracking
        self.frame_scores = []
        self.frame_times = []
        self.response_triggers = []
        self.response_times = []
        self.generation_state = None
        self.kv_cache_location = 'cpu'
        self._kv_reload_time = 0.0
    
    def set_conversation_context(self, conversation_id):
        """Set the current conversation context for feature tracking."""
        self.current_conversation_id = conversation_id
        # Initialize storage for this conversation if not exists
        if conversation_id not in self.frame_features_data:
            self.frame_features_data[conversation_id] = []
        if conversation_id not in self.prev_frame_per_conversation:
            self.prev_frame_per_conversation[conversation_id] = None
        self._kv_offload_time = 0.0
        # self.generation_event_pending = False
        # self.pending_frame_events = collections.deque()
    
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
        # Measure visual embedding time: RGB frames → Visual token embeddings
        visual_start = time.time()
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            # Process frames one at a time to avoid OOM
            with torch.no_grad():
                for single_frame_idx in range(self.last_frame_idx + 1, frame_idx + 1):
                    # Stream single frame from CPU to GPU as needed
                    if self.video_tensor is not None:
                        # Get single frame from CPU memory and move to GPU
                        cpu_frame = self.video_tensor[single_frame_idx:single_frame_idx+1]  # Keep batch dimension
                        gpu_frame = cpu_frame.to(self.device)
                        
                        # Calculate frame difference features (per conversation)
                        if self.current_conversation_id is not None:
                            current_frame = cpu_frame[0]  # Remove batch dimension for feature calculation
                            prev_frame = self.prev_frame_per_conversation.get(self.current_conversation_id, None)
                            frame_features = calculate_frame_diff_features(current_frame, prev_frame)
                            
                            # Store all 12 lightweight features
                            self.frame_features_data[self.current_conversation_id].append({
                                'frame_idx': single_frame_idx,
                                'video_time': single_frame_idx / self.frame_fps,
                                'conversation_id': self.current_conversation_id,
                                'response_length': 0,  # Will be updated when response is generated
                                # Single-frame features (always available)
                                'brightness': frame_features['brightness'],
                                'contrast': frame_features['contrast'],
                                'edge_density': frame_features['edge_density'],
                                'corner_count': frame_features['corner_count'],
                                'blur_score': frame_features['blur_score'],
                                'color_variance': frame_features['color_variance'],
                                # Difference features (None for first frame)
                                'pixel_diff': frame_features['pixel_diff'],
                                'edge_diff': frame_features['edge_diff'],
                                'corner_diff': frame_features['corner_diff'],
                                'histogram_diff': frame_features['histogram_diff'],
                                'optical_flow_mag': frame_features['optical_flow_mag'],
                                'motion_energy': frame_features['motion_energy'],
                            })
                            
                            # Update prev_frame for this conversation
                            self.prev_frame_per_conversation[self.current_conversation_id] = current_frame.clone()
                        
                        # Process single frame on GPU with OOM handling
                        try:
                            frame_embeds = self.model.visual_embed(gpu_frame).split(self.frame_num_tokens)
                            self.frame_embeds_queue.extend([
                                (single_frame_idx / self.frame_fps, embed.cpu())
                                for embed in frame_embeds
                            ])
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                print("Out of memory in visual_embed")
                                self.oom_occurred = True
                                # Free GPU frame and break out of frame processing loop
                                del gpu_frame
                                del frame_embeds
                                # release all data on gpu for this liveinfer
                                defragment_gpu_memory()
                                break
                            raise e
                        
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

        # print("input_video_stream", len(self.frame_embeds_queue), video_time, frame_idx, self.last_frame_idx)
    
    def _call_for_response(self, video_time, query):
        # Lazily initialise generation state on first invocation
        if self.generation_state is None:
            self._initialize_generation_state(video_time, query)

        state = self.generation_state
        state['chunk_invocations'] += 1

        generation_start = time.time()
        result = self._execute_generation_chunk(state)
        chunk_duration = time.time() - generation_start
        
        # Handle OOM case
        if isinstance(result, tuple) and len(result) == 2:
            response_tokens, oom_occurred = result
            if oom_occurred:
                self.oom_occurred = True
        else:
            # Fallback for old format
            response_tokens = result
        state['total_generation_time'] += chunk_duration

        self.timing_data['generation_time'] = state['total_generation_time']
        self.timing_data['generation_chunk_time'] = chunk_duration

        if response_tokens is None:
            return (state['formatted_query'] if state['chunk_invocations'] == 1 else None), None

        decode_start = time.time()
        decoded_response = self.tokenizer.decode(
            response_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        response_text = f"(Video Time = {state['video_time']}s) Assistant:{decoded_response}"
        decode_time = time.time() - decode_start
        self.timing_data['decode_time'] = decode_time

        final_query = state['formatted_query']
        self.generation_state = None
        return final_query, response_text
    
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
    
    def _get_sample_tensor_from_cache(self, cache):
        """Extract a sample tensor from KV cache to check device, handling different cache types."""
        if cache is None:
            return None
        
        # Handle transformers DynamicCache and similar objects with iteration
        if hasattr(cache, '__iter__') and not isinstance(cache, (str, bytes)):
            try:
                for layer in cache:
                    if layer is None:
                        continue
                    # Check if layer is a tensor
                    if torch.is_tensor(layer):
                        return layer
                    # Check if layer is a dict
                    if isinstance(layer, dict):
                        for value in layer.values():
                            if torch.is_tensor(value):
                                return value
                    # Check if layer is iterable (tuple/list)
                    if hasattr(layer, '__iter__') and not isinstance(layer, (str, bytes)):
                        for item in layer:
                            if torch.is_tensor(item):
                                return item
            except (TypeError, StopIteration):
                pass
        
        # Handle direct tensor
        if torch.is_tensor(cache):
            return cache
        
        return None

    def _ensure_kv_on_device(self, past_key_values=None):
        """Ensure KV cache is on the correct device (GPU), moving from CPU if needed."""
        target = past_key_values if past_key_values is not None else self.past_key_values
        if target is None:
            return target

        # Get a sample tensor to check current device
        sample_tensor = self._get_sample_tensor_from_cache(target)
        if self.device == 'cuda:0':
            # Only move if we have a tensor and it's not on the target device
            if sample_tensor is not None and sample_tensor.device != torch.device(self.device):
                start = time.time()
                target = move_kv_cache_to_device(target, self.device)
                self._kv_reload_time += time.time() - start
        return target

    def _offload_kv_cache(self, past_key_values=None, non_blocking: bool = None):
        """Offload KV cache to CPU to save GPU memory."""
        target = past_key_values if past_key_values is not None else self.past_key_values
        if target is None:
            return target
        
        # Use configuration default if not specified
        if non_blocking is None:
            non_blocking = Config.ASYNC_KV_OFFLOAD
        
        # Get a sample tensor to check current device
        sample_tensor = self._get_sample_tensor_from_cache(target)
        
        # Only move if we have a tensor and it's not already on CPU
        if sample_tensor is not None and sample_tensor.device != torch.device('cpu'):
            start = time.time()
            target = move_kv_cache_to_device(target, 'cpu', non_blocking=non_blocking)
            # No synchronization here - let it complete asynchronously
            # torch.cuda.synchronize()
            self._kv_offload_time += time.time() - start
            self.timing_data['kv_offload_time'] = self._kv_offload_time
            
        return target

    def _initialize_generation_state(self, video_time, query):
        # Track trigger metadata
        if query is not None:
            self.response_triggers.append('query')
            self._current_query = query
        else:
            self.response_triggers.append('frame')
            self._current_query = None
        self.response_times.append(video_time)

        # Prepare initial token context
        if query is not None:
            formatted_query = f'(Video Time = {video_time}s) User: {query}'
            self.last_ids = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': query}],
                add_stream_query_prompt=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).to(self.device)
        else:
            formatted_query = None
            self.last_ids = self._added_stream_generation_ids

        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
            next_inputs_cpu = inputs_embeds.detach().cpu()

        self.generation_state = {
            'video_time': video_time,
            'formatted_query': formatted_query,
            'tokens': [],
            'tokens_generated': 0,
            'total_generation_time': 0.0,
            'next_inputs_embeds_cpu': next_inputs_cpu,
            'past_key_values_cpu': self.past_key_values,
            'finished': False,
            'chunk_invocations': 0,
        }

    def _execute_generation_chunk(self, state):
        chunk_size = self.generation_chunk_size or Config.INPLACE_OUTPUT_SIZE
        chunk_size = min(chunk_size, Config.INPLACE_OUTPUT_SIZE)
        if chunk_size <= 0:
            chunk_size = Config.INPLACE_OUTPUT_SIZE

        with torch.no_grad():
            next_inputs = state['next_inputs_embeds_cpu'].to(self.device)
            past_key_values = self._ensure_kv_on_device(state['past_key_values_cpu'])
            past_key_values = state['past_key_values_cpu']

            buffer = torch.zeros(1, chunk_size, dtype=torch.long, device=self.device)
            # handle OOM with try and except
            try:
                output_ids, past_key_values, next_inputs_embeds, finished = fast_greedy_generate(
                model=self.model,
                inputs_embeds=next_inputs,
                past_key_values=past_key_values,
                eos_token_id=self.eos_token_id,
                inplace_output_ids=buffer,
                max_new_tokens=chunk_size,
                    )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("Out of memory in execute generation chunk")
                    del past_key_values
                    del next_inputs
                    defragment_gpu_memory()
                    # mark the generation as finished
                    state['finished'] = True
                    all_tokens = torch.cat(state['tokens'], dim=1) if state['tokens'] else torch.empty((1, 0), dtype=torch.long)
                    # Return a special indicator for OOM
                    return all_tokens, True  # (tokens, oom_occurred)
                raise e

            self.timing_data['generation_video_time'] = state['video_time']

            if output_ids.numel() > 0:
                state['tokens'].append(output_ids.detach().cpu())
                state['tokens_generated'] += output_ids.size(1)
                self.last_ids = output_ids[:, -1:].to(self.device)
                # decode partial response
                self.texts_generated_previous = self.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            if next_inputs_embeds is not None:
                state['next_inputs_embeds_cpu'] = next_inputs_embeds.detach().cpu()
            else:
                state['next_inputs_embeds_cpu'] = None

            if finished or state['tokens_generated'] >= Config.INPLACE_OUTPUT_SIZE:
                state['finished'] = True
                all_tokens = torch.cat(state['tokens'], dim=1) if state['tokens'] else torch.empty((1, 0), dtype=torch.long)
                return all_tokens, False  # (tokens, oom_occurred)

        return None, False  # (tokens, oom_occurred)

    def _call_for_streaming(self):
        while self.frame_embeds_queue:
            # 1. if query is before next frame, response
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                self.trigger_method = 'prompt'
                return video_time, query
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            if self.past_key_values is None:
                self.last_ids = self._start_ids
            elif self.last_ids.numel() == 1 and int(self.last_ids.item()) == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)

            # MEASURE MODEL FORWARD PASS TIME (this is the main VLM computation)
            with torch.no_grad():
                model_forward_start = time.time()
                self.past_key_values = self._ensure_kv_on_device(self.past_key_values)
                frame_embeds = frame_embeds.to(self.device)
                inputs_embeds = torch.cat([
                    self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                    frame_embeds.view(1, -1, self.hidden_size),
                ], dim=1)
                del frame_embeds
                
                # Handle OOM in model forward pass
                try:
                    outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("Out of memory in call for streaming")
                        self.oom_occurred = True
                        # Free inputs_embeds and return None to indicate OOM
                        del inputs_embeds
                        del self.past_key_values
                        defragment_gpu_memory()
                        return None, None
                    raise e
                
                # Free inputs_embeds immediately after forward pass
                del inputs_embeds
                
                model_forward_time = time.time() - model_forward_start
                
                # Store the actual model forward time
                self.timing_data['model_forward_time'] = model_forward_time
                
                self.past_key_values = outputs.past_key_values
                
                # Extract logits and break reference to outputs immediately
                # Clone to create independent tensor, breaking reference to model buffers
                last_logits = outputs.logits[:, -1:].clone()
                del outputs  # Free outputs immediately to release model buffers
                
                # 2. if the same time, response after frame at that time
                if self.query_queue and video_time >= self.query_queue[0][0]:
                    video_time, query = self.query_queue.popleft()
                    self.trigger_method = 'prompt'
                    # Note: Response triggers are now tracked in call for response
                    del last_logits  # Clean up
                    return video_time, query
                
                # 3. if the next is frame but next is not interval, then response
                next_score = last_logits.softmax(dim=-1)
                frame_token_interval_score = next_score[:,:,self.frame_token_interval_id].item()
                
                # Track frame_token_interval_score for analysis
                self.frame_scores.append(frame_token_interval_score)
                self.frame_times.append(video_time)
                
                if frame_token_interval_score < self.frame_token_interval_threshold:
                    next_score[:,:,self.frame_token_interval_id].zero_()
                self.last_ids = next_score.argmax(dim=-1)
                
                # Free tensors immediately
                del next_score
                del last_logits
                torch.cuda.empty_cache()  # Force PyTorch to release memory to CUDA
                
                if self.last_ids.numel() == 1 and int(self.last_ids.item()) != self.frame_token_interval_id: 
                    # Note: Response triggers are now tracked in call for response
                    self.trigger_method = 'score'
                    return video_time, None
        
        return None, None

    def __call__(self):
        """Main call method that processes video and generates responses with timing."""
        self._kv_reload_time = 0.0
        self._kv_offload_time = 0.0
        self.timing_data['generation_chunk_time'] = 0.0
        self.timing_data['model_forward_time'] = 0.0
        video_time = None
        query = None
        response = None

        streaming_time = 0.0
        if self.frame_embeds_queue:
            streaming_start = time.time()
            result = self._call_for_streaming()
            streaming_time = time.time() - streaming_start
            self.timing_data['streaming_time'] = streaming_time
            
            if result is None or result == (None, None):
                return None, None
            
            video_time, query = result
        elif self.generation_state is None and self.query_queue:
            video_time, query = self.query_queue.popleft()


        needs_generation = (
            query is not None
            or self.generation_state is not None
            or video_time is not None
        )
        
        if needs_generation:
            target_time = video_time if video_time is not None else (
                self.generation_state['video_time'] if self.generation_state else video_time
            )
            query, response = self._call_for_response(target_time, query)

        # Record timing metadata for diagnostics
        kv_cache_mb = calculate_kv_cache_memory_mb(self.past_key_values)
        transfer_record = {
            'kv_cache_mb': kv_cache_mb,
            'offload_time': self._kv_offload_time,
            'reload_time': self._kv_reload_time,
        }
        self.kv_transfer_metrics.append(transfer_record)

        self.timing_data['kv_cache_mb'] = kv_cache_mb
        self.timing_data['kv_offload_time'] = self._kv_offload_time
        self.timing_data['kv_reload_time'] = self._kv_reload_time

        return query, response

    def offload_kv_cache(self, non_blocking: bool = None):
        """Offload KV cache to CPU, using asynchronous transfer by default."""
        self.past_key_values = self._offload_kv_cache(self.past_key_values, non_blocking=non_blocking)
        if self.generation_state is not None:
            self.generation_state['past_key_values_cpu'] = self.past_key_values
    
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
    
    def get_frame_features_data(self):
        """Get frame difference features data from all conversations."""
        all_features = []
        for conversation_id, features_list in self.frame_features_data.items():
            all_features.extend([entry.copy() for entry in features_list])
        return all_features
    
    def update_frame_response_length(self, frame_idx, response_length, conversation_id):
        """Update the response length for a specific frame in a specific conversation."""
        if conversation_id not in self.frame_features_data:
            return  # Conversation not found
        
        for entry in self.frame_features_data[conversation_id]:
            if entry['frame_idx'] == frame_idx:
                entry['response_length'] += response_length
                break


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
    data_source = Config.DATA_SOURCE  # default
    num_videos = Config.DEFAULT_NUM_VIDEOS
    custom_start_times = None

    if '--data_source' in sys.argv:
        idx = sys.argv.index('--data_source')
        if idx + 1 < len(sys.argv):
            data_source = sys.argv[idx + 1]
            # Remove these args to avoid conflicts with parse_args()
            sys.argv.pop(idx)
            sys.argv.pop(idx)

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

    if '--config_id' in sys.argv:
        idx = sys.argv.index('--config_id')
        if idx + 1 < len(sys.argv):
            config_id = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        else:
            print(f"⚠️  Invalid --config_id value '{config_id}', falling back to {config_id}")

    if '--iteration' in sys.argv:
        idx = sys.argv.index('--iteration')
        if idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            iteration = max(1, int(value))
            sys.argv.pop(idx)
            sys.argv.pop(idx)
    
    # Parse the main args
    args = parse_args()
    
    # Add custom args to args object
    args.data_source = data_source
    args.num_videos = num_videos
    args.iteration = iteration
    
    # modify config based on config_id
    config_id = getattr(args, 'config_id', 'base')
    if config_id == 'base':
        pass
    elif config_id == 'round_robin_m':
        Config.SCHEDULING_METHOD = 'round_robin'
    elif config_id == 'round_robin_2':
        Config.SCHEDULING_METHOD = 'round_robin'
        Config.GENERATION_CHUNK_SIZE = 2
    elif config_id == 'random_m':
        Config.SCHEDULING_METHOD = 'random'
    elif config_id == 'random_2':
        Config.SCHEDULING_METHOD = 'random'
        Config.GENERATION_CHUNK_SIZE = 2
    else:
        print(f"⚠️  Invalid --config_id value '{config_id}', falling back to {config_id}")

    # create output directory based on dataset, num_videos, config_id, and iteration
    output_dir = f'figures/{data_source}/{num_videos}/{config_id}/iteration_{iteration}'
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Starting Streaming Evaluation")
    print("=" * 50)
    
    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()  # Explicitly set to eval mode to disable dropout, etc.
    
    print(f"🔧 Device: {device}")
    print(f"🤖 Model loaded successfully")
    
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
    
    # print(f"📊 Dataset loaded: {len(dataset)} conversation turn(s) from validation set")
    
    # Create ground truth word count analysis (skipped for speed)
    # print(f"\n📊 Creating ground truth word count analysis...")
    # create_gt_word_count_analysis(data_source)
    
    print("-" * 50)

    # Regular single-threshold evaluation
    default_start_times = custom_start_times
    if default_start_times is None:
        default_start_times = [0.0] * num_videos
    
    # Evaluate more conversations for better coverage
    try:
        # Use config default number of videos
        num_videos = getattr(args, 'num_videos', Config.DEFAULT_NUM_VIDEOS)
        print(f"💬 Processing {num_videos} conversation{'s' if num_videos > 1 else ''} for PPL analysis...")
        random.seed(42 + args.iteration)  # For reproducibility

        overall_summary = streaming_evaluate_conversations(
            model,
            tokenizer,
            dataset,
            device,
            num_conversations=num_videos,
            random_selection=True,
            data_source=data_source,
            conversation_start_times=default_start_times
        )
        
        # # Create timing analysis
        # conversation_timings = [r for r in results if 'visual_embedding_time' in r]
        
        # if conversation_timings:
        #     # Create individual conversation timing plots
        #     create_individual_conversation_timing_plots(conversation_timings, data_source=data_source)
        
        # Create PPL analysis visualization (includes PPL over time visualization)
        # print(f"\n📊 Creating dual PPL analysis...")
        # create_dual_ppl_frame_visualization(results, data_source=data_source)

        # Create response length distribution analysis
        # print(f"\n📊 Creating response length distribution analysis...")
        # create_response_length_distribution_analysis(overall_summary, data_source=data_source)

        # print("\n📊 Creating processor timeline...")
        # create_processor_timeline(overall_summary, data_source=data_source)

        # print("\n📊 Creating memory usage analysis...")
        # create_memory_visualization(overall_summary, data_source=data_source)
        
        # Create aggregated metrics visualization
        # print(f"\n📊 Creating aggregated metrics visualization...")
        create_aggregated_metrics_visualization(overall_summary, output_dir=output_dir, data_source=data_source)
        
        # Create time per token analysis (skipped for speed)
        # print(f"\n📊 Creating time per token analysis...")
        # create_time_per_token_analysis(results, data_source=data_source)
        
        # Create generated word count analysis (skipped for speed)
        # print(f"\n📊 Creating generated word count analysis...")
        # create_generated_word_count_analysis(results, data_source=data_source)

        # save overall_summary to json in output_dir
        with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
            json.dump(overall_summary, f)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()

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

def calculate_ppl_for_response(model, tokenizer, conversation, video_tensor, device, data_source='goalstep', use_visual=True, custom_threshold=None, frame_index=None):
    """Calculate PPL for a single response using the proper conversation format."""
    try:
        
        # Tokenize the conversation
        input_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
        
        # Get learn ranges for this conversation
        learn_ranges = tokenizer.get_learn_ranges(conversation)
        
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
                defragment_gpu_memory()
                
                # Use the same frame-by-frame approach as VLM streaming
                # Process only the first frame to minimize memory usage
                if frame_index is not None:
                    frame_range = range(frame_index, frame_index + 1)
                else:
                    frame_range = range(0, 1)  # Only first frame
                
                # Move frame from CPU to GPU one by one (like VLM streaming)
                single_frame_cpu = video_tensor[frame_range]  # Shape: [1, 3, H, W] on CPU
                single_frame_gpu = single_frame_cpu.to(device)  # Move to GPU
            
                # Use custom threshold if provided, otherwise use data source default
                threshold = Config.STREAMING_THRESHOLD
                
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
                
                # Extract PPL for this response
                lm_ppl, _, _, _ = raw_metrics.tolist()
                return float(lm_ppl)
            else:
                return None
        else:
            # No visual PPL calculation removed for simplification
            return None
            
    except Exception as e:
        assert 'CUDA out of memory' in str(e), f"PPL calculation error: {e}"
        print(f"CUDA out of memory. Skip the remaining responses of this conversation")
        defragment_gpu_memory()
        return None
        # print(f"PPL calculation error: {e}")
        # traceback.print_exc()
        # return None

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
            if turn['time'] <= gt_time + 1e-6:  # Only previous responses
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
        # print(f"📊 Turn", turn)
        if turn['time'] < gt_time:  # Only previous responses
            # Extract user prompt from turn
            turn_user_prompt = turn.get('user_prompt', 'Please help with the video analysis.')
            # Include ALL responses, not just the ones with user queries
            # This ensures VLM context changes with threshold
            if turn_user_prompt != "Frame processing":
                # Only add user prompt if last added was not user
                if last_added_role != 'user':
                    if 'User:' in turn_user_prompt:
                        turn_user_prompt = turn_user_prompt.split('User:', 1)[1].strip()
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
            # print(f"📊 Assistant Content: {assistant_content}")
            conversation.append({'role': 'assistant', 'content': assistant_content, 'learn': True})
            last_added_role = 'assistant'
    
    # Add current user prompt and GT response
    if last_added_role != 'user':
        conversation.append({'role': 'user', 'content': user_prompt})
    conversation.append({'role': 'assistant', 'content': gt_content, 'learn': True})
    
    return conversation

def create_dual_ppl_frame_visualization(results, output_dir="figures", data_source="goalstep"):
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
    # print(f"\n📊 DUAL PPL ANALYSIS SUMMARY ({data_source.upper()}):")
    # print(f"   • Total responses analyzed: {len(all_gt_prefix_visual)}")
    # print(f"   • GT Prefix PPL: {np.mean(all_gt_prefix_visual):.3f} ± {np.std(all_gt_prefix_visual):.3f}")
    # print(f"   • VLM Prefix PPL: {np.mean(all_vlm_prefix_visual):.3f} ± {np.std(all_vlm_prefix_visual):.3f}")
    
    # Calculate differences and correlation
    # if len(all_gt_prefix_visual) == len(all_vlm_prefix_visual):
    #     differences = [vlm - gt for gt, vlm in zip(all_gt_prefix_visual, all_vlm_prefix_visual)]
    #     mean_diff = np.mean(differences)
    #     std_diff = np.std(differences)
    #     correlation = np.corrcoef(all_gt_prefix_visual, all_vlm_prefix_visual)[0, 1]
    #     print(f"   • Average difference (VLM - GT): {mean_diff:.3f} ± {std_diff:.3f}")
    #     print(f"   • Correlation coefficient: {correlation:.3f}")

def create_ppl_over_time_visualization(video_data, output_dir="figures", data_source="goalstep"):
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

def calculate_metrics(model, tokenizer, video_tensor, normalized_conversation, generated_turns, device, data_source='goalstep'):
    """Calculate metrics exactly like evaluate.py using stream_evaluate and compute_metrics."""
    
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
    
    
    # print(f"📊 Calculating PPL for {len(ground_truth_responses)} ground truth responses...")
    
    for i, gt_response in enumerate(ground_truth_responses):
        
        gt_content = gt_response['content']
        gt_time = gt_response['time']

        # convert time to frame index
        frame_index = int(gt_time * Config.FRAME_FPS)

        # skip if frame index is out of range
        if frame_index >= video_tensor.shape[0]:
            continue
        
        # Find the user prompt that this ground truth response was answering
        user_prompt = "Please help with the video analysis."  # Default fallback
        if normalized_conversation:
            # Find the most recent user prompt before this response time
            for conv_turn in reversed(normalized_conversation):
                if conv_turn['role'] == 'user' and conv_turn['time'] <= gt_time + 1e-6:
                    user_prompt = conv_turn['content']
                    break
        # print(f"📊 GT Time: {gt_time} GT Content: {gt_content} User Prompt: {user_prompt}")
        
        # 1. PPL with GT prefix (golden context) - WITH VISUAL
        gt_conversation = create_conversation_with_gt_prefix(
            normalized_conversation, gt_time, user_prompt, gt_content
        )
        # print(f"📊 GT Conversation: {gt_conversation}")
        ppl_gt_prefix_visual = calculate_ppl_for_response(model, tokenizer, gt_conversation, video_tensor, device, data_source, use_visual=True, custom_threshold=None, frame_index=frame_index)
        
        # 2. PPL with VLM prefix (actual generated responses as context) - WITH VISUAL
        vlm_conversation = create_conversation_with_vlm_prefix(
            generated_turns, gt_time, user_prompt, gt_content
        )
        # print(f"📊 VLM Conversation: {vlm_conversation}")
        ppl_vlm_prefix_visual = calculate_ppl_for_response(model, tokenizer, vlm_conversation, video_tensor, device, data_source, use_visual=True, custom_threshold=None, frame_index=frame_index)
                    
        if ppl_gt_prefix_visual is None or ppl_vlm_prefix_visual is None:
            print(f"🚨 OOM occurred for conversation {i}")
            print(f"📊 Video Tensor Shape: {video_tensor.shape}")
            print(f"📊 Frame Index: {frame_index}")
            print(f"📊 GT Content: {gt_content}")
            print(f"📊 GT Time: {gt_time}")
            print(f"📊 User Prompt: {user_prompt}")
            print(f"📊 GT Conversation: {gt_conversation}")
            print(f"📊 VLM Conversation: {vlm_conversation}")
            break
        
        gt_ppls_gt_prefix_visual.append(ppl_gt_prefix_visual)
        gt_ppls_vlm_prefix_visual.append(ppl_vlm_prefix_visual)
            
        # Clean up GPU memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up memory after each response to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate average PPLs for all four contexts
    # avg_gt_ppl_gt_prefix_visual = sum(gt_ppls_gt_prefix_visual) / len(gt_ppls_gt_prefix_visual)
    if gt_ppls_vlm_prefix_visual:
        avg_gt_ppl_vlm_prefix_visual = sum(gt_ppls_vlm_prefix_visual) / len(gt_ppls_vlm_prefix_visual)
    else:
        avg_gt_ppl_vlm_prefix_visual = 0.0
    # print(f"📊 GT Prefix PPL (Visual): {len(gt_ppls_gt_prefix_visual)} responses, avg PPL: {avg_gt_ppl_gt_prefix_visual:.3f}")
    # print(f"📊 GT Prefix PPL (Visual) range: {min(gt_ppls_gt_prefix_visual):.3f} - {max(gt_ppls_gt_prefix_visual):.3f}")
    # print(f"📊 VLM Prefix PPL (Visual): {len(gt_ppls_vlm_prefix_visual)} responses, avg PPL: {avg_gt_ppl_vlm_prefix_visual:.3f}")
    # print(f"📊 VLM Prefix PPL (Visual) range: {min(gt_ppls_vlm_prefix_visual):.3f} - {max(gt_ppls_vlm_prefix_visual):.3f}")
    
    return {
        'lm_ppl': avg_gt_ppl_vlm_prefix_visual,
    'fluency': 1.0,  # Will be overridden by actual fluency calculation
        'ppl_data': {
        'gt_ppls_gt_prefix_visual': gt_ppls_gt_prefix_visual,
        'gt_ppls_vlm_prefix_visual': gt_ppls_vlm_prefix_visual,
            'generated_responses': len(generated_turns),
            'total_gt_responses': len(ground_truth_responses)
        }
    }

def create_aggregated_metrics_visualization(overall_summary, output_dir="figures", data_source="goalstep"):
    """Create aggregated metrics visualization with 4 vertical bar plots in scientific style."""
    
    results = overall_summary['results']
    onthefly_buffer_data = overall_summary['onthefly_buffer_data']
    scheduling_data = overall_summary['scheduling_data']
    all_memory_data = overall_summary['all_memory_data']

    os.makedirs(output_dir, exist_ok=True)

    buffer_data = {}
    for cid, state in onthefly_buffer_data.items():
        buffer_data[cid] = {
            'listening': {
                'times': state['times'],
                'values': state['values'],
                'rebuffer_times': state['rebuffer_times'],
                'rebuffer_values': state['rebuffer_values'],
                'final_time': state['last_update_time'],
                'total_rebuffer': state['total_rebuffer'],
                'processing_delays': state['processing_delays'],
                'queuing_delays': state['queuing_delays'],
                'networking_delays': state['networking_delays'],
            },
            'conversation_id': cid
        }
    
    # Extract data from all conversations
    vlm_ppls = []
    gt_ppls = []
    listening_rebuffering_times = []
    fluencies = []
    response_latencies = []
    
    # Process each conversation result
    for result in results:
        # 1. VLM Perplexity data
        if 'ppl_data' in result and 'gt_ppls_vlm_prefix_visual' in result['ppl_data']:
            vlm_ppls.extend(result['ppl_data']['gt_ppls_vlm_prefix_visual'])
        
        # 2. GT Perplexity data (for reference line)
        if 'ppl_data' in result and 'gt_ppls_gt_prefix_visual' in result['ppl_data']:
            gt_ppls.extend(result['ppl_data']['gt_ppls_gt_prefix_visual'])
        
        # 3. Fluency data
        if 'fluency' in result:
            fluencies.append(result['fluency'])
        
        # 4. Response latency data (calculated in aggregate metrics)
        if 'generated_turns' in result and len(result['generated_turns']) > 0:
            # Calculate per-response latency for this conversation
            response_time = result.get('response_time', 0)
            response_latencies.append(response_time)
        
        # 5. Rebuffering time data - will be extracted from buffer_data after processing all results
    
    # Extract rebuffering data from buffer_data (listening mode only)
    if buffer_data:
        for cid, conversation_buffer in buffer_data.items():
            listening_traj = conversation_buffer.get('listening', {})
            
            if 'rebuffer_values' in listening_traj and listening_traj['rebuffer_values']:
                final_listening_rebuffer = listening_traj['rebuffer_values'][-1]
                listening_rebuffering_times.append(final_listening_rebuffer)
    
    # Calculate aggregate statistics
    vlm_ppl_mean = np.mean(vlm_ppls) if vlm_ppls else 0.0
    vlm_ppl_std = np.std(vlm_ppls) if vlm_ppls else 0.0
    gt_ppl_mean = np.mean(gt_ppls) if gt_ppls else 0.0
    
    listening_rebuffer_mean = np.mean(listening_rebuffering_times) if listening_rebuffering_times else 0.0
    listening_rebuffer_std = np.std(listening_rebuffering_times) if listening_rebuffering_times else 0.0
    
    fluency_mean = np.mean(fluencies) if fluencies else 0.0
    fluency_std = np.std(fluencies) if fluencies else 0.0
    
    latency_mean = np.mean(response_latencies) if response_latencies else 0.0
    latency_std = np.std(response_latencies) if response_latencies else 0.0
    
    # Set scientific style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    })
    
    # Create figure with 6 subplots in 1 row
    fig, axes = plt.subplots(1, 6, figsize=(30, 6))
    fig.suptitle(f'Aggregated Performance Metrics - {data_source.title()}', fontsize=14, fontweight='bold', y=0.95)
    
    # Define colors and positions
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    x_pos = [0, 1, 2, 3]
    labels = ['VLM PPL', 'Rebuffering', 'Fluency', 'Latency']
    
    # 3. Frame Timing Analysis (3rd position)
    ax_timing = axes[2]
    
    # Calculate frame timing statistics from results
    visual_embedding_times = []
    model_forward_times = []
    generation_times = []
    kv_offload_times = []
    kv_reload_times = []
    total_processing_times = []
    total_sending_times = []
    total_receiving_times = []

    # Create timing analysis
    conversation_timings = [r for r in results if 'visual_embedding_time' in r]

    for conversation_timing in conversation_timings:
        frame_count = len(conversation_timing['frame_processing_times'])
        visual_per_frame = conversation_timing['visual_embedding_time'] / frame_count
        model_per_frame = conversation_timing['model_forward_time'] / frame_count
        generation_per_response = conversation_timing['generation_time'] / frame_count
        kv_offload_time = conversation_timing['kv_offload_time'] / frame_count
        kv_reload_time = conversation_timing['kv_reload_time'] / frame_count
        total_processing_time = conversation_timing['total_processing_time'] / frame_count
        total_sending_time = conversation_timing['total_sending_time'] / frame_count

        visual_embedding_times.append(visual_per_frame*1000)
        model_forward_times.append(model_per_frame*1000)
        generation_times.append(generation_per_response*1000)
        total_processing_times.append(total_processing_time*1000)
        kv_offload_times.append(kv_offload_time*1000)
        kv_reload_times.append(kv_reload_time*1000)
        total_sending_times.append(total_sending_time*1000)

    # Calculate means and standard deviations
    visual_embedding_mean = np.mean(visual_embedding_times) if visual_embedding_times else 0.0
    visual_embedding_std = np.std(visual_embedding_times) if visual_embedding_times else 0.0
    model_forward_mean = np.mean(model_forward_times) if model_forward_times else 0.0
    model_forward_std = np.std(model_forward_times) if model_forward_times else 0.0
    generation_mean = np.mean(generation_times) if generation_times else 0.0
    generation_std = np.std(generation_times) if generation_times else 0.0
    total_processing_mean = np.mean(total_processing_times) if total_processing_times else 0.0
    total_processing_std = np.std(total_processing_times) if total_processing_times else 0.0
    kv_offload_mean = np.mean(kv_offload_times) if kv_offload_times else 0.0
    kv_offload_std = np.std(kv_offload_times) if kv_offload_times else 0.0
    kv_reload_mean = np.mean(kv_reload_times) if kv_reload_times else 0.0
    kv_reload_std = np.std(kv_reload_times) if kv_reload_times else 0.0
    total_sending_mean = np.mean(total_sending_times) if total_sending_times else 0.0
    total_sending_std = np.std(total_sending_times) if total_sending_times else 0.0

    # Create grouped bars for frame timing metrics
    x_positions = [0, 1, 2, 3, 4, 5, 6]
    values = [visual_embedding_mean, model_forward_mean, generation_mean, total_processing_mean, kv_offload_mean, kv_reload_mean, total_sending_mean]
    errors = [visual_embedding_std, model_forward_std, generation_std, total_processing_std, kv_offload_std, kv_reload_std, total_sending_std]
    labels_timing = ['Prefilling', 'Scoring', 'Gen', 'Total', 'Offload', 'Reload', 'Network']
    colors_timing = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#9999ff', '#ff99ff', '#9999ff']
    
    bars_timing = ax_timing.bar(x_positions, values, yerr=errors,
                               color=colors_timing, alpha=0.8, capsize=4, width=0.6,
                               edgecolor='black', linewidth=0.5)
    ax_timing.set_ylabel('Time (ms)')
    ax_timing.set_xticks(x_positions)
    ax_timing.set_xticklabels(labels_timing, fontsize=8)
    ax_timing.grid(True, alpha=0.3, axis='y')
    ax_timing.spines['top'].set_visible(False)
    ax_timing.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (val, err) in enumerate(zip(values, errors)):
        ax_timing.text(i, val + err + 0.01, f'{val:.1f}±{err:.1f}', 
                      ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # 2. Time Metrics - Rebuffering, Processing, Queuing, and Latency (top-right)
    ax2 = axes[0]
    
    # Calculate delay statistics from buffer_data
    processing_delays = []
    queuing_delays = []
    networking_delays = []
    if buffer_data:
        for cid, conversation_buffer in buffer_data.items():
            listening_traj = conversation_buffer.get('listening', {})
            if 'processing_delays' in listening_traj and listening_traj['processing_delays']:
                processing_delays.append(listening_traj['processing_delays'][-1])
            if 'queuing_delays' in listening_traj and listening_traj['queuing_delays']:
                queuing_delays.append(listening_traj['queuing_delays'][-1])
            if 'networking_delays' in listening_traj and listening_traj['networking_delays']:
                networking_delays.append(listening_traj['networking_delays'][-1])
    processing_mean = np.mean(processing_delays) if processing_delays else 0.0
    processing_std = np.std(processing_delays) if processing_delays else 0.0
    queuing_mean = np.mean(queuing_delays) if queuing_delays else 0.0
    queuing_std = np.std(queuing_delays) if queuing_delays else 0.0
    networking_mean = np.mean(networking_delays) if networking_delays else 0.0
    networking_std = np.std(networking_delays) if networking_delays else 0.0

    # Create grouped bars for latency, rebuffering, processing, and queuing delays
    x_positions = [0, 1, 2, 3, 4]
    values = [latency_mean, listening_rebuffer_mean, processing_mean, queuing_mean, networking_mean]
    errors = [latency_std, listening_rebuffer_std, processing_std, queuing_std, networking_std]
    labels_delays = ['TTFT', 'Rebuffering', 'Processing', 'Queuing', 'Networking']
    colors_delays = [colors[3], colors[1], '#e377c2', '#17becf', '#1f77b4']
    
    bars2 = ax2.bar(x_positions, values, yerr=errors,
                    color=colors_delays, alpha=0.8, capsize=4, width=0.6,
                    edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Time (s)')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels_delays)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (val, err) in enumerate(zip(values, errors)):
        ax2.text(i, val + err + 0.05, f'{val:.2f}±{err:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. VLM Perplexity with GT reference (4th position)
    ax1 = axes[4]
    bars1 = ax1.bar([0], [vlm_ppl_mean], yerr=[vlm_ppl_std], 
                    color=colors[0], alpha=0.8, capsize=4, width=0.4, 
                    edgecolor='black', linewidth=0.5)
    ax1.axhline(gt_ppl_mean, color='red', linestyle='--', linewidth=1.5, 
                label=f'GT: {gt_ppl_mean:.2f}')
    ax1.set_ylabel('Perplexity')
    ax1.set_xticks([0])
    ax1.set_xticklabels(['VLM'])
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    ax1.text(0, vlm_ppl_mean + vlm_ppl_std + 0.2, f'{vlm_ppl_mean:.2f}±{vlm_ppl_std:.2f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Memory Usage Analysis (5th position)
    ax_memory = axes[3]
    
    # Calculate memory statistics from all_memory_data
    total_gpu_memory = []
    model_memory = []
    kv_cache_memory = []
    activation_memory = []
    other_memory = []
    cpu_memory_growth = []
    
    # Extract memory data from all conversations
    for conversation_key, memory_data in all_memory_data.items():
        if 'memory_usage' in memory_data and memory_data['memory_usage']:
            # Peak memory usage (like in memory_usage_analysis and memory_comparison)
            total_gpu_memory.append(max(memory_data['memory_usage']))
            model_memory.append(max(memory_data['model_memory']))
            kv_cache_memory.append(max(memory_data['kv_cache_memory']))
            activation_memory.append(max(memory_data['activation_memory']))
            other_memory.append(max(memory_data['other_memory']))
            # CPU memory growth (peak growth)
            cpu_growth_data = memory_data.get('cpu_memory_growth', [])
            cpu_memory_growth.append(max(cpu_growth_data) if cpu_growth_data else 0.0)
    
    # Calculate means and standard deviations
    total_gpu_mean = np.mean(total_gpu_memory) if total_gpu_memory else 0.0
    total_gpu_std = np.std(total_gpu_memory) if total_gpu_memory else 0.0
    model_memory_mean = np.mean(model_memory) if model_memory else 0.0
    model_memory_std = np.std(model_memory) if model_memory else 0.0
    kv_cache_memory_mean = np.mean(kv_cache_memory) if kv_cache_memory else 0.0
    kv_cache_memory_std = np.std(kv_cache_memory) if kv_cache_memory else 0.0
    activation_memory_mean = np.mean(activation_memory) if activation_memory else 0.0
    activation_memory_std = np.std(activation_memory) if activation_memory else 0.0
    other_memory_mean = np.mean(other_memory) if other_memory else 0.0
    other_memory_std = np.std(other_memory) if other_memory else 0.0
    cpu_memory_growth_mean = np.mean(cpu_memory_growth) if cpu_memory_growth else 0.0
    cpu_memory_growth_std = np.std(cpu_memory_growth) if cpu_memory_growth else 0.0
    
    # Calculate combined memory (KV cache + activation + other)
    combined_memory = []
    for i in range(len(kv_cache_memory)):
        combined = kv_cache_memory[i] + activation_memory[i] + other_memory[i]
        combined_memory.append(combined)
    combined_memory_mean = np.mean(combined_memory) if combined_memory else 0.0
    combined_memory_std = np.std(combined_memory) if combined_memory else 0.0
    
    # Create grouped bars for memory metrics
    x_positions = [0, 1, 2, 3, 4, 5, 6]
    values = [total_gpu_mean, model_memory_mean, kv_cache_memory_mean, activation_memory_mean, other_memory_mean, combined_memory_mean, cpu_memory_growth_mean]
    errors = [total_gpu_std, model_memory_std, kv_cache_memory_std, activation_memory_std, other_memory_std, combined_memory_std, cpu_memory_growth_std]
    labels_memory = ['Total\nGPU', 'Model\nParams', 'KV\nCache', 'Activations', 'Other\nGPU', 'Combined\nDynamic', 'CPU\nGrowth']
    colors_memory = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9500', '#ff9ff3']
    
    bars_memory = ax_memory.bar(x_positions, values, yerr=errors,
                               color=colors_memory, alpha=0.8, capsize=4, width=0.6,
                               edgecolor='black', linewidth=0.5)
    ax_memory.set_ylabel('Memory (MB)')
    ax_memory.set_yscale('log')  # Use log scale for better visualization of wide range
    ax_memory.set_xticks(x_positions)
    ax_memory.set_xticklabels(labels_memory, fontsize=8)
    ax_memory.grid(True, alpha=0.3, axis='y')
    ax_memory.spines['top'].set_visible(False)
    ax_memory.spines['right'].set_visible(False)
    
    # Add value labels (adjusted for log scale)
    for i, (val, err) in enumerate(zip(values, errors)):
        # For log scale, multiply by a small factor instead of adding
        label_y = val * 1.1 if val > 0 else 1.0
        ax_memory.text(i, label_y, f'{val:.0f}±{err:.0f}', 
                      ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # 6. Fluency (6th position)
    ax3 = axes[5]
    bars3 = ax3.bar([0], [fluency_mean], yerr=[fluency_std],
                    color=colors[2], alpha=0.8, capsize=4, width=0.4,
                    edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.0)
    ax3.set_xticks([0])
    ax3.set_xticklabels(['Fluency'])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels
    ax3.text(0, fluency_mean + fluency_std + 0.02, f'{fluency_mean:.3f}±{fluency_std:.3f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Scheduling Metrics (bottom-right)
    ax4 = axes[1]
    
    # Calculate scheduling statistics from scheduling_data
    scheduling_scores = []
    selected_lowest_buffer = []
    selected_nonzero = []
    selected_ending = []
    
    if scheduling_data:
        scheduling_scores = scheduling_data.get('selected_score', [])
        selected_lowest_buffer = scheduling_data.get('selected_lowest', [])
        selected_nonzero = scheduling_data.get('selected_increment', [])
        selected_ending = scheduling_data.get('selected_ending', [])
    
    # Calculate means and standard deviations
    scheduling_score_mean = np.mean(scheduling_scores) if scheduling_scores else 0.0
    selected_lowest_mean = np.mean(selected_lowest_buffer) if selected_lowest_buffer else 0.0
    selected_nonzero_mean = np.mean(selected_nonzero) if selected_nonzero else 0.0
    selected_ending_mean = np.mean(selected_ending) if selected_ending else 0.0
    
    # Create grouped bars for scheduling metrics
    x_positions = [0, 1, 2, 3]
    values = [scheduling_score_mean, selected_lowest_mean, selected_nonzero_mean, selected_ending_mean]
    labels_scheduling = ['Scheduling\nScore', 'Selected\nLowest Buffer', 'Selected\nNonzero', 'Selected\nEnding']
    colors_scheduling = ['#9467bd', '#ff7f0e', '#2ca02c', '#8c564b']
    
    bars4 = ax4.bar(x_positions, values, 
                    color=colors_scheduling, alpha=0.8, capsize=4, width=0.6,
                    edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Count/Score')
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(labels_scheduling, fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add value labels
    for i, val in enumerate(values):
        ax4.text(i, val + 0.05, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'aggregated_metrics_{data_source}.png')
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Aggregated metrics visualization saved to: {output_path}")
    

def create_response_length_distribution_analysis(overall_summary, output_dir="figures", data_source="goalstep"):
    """Create analysis and visualization of response length distribution from generated_turns."""

    results = overall_summary['results']
    if not results:
        print("⚠️ No results data available for response length analysis")
        return
    
    # Extract all response lengths from generated_turns across all conversations
    all_response_lengths = []
    conversation_response_lengths = {}
    
    for result in results:
        conversation_id = result['conversation_id']
        generated_turns = result.get('generated_turns', [])
        
        conv_lengths = []
        for turn in generated_turns:
            if turn.get('text'):
                # Count words in the response text
                word_count = len(re.findall(r"\b\w+\b", turn['text']))
                all_response_lengths.append(word_count)
                conv_lengths.append(word_count)
        
        if conv_lengths:
            conversation_response_lengths[conversation_id] = conv_lengths
    
    if not all_response_lengths:
        print("⚠️ No response text found in generated_turns")
        return
    
    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall response length distribution (histogram)
    ax1 = plt.subplot(2, 4, 1)
    plt.hist(all_response_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Response Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Response Lengths')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_length = np.mean(all_response_lengths)
    median_length = np.median(all_response_lengths)
    std_length = np.std(all_response_lengths)
    plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.1f}')
    plt.axvline(median_length, color='orange', linestyle='--', label=f'Median: {median_length:.1f}')
    plt.legend()
    
    # 2. Box plot of response lengths by conversation
    ax2 = plt.subplot(2, 4, 2)
    conversation_ids = list(conversation_response_lengths.keys())
    conversation_ids.sort()
    
    lengths_by_conversation = []
    for conv_id in conversation_ids:
        lengths_by_conversation.append(conversation_response_lengths[conv_id])
    
    plt.boxplot(lengths_by_conversation)
    plt.xlabel('Conversation')
    plt.ylabel('Response Length (words)')
    plt.title('Response Length Distribution by Conversation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = plt.subplot(2, 4, 3)
    sorted_lengths = np.sort(all_response_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cumulative, 'b-', linewidth=2)
    plt.xlabel('Response Length (words)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Response Lengths')
    plt.grid(True, alpha=0.3)
    
    # Add percentiles
    p50 = np.percentile(all_response_lengths, 50)
    p90 = np.percentile(all_response_lengths, 90)
    p95 = np.percentile(all_response_lengths, 95)
    plt.axvline(p50, color='red', linestyle='--', alpha=0.7, label=f'50th: {p50:.1f}')
    plt.axvline(p90, color='orange', linestyle='--', alpha=0.7, label=f'90th: {p90:.1f}')
    plt.axvline(p95, color='purple', linestyle='--', alpha=0.7, label=f'95th: {p95:.1f}')
    plt.legend()
    
    # 4. Response length vs time (scatter plot)
    ax4 = plt.subplot(2, 4, 4)
    response_times = []
    response_lengths_time = []
    
    for result in results:
        generated_turns = result.get('generated_turns', [])
        for turn in generated_turns:
            if turn.get('text'):
                word_count = len(re.findall(r"\b\w+\b", turn['text']))
                response_times.append(turn.get('time', 0))
                response_lengths_time.append(word_count)
    
    if response_times and response_lengths_time:
        plt.scatter(response_times, response_lengths_time, alpha=0.6, color='green')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Response Length (words)')
        plt.title('Response Length Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(response_times) > 1:
            z = np.polyfit(response_times, response_lengths_time, 1)
            p = np.poly1d(z)
            plt.plot(response_times, p(response_times), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
            plt.legend()
    
    # 5. Log-scale histogram (for better visualization if there's wide range)
    ax5 = plt.subplot(2, 4, 5)
    plt.hist(all_response_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Response Length (words)')
    plt.ylabel('Frequency')
    plt.title('Response Length Distribution (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 6. Response length vs generation time
    ax6 = plt.subplot(2, 4, 6)
    generation_times = []
    response_lengths_gen = []
    
    for result in results:
        generated_turns = result.get('generated_turns', [])
        for turn in generated_turns:
            if turn.get('text'):
                word_count = len(re.findall(r"\b\w+\b", turn['text']))
                gen_time = turn.get('generation_time', 0)
                generation_times.append(gen_time)
                response_lengths_gen.append(word_count)
    
    if generation_times and response_lengths_gen:
        plt.scatter(generation_times, response_lengths_gen, alpha=0.6, color='purple')
        plt.xlabel('Generation Time (seconds)')
        plt.ylabel('Response Length (words)')
        plt.title('Response Length vs Generation Time')
        plt.grid(True, alpha=0.3)
        
        # Add correlation
        if len(generation_times) > 1:
            correlation = np.corrcoef(generation_times, response_lengths_gen)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 7. Violin plot for better distribution visualization
    ax7 = plt.subplot(2, 4, 7)
    if len(conversation_ids) > 1:
        data_for_violin = [conversation_response_lengths[conv_id] for conv_id in conversation_ids]
        parts = plt.violinplot(data_for_violin, positions=range(1, len(conversation_ids) + 1), 
                              showmeans=True, showmedians=True)
        plt.xlabel('Conversation')
        plt.ylabel('Response Length (words)')
        plt.title('Response Length Distribution (Violin Plot)')
        plt.xticks(range(1, len(conversation_ids) + 1), [f'Conv {i+1}' for i in range(len(conversation_ids))], rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Need multiple\nconversations\nfor violin plot', 
                ha='center', va='center', transform=ax7.transAxes)
        plt.title('Response Length Distribution (Violin Plot)')
    
    # 8. Summary statistics table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate comprehensive statistics
    stats_text = f"""
    Response Length Analysis Summary
    
    Total Responses: {len(all_response_lengths)}
    Total Conversations: {len(conversation_ids)}
    
    Length Statistics (words):
    • Mean: {mean_length:.1f}
    • Median: {median_length:.1f}
    • Std Dev: {std_length:.1f}
    • Min: {min(all_response_lengths)}
    • Max: {max(all_response_lengths)}
    
    Percentiles (words):
    • 25th: {np.percentile(all_response_lengths, 25):.1f}
    • 50th: {p50:.1f}
    • 75th: {np.percentile(all_response_lengths, 75):.1f}
    • 90th: {p90:.1f}
    • 95th: {p95:.1f}
    • 99th: {np.percentile(all_response_lengths, 99):.1f}
    
    Responses per Conversation:
    • Mean: {np.mean([len(conv_lengths) for conv_lengths in conversation_response_lengths.values()]):.1f}
    • Range: {min([len(conv_lengths) for conv_lengths in conversation_response_lengths.values()])} - {max([len(conv_lengths) for conv_lengths in conversation_response_lengths.values()])}
    """
    
    plt.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Response Length Distribution Analysis - {data_source.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'response_length_distribution_{data_source}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎯 Response length distribution analysis saved to {os.path.join(output_dir, f'response_length_distribution_{data_source}.png')}")

if __name__ == "__main__":
    main()
