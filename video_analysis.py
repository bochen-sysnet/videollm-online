#!/usr/bin/env python3
"""
Video Analysis Module

This module contains analysis functions for video datasets including:
- Ground truth word count analysis
- Conversation distribution analysis
- Time per token analysis
- Generated word count analysis

Can be run standalone to process both datasets in one run.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


class Config:
    """Configuration constants for analysis."""
    OUTPUT_DIR = "timing_plots"
    PLOT_STYLE = "seaborn-v0_8"
    FIGURE_DPI = 100
    FONT_SIZE = 10


def create_gt_word_count_analysis(data_source, output_dir=Config.OUTPUT_DIR):
    """
    Analyze and visualize word counts in ground truth responses.
    
    Args:
        data_source (str): 'goalstep' or 'narration'
        output_dir (str): Output directory for plots
    """
    print(f"📊 Creating ground truth word count analysis for {data_source}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if data_source == 'goalstep':
        print("📊 Loading goalstep conversation data from JSON files...")
        data_path = "datasets/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json"
        with open(data_path, 'r') as f:
            goalstep_data = json.load(f)
        
        # Collect word counts from goalstep data
        all_word_counts = []
        video_word_counts = {}
        
        for conversation_data in goalstep_data:
            if isinstance(conversation_data, dict):
                video_uid = conversation_data.get('video_uid', 'unknown')
                
                if video_uid not in video_word_counts:
                    video_word_counts[video_uid] = []
                
                # Count words in ground truth assistant responses
                for turn in conversation_data.get('conversation', []):
                    if turn.get('role') == 'assistant' and 'content' in turn:
                        word_count = len(turn['content'].split())
                        all_word_counts.append(word_count)
                        video_word_counts[video_uid].append(word_count)
        
        print(f"📊 Processed {len(all_word_counts)} ground truth responses from {len(video_word_counts)} videos")
        
    else:  # narration
        print("📊 Loading narration conversation data from JSON files...")
        data_path = "datasets/ego4d/v2/annotations/refined_narration_stream_val.json"
        with open(data_path, 'r') as f:
            narration_data = json.load(f)
        
        # Collect word counts from narration data
        all_word_counts = []
        video_word_counts = {}
        
        for video_uid, conversations in narration_data.items():
            if video_uid not in video_word_counts:
                video_word_counts[video_uid] = []
            
            # Each video can have multiple conversations
            for conversation_id, conversation_data in conversations.items():
                # conversation_data is a list of narration entries
                if isinstance(conversation_data, list):
                    # Count words in ground truth assistant responses (narration entries)
                    for entry in conversation_data:
                        if isinstance(entry, dict) and 'text' in entry:
                            word_count = len(entry['text'].split())
                            all_word_counts.append(word_count)
                            video_word_counts[video_uid].append(word_count)
    
    if not all_word_counts:
        print("⚠️ No ground truth word count data found")
        return None
    
    # Create simplified analysis plot
    plt.style.use(Config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Ground Truth Word Count Analysis - {data_source.title()}', fontsize=14, fontweight='bold')
    
    # 1. Word count distribution
    ax1 = axes[0]
    ax1.hist(all_word_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Word Count Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_count = np.mean(all_word_counts)
    median_count = np.median(all_word_counts)
    ax1.axvline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
    ax1.axvline(median_count, color='orange', linestyle='--', label=f'Median: {median_count:.1f}')
    ax1.legend()
    
    # 2. Cumulative distribution
    ax2 = axes[1]
    sorted_counts = np.sort(all_word_counts)
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax2.plot(sorted_counts, cumulative_pct, linewidth=2, color='darkblue')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Distribution of Word Counts')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'gt_word_count_analysis_{data_source}.png')
    plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Ground truth word count analysis saved to: {output_file}")
    return output_file


def create_initial_distribution_analysis(conversations_per_video, unique_videos, data_source, output_dir=Config.OUTPUT_DIR):
    """
    Create initial conversation distribution analysis.
    
    Args:
        conversations_per_video (list): List of conversation counts per video
        unique_videos (list): List of unique video UIDs
        data_source (str): 'goalstep' or 'narration'
        output_dir (str): Output directory for plots
    """
    print(f"📊 Creating initial conversation distribution analysis for {data_source}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use(Config.PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Conversation Distribution Analysis - {data_source.title()}', fontsize=16, fontweight='bold')
    
    # 1. Overall distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(conversations_per_video, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Conversations per Video')
    ax1.set_ylabel('Number of Videos')
    ax1.set_title('Distribution of Conversations per Video')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conv = np.mean(conversations_per_video)
    median_conv = np.median(conversations_per_video)
    ax1.axvline(mean_conv, color='red', linestyle='--', label=f'Mean: {mean_conv:.1f}')
    ax1.axvline(median_conv, color='orange', linestyle='--', label=f'Median: {median_conv:.1f}')
    ax1.legend()
    
    # 2. Cumulative distribution
    ax2 = axes[0, 1]
    sorted_conv = np.sort(conversations_per_video)
    cumulative_pct = np.arange(1, len(sorted_conv) + 1) / len(sorted_conv) * 100
    ax2.plot(sorted_conv, cumulative_pct, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Conversations per Video')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Distribution of Conversations')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Box plot
    ax3 = axes[1, 0]
    box_data = [conversations_per_video]
    bp = ax3.boxplot(box_data, patch_artist=True, tick_labels=['All Videos'])
    bp['boxes'][0].set_facecolor('lightcoral')
    ax3.set_ylabel('Conversations per Video')
    ax3.set_title('Box Plot of Conversations per Video')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Dataset Statistics:
    
    Total Videos: {len(unique_videos):,}
    Total Conversations: {sum(conversations_per_video):,}
    
    Conversations per Video:
    • Mean: {np.mean(conversations_per_video):.2f}
    • Median: {np.median(conversations_per_video):.2f}
    • Min: {np.min(conversations_per_video)}
    • Max: {np.max(conversations_per_video)}
    • Std: {np.std(conversations_per_video):.2f}
    
    Percentiles:
    • 25th: {np.percentile(conversations_per_video, 25):.1f}
    • 75th: {np.percentile(conversations_per_video, 75):.1f}
    • 90th: {np.percentile(conversations_per_video, 90):.1f}
    • 95th: {np.percentile(conversations_per_video, 95):.1f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'initial_conversation_distribution_{data_source}.png')
    plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Initial conversation distribution analysis saved to: {output_file}")
    return output_file


def create_time_per_token_analysis(results, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """
    Analyze and visualize time per token for VLM generation.
    
    Args:
        results (list): List of evaluation results
        output_dir (str): Output directory for plots
    """
    print("📊 Creating time per token analysis...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract time per token data
    time_per_token_data = []
    
    for result in results:
        generated_turns = result.get('generated_turns', [])
        if isinstance(generated_turns, int):
            generated_turns = []
        
        for turn in generated_turns:
            if 'generation_time' in turn and 'text' in turn:
                generation_time = turn['generation_time']
                text = turn['text']
                
                # Extract actual response text (remove video time prefix)
                if '(Video Time =' in text:
                    response_text = text.split('Assistant:', 1)[-1].strip()
                else:
                    response_text = text
                
                # Estimate token count from word count (rough approximation)
                word_count = len(response_text.split())
                token_count = word_count * 1.3  # Approximate tokens per word
                
                if token_count > 0:
                    time_per_token = generation_time / token_count
                    time_per_token_data.append(time_per_token)
    
    if not time_per_token_data:
        print("⚠️ No time per token data found")
        return None
    
    # Create simplified analysis plot
    plt.style.use(Config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('VLM Generation Time per Token Analysis', fontsize=14, fontweight='bold')
    
    # 1. Time per token distribution
    ax1 = axes[0]
    ax1.hist(time_per_token_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.set_xlabel('Time per Token (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Time per Token Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_time = np.mean(time_per_token_data)
    median_time = np.median(time_per_token_data)
    ax1.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.3f}s')
    ax1.axvline(median_time, color='orange', linestyle='--', label=f'Median: {median_time:.3f}s')
    ax1.legend()
    
    # 2. Cumulative distribution
    ax2 = axes[1]
    sorted_times = np.sort(time_per_token_data)
    cumulative_pct = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
    ax2.plot(sorted_times, cumulative_pct, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Time per Token (seconds)')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Distribution of Time per Token')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'time_per_token_analysis_{data_source}.png')
    plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Time per token analysis saved to: {output_file}")
    return output_file


def create_generated_word_count_analysis(results, output_dir=Config.OUTPUT_DIR, data_source='goalstep'):
    """
    Analyze and visualize word counts in VLM generated responses.
    
    Args:
        results (list): List of evaluation results
        output_dir (str): Output directory for plots
    """
    print("📊 Creating generated word count analysis...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract generated word count data
    generated_word_counts = []
    
    for result in results:
        generated_turns = result.get('generated_turns', [])
        if isinstance(generated_turns, int):
            generated_turns = []
        
        for turn in generated_turns:
            if 'text' in turn:
                text = turn['text']
                
                # Extract actual response text (remove video time prefix)
                if '(Video Time =' in text:
                    response_text = text.split('Assistant:', 1)[-1].strip()
                else:
                    response_text = text
                
                # Count words in generated response
                word_count = len(response_text.split())
                if word_count > 0:
                    generated_word_counts.append(word_count)
    
    if not generated_word_counts:
        print("⚠️ No generated word count data found")
        return None
    
    # Create simplified analysis plot
    plt.style.use(Config.PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('VLM Generated Response Word Count Analysis', fontsize=14, fontweight='bold')
    
    # 1. Word count distribution
    ax1 = axes[0]
    ax1.hist(generated_word_counts, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Generated Word Count Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_count = np.mean(generated_word_counts)
    median_count = np.median(generated_word_counts)
    ax1.axvline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
    ax1.axvline(median_count, color='orange', linestyle='--', label=f'Median: {median_count:.1f}')
    ax1.legend()
    
    # 2. Cumulative distribution
    ax2 = axes[1]
    sorted_counts = np.sort(generated_word_counts)
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax2.plot(sorted_counts, cumulative_pct, linewidth=2, color='darkred')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Distribution of Generated Word Counts')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'generated_word_count_analysis_{data_source}.png')
    plt.savefig(output_file, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Generated word count analysis saved to: {output_file}")
    return output_file


def analyze_datasets():
    """
    Analyze both goalstep and narration datasets in one run.
    """
    print("🚀 Starting Video Dataset Analysis")
    print("=" * 60)
    
    datasets = ['goalstep', 'narration']
    
    for data_source in datasets:
        print(f"\n📊 Processing {data_source} dataset...")
        print("-" * 40)
        
        try:
            # Create ground truth word count analysis
            create_gt_word_count_analysis(data_source)
            
            print(f"✅ {data_source} analysis completed successfully")
            
        except Exception as e:
            print(f"❌ Error processing {data_source}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ Dataset analysis completed!")
    print("📊 Check timing_plots/ directory for generated plots")


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description='Video Dataset Analysis')
    parser.add_argument('--data_source', choices=['goalstep', 'narration', 'both'], 
                       default='both', help='Dataset to analyze')
    parser.add_argument('--output_dir', default='timing_plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Update config
    Config.OUTPUT_DIR = args.output_dir
    
    if args.data_source == 'both':
        analyze_datasets()
    else:
        print(f"📊 Analyzing {args.data_source} dataset...")
        create_gt_word_count_analysis(args.data_source)


if __name__ == "__main__":
    main()