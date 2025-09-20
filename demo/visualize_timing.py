#!/usr/bin/env python3
"""
Visualize timing components over video frames
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_timing_data(file_path):
    """Load timing data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_timing_visualization(data, output_path='timing_visualization.png'):
    """Create visualization of timing components over frames."""
    
    # Extract frame data
    frames = data['detailed_frame_metrics']
    frame_indices = [frame['frame_idx'] for frame in frames]
    video_times = [frame['video_time'] for frame in frames]
    visual_embedding_times = [frame['visual_embedding_time'] for frame in frames]
    model_forward_times = [frame['model_forward_time'] for frame in frames]
    generation_times = [frame['generation_time'] for frame in frames]
    total_times = [frame['total_time'] for frame in frames]
    fps_values = [frame['fps'] for frame in frames]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VideoLLM-online Timing Components Over Frames', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual timing components
    ax1.plot(frame_indices, visual_embedding_times, 'b-', label='Visual Embedding', linewidth=2, alpha=0.8)
    ax1.plot(frame_indices, model_forward_times, 'g-', label='Model Forward', linewidth=2, alpha=0.8)
    ax1.plot(frame_indices, generation_times, 'r-', label='Generation', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Individual Timing Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stacked area chart showing component contributions
    ax2.fill_between(frame_indices, 0, visual_embedding_times, alpha=0.7, color='blue', label='Visual Embedding')
    ax2.fill_between(frame_indices, visual_embedding_times, 
                     np.array(visual_embedding_times) + np.array(model_forward_times), 
                     alpha=0.7, color='green', label='Model Forward')
    ax2.fill_between(frame_indices, 
                     np.array(visual_embedding_times) + np.array(model_forward_times),
                     total_times, 
                     alpha=0.7, color='red', label='Generation')
    ax2.plot(frame_indices, total_times, 'k-', linewidth=2, label='Total Time')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Stacked Timing Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: FPS over time
    ax3.plot(frame_indices, fps_values, 'purple', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('FPS')
    ax3.set_title('Processing FPS Over Frames')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Component percentages
    visual_pct = np.array(visual_embedding_times) / np.array(total_times) * 100
    model_pct = np.array(model_forward_times) / np.array(total_times) * 100
    generation_pct = np.array(generation_times) / np.array(total_times) * 100
    
    ax4.fill_between(frame_indices, 0, visual_pct, alpha=0.7, color='blue', label='Visual Embedding %')
    ax4.fill_between(frame_indices, visual_pct, visual_pct + model_pct, 
                     alpha=0.7, color='green', label='Model Forward %')
    ax4.fill_between(frame_indices, visual_pct + model_pct, 100, 
                     alpha=0.7, color='red', label='Generation %')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Percentage of Total Time (%)')
    ax4.set_title('Component Time Percentages')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig

def print_timing_summary(data):
    """Print summary statistics."""
    frames = data['detailed_frame_metrics']
    
    visual_times = [frame['visual_embedding_time'] for frame in frames]
    model_times = [frame['model_forward_time'] for frame in frames]
    generation_times = [frame['generation_time'] for frame in frames]
    total_times = [frame['total_time'] for frame in frames]
    
    print("\n" + "="*60)
    print("TIMING COMPONENT ANALYSIS")
    print("="*60)
    
    # Find frames with generation (responses)
    response_frames = [i for i, frame in enumerate(frames) if frame['generation_time'] > 0]
    non_response_frames = [i for i, frame in enumerate(frames) if frame['generation_time'] == 0]
    
    print(f"Total frames processed: {len(frames)}")
    print(f"Frames with responses: {len(response_frames)}")
    print(f"Frames without responses: {len(non_response_frames)}")
    
    print(f"\nVisual Embedding Time:")
    print(f"  Average: {np.mean(visual_times):.4f}s")
    print(f"  Min: {np.min(visual_times):.4f}s")
    print(f"  Max: {np.max(visual_times):.4f}s")
    print(f"  Std: {np.std(visual_times):.4f}s")
    
    print(f"\nModel Forward Time:")
    print(f"  Average: {np.mean(model_times):.4f}s")
    print(f"  Min: {np.min(model_times):.4f}s")
    print(f"  Max: {np.max(model_times):.4f}s")
    print(f"  Std: {np.std(model_times):.4f}s")
    
    print(f"\nGeneration Time (response frames only):")
    response_generation_times = [generation_times[i] for i in response_frames]
    if response_generation_times:
        print(f"  Average: {np.mean(response_generation_times):.4f}s")
        print(f"  Min: {np.min(response_generation_times):.4f}s")
        print(f"  Max: {np.max(response_generation_times):.4f}s")
        print(f"  Std: {np.std(response_generation_times):.4f}s")
    
    print(f"\nTotal Frame Time:")
    print(f"  Average: {np.mean(total_times):.4f}s")
    print(f"  Min: {np.min(total_times):.4f}s")
    print(f"  Max: {np.max(total_times):.4f}s")
    print(f"  Std: {np.std(total_times):.4f}s")
    
    # Show first few frames in detail
    print(f"\nFirst 10 frames detail:")
    print("Frame | Video Time | Visual | Model | Generation | Total | FPS")
    print("-" * 70)
    for i in range(min(10, len(frames))):
        frame = frames[i]
        print(f"{frame['frame_idx']:5d} | {frame['video_time']:8.1f}s | {frame['visual_embedding_time']:6.3f}s | {frame['model_forward_time']:5.3f}s | {frame['generation_time']:10.3f}s | {frame['total_time']:5.3f}s | {frame['fps']:4.1f}")

def main():
    """Main function."""
    # prepare the data
    # python -m demo.benchmark --video_path demo/assets/cooking.mp4 --question "What do you see in this video?" --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus --output_path /tmp/full_video_timing.json
    
    # Load data
    data = load_timing_data('/tmp/full_video_timing.json')
    
    # Print summary
    print_timing_summary(data)
    
    # Create visualization
    fig = create_timing_visualization(data, 'timing_components_visualization.png')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()