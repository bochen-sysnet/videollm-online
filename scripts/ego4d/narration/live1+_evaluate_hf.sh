#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=1

# Basic Ego4D Narration Evaluation with Hugging Face Model
# Single GPU evaluation for perplexity, time difference, and fluency metrics
# 
# NOTE: This requires refined Ego4D data structure with:
# - datasets/ego4d/v2/annotations/refined_narration_stream_val_filtered.json (146 videos with both embeddings and narration)
# - datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384/ (video embeddings)
# - datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384_metadata.json

source .venv/bin/activate && torchrun --nproc_per_node=1 --standalone evaluate.py \
    --live_version live1+ \
    --eval_datasets ego4d_refined_narration_stream_val_single \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_pin_memory False \
    --prediction_loss_only False \
    --dataloader_num_workers 2 \
    --bf16 True \
    --tf32 True \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --output_dir outputs/ego4d_narration_eval_hf/live1+