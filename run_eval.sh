#!/bin/bash

# Define lists
data_sources=("goalstep" "narration")
num_videos_list=(5 10 15 20)
# num_videos_list=(1 3 5 8 10 15 20)
iterations=(1 2 3 4 5)

# full evaluation
# config_ids=("base" "random_m" "random_2" "round_robin_m" "round_robin_2")
# ablation study on remaining length
# component ablation study
config_ids=("rl_ablation1" "rl_ablation2" "rl_ablation3" "comp_ablation1" "comp_ablation2" "comp_ablation3" "comp_ablation4" "chunk_ablation1" "chunk_ablation2" "chunk_ablation3" "chunk_ablation4" "chunk_ablation5")

cd /home/ryan/bo/videollm-online
source .venv/bin/activate

# Main loop
for data_source in "${data_sources[@]}"; do
  for num_videos in "${num_videos_list[@]}"; do
    for config_id in "${config_ids[@]}"; do
      for iteration in "${iterations[@]}"; do
        echo "Running: data_source=$data_source, num_videos=$num_videos, config_id=$config_id, iteration=$iteration"
        python streaming_evaluate_event_driven.py \
          --live_version live1+ \
          --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
          --data_source "$data_source" \
          --num_videos "$num_videos" \
          --config_id "$config_id" \
          --iteration "$iteration"
      done
    done
  done
done
