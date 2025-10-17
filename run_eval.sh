#!/bin/bash

# Define lists
data_sources=("goalstep")
num_videos_list=(3 5 10)
iterations=(1)

# ablation study on remaining length
config_ids=("base" "rl_ablation1" "rl_ablation2" "rl_ablation3")
# 

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
