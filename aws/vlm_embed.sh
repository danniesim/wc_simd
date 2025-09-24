#!/bin/bash

# Script: run_vlm.sh
# Purpose: Launch 8 wc_simd.vlm_embed instances in one screen session + GPU monitor

SESSION_NAME="vlm_all"

# Start first window with instance 0
screen -dmS $SESSION_NAME \
  bash -c "python -m wc_simd.vlm_embed --instances 8 --instance-no 0"

# Add more windows for instances 1..7
for i in {1..7}; do
  screen -S $SESSION_NAME -X screen bash -c "python -m wc_simd.vlm_embed --instances 8 --instance-no $i"
done

# Add one more window running GPU monitor
screen -S $SESSION_NAME -X screen bash -c "watch -n 1 nvidia-smi"