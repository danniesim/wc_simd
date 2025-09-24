#!/bin/bash

# Script: run_vlm.sh
# Purpose: Launch 8 wc_simd.vlm_embed instances in one screen session + GPU monitor

SESSION_NAME="vlm_all"

# Start first window with instance 0
screen -dmS $SESSION_NAME \
  bash -c "python -m wc_simd.vlm_embed --instances 8 --instance-no 0 --instances 8 --instance-no 0 --batch-size 16 --prefetch-workers 4 --fetch-max-inflight 16 --prefetch-buffer 128"

# Add more windows for instances 1..7
for i in {1..7}; do
  screen -S $SESSION_NAME -X screen bash -c "python -m wc_simd.vlm_embed --instances 8 --instance-no $i --batch-size 16 --prefetch-workers 4 --fetch-max-inflight 16 --prefetch-buffer 128"
done

# Add one more window running GPU monitor
screen -S $SESSION_NAME -X screen bash -c "watch -n 1 nvidia-smi"