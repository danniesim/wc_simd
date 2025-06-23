#!/bin/sh

docker run --gpus all -p 8080:80 -v /home/ubuntu/hf_data:/data  --pull always ghcr.io/huggingface/text-embeddings-inference:1.7.2 --model-id Qwen/Qwen3-Embedding-0.6B --dtype float16