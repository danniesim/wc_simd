#!/bin/sh

# Ref: https://github.com/huggingface/text-embeddings-inference/issues/87

volume=/home/ubuntu/hf_data
model=Qwen/Qwen3-Embedding-0.6B

# Create a network that the containers will sahre
docker network create tei-net

# Start N replicas of the model in detach mode, each on one gpu, each with a different name

for i in $(seq 0 7); do
    docker run --gpus '"device='$i'"' --net tei-net --name tei-$i -v $volume:/data --pull always --rm -d ghcr.io/huggingface/text-embeddings-inference:1.7.2 --model-id $model --dtype float16
done

# Create nginx.conf file

cat << 'EOF' | sudo tee "$volume/nginx.conf" > /dev/null
upstream tei {
    server tei-0;
    server tei-1;
    server tei-2;
    server tei-3;
    server tei-4;
    server tei-5;
    server tei-6;
    server tei-7;
}
    
server {
    location / {
        proxy_pass http://tei;
    }
}
EOF

# Start nginx container
docker run -d -v $volume/nginx.conf:/etc/nginx/conf.d/default.conf:ro  -p 8080:80 --net tei-net --name balancer nginx