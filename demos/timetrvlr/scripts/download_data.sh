#!/bin/bash

files=( 
    "demos/timetrvlr/public/iiif_no_text_embedding_index.json" 
    "demos/timetrvlr/public/iiif_no_text_embedding_matrix_vlm_embed_ae3d_hires_1.npy"
    "runs/vlm_embed_vae3d_hires_1/ae.pt"
)

# activate root virtual environment
cd ../../
source .venv/bin/activate

# if argument is "up"
if [ "$1" == "up" ]; then
    for file in "${files[@]}"; do
        python aws/upload_to_s3.py --overwrite "$file"
    done
# else if
elif [ "$1" == "down" ]; then
    for file in "${files[@]}"; do
        python aws/upload_to_s3.py --overwrite --download "$file"
    done
# else error
else
    echo "Usage: $0 [up|down]"
    exit 1
fi