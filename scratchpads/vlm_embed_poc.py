from io import BytesIO
from PIL import Image
import requests
import torch
from transformers.utils.versions import require_version
from transformers import AutoModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


# Detect available device
if torch.cuda.is_available():
    device_map = 'cuda'
    torch_dtype = "float16"
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device_map = 'mps'  # Apple Silicon GPU
    torch_dtype = "float16"
    print("Using Apple Silicon MPS")
else:
    device_map = None  # Don't use device_map for CPU to avoid accelerate dependency
    torch_dtype = "float32"  # float16 may not be well supported on CPU
    print("Using CPU")


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL with proper headers to avoid 403 errors."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Python script)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


t2i_prompt = 'Find an image that matches the given text.'
texts = [
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "Alibaba office.",
]

# Load images from URLs with proper headers
print("Loading images...")
images = [
    load_image_from_url(
        'https://upload.wikimedia.org/wikipedia/commons/e/e9/Tesla_Cybertruck_damaged_window.jpg'),
    load_image_from_url(
        'https://upload.wikimedia.org/wikipedia/commons/e/e0/TaobaoCity_Alibaba_Xixi_Park.jpg'),]
print(f"Loaded {len(images)} images successfully")


gme = AutoModel.from_pretrained(
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
)


# Single-modal embedding
e_text = gme.get_text_embeddings(texts=texts)
e_image = gme.get_image_embeddings(images=images)
print('Single-modal', (e_text @ e_image.T).tolist())
# Single-modal [[0.359619140625, 0.0655517578125], [0.04180908203125, 0.374755859375]]

# How to set embedding instruction
e_query = gme.get_text_embeddings(texts=texts, instruction=t2i_prompt)
# If is_query=False, we always use the default instruction.
e_corpus = gme.get_image_embeddings(images=images, is_query=False)
print('Single-modal with instruction', (e_query @ e_corpus.T).tolist())
# Single-modal with instruction [[0.429931640625, 0.11505126953125],
# [0.049835205078125, 0.409423828125]]

# Fused-modal embedding
e_fused = gme.get_fused_embeddings(texts=texts, images=images)
print('Fused-modal', (e_fused @ e_fused.T).tolist())
# Fused-modal [[1.0, 0.05511474609375], [0.05511474609375, 1.0]]
