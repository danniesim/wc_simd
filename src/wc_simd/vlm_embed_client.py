import requests
import base64


def encode_image(image_url):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Python script)'}
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    base64_str = base64.b64encode(response.content).decode('utf-8')
    return base64_str


def get_embedding(content):
    response = requests.post(
        "http://localhost:8080/v1/embeddings",
        json={
            "model": "gme",
            "messages": [
                {
                    "role": "user",
                    "content": content}],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))


t2i_prompt = 'Find an image that matches the given text.'
texts = [
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "Alibaba office.",
]

# Load images from URLs
print("Loading images...")
image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/e/e9/Tesla_Cybertruck_damaged_window.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/e/e0/TaobaoCity_Alibaba_Xixi_Park.jpg']
images_base64 = [encode_image(url) for url in image_urls]
print(f"Loaded {len(images_base64)} images successfully")

# Single-modal embedding
print("Computing single-modal embeddings...")
e_text = [get_embedding([{"type": "text", "text": text}]) for text in texts]
e_image = [
    get_embedding(
        [{"type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}])
    for b64 in images_base64]

similarity_matrix = [[dot_product(t, i) for i in e_image] for t in e_text]
print('Single-modal', similarity_matrix)

# Single-modal with instruction
print("Computing single-modal embeddings with instruction...")
e_query = [
    get_embedding([{"type": "text", "text": t2i_prompt + " " + text}])
    for text in texts]
e_corpus = [
    get_embedding(
        [{"type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}])
    for b64 in images_base64]

similarity_matrix_instr = [
    [dot_product(q, c) for c in e_corpus] for q in e_query]
print('Single-modal with instruction', similarity_matrix_instr)

# Fused-modal embedding
print("Computing fused-modal embeddings...")
e_fused = [
    get_embedding(
        [{"type": "text", "text": text},
         {"type": "image_url",
          "image_url":
          {"url": f"data:image/jpeg;base64,{images_base64[i]}"}}])
    for i, text in enumerate(texts)]

fused_similarity = [[dot_product(f1, f2) for f2 in e_fused] for f1 in e_fused]
print('Fused-modal', fused_similarity)
