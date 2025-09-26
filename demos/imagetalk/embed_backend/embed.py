"""Flask backend exposing text embedding inference using the GME model."""
from __future__ import annotations

import logging
from typing import Iterable, List

import torch
from flask import Flask, jsonify, request
from transformers import AutoModel

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_INSTRUCTION = "Find an image that matches the given text."


class EmbeddingService:
    """Lazy-load the HF model and provide embedding helpers."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None

    @property
    def model(self):
        if self._model is None:
            LOGGER.info("Loading model %s", self._model_id)
            self._model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True)
        return self._model

    def embed(self, texts: Iterable[str],
              instruction: str = DEFAULT_INSTRUCTION) -> List[List[float]]:
        with torch.inference_mode():
            embeddings = self.model.get_text_embeddings(
                texts=list(texts), instruction=instruction)
        return embeddings.tolist()


service = EmbeddingService()


@app.post("/embed")
def embed_endpoint():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    texts = payload.get("texts")
    if texts is None:
        return jsonify({"error": "Missing 'texts' in request body."}), 400

    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return jsonify(
            {"error": "'texts' must be a string or a list of strings."}), 400

    instruction = payload.get("instruction", DEFAULT_INSTRUCTION)
    if not isinstance(instruction, str):
        return jsonify({"error": "'instruction' must be a string."}), 400

    try:
        embeddings = service.embed(texts, instruction=instruction)
    except Exception as exc:  # pragma: no cover - bubble up inference issues
        LOGGER.exception("Embedding generation failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"embeddings": embeddings, "count": len(embeddings)})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8001, debug=False)
