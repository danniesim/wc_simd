"""Flask backend exposing text embedding inference using the GME model.

Augmented to project HF embeddings into 3D using a trained VAE3D.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from flask import Flask, jsonify, request
from transformers import AutoModel

from wc_simd.vlm_embed_vae import VAE3DWrapper

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_INSTRUCTION = "Find an image that matches the given text."
# Default VAE checkpoint relative to repo root
# (runs/vlm_embed_vae3d_hires_1/vae3d.pt)
DEFAULT_VAE_CKPT: Optional[Path] = (
    Path(__file__).resolve().parents[4] /
    "runs" /
    "vlm_embed_vae3d_hires_1" /
    "vae3d.pt")


class EmbeddingService:
    """Lazy-load the HF model and provide embedding + 3D projection helpers."""

    def __init__(self, model_id: str = MODEL_ID,
                 vae_ckpt: Optional[str] = None) -> None:
        self._model_id = model_id
        self._model = None
        # Always use the default VAE checkpoint; error if missing
        self._vae_ckpt_path: Optional[Path] = None
        if DEFAULT_VAE_CKPT is not None:
            self._vae_ckpt_path = DEFAULT_VAE_CKPT.resolve()
        if self._vae_ckpt_path is None or not self._vae_ckpt_path.exists():
            raise FileNotFoundError(
                f"Required VAE checkpoint missing: {DEFAULT_VAE_CKPT}. "
                "Ensure the file exists before starting the server.")
        self._vae_wrapper: Optional[VAE3DWrapper] = None

    @property
    def model(self):
        if self._model is None:
            LOGGER.info("Loading model %s", self._model_id)
            self._model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True)
        return self._model

    def _get_vae(self) -> Optional[VAE3DWrapper]:
        if self._vae_wrapper is None and self._vae_ckpt_path is not None:
            if not self._vae_ckpt_path.exists():
                raise FileNotFoundError(
                    f"VAE checkpoint not found: {self._vae_ckpt_path}")
            LOGGER.info("Loading VAE3D checkpoint: %s", self._vae_ckpt_path)
            self._vae_wrapper = VAE3DWrapper(str(self._vae_ckpt_path))
        return self._vae_wrapper

    def embed(self, texts: Iterable[str],
              instruction: str = DEFAULT_INSTRUCTION) -> List[List[float]]:
        """Return raw HF embeddings as lists of floats."""
        with torch.inference_mode():
            embeddings = self.model.get_text_embeddings(
                texts=list(texts), instruction=instruction)
        return embeddings.tolist()

    def embed3d(self, texts: Iterable[str],
                instruction: str = DEFAULT_INSTRUCTION) -> List[List[float]]:
        """Return 3D vectors by projecting HF embeddings through VAE3D.
        """
        # Use the same instruction used for retrieval to keep text embeddings
        # aligned with the image embedding space the VAE was trained on.
        with torch.inference_mode():
            embs = self.model.get_text_embeddings(
                texts=list(texts), instruction=instruction)

        vae = self._get_vae()
        if vae is None:
            raise RuntimeError(
                f"VAE checkpoint not available. Expected at {DEFAULT_VAE_CKPT}.")

        try:
            X = np.asarray(embs, dtype=np.float32)
            Z = vae.to3d(
                X, use_mu=True, batch_size=max(
                    64, min(
                        4096, len(X) or 1)))
            return Z.tolist()
        except Exception:
            # Log internal error; convert to 500 at boundary
            LOGGER.exception("VAE projection failed")
            raise


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
        # Produce 3D coordinates instead of full-dim embeddings, ensuring the
        # same instruction is used for text embeddings as during training.
        embeddings = service.embed3d(texts, instruction=instruction)
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
