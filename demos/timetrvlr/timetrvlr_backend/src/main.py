"""Flask backend exposing text embedding inference using the GME model.

Augmented to project HF embeddings into 3D using a trained AE3D.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from flask import Flask, jsonify, request
from transformers import AutoModel

from wc_simd.vlm_embed_ae import load_model, resolve_device, AE

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_INSTRUCTION = "Find an image that matches the given text."
# Default AE checkpoint relative to repo root
# (runs/vlm_embed_ae3d_hires_1/ae.pt)
DEFAULT_AE_CKPT: Optional[Path] = (
    Path(__file__).resolve().parents[4] /
    "runs" /
    "vlm_embed_ae3d_hires_1" /
    "ae.pt")


class EmbeddingService:
    """Lazy-load the HF model and provide embedding + 3D projection helpers."""

    def __init__(self, model_id: str = MODEL_ID,
                 ae_ckpt: Optional[str] = None) -> None:
        self._model_id = model_id
        self._model = None
        self._device: Optional[torch.device] = None
        # Always use the default AE checkpoint; error if missing
        self._ae_ckpt_path: Optional[Path] = None
        if DEFAULT_AE_CKPT is not None:
            self._ae_ckpt_path = DEFAULT_AE_CKPT.resolve()
        if self._ae_ckpt_path is None or not self._ae_ckpt_path.exists():
            raise FileNotFoundError(
                f"Required AE checkpoint missing: {DEFAULT_AE_CKPT}. "
                "Ensure the file exists before starting the server.")
        self._ae_model: Optional[AE] = None

    @property
    def model(self):
        if self._model is None:
            LOGGER.info("Loading model %s", self._model_id)
            self._model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True)
        return self._model

    def _get_ae(self) -> Optional[AE]:
        if self._ae_model is None and self._ae_ckpt_path is not None:
            if not self._ae_ckpt_path.exists():
                raise FileNotFoundError(
                    f"AE checkpoint not found: {self._ae_ckpt_path}")
            LOGGER.info("Loading AE3D checkpoint: %s", self._ae_ckpt_path)

            self._device = resolve_device("auto")
            self._ae_model, _ = load_model(
                self._ae_ckpt_path, self._device)

        return self._ae_model

    def embed3d(self, texts: Iterable[str],
                instruction: str = DEFAULT_INSTRUCTION) -> List[List[float]]:
        """Return 3D vectors by projecting HF embeddings through AE3D.
        """
        # Use the same instruction used for retrieval to keep text embeddings
        # aligned with the image embedding space the AE was trained on.
        with torch.inference_mode():
            embs = self.model.get_text_embeddings(
                texts=list(texts), instruction=instruction)

        ae = self._get_ae()
        if ae is None:
            raise RuntimeError(
                f"AE checkpoint not available. Expected at {DEFAULT_AE_CKPT}.")

        try:
            # Accept either a tensor or a list/array. Avoid unnecessary
            # CPU<->GPU round trips: if the model already returned a tensor
            # keep it on device.
            if torch.is_tensor(embs):
                emb_tensor = embs.detach()
                if emb_tensor.dtype != torch.float32:
                    emb_tensor = emb_tensor.float()
                emb_tensor = emb_tensor.to(self._device)
            else:
                emb_array = np.asarray(embs, dtype=np.float32)
                emb_tensor = torch.from_numpy(emb_array).to(self._device)

            # Validate dimensionality against first Linear layer of encoder.
            expected_in: Optional[int] = None
            for module in self._ae_model.encoder.modules():
                if isinstance(module, torch.nn.Linear):
                    expected_in = module.in_features
                    break
            if expected_in is not None and emb_tensor.shape[1] != expected_in:
                raise ValueError(
                    f"Embedding dim {emb_tensor.shape[1]} != AE input {expected_in}.")

            with torch.inference_mode():
                Z = self._ae_model.encoder(emb_tensor).detach().cpu().numpy()
            return Z.tolist()
        except Exception:
            # Log internal error; convert to 500 at boundary
            LOGGER.exception("AE projection failed")
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
