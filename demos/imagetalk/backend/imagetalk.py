"""Audio-in / audio-out backend using HF Qwen2.5 Omni.

POST /api/v1/audio (multipart/form-data)
    field 'audio' (WAV)
    optional field 'session_id'

If model produces audio -> returns raw WAV bytes with headers:
    X-Session-Id, X-Model, X-Transcript

If model produces no audio -> JSON fallback containing transcript.
"""

from __future__ import annotations

import json
import io
import os
import tempfile
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request
import torch
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from transformers.tokenization_utils_base import BatchEncoding
from qwen_omni_utils import process_mm_info

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("IMAGETALK_MODEL", "Qwen/Qwen2.5-Omni-7B")
SYSTEM_PROMPT = os.environ.get(
    "IMAGETALK_SYSTEM_PROMPT",
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
)
SYSTEM_PROMPT_TRANSCRIBE = os.environ.get(
    "IMAGETALK_SYSTEM_PROMPT_TRANSCRIBE",
    f"{SYSTEM_PROMPT} Your task is to accurately transcribe the provided audio input into english text. Only provide the transcribed text as output, without any additional commentary, information or questions.",
)
TARGET_SAMPLE_RATE = int(
    os.environ.get(
        "IMAGETALK_INPUT_SAMPLE_RATE",
        "16000"))
OUTPUT_SAMPLE_RATE = int(
    os.environ.get(
        "IMAGETALK_OUTPUT_SAMPLE_RATE",
        "24000"))
ALLOW_ORIGINS = os.environ.get("IMAGETALK_ALLOWED_ORIGINS", "*")
EMBED_SERVICE_URL = os.environ.get(
    "IMAGETALK_EMBED_URL",
    "http://localhost:8001/embed")
EMBED_TIMEOUT_SECONDS = float(os.environ.get("IMAGETALK_EMBED_TIMEOUT", "10"))
EMBED_INSTRUCTION = os.environ.get(
    "IMAGETALK_EMBED_INSTRUCTION",
    "Find an image that matches the given text.",
)
MONGODB_CONNECTION_STRING = os.environ.get(
    "IMAGETALK_MONGODB_CONNECTION_STRING",
    "mongodb://user:pass@localhost:27017/?directConnection=true",
)
MONGODB_DB_NAME = os.environ.get("IMAGETALK_MONGODB_DB_NAME", "imagetalk")
IMAGE_COLLECTION_NAME = os.environ.get("IMAGETALK_IMAGE_COLLECTION", "images")
VECTOR_INDEX_NAME = os.environ.get(
    "IMAGETALK_VECTOR_INDEX",
    "embedding_vector_index")
VECTOR_FIELD_NAME = os.environ.get("IMAGETALK_VECTOR_FIELD", "embedding")
VECTOR_NUM_CANDIDATES = int(
    os.environ.get(
        "IMAGETALK_VECTOR_CANDIDATES",
        "200"))
DEFAULT_TOP_K = int(os.environ.get("IMAGETALK_VECTOR_TOP_K", "4"))

_app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("imagetalk")
_model_lock = threading.Lock()
_model: Optional[Qwen2_5OmniForConditionalGeneration] = None
_processor: Optional[Qwen2_5OmniProcessor] = None
_mongo_lock = threading.Lock()
_mongo_client: Optional[MongoClient] = None
_image_collection: Optional[Collection] = None


@dataclass
class GenerationResult:
    text: str
    # mono float32 in OUTPUT_SAMPLE_RATE (if present)
    audio: Optional[np.ndarray]


def _load_model(
) -> tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]:
    """Lazy-load the HF model and processor (thread-safe)."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    with _model_lock:
        if _model is None or _processor is None:
            _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16
                if torch.cuda.is_available() else torch.float16,
                device_map="auto", attn_implementation="flash_attention_2",)
            _processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def _decode_audio(file_storage) -> tuple[np.ndarray, int]:
    """Decode uploaded WAV -> mono float32 array plus sample rate."""
    file_storage.stream.seek(0)
    payload = file_storage.read()
    if not payload:
        raise ValueError("No audio data received")
    buf = io.BytesIO(payload)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:  # noqa: BLE001 - best effort cleanup
        if os.getenv("IMAGETALK_DEBUG") == "1":
            _log.warning("Failed to remove temp audio file: %s", path)


def _prepare_generation_inputs(
    user_audio: np.ndarray,
    sample_rate: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BatchEncoding, str]:
    model, processor = _load_model()
    # Many helper utilities (process_mm_info) expect an audio path.
    # To be faithful to the reference notebook we materialize a temporary WAV.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, user_audio, samplerate=sample_rate, format="WAV")
    audio_path = tmp_path
    if os.getenv("IMAGETALK_DEBUG") == "1":
        _log.info(
            "Wrote temp user audio: %s (samples=%d sr=%d)",
            audio_path,
            user_audio.size,
            sample_rate,
        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=chat_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    return model, processor, inputs, audio_path


def _generate(user_audio: np.ndarray, sample_rate: int) -> GenerationResult:
    model, processor, inputs, audio_path = _prepare_generation_inputs(
        user_audio, sample_rate
    )

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=True,
            )
    finally:
        _cleanup_temp_file(audio_path)

    decoded = processor.batch_decode(
        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text = decoded[0] if decoded else ""
    audio_tensor = output[1]
    # audio_tensor expected shape: (batch, samples) or (samples,) -> flatten
    audio_np: Optional[np.ndarray] = None
    try:
        if audio_tensor is not None:
            if hasattr(audio_tensor, "detach"):
                audio_tensor = audio_tensor.detach().cpu()
            audio_np = np.array(audio_tensor).reshape(-1).astype(np.float32)
    except Exception:  # noqa: BLE001
        audio_np = None

    return GenerationResult(text=text, audio=audio_np)


# ---------------------------------------------------------------------------
# Packaging helpers
# ---------------------------------------------------------------------------
def _encode_wav(audio_samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_samples, samplerate=sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _resolve_cors_origin(req_origin: str | None) -> str:
    if ALLOW_ORIGINS == "*":
        return "*"
    allowed = {o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()}
    if req_origin and req_origin in allowed:
        return req_origin
    return next(iter(allowed), "*")


@_app.after_request
def _add_cors_headers(resp: Response) -> Response:
    origin = _resolve_cors_origin(request.headers.get("Origin"))
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    if origin != "*":
        resp.headers.add("Vary", "Origin")
    return resp


@_app.route("/api/v1/audio", methods=["POST", "OPTIONS"])
def converse() -> Response:  # noqa: D401 - simple endpoint
    if request.method == "OPTIONS":
        return _app.make_response(("", 204))

    if "audio" not in request.files:
        return jsonify({"error": "Expected form field 'audio'"}), 400

    session_id = request.form.get("session_id") or str(uuid.uuid4())

    try:
        user_audio, in_sr = _decode_audio(request.files["audio"])  # noqa: F841
        gen = _generate(user_audio, in_sr)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    if gen.audio is None or gen.audio.size == 0:
        return jsonify({
            "error": "Model produced no audio output",
            "session_id": session_id,
            "model": MODEL_NAME,
            "transcript": gen.text,
        }), 502

    wav_bytes = _encode_wav(gen.audio, OUTPUT_SAMPLE_RATE)
    resp = Response(wav_bytes, mimetype="audio/wav")
    resp.headers["X-Session-Id"] = session_id
    resp.headers["X-Model"] = MODEL_NAME
    transcript = gen.text.replace("\n", " ").strip()
    if len(transcript) > 512:
        transcript = transcript[:509] + "..."
    resp.headers["X-Transcript"] = transcript
    resp.headers["Content-Length"] = str(len(wav_bytes))
    return resp


def _transcribe(user_audio: np.ndarray, sample_rate: int) -> str:
    model, processor, inputs, audio_path = _prepare_generation_inputs(
        user_audio, sample_rate, system_prompt=SYSTEM_PROMPT_TRANSCRIBE
    )
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
            )
    finally:
        _cleanup_temp_file(audio_path)

    sequences = output
    if isinstance(output, (list, tuple)):
        sequences = output[0]
    if torch.is_tensor(sequences):
        sequences = sequences.cpu()
    decoded = processor.batch_decode(
        sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    transcript = decoded[0] if decoded else ""
    marker = "\nuser\n\nassistant\n"
    if marker in transcript:
        result = transcript.split(marker, 1)[-1]
    else:
        result = transcript
    return result.strip()


def _get_image_collection() -> Collection:
    global _mongo_client, _image_collection
    if _image_collection is not None:
        return _image_collection
    with _mongo_lock:
        if _image_collection is not None:
            return _image_collection
        _mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
        db = _mongo_client[MONGODB_DB_NAME]
        _image_collection = db[IMAGE_COLLECTION_NAME]
    return _image_collection


def _call_embed_service(
    texts: List[str], instruction: Optional[str] = None
) -> List[List[float]]:
    payload = {"texts": texts}
    if instruction:
        payload["instruction"] = instruction
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        EMBED_SERVICE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=EMBED_TIMEOUT_SECONDS) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"Embed service error: HTTP {resp.status}")
        body = resp.read()
    parsed = json.loads(body.decode("utf-8"))
    embeddings = parsed.get("embeddings")
    if not embeddings:
        raise RuntimeError("Embed service returned no embeddings")
    return embeddings


def _query_similar_images(vector: List[float], limit: int) -> List[dict]:
    collection = _get_image_collection()
    query_vector = [float(v) for v in vector]
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD_NAME,
                "queryVector": query_vector,
                "numCandidates": VECTOR_NUM_CANDIDATES,
                "limit": limit,
            }
        },
        {
            "$project": {
                "image_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = []
    for doc in collection.aggregate(pipeline):
        image_id = doc.get("image_id")
        score = doc.get("score")
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        results.append({"image_id": image_id, "score": score_val})
    return results


@_app.route("/api/v1/images/from-audio", methods=["POST", "OPTIONS"])
def image_search_from_audio() -> Response:
    if request.method == "OPTIONS":
        return _app.make_response(("", 204))

    if "audio" not in request.files:
        return jsonify({"error": "Expected form field 'audio'"}), 400

    top_k_param = request.form.get("top_k") or request.args.get("top_k")
    top_k = DEFAULT_TOP_K
    if top_k_param:
        try:
            parsed = int(top_k_param)
            if parsed > 0:
                top_k = min(parsed, 20)
        except ValueError:
            return jsonify({"error": "'top_k' must be an integer"}), 400

    embed_instruction = (
        request.form.get("instruction")
        or request.args.get("instruction")
        or EMBED_INSTRUCTION
    )

    try:
        user_audio, in_sr = _decode_audio(request.files["audio"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    try:
        transcript = _transcribe(user_audio, in_sr)
    except Exception as exc:  # noqa: BLE001
        _log.exception("Transcription failed")
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    if not transcript:
        return jsonify({"error": "Transcript was empty"}), 502

    try:
        embeddings = _call_embed_service(
            [transcript], instruction=embed_instruction)
    except (HTTPError, URLError, RuntimeError) as exc:
        _log.exception("Embedding service call failed")
        return jsonify({"error": f"Embedding request failed: {exc}"}), 502

    query_vector = embeddings[0]
    if not isinstance(query_vector, list) or not query_vector:
        return jsonify({"error": "Embedding vector missing or invalid"}), 502

    try:
        results = _query_similar_images(query_vector, limit=top_k)
    except PyMongoError as exc:
        _log.exception("Vector search failed")
        return jsonify({"error": f"Vector search failed: {exc}"}), 502

    image_ids = [r.get("image_id") for r in results if r.get("image_id")]

    return jsonify(
        {
            "transcript": transcript,
            "image_ids": image_ids,
            "results": results,
            "count": len(results),
        }
    )


if __name__ == "__main__":  # pragma: no cover - manual run
    host = os.environ.get("IMAGETALK_HOST", "0.0.0.0")
    port = int(os.environ.get("IMAGETALK_PORT", "8000"))
    debug = os.environ.get("IMAGETALK_DEBUG") == "1"
    _app.run(host=host, port=port, debug=debug)
