import io
import os
from pathlib import Path

import json
import urllib.request

import numpy as np
import pytest
import soundfile as sf

# Direct local import of backend module.
import imagetalk


@pytest.fixture()
def client():
    return imagetalk._app.test_client()


def test_audio_in_audio_out_endpoint(client):
    """Integration test for mandatory audio -> audio behavior.

    Contract: successful response MUST be audio/wav with transcript header.
    If backend cannot produce audio it should return 5xx.

    Skips: env flag or missing fixture.
    """
    if os.getenv("IMAGETALK_SKIP_MODEL_TEST") == "1":
        pytest.skip("Skipping heavy model test via IMAGETALK_SKIP_MODEL_TEST")

    wav_fixture = Path(__file__).parent / "bcn_weather_short.wav"
    if not wav_fixture.exists():
        pytest.skip("Test audio sample missing")

    with wav_fixture.open("rb") as f:
        wav_buf = io.BytesIO(f.read())
        wav_buf.seek(0)

    response = client.post(
        "/api/v1/audio",
        data={"audio": (wav_buf, wav_fixture.name)},
        content_type="multipart/form-data",
    )

    if response.status_code != 200:
        # Surface backend error payload for debugging
        raise AssertionError(
            f"Expected 200, got {response.status_code}: {response.get_data(as_text=True)}")

    ctype = response.headers.get("Content-Type", "")
    assert ctype.startswith(
        "audio/wav"), f"Unexpected content type: {ctype} body={response.get_data()[:120]}"  # limit body preview

    wav_bytes = response.data
    assert len(wav_bytes) > 44  # WAV header minimal size
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    assert audio.ndim == 1 and audio.size > 0
    assert np.isfinite(audio).all()
    assert isinstance(sr, int)
    # Persist generated audio for manual inspection (not treated as a golden
    # file)
    out_dir = Path(__file__).parent / "_outputs"
    out_dir.mkdir(exist_ok=True)
    session_id = response.headers.get("X-Session-Id", "unknown_session")
    out_path = out_dir / f"generated_{session_id}.wav"
    with out_path.open("wb") as f_out:
        f_out.write(wav_bytes)
    # Provide a lightweight assertion that file was written
    assert out_path.exists() and out_path.stat().st_size == len(wav_bytes)
    # Headers verification
    assert response.headers.get("X-Session-Id")
    assert response.headers.get("X-Model") == imagetalk.MODEL_NAME
    transcript = response.headers.get("X-Transcript", "")
    assert transcript.strip() != ""
    # Optionally store transcript alongside audio for later review
    transcript_path = out_path.with_suffix(".txt")
    with transcript_path.open("w", encoding="utf-8") as tf:
        tf.write(transcript + "\n")


def _embed_service_available() -> bool:
    try:
        req = urllib.request.Request(
            imagetalk.EMBED_SERVICE_URL,
            data=json.dumps({"texts": ["ping"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=imagetalk.EMBED_TIMEOUT_SECONDS) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - best effort probe
        return False


def _mongo_available() -> bool:
    try:
        collection = imagetalk._get_image_collection()
        # Provide a lightweight query to confirm DB reachability.
        collection.find_one()
        return True
    except Exception:  # noqa: BLE001 - include connection/auth failures
        return False


def test_images_from_audio_endpoint(client):
    # if os.getenv("IMAGETALK_SKIP_MODEL_TEST") == "1":
    #     pytest.skip("Skipping heavy model test via IMAGETALK_SKIP_MODEL_TEST")

    # if not _embed_service_available():
    #     pytest.skip("Embed service not reachable")

    # if not _mongo_available():
    #     pytest.skip("MongoDB image collection not reachable")

    wav_fixture = Path(__file__).parent / "bcn_weather_short.wav"
    # if not wav_fixture.exists():
    #     pytest.skip("Test audio sample missing")

    with wav_fixture.open("rb") as f:
        wav_buf = io.BytesIO(f.read())
        wav_buf.seek(0)

    response = client.post(
        "/api/v1/images/from-audio",
        data={"audio": (wav_buf, wav_fixture.name)},
        content_type="multipart/form-data",
    )

    if response.status_code != 200:
        raise AssertionError(
            f"Expected 200, got {response.status_code}: {response.get_data(as_text=True)}"
        )

    payload = response.get_json()
    assert payload is not None
    assert "transcript" in payload and payload["transcript"].strip() != ""
    assert "image_ids" in payload
    image_ids = payload["image_ids"]
    assert isinstance(image_ids, list)
    assert 0 < len(image_ids) <= imagetalk.DEFAULT_TOP_K
    for iid in image_ids:
        assert isinstance(iid, str) and iid.strip() != ""

    assert "results" in payload
    for result in payload["results"]:
        assert "image_id" in result
        assert result.get("image_id") in image_ids

    assert payload.get("count") == len(payload["results"])
