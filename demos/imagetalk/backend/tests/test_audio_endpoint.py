import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Direct local import of backend module.
import imagetalk


@pytest.fixture()
def client():
    return imagetalk._app.test_client()


@pytest.mark.slow
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
