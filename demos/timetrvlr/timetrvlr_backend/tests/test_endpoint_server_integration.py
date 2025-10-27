import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import requests


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    addr, port = s.getsockname()
    s.close()
    return port


@pytest.mark.integration
# @pytest.mark.skipif(
#     os.getenv("LOAD_REAL_MODEL") not in {"1", "true", "TRUE", "yes", "on"},
#     reason="Set LOAD_REAL_MODEL=1 to run real server + HTTP inference test",
# )
def test_embed_endpoint_via_http_with_real_model(tmp_path):
    """Starts the Flask server and calls /embed over HTTP.

    - Uses a subprocess to avoid mocked modules from unit tests.
    - Binds to 127.0.0.1:<random-port> and probes /health until ready.
    - Posts to /embed which triggers real model load + inference.
    - Skipped by default; opt-in with LOAD_REAL_MODEL=1.
    - Override model via TIMETRVLR_MODEL_ID to use a smaller/cached model.
    """

    backend_main_path = (
        Path(__file__).resolve().parents[1] / "src" / "main.py"
    )
    assert backend_main_path.exists(
    ), f"Backend main not found: {backend_main_path}"

    port = _pick_free_port()

    server_script = textwrap.dedent(
        f"""
        import os, sys, importlib.util
        from pathlib import Path

        module_path = Path(r"{str(backend_main_path)}")
        spec = importlib.util.spec_from_file_location("timetrvlr_backend_main_server", str(module_path))
        assert spec and spec.loader, f"Unable to load spec for {{module_path}}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        model_id = os.getenv("TIMETRVLR_MODEL_ID", module.MODEL_ID)
        module.service = module.EmbeddingService(model_id=model_id)

        host = os.environ.get("HOST", "127.0.0.1")
        port = int(os.environ["PORT"])
        # Run a single process without reloader so the parent can manage it
        module.app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        """
    )

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HOST", "127.0.0.1")
    env["PORT"] = str(port)

    # Start server
    server = subprocess.Popen(
        [sys.executable, "-c", server_script],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"

    try:
        # Wait for health endpoint to be ready (server startup only)
        t0 = time.time()
        health_ok = False
        while time.time() - t0 < 60:
            if server.poll() is not None:
                raise RuntimeError(
                    "Server process exited before becoming ready")
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    health_ok = True
                    break
            except Exception:
                time.sleep(0.5)
        assert health_ok, "Server did not become healthy in time"

        # Call /embed: this will trigger real model load and inference
        payload = {"texts": ["hello world"]}
        resp = requests.post(f"{base_url}/embed", json=payload, timeout=1200)
        assert resp.status_code == 200, f"/embed failed: {resp.status_code} {resp.text}"

        data = resp.json()
        assert data.get("count") == 1
        embs = data.get("embeddings")
        assert isinstance(embs, list) and len(embs) == 1
        assert isinstance(embs[0], list) and len(embs[0]) > 0
        assert all(isinstance(v, (int, float)) for v in embs[0])
    finally:
        # Gracefully stop the server
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
