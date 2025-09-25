# ImageTalk Frontend

This Next.js application provides the push-to-talk experience for the ImageTalk demo. It records microphone audio in the browser, sends the audio clip to the backend running Qwen2.5-Omni, and plays the returned synthetic speech immediately.

## Prerequisites

- Node.js 18+ (for the Next.js app)
- Python 3.10 with the dependencies defined in `pyproject.toml` (for the backend)

## Running the backend

From the repository root:

```bash
cd demos/imagetalk/backend
python3 imagetalk.py
```

Environment variables:

- `IMAGETALK_HOST` and `IMAGETALK_PORT` (optional) control the bind address. Defaults to `0.0.0.0:8000`.
- `IMAGETALK_ALLOWED_ORIGINS` (optional) can list the frontend origin, e.g. `http://localhost:3000`.
- `IMAGETALK_SPEAKER` (optional) selects the talker voice (defaults to `Chelsie`).

## Running the frontend

1. Install dependencies once:
   ```bash
   cd demos/imagetalk/frontend
   npm install
   ```

2. Create `.env.local` alongside `package.json` if you need to override the backend URL:
   ```bash
   echo "NEXT_PUBLIC_IMAGETALK_API_BASE_URL=http://localhost:8000" > .env.local
   ```
   The value should match the host/port where the Flask backend is listening.

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open the demo at `http://localhost:3000/demo` and follow the on-screen instructions.

## Key files

- `app/demo/page.tsx` – client-side logic for microphone capture and playback.
- `components/Header.tsx` – shared navigation with a shortcut to the demo page.
- `app/page.tsx` – landing page with entry points into the experience.

## Known limitations

- Browser recording relies on the Web Audio API; ensure you test on a browser that supports `MediaRecorder` (Chrome, Edge, Firefox).
- The repo does not bundle the large Qwen model weights; make sure you have network access or a local snapshot available when the backend starts.
