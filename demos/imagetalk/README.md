# ImageTalk - Wellcome Collection

A Wellcome D&T Firebreak Project - Presented on 26 Sep 2025

Talking with Images (Multi-modal Speech-to-Speech)

## Pitch

### Brief description of the idea

A system that allows people to engage in natural, spoken conversations with images, artworks, or objects. Instead of typing or reading, users can ask questions verbally and hear spoken responses, with the image itself serving as the focal point of the exchange. The model links visual analysis with curated knowledge (metadata, archives, expert commentary) to give context, stories, and insights directly through speech.

### Why is this important to you?

Wellcome Collection’s holdings are rich in imagery, artefacts, and visual materials. Many visitors—onsite or online—struggle to access or interpret these collections without reading-intensive interfaces. A multimodal speech-to-speech system lowers barriers for those with visual, cognitive, or literacy challenges, and it creates a more human, conversational way of exploring cultural heritage.

For me, it’s important because it aligns with Wellcome’s mission of accessibility and engagement: opening collections to broader audiences, sparking curiosity, and making knowledge discovery feel as natural as talking with a guide or friend.

### Output, what did you do/make? (to be filled in at the end of the week)

☐ Prototype demo where a user can ask a question aloud about a chosen Wellcome Collection image and receive a spoken response.

---

## How to run

### Run MongoDB Atlas

From `demos/imagetalk` directory:

```sh
cd backend/docker
bash run_mongodb_atlas.sh
```

Use `indeximages.ipynb` to index images into MongoDB. Note it requires the output parquets of `src/wc_simd/vlm_embed.py`

### Run embedding backend

From `demos/imagetalk` directory:

```sh
cd embed_backend
source .venv/bin/activate
python embed.py
```

### Run imagetalk backend

In another terminal and from `demos/imagetalk` directory:

```sh
cd backend
source .venv/bin/activate
python imagetalk.py
```

### Run frontend

In another terminal and from `demos/imagetalk` directory:

```sh
cd frontend
yarn next dev
```

Note the localhost url and use your browswer to view it.

## Installation and Setup

### Tools Required

- Astral UV

  ```sh
  sudo apt install pipx
  pipx ensurepath
  pipx install uv
  ```

- Node v24.8.0 and Yarn

  ```sh
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  # Restart terminal
  nvm install 24.8.0
  npm install -g yarn
  ```

### Install Dependencies

From `demos/imagetalk/embed_backend` directory:

```sh
uv sync
```

From `demos/imagetalk/backend` directory:

```sh
uv sync
```

From `demos/imagetalk/frontend` directory:

```sh
yarn install
```
