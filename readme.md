# Semantic Image Search

Search 82,612 MS COCO images by meaning instead of keywords — type "a dog running on the beach" and get back images whose *content* matches, not just images whose captions happen to contain those words.

**Live demo:** backend at [semanticimagesearch.onrender.com](https://semanticimagesearch.onrender.com) (Render free tier — cold-starts after ~15 min idle, first request can take 30-60s). Frontend is deployed on Vercel.

## How it works

Captions and images are mapped into a shared 128-dimensional embedding space, so a piece of text and an image can be compared directly by cosine similarity.

**Text → vector.** A caption or query is tokenized, then embedded as an IDF-weighted average of its words' [GloVe-200](https://nlp.stanford.edu/projects/glove/) vectors, L2-normalized. IDF weights are computed once over the ~413K-caption COCO corpus (`semantic_search/text_embedding.py`, cached in `data/idf_map.pkl`).

**Image → vector.** Each image already has a 512-dim descriptor from a pretrained ResNet-18 (provided by the dataset). No image processing happens at query time — everything downstream is fast, precomputed dense math.

**Joint embedding model.** Two independent dense layers — `caption_embed: 200→128` and `image_embed: 512→128` — project captions and images into the same space (`semantic_search/image_model.py`). Trained with margin ranking loss on (caption, true image, random confusor image) triples: 500 epochs over all 413,258 COCO captions, reaching 98.96% pairwise-ranking accuracy (`train.py`). Known limitation: training uses *random* negatives, which is an easier task than real top-k retrieval against all 82K images — replacing it with in-batch negatives (treating every other example in a batch as an additional negative) is the natural next step for search-quality improvement.

**Search.** `build_database.py` runs every image's descriptor through the trained `image_embed` layer once, offline, and stores the resulting 82,612 embeddings (`data/database.pkl`). At request time the API just embeds the query text, runs it through `caption_embed`, and does a cosine-similarity scan against that precomputed matrix (`semantic_search/database.py`) — no model inference over images ever happens live.

## Serving

The FastAPI backend (`api/main.py`) is deliberately lean: it never imports `gensim` or loads the full 660MB GloVe file. Instead it loads a ~17MB cache of just the GloVe vectors for words that actually appear in the COCO vocabulary (`data/query_vectors.pkl`) — safe because any word outside that vocabulary already contributes zero weight to the embedding.

COCO's image host (`images.cocodataset.org`) is HTTP-only, which browsers treat as mixed content on the HTTPS-served frontend. `/image-proxy` re-fetches each image server-side (a pooled `httpx.AsyncClient`) and re-serves it from the API's own HTTPS origin.

| Endpoint | Purpose |
|---|---|
| `POST /search` | Embed a query, return top-k images by cosine similarity + nearest GloVe-space words |
| `GET /random` | N random images, for the idle-state grid |
| `GET /image-proxy` | HTTPS relay for an allowlisted COCO image URL |
| `GET /health` | Liveness check |

## Frontend

React 19 + Vite, plain CSS (no framework) — a grayscale, square-grid, editorial look. The result grid loads a random set of images on page load; submitting a search flips each card over (a real CSS 3D transform) to its new image at its own randomized moment rather than swapping everything at once. Hovering a card fades in its MS COCO caption over a dark scrim.

## Project structure

```
semantic_search/          importable pipeline package
  coco_organizer.py         COCO annotations -> image/caption ID mappings
  text_embedding.py         tokenization, IDF, GloVe caption/query embedding
  image_model.py            ImageDescriptors model, loss, save/load
  database.py                embedding storage + cosine-similarity search
train.py                  builds (caption, image, confusor) triples, trains the model
build_database.py         runs the trained model over all images -> data/database.pkl
data/                      cached artifacts (idf_map, trimmed GloVe cache, image database/metadata)
mynn_model_weights_final.pkl   trained model weights
api/                       FastAPI backend (Dockerfile included)
frontend/                  React + Vite frontend
requirements.txt          data-pipeline / training dependencies
api/requirements.txt      lean serving-only dependencies
```

`SemanticImageSearchOverview_STUDENT.ipynb`, `makingCaptionVectors.ipynb`, and `profile.ipynb` are early exploratory notebooks, kept for reference.

## Running locally

**Backend**
```
python3 -m venv venv && source venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

**Frontend**
```
cd frontend
npm install
npm run dev
```
Set `VITE_API_URL` in `frontend/.env` to point at the backend (defaults to `http://127.0.0.1:8000`).

**Rebuilding the model / database from scratch** needs the full pipeline dependencies (`pip install -r requirements.txt`, includes `gensim`, `mygrad`, `mynn`) plus the MS COCO caption/ResNet data — see `train.py` and `build_database.py`.
