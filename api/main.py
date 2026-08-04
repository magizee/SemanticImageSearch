import os
import pickle
import random
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mygrad import Tensor

from semantic_search.text_embedding import embed_query_from_cache, to_token
from semantic_search.image_model import ImageDescriptors, load_model
from semantic_search.database import Database

# macOS's Accelerate BLAS backend raises spurious divide-by-zero/overflow/
# invalid-value RuntimeWarnings on ordinary, finite matmuls (verified: output
# has no actual NaN/Inf). Harmless false positive, not a real numeric issue.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = REPO_ROOT / "mynn_model_weights_final.pkl"

CAPTION_DIM = 200
IMAGE_DIM = 512
EMBEDDING_DIM = 128

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

# only these prefixes may be fetched by /image-proxy -- without this
# allowlist, a URL-fetching proxy is an open SSRF vector
ALLOWED_IMAGE_URL_PREFIXES = (
    "http://images.cocodataset.org/",
    "https://images.cocodataset.org/",
)

app = FastAPI(title="Semantic Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- load everything once at startup (all small/fast thanks to the
# trimmed query_vectors cache -- no gensim/full GloVe needed here) ---
with open(DATA_DIR / "idf_map.pkl", "rb") as f:
    idf_map = pickle.load(f)["idf_map"]
with open(DATA_DIR / "query_vectors.pkl", "rb") as f:
    query_vectors = pickle.load(f)
with open(DATA_DIR / "image_metadata.pkl", "rb") as f:
    image_metadata = pickle.load(f)

model = ImageDescriptors(caption_dimension=CAPTION_DIM, image_dimension=IMAGE_DIM, embedding_dimension=EMBEDDING_DIM)
load_model(model, str(MODEL_PATH))

_db = Database()
_db.image_embeddings = _db.load_image_database(str(DATA_DIR / "database.pkl"))
_image_ids = list(_db.image_embeddings.keys())
_embedding_matrix = np.array(list(_db.image_embeddings.values()))

# precomputed once for nearest-word lookups (see nearest_words below)
_vocab_words = list(query_vectors.keys())
_vocab_vectors = np.stack([query_vectors[w] for w in _vocab_words])
_vocab_norms = np.linalg.norm(_vocab_vectors, axis=1)


def nearest_words(query_vec: np.ndarray, top_k: int = 24, exclude=frozenset()) -> List[str]:
    """Finds the vocab words whose GloVe vectors are most similar (by cosine
    similarity) to query_vec -- a lightweight way to surface what the
    embedding space "thinks" is near a query, for decorative display.
    """
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    sims = (_vocab_vectors @ query_vec) / (_vocab_norms * query_norm)
    order = np.argsort(sims)[::-1]
    words = []
    for idx in order:
        word = _vocab_words[idx]
        if word not in exclude:
            words.append(word)
        if len(words) >= top_k:
            break
    return words


class SearchRequest(BaseModel):
    query: str
    top_k: int = 12


class SearchResult(BaseModel):
    image_url: Optional[str]
    caption: Optional[str]
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    similar_words: List[str]


@app.get("/")
def root():
    return {"service": "Semantic Image Search API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "num_images": len(_image_ids)}


@app.get("/image-proxy")
def image_proxy(url: str):
    # COCO's image host is http-only; a browser on the https-served
    # frontend can silently fail to load http <img> resources (mixed
    # content), so we fetch server-side and re-serve from our own
    # https origin instead.
    if not url.startswith(ALLOWED_IMAGE_URL_PREFIXES):
        raise HTTPException(status_code=400, detail="URL not allowed")

    upstream = requests.get(url, timeout=10)
    upstream.raise_for_status()
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get("Content-Type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/random", response_model=List[SearchResult])
def random_images(count: int = 16):
    count = max(1, min(count, len(_image_ids)))
    results = []
    for image_id in random.sample(_image_ids, count):
        meta = image_metadata.get(image_id, {})
        results.append(SearchResult(
            image_url=meta.get("url"),
            caption=meta.get("caption"),
            score=0.0,
        ))
    return results


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    query_vec = embed_query_from_cache(req.query, query_vectors, idf_map)
    caption_emb = model.caption_embed(Tensor(query_vec.reshape(1, -1))).data.reshape(-1)
    sims = _db.cosine_similarity_matrix(_embedding_matrix, caption_emb)

    top_k = max(1, min(req.top_k, len(_image_ids)))
    top_indices = np.argsort(sims)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        image_id = _image_ids[idx]
        meta = image_metadata.get(image_id, {})
        results.append(SearchResult(
            image_url=meta.get("url"),
            caption=meta.get("caption"),
            score=float(sims[idx]),
        ))

    query_tokens = set(to_token(req.query))
    similar_words = nearest_words(query_vec, top_k=24, exclude=query_tokens)

    return SearchResponse(results=results, similar_words=similar_words)
