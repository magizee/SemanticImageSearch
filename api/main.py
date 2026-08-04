import os
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mygrad import Tensor

from semantic_search.text_embedding import embed_query_from_cache
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


class SearchRequest(BaseModel):
    query: str
    top_k: int = 12


class SearchResult(BaseModel):
    image_url: Optional[str]
    caption: Optional[str]
    score: float


@app.get("/")
def root():
    return {"service": "Semantic Image Search API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "num_images": len(_image_ids)}


@app.post("/search", response_model=List[SearchResult])
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
    return results
