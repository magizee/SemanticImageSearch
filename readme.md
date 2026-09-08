# Semantic Image Search

Search 82,612 MS COCO images by meaning instead of keywords — type "grassy field" and get back images of grassy fields, rather than captions that contain grassy fields.

**Live demo:** https://semantic-image-search-pied.vercel.app 

This repository is adapted from a semantic image search project originally developed by my team through MIT Beaver Works Summer Institute (BWSI). I later revisited the project to complete the remaining setup and deploy it as a standalone web application. The original repository can be found here: https://github.com/magizee/cogworks-langproj

## How it works

Captions and images are mapped into a shared 128-dimensional embedding space, so a piece of text and an image can be compared directly by cosine similarity.

**Text → vector.** A caption or query is tokenized, then embedded as an IDF-weighted average of its words' [GloVe-200](https://nlp.stanford.edu/projects/glove/) vectors, L2-normalized. IDF weights are computed once over the ~413K-caption COCO corpus (`semantic_search/text_embedding.py`, cached in `data/idf_map.pkl`).

**Image → vector.** Each image already has a 512-dim descriptor from a pretrained ResNet-18 (provided by the dataset). No image processing happens at query time.

**Joint embedding model.** Two independent dense layers — `caption_embed: 200→128` and `image_embed: 512→128` — project captions and images into the same space (`semantic_search/image_model.py`). Trained with margin ranking loss on (caption, true image, random confusor image) triples: 500 epochs over all 413,258 COCO captions, reaching 98.96% pairwise-ranking accuracy (`train.py`). Known limitation: training uses *random* negatives, which is an easier task than real top-k retrieval against all 82K images — replacing it with in-batch negatives (treating every other example in a batch as an additional negative) is the natural next step for search-quality improvement.

**Search.** `build_database.py` runs every image's descriptor through the trained `image_embed` layer once, offline, and stores the resulting 82,612 embeddings (`data/database.pkl`). At request time the API just embeds the query text, runs it through `caption_embed`, and does a cosine-similarity scan against that precomputed matrix (`semantic_search/database.py`) — no model inference over images ever happens live.


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
