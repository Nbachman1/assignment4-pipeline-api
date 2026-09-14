"""FastAPI service for the CS-quotes semantic search pipeline.

Loads pipeline.joblib once at import time (never per-request) and exposes:
  GET  /health  - liveness + artifact-loaded status
  GET  /info    - artifact metadata (steps, corpus size, sklearn version, ...)
  POST /search  - runs a query through the fitted pipeline and returns the
                  top_k nearest quotes by cosine similarity

Local dev:
    uvicorn serve:app --reload
    -> http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# pipeline_def must be imported so joblib can resolve TextCleaner when
# unpickling the bundle below.
from pipeline_def import TextCleaner  # noqa: F401

ARTIFACT_PATH = Path(__file__).parent / "pipeline.joblib"

_bundle: dict | None = None
_load_error: str | None = None

try:
    _bundle = joblib.load(ARTIFACT_PATH)
except Exception as exc:  # noqa: BLE001 - we want to surface any load failure as 503
    _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="CS Quotes Search API",
    description="Fitted TF-IDF search pipeline over a corpus of programming/CS quotes.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_bundle() -> dict:
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline artifact not loaded: {_load_error or 'unknown error'}",
        )
    return _bundle


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, description="Free-text search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    text: str
    author: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/health")
def health():
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline artifact not loaded: {_load_error or 'unknown error'}",
        )
    return {"status": "ok"}


@app.get("/info")
def info():
    bundle = _require_bundle()
    return bundle["metadata"]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    bundle = _require_bundle()
    pipeline = bundle["pipeline"]
    matrix = bundle["matrix"]
    documents = bundle["documents"]

    query_vec = pipeline.transform([req.query])
    scores = cosine_similarity(query_vec, matrix).ravel()
    top_idx = scores.argsort()[::-1][: req.top_k]

    results = [
        SearchResult(text=documents[i]["text"], author=documents[i]["author"], score=round(float(scores[i]), 4))
        for i in top_idx
    ]
    return SearchResponse(query=req.query, results=results)
