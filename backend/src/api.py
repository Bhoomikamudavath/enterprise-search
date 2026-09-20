import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.hybrid_retrieval import HybridRetriever
from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("enterprise_search")

limiter = Limiter(key_func=get_remote_address)

retriever: Optional[HybridRetriever] = None
retriever_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, retriever_load_error
    try:
        logger.info("Loading retrieval indexes...")
        retriever = HybridRetriever()
        logger.info("Indexes loaded successfully.")
    except Exception as e:
        retriever_load_error = str(e)
        logger.error(f"Failed to load indexes: {e}")
    yield


app = FastAPI(title="Enterprise Knowledge Search", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms}"
    )
    return response


class SearchResult(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: int
    url: str
    tags: List[str]


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: List[SearchResult]


def make_snippet(doc, query_terms, max_len=250):
    text = doc["question_body"] or doc["answer_body"]
    lower_text = text.lower()

    best_pos = 0
    for term in query_terms:
        pos = lower_text.find(term.lower())
        if pos != -1:
            best_pos = max(0, pos - 50)
            break

    snippet = text[best_pos:best_pos + max_len]
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + max_len < len(text):
        snippet = snippet + "..."
    return snippet


@app.get("/search", response_model=SearchResponse)
@limiter.limit(settings.rate_limit)
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=300, description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    mode: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    api_key: str = Header(..., alias="x-api-key"),
):
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if retriever is None:
        raise HTTPException(status_code=503, detail="Search index is not available")

    try:
        hits = retriever.search(q, top_k=top_k * 3 if tag else top_k, mode=mode)
    except Exception as e:
        logger.error(f"Search failed for query='{q}': {e}")
        raise HTTPException(status_code=500, detail="Search failed unexpectedly")

    if tag:
        hits = [h for h in hits if tag.lower() in [t.lower() for t in h["tags"]]]
        hits = hits[:top_k]

    query_terms = q.split()
    results = [
        SearchResult(
            doc_id=h["doc_id"],
            title=h["title"],
            snippet=make_snippet(h, query_terms),
            score=h["score"],
            url=h["url"],
            tags=h["tags"],
        )
        for h in hits
    ]

    return SearchResponse(query=q, mode=mode, results=results)


@app.get("/health")
def health():
    if retriever is None:
        return {
            "status": "unhealthy",
            "detail": retriever_load_error or "Retriever not yet loaded",
        }
    return {"status": "ok"}