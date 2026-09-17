from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from src.hybrid_retrieval import HybridRetriever

app = FastAPI(title="Enterprise Knowledge Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

retriever = HybridRetriever()


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
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    mode: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    hits = retriever.search(q, top_k=top_k * 3 if tag else top_k, mode=mode)

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
    return {"status": "ok"}