# Enterprise Knowledge Search
### Hybrid (Keyword + Semantic) Retrieval vs Single-Mode Baselines

## Overview

This project implements and evaluates a hybrid retrieval search service that combines sparse keyword search (BM25, via OpenSearch) with dense semantic search (embeddings, via Qdrant), fused with Reciprocal Rank Fusion (RRF), and exposed behind an authenticated FastAPI backend with a React + TypeScript frontend.

The analysis demonstrates that **hybrid retrieval beats both single-mode baselines on overall ranking quality**, achieving 0.831 MRR vs 0.722 (BM25-only) and 0.818 (dense-only), while also surfacing an honest, query-type-specific finding: naive RRF fusion underperforms dense-only search specifically on paraphrased natural-language queries — a real trade-off documented below rather than hidden.

Built as a portfolio project simulating an internal company knowledge search tool, indexed against a public AI/ML Q&A corpus (Stack Exchange) as a stand-in for real internal engineering docs, runbooks, and Q&A threads.

---

## Project Structure

```text
enterprise-search/
│
├── backend/
│   ├── data/
│   │   ├── documents.jsonl            # Ingested corpus (question + best answer merged)
│   │   └── eval_queries.json          # 40 hand-labelled evaluation queries
│   ├── src/
│   │   ├── ingest.py                  # Posts.xml -> documents.jsonl
│   │   ├── opensearch_index.py        # Index into OpenSearch (BM25)
│   │   ├── qdrant_index.py            # Index embeddings into Qdrant (dense)
│   │   ├── hybrid_retrieval.py        # RRF fusion + graceful degradation
│   │   ├── update_document.py         # Incremental single-document reindex
│   │   ├── evaluate.py                # Evaluation harness (MRR, Recall@K)
│   │   ├── config.py                  # Environment-based settings
│   │   └── api.py                     # FastAPI app (auth, rate limit, logging, metrics)
│   ├── tests/
│   │   ├── test_hybrid_retrieval.py   # RRF fusion unit tests
│   │   └── test_api.py                # Full API surface tests
│   └── .env.example
│
├── frontend/
│   └── src/
│       ├── App.tsx                    # Search UI, mode toggle, highlighted snippets
│       ├── App.css
│       └── api.ts                     # Typed API client
│
├── docker-compose.yml                 # OpenSearch + Qdrant
├── requirements.txt
└── README.md
```

---

## Key Results

### Model Comparison (K=10, n=40 labelled queries)

| Mode | MRR | Recall@10 |
|------|-----|-----------|
| BM25 (keyword only) | 0.722 | 0.825 |
| Dense (semantic only) | 0.818 | 0.975 |
| Hybrid (RRF fusion) | 0.831 | 0.950 |

### By Query Type

| Type | BM25 MRR | Dense MRR | Hybrid MRR | BM25 R@10 | Dense R@10 | Hybrid R@10 |
|------|----------|-----------|------------|-----------|------------|-------------|
| Direct (n=8) | 1.000 | 0.906 | 1.000 | 1.000 | 1.000 | 1.000 |
| Identifier (n=17) | 0.833 | 0.803 | 0.873 | 0.941 | 1.000 | 1.000 |
| Paraphrase (n=15) | 0.447 | 0.789 | 0.694 | 0.600 | 0.933 | 0.867 |

### An Honest Finding: Hybrid Underperforms Dense-Only on Paraphrased Queries

On paraphrase-type queries specifically, dense-only search (MRR 0.789) beats hybrid (MRR 0.694), even though hybrid still comfortably beats BM25-only. This is a real, explainable effect, not noise.

BM25 is genuinely weak on paraphrased queries by construction — little to no word overlap exists between a paraphrased question and its matching document. When RRF fuses BM25's ranking into the mix, it has no way to know BM25's signal is unreliable for this query type; it blindly adds BM25's weak rank contribution to every candidate. This can push a document BM25 ranked moderately (coincidental keyword overlap, not the true best semantic match) above the correct document that dense search alone had ranked first.

**Takeaway:** naive RRF fusion is not free. When one retriever's signal is unreliable for a given query type, it can act as noise and degrade ranking quality even while preserving recall. A natural follow-up — not implemented here — would be query-type detection to route toward dense-weighted fusion for natural-language-looking queries, or a learned/weighted RRF variant.

### Where Hybrid Concretely Wins

For the query `"backpropagation"`, hybrid mode surfaces `"What is 'backprop'?"` in its top results — a document that ranked outside the top 5 in *both* individual BM25-only and dense-only result sets, but whose combined rank score across both retrievers was high enough to surface it after fusion. This is the direct, observable payoff of fusing two signals instead of relying on either alone.

### Infrastructure Migration Affected Retrieval Quality

The project was built first on in-memory `rank_bm25` + a flat FAISS file, then migrated to persistent OpenSearch + Qdrant for production-readiness. The migration was not score-neutral: original in-memory BM25 showed higher Recall@10 (0.875 overall, 0.733 on paraphrase) than OpenSearch-backed BM25 (0.825 overall, 0.600 on paraphrase). OpenSearch's default `k1`/`b` scoring parameters and analysis pipeline are not identical to `rank_bm25`'s defaults. Tuning these to recover parity is a reasonable follow-up, not pursued here in favor of completing the broader production-hardening scope.

---

## Technology Stack

* **Python 3.10+**
* **OpenSearch** — persistent BM25 keyword index, custom technical tokenizer
* **Qdrant** — persistent dense vector index (cosine similarity)
* **sentence-transformers** (`all-MiniLM-L6-v2`) — query/document embeddings
* **FastAPI / Uvicorn** — search API (auth, rate limiting, logging, metrics)
* **Pydantic / pydantic-settings** — request validation and environment config
* **slowapi** — rate limiting
* **prometheus-client** — metrics endpoint
* **pytest / httpx** — automated testing
* **React + TypeScript (Vite)** — frontend search UI
* **Docker / Docker Compose** — OpenSearch + Qdrant orchestration

---

## Dataset

**AI Stack Exchange** data dump, from the Internet Archive's Stack Exchange collection.

* **Source:** https://archive.org/details/stackexchange
* **Site used:** ai.stackexchange.com
* **Questions indexed:** 12,380 (each merged with its accepted or highest-scored answer into one knowledge unit)

Download `Posts.xml` for the site and place it in `backend/data/`.

---

## Environment Setup

Navigate to the project directory:

```bash
cd enterprise-search
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install backend dependencies (from project root):

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

---

## Running It Locally

```bash
# 1. Start OpenSearch and Qdrant (from project root)
docker compose up -d

# 2. Configure environment
cd backend
cp .env.example .env
# edit .env — set SEARCH_API_KEY to a real value

# 3. Ingest and index the corpus (one-time)
python -m src.ingest
python -m src.opensearch_index
python -m src.qdrant_index

# 4. Run the test suite
python -m pytest tests/ -v

# 5. Start the backend
python -m uvicorn src.api:app --reload

# 6. In a separate terminal, start the frontend
cd frontend
npm run dev
```

Frontend: `http://localhost:5173`
API interactive docs: `http://127.0.0.1:8000/docs`

---

## API

### Authentication

All `/search` requests require an `x-api-key` header, matching `SEARCH_API_KEY` in `backend/.env`.

### `GET /search`

| Endpoint | Description | Example |
|----------|-------------|---------|
| GET /search | Hybrid/keyword/semantic search | /search?q=backpropagation&mode=hybrid&top_k=5 |
| GET /health | Health check (OpenSearch + Qdrant connectivity) | /health |
| GET /metrics | Prometheus metrics | /metrics |

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | required | Search query |
| `top_k` | int | 10 | Number of results (1-50) |
| `mode` | string | `hybrid` | `bm25`, `dense`, or `hybrid` |
| `tag` | string | none | Filter results by tag |

### Example Request

```
GET /search?q=backpropagation&mode=hybrid&top_k=5
```

### Example Response

```json
{
  "query": "backpropagation",
  "mode": "hybrid",
  "results": [
    {
      "doc_id": "3962",
      "title": "How do I know if my backpropagation is implemented correctly?",
      "snippet": "I'm working on an implementation of the backpropagation algorithm...",
      "score": 8,
      "url": "https://ai.stackexchange.com/questions/3962",
      "tags": ["neural-networks", "backpropagation", "algorithm-request"]
    }
  ]
}
```

If a backing retriever is temporarily unavailable, `mode` in the response reflects the actual mode used, e.g. `"bm25-only (qdrant unavailable)"`, rather than failing the request outright.

---

## Reusable Modules

### hybrid_retrieval.py

Contains the retriever with a shared interface across all three modes:

```python
from src.hybrid_retrieval import HybridRetriever

retriever = HybridRetriever()
results, actual_mode = retriever.search("backpropagation", top_k=10, mode="hybrid")
```

### evaluate.py

Contains the evaluation harness and ranking metrics:

```python
from src.evaluate import evaluate, load_eval_set

eval_set = load_eval_set()
results = evaluate(retriever, eval_set)
```

---

## Key Decisions

1. **Document unit = question + best answer, merged.** Each indexed document combines a question with its accepted (or highest-scored) answer into one self-contained knowledge unit — closer to what a real runbook search needs to return than a bare question or an unlabelled answer.

2. **Custom tokenizer for technical terms.** Both the BM25 implementation and OpenSearch's index use a tokenizer that treats hyphenated/underscored terms (`t-SNE`, `k-means`, `L1-regularization`) as single tokens instead of splitting on `-`/`_`. Default tokenization would shred exactly the identifier-style queries this project is meant to handle well.

3. **RRF over score-normalized fusion.** BM25 scores and cosine similarity live on incompatible scales. Reciprocal Rank Fusion sidesteps calibration entirely by fusing on rank position, which is simpler and more robust at this scope.

4. **Real persistent infrastructure, not in-memory demo structures.** The project was migrated mid-build from `rank_bm25` + a flat FAISS file to OpenSearch + Qdrant, both running via Docker with persistent volumes — enabling incremental single-document updates instead of full corpus rebuilds.

5. **Graceful degradation over hard failure.** If one retriever backend is down, hybrid mode falls back to whichever is healthy rather than failing the whole request, and the response is explicit about the degraded mode used.

6. **API key auth over no auth.** A single shared API key is a deliberately minimal, scope-appropriate choice — real per-user identity (OAuth/SSO) is called out explicitly as future work, not silently skipped.

---

## Trade-Off Analysis

| Dimension | BM25 (Keyword) | Dense (Semantic) | Hybrid (RRF) |
|-----------|-----------------|-------------------|----------------|
| Exact identifiers / error codes | Strong | Weak | Strong |
| Paraphrased natural language | Weak | Strong | Moderate (weaker than dense-only, see finding above) |
| Interpretability | High (term overlap) | Low (embedding distance) | Moderate |
| Infra requirement | Search engine (OpenSearch) | Vector DB (Qdrant) + embedding model | Both |
| Failure mode | Fails on synonyms | Fails on rare exact tokens | Degrades gracefully to whichever retriever is healthy |
| Latency | Low | Moderate (embedding + ANN search) | Highest (both retrievers + fusion) |

---

## Conclusion: Which Mode to Ship

Ship **hybrid as the default mode**, with the honest caveat documented above rather than hidden.

Hybrid wins on overall MRR and near-ties dense-only on Recall@10, while meaningfully beating BM25-only on both metrics across every query type except pure paraphrase. For an internal knowledge tool where queries mix exact error codes, method names, and natural-language questions — the exact problem this project targets — hybrid's balance across query types is the right default, even though it is not uniformly the best mode for every query type individually.

### Production Strategy

* **Default:** hybrid mode
* **Power users searching a known exact term (error code, method name):** BM25-only mode remains available and is measurably strongest here
* **Follow-up worth prioritizing:** query-type detection to route paraphrase-shaped queries toward dense-weighted or dense-only scoring, directly motivated by the finding above

### Why the Infrastructure Migration Mattered

Moving from in-memory `rank_bm25`/FAISS to OpenSearch/Qdrant was necessary for real production use (persistence, incremental updates, no full-reload-on-restart) but was not score-neutral. This is documented explicitly rather than treated as a footnote — infrastructure choices affect retrieval quality, not just operational characteristics, and pretending otherwise would be dishonest about what "production-ready" cost here.

---

## Production Hardening Implemented

* API key authentication on all search requests
* Rate limiting (configurable, default 30 requests/minute per client)
* Structured request logging (method, path, status, duration, per-request ID)
* Environment-based configuration (`.env` / `pydantic-settings`), no hardcoded secrets in source
* Persistent search infrastructure (OpenSearch + Qdrant via Docker, not in-memory files)
* Graceful degradation when a search backend is unavailable, verified by manually stopping each service
* Prometheus metrics endpoint, verified against real traffic
* Incremental document re-indexing (single-document upsert, not full rebuild)
* 9 automated tests (pytest) covering RRF fusion logic and the full API surface, including auth and degradation paths

---

## Explicitly Out of Scope, and Why

* **Real company data / ingestion from Confluence, Notion, Slack, etc.** This project indexes a public Stack Exchange corpus as a stand-in. Indexing real internal company knowledge requires access to those systems — a separate, larger scoping conversation.
* **Cloud deployment / CI-CD pipeline.** Everything here runs locally via Docker Compose. A real deployment (e.g. to Cloud Run) needs a deployment target and pipeline setup outside this project's scope.
* **Per-user authentication / document-level permissions.** A single shared API key is used rather than real user identity or per-document access control. A real internal tool indexing sensitive company docs would need this before production use.
* **Automated re-indexing triggers.** `update_document.py` proves the incremental-update mechanism works, but nothing currently triggers it automatically — a real system needs webhooks from source systems or a scheduled job, both dependent on infrastructure decisions out of scope here.

---

## Known Limitations

* Ground truth is one correct document per query — real queries often have multiple valid answers, which Recall@K/MRR as implemented here don't capture
* OpenSearch's BM25 parameters (`k1`, `b`) are left at defaults, not tuned to match `rank_bm25`'s original scoring behavior
* RRF fusion is unweighted — the paraphrase-query finding above suggests a weighted or query-type-aware variant would help
* No cross-encoder re-ranking stage on the fused candidate pool
* No latency benchmarking across modes, despite metrics infrastructure being in place to support it
* Frontend uses a single hardcoded shared API key, matching the backend's single-key auth model — not per-user
* No automated end-to-end (frontend + backend integration) tests — only backend unit/API tests