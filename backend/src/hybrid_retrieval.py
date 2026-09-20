import json
import logging
from pathlib import Path
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("enterprise_search")

DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"
OPENSEARCH_INDEX = "knowledge-docs"
QDRANT_COLLECTION = "knowledge-docs"
RRF_K = 60


class RetrieverUnavailable(Exception):
    def __init__(self, source, original_error):
        self.source = source
        self.original_error = original_error
        super().__init__(f"{source} is unavailable: {original_error}")


class HybridRetriever:
    def __init__(self):
        self.opensearch = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            use_ssl=False,
            verify_certs=False,
        )
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.model = SentenceTransformer(MODEL_NAME)

        self.docs_by_id = {}
        with open(DATA_DIR / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs_by_id[d["doc_id"]] = d

    def check_health(self):
        status = {"opensearch": "ok", "qdrant": "ok"}
        try:
            self.opensearch.cluster.health(request_timeout=2)
        except Exception as e:
            status["opensearch"] = f"unreachable: {e}"
        try:
            self.qdrant.get_collections()
        except Exception as e:
            status["qdrant"] = f"unreachable: {e}"
        return status

    def search_bm25(self, query, top_k):
        try:
            response = self.opensearch.search(
                index=OPENSEARCH_INDEX,
                body={"query": {"match": {"full_text": query}}, "size": top_k},
            )
            return [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]
        except Exception as e:
            raise RetrieverUnavailable("opensearch", e)

    def search_dense(self, query, top_k):
        try:
            query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
            hits = self.qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=top_k,
            ).points
            return [hit.payload["doc_id"] for hit in hits]
        except Exception as e:
            raise RetrieverUnavailable("qdrant", e)

    def search_hybrid(self, query, top_k, candidate_pool=50):
        bm25_ranked = None
        dense_ranked = None
        degraded_from = None

        try:
            bm25_ranked = self.search_bm25(query, candidate_pool)
        except RetrieverUnavailable as e:
            logger.warning(f"BM25 unavailable during hybrid search: {e}")
            degraded_from = "opensearch"

        try:
            dense_ranked = self.search_dense(query, candidate_pool)
        except RetrieverUnavailable as e:
            logger.warning(f"Dense retrieval unavailable during hybrid search: {e}")
            degraded_from = "qdrant"

        if bm25_ranked is None and dense_ranked is None:
            raise RetrieverUnavailable("both opensearch and qdrant", "both retrievers failed")

        if bm25_ranked is None:
            logger.info("Degraded hybrid search: returning dense-only results")
            return dense_ranked[:top_k], "dense-only (opensearch unavailable)"

        if dense_ranked is None:
            logger.info("Degraded hybrid search: returning bm25-only results")
            return bm25_ranked[:top_k], "bm25-only (qdrant unavailable)"

        rrf_scores = {}
        for rank, doc_id in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(dense_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [doc_id for doc_id, score in ranked[:top_k]], "hybrid"

    def search(self, query, top_k=10, mode="hybrid"):
        if mode == "bm25":
            doc_ids = self.search_bm25(query, top_k)
            actual_mode = "bm25"
        elif mode == "dense":
            doc_ids = self.search_dense(query, top_k)
            actual_mode = "dense"
        elif mode == "hybrid":
            doc_ids, actual_mode = self.search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results = [self.docs_by_id[doc_id] for doc_id in doc_ids]
        return results, actual_mode