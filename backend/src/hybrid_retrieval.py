import json
from pathlib import Path
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"
OPENSEARCH_INDEX = "knowledge-docs"
QDRANT_COLLECTION = "knowledge-docs"
RRF_K = 60


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

    def search_bm25(self, query, top_k):
        response = self.opensearch.search(
            index=OPENSEARCH_INDEX,
            body={
                "query": {"match": {"full_text": query}},
                "size": top_k,
            },
        )
        return [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]

    def search_dense(self, query, top_k):
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
        ).points
        return [hit.payload["doc_id"] for hit in hits]

    def search_hybrid(self, query, top_k, candidate_pool=50):
        bm25_ranked = self.search_bm25(query, candidate_pool)
        dense_ranked = self.search_dense(query, candidate_pool)

        rrf_scores = {}
        for rank, doc_id in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(dense_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [doc_id for doc_id, score in ranked[:top_k]]

    def search(self, query, top_k=10, mode="hybrid"):
        if mode == "bm25":
            doc_ids = self.search_bm25(query, top_k)
        elif mode == "dense":
            doc_ids = self.search_dense(query, top_k)
        elif mode == "hybrid":
            doc_ids = self.search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return [self.docs_by_id[doc_id] for doc_id in doc_ids]