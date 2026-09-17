import pickle
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.bm25_index import tokenize

DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60


class HybridRetriever:
    def __init__(self):
        bm25_data = pickle.load(open(DATA_DIR / "bm25_index.pkl", "rb"))
        self.bm25 = bm25_data["bm25"]
        self.bm25_doc_ids = bm25_data["doc_ids"]

        self.faiss_index = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
        self.faiss_doc_ids = pickle.load(open(DATA_DIR / "faiss_doc_ids.pkl", "rb"))

        self.model = SentenceTransformer(MODEL_NAME)

        self.docs_by_id = {}
        with open(DATA_DIR / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs_by_id[d["doc_id"]] = d

    def search_bm25(self, query, top_k):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.bm25_doc_ids, scores), key=lambda x: -x[1])
        return [doc_id for doc_id, score in ranked[:top_k]]

    def search_dense(self, query, top_k):
        query_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [self.faiss_doc_ids[idx] for idx in indices[0]]

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