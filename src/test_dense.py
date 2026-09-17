import pickle
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(f"{DATA_DIR}/faiss_index.bin")
doc_ids = pickle.load(open(f"{DATA_DIR}/faiss_doc_ids.pkl", "rb"))

docs = [json.loads(l) for l in open(f"{DATA_DIR}/documents.jsonl", encoding="utf-8")]
docs_by_id = {d["doc_id"]: d for d in docs}

query = "why does my model overfit on training data"
query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

k = 5
scores, indices = index.search(query_vec, k)

for score, idx in zip(scores[0], indices[0]):
    doc_id = doc_ids[idx]
    print(f"{score:.3f}  {docs_by_id[doc_id]['title']}")