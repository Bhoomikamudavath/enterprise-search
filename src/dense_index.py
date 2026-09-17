import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "documents.jsonl"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
DOC_IDS_FILE = DATA_DIR / "faiss_doc_ids.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    docs = []
    with open(DOCS_FILE, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_index(docs, model):
    texts = [d["full_text"] for d in docs]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding and indexing...")
    index = build_index(docs, model)

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    doc_ids = [d["doc_id"] for d in docs]
    with open(DOC_IDS_FILE, "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"FAISS index saved to {FAISS_INDEX_FILE}")


if __name__ == "__main__":
    main()