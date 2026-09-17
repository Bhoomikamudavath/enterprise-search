import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "documents.jsonl"
INDEX_FILE = DATA_DIR / "bm25_index.pkl"

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-\.]*")


def tokenize(text):
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def load_documents():
    docs = []
    with open(DOCS_FILE, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_index(docs):
    tokenized_corpus = [tokenize(d["full_text"]) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    bm25 = build_index(docs)

    doc_ids = [d["doc_id"] for d in docs]
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)

    print(f"BM25 index saved to {INDEX_FILE}")


if __name__ == "__main__":
    main()