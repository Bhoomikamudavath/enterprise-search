import json
import sys
from pathlib import Path
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "documents.jsonl"
OPENSEARCH_INDEX = "knowledge-docs"
QDRANT_COLLECTION = "knowledge-docs"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_all_documents():
    docs = {}
    with open(DOCS_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def save_all_documents(docs):
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d) + "\n")


def upsert_document(doc, qdrant_point_id):
    opensearch = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False, verify_certs=False)
    qdrant = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(MODEL_NAME)

    opensearch.index(
        index=OPENSEARCH_INDEX,
        id=doc["doc_id"],
        body={
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "full_text": doc["full_text"],
            "tags": doc["tags"],
            "score": doc["score"],
            "url": doc["url"],
        },
    )
    print(f"Upserted doc_id={doc['doc_id']} into OpenSearch")

    vector = model.encode(doc["full_text"], normalize_embeddings=True).tolist()
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[PointStruct(
            id=qdrant_point_id,
            vector=vector,
            payload={"doc_id": doc["doc_id"], "title": doc["title"], "url": doc["url"]},
        )],
    )
    print(f"Upserted doc_id={doc['doc_id']} into Qdrant")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.update_document <doc_id>")
        print("Updates a single document's title by appending ' [UPDATED]' as a demo,")
        print("then re-indexes it into both OpenSearch and Qdrant without touching")
        print("any other document.")
        sys.exit(1)

    doc_id = sys.argv[1]
    docs = load_all_documents()

    if doc_id not in docs:
        print(f"No document with doc_id={doc_id} found")
        sys.exit(1)

    docs[doc_id]["title"] = docs[doc_id]["title"] + " [UPDATED]"
    docs[doc_id]["full_text"] = docs[doc_id]["title"] + " " + docs[doc_id]["question_body"] + " " + docs[doc_id]["answer_body"]

    save_all_documents(docs)

    doc_ids_ordered = list(docs.keys())
    qdrant_point_id = doc_ids_ordered.index(doc_id)

    upsert_document(docs[doc_id], qdrant_point_id)
    print(f"\nDone. Document {doc_id} updated without re-indexing the other {len(docs) - 1} documents.")


if __name__ == "__main__":
    main()