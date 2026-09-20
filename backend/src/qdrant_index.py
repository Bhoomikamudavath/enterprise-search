import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "documents.jsonl"
COLLECTION_NAME = "knowledge-docs"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

client = QdrantClient(host="localhost", port=6333)


def load_documents():
    docs = []
    with open(DOCS_FILE, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def create_collection(dim):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def index_documents(docs, model):
    texts = [d["full_text"] for d in docs]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    points = []
    for i, (doc, vector) in enumerate(zip(docs, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                },
            )
        )

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    print("Creating Qdrant collection...")
    create_collection(dim)

    print("Encoding and upserting into Qdrant...")
    index_documents(docs, model)

    count = client.count(COLLECTION_NAME).count
    print(f"Indexed {count} documents into Qdrant")


if __name__ == "__main__":
    main()