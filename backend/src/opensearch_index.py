import json
from pathlib import Path
from opensearchpy import OpenSearch, helpers

DATA_DIR = Path("data")
DOCS_FILE = DATA_DIR / "documents.jsonl"
INDEX_NAME = "knowledge-docs"

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    use_ssl=False,
    verify_certs=False,
)

INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "technical_analyzer": {
                    "type": "custom",
                    "tokenizer": "technical_tokenizer",
                    "filter": ["lowercase"],
                }
            },
            "tokenizer": {
                "technical_tokenizer": {
                    "type": "pattern",
                    "pattern": "[^a-zA-Z0-9_\\-\\.]+",
                }
            },
        }
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "technical_analyzer"},
            "full_text": {"type": "text", "analyzer": "technical_analyzer"},
            "tags": {"type": "keyword"},
            "score": {"type": "integer"},
            "url": {"type": "keyword"},
        }
    },
}


def load_documents():
    docs = []
    with open(DOCS_FILE, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def create_index():
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


def index_documents(docs):
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": d["doc_id"],
            "_source": {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "full_text": d["full_text"],
                "tags": d["tags"],
                "score": d["score"],
                "url": d["url"],
            },
        }
        for d in docs
    ]
    helpers.bulk(client, actions)


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("Creating index with technical tokenizer...")
    create_index()

    print("Bulk indexing into OpenSearch...")
    index_documents(docs)

    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    print(f"Indexed {count} documents into OpenSearch")


if __name__ == "__main__":
    main()