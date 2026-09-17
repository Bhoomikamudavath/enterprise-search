import pickle
import json
from src.bm25_index import tokenize

data = pickle.load(open("data/bm25_index.pkl", "rb"))
bm25, doc_ids = data["bm25"], data["doc_ids"]

docs = [json.loads(l) for l in open("data/documents.jsonl", encoding="utf-8")]
docs_by_id = {d["doc_id"]: d for d in docs}

query = "backpropagation"
scores = bm25.get_scores(tokenize(query))
top5 = sorted(zip(doc_ids, scores), key=lambda x: -x[1])[:5]
for doc_id, score in top5:
    print(f"{score:.2f}  {docs_by_id[doc_id]['title']}")