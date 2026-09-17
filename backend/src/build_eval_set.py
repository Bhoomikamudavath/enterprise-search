import json
import random

random.seed(42)

docs = [json.loads(l) for l in open("data/documents.jsonl", encoding="utf-8")]
sample = random.sample(docs, 40)

for i, d in enumerate(sample):
    print(f"\n[{i}] doc_id={d['doc_id']}")
    print(f"Title: {d['title']}")
    print(f"Body preview: {d['question_body'][:200]}")