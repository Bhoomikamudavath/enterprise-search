from src.hybrid_retrieval import HybridRetriever

retriever = HybridRetriever()

queries = [
    "why does my model overfit on training data",
    "backpropagation",
]

for query in queries:
    print(f"\n=== Query: {query} ===")
    for mode in ["bm25", "dense", "hybrid"]:
        print(f"\n--- {mode} ---")
        results = retriever.search(query, top_k=5, mode=mode)
        for r in results:
            print(f"  {r['title']}")