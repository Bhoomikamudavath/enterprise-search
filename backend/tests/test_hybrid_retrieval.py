from src.hybrid_retrieval import HybridRetriever


def test_rrf_fusion_combines_scores_from_both_lists():
    retriever = HybridRetriever.__new__(HybridRetriever)

    def fake_bm25(query, top_k):
        return ["docA", "docB", "docC"]

    def fake_dense(query, top_k):
        return ["docB", "docD", "docA"]

    retriever.search_bm25 = fake_bm25
    retriever.search_dense = fake_dense

    results = retriever.search_hybrid("test query", top_k=4, candidate_pool=3)

    assert results[0] == "docB"
    assert set(results) == {"docA", "docB", "docC", "docD"}


def test_rrf_respects_top_k():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.search_bm25 = lambda q, k: ["d1", "d2", "d3", "d4"]
    retriever.search_dense = lambda q, k: ["d5", "d6", "d7", "d8"]

    results = retriever.search_hybrid("test query", top_k=2, candidate_pool=4)

    assert len(results) == 2