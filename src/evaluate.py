import json
from collections import defaultdict
from src.hybrid_retrieval import HybridRetriever

EVAL_FILE = "data/eval_queries.json"
TOP_K = 10
MODES = ["bm25", "dense", "hybrid"]


def load_eval_set():
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def reciprocal_rank(ranked_doc_ids, correct_doc_id):
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == correct_doc_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_doc_ids, correct_doc_id):
    return 1.0 if correct_doc_id in ranked_doc_ids else 0.0


def evaluate(retriever, eval_set):
    results = {mode: {"mrr": [], "recall": [], "by_type": defaultdict(lambda: {"mrr": [], "recall": []})}
               for mode in MODES}

    for item in eval_set:
        query = item["query"]
        correct_doc_id = item["doc_id"]
        query_type = item["type"]

        for mode in MODES:
            hits = retriever.search(query, top_k=TOP_K, mode=mode)
            ranked_doc_ids = [h["doc_id"] for h in hits]

            rr = reciprocal_rank(ranked_doc_ids, correct_doc_id)
            rec = recall_at_k(ranked_doc_ids, correct_doc_id)

            results[mode]["mrr"].append(rr)
            results[mode]["recall"].append(rec)
            results[mode]["by_type"][query_type]["mrr"].append(rr)
            results[mode]["by_type"][query_type]["recall"].append(rec)

    return results


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def print_report(results, eval_set):
    query_types = sorted(set(item["type"] for item in eval_set))

    print(f"\n{'=' * 60}")
    print(f"OVERALL (n={len(eval_set)} queries, K={TOP_K})")
    print(f"{'=' * 60}")
    print(f"{'Mode':<10} {'MRR':>8} {'Recall@' + str(TOP_K):>10}")
    for mode in MODES:
        mrr = avg(results[mode]["mrr"])
        rec = avg(results[mode]["recall"])
        print(f"{mode:<10} {mrr:>8.3f} {rec:>10.3f}")

    for qtype in query_types:
        n = sum(1 for item in eval_set if item["type"] == qtype)
        print(f"\n{'-' * 60}")
        print(f"BY TYPE: {qtype} (n={n})")
        print(f"{'-' * 60}")
        print(f"{'Mode':<10} {'MRR':>8} {'Recall@' + str(TOP_K):>10}")
        for mode in MODES:
            mrr = avg(results[mode]["by_type"][qtype]["mrr"])
            rec = avg(results[mode]["by_type"][qtype]["recall"])
            print(f"{mode:<10} {mrr:>8.3f} {rec:>10.3f}")


def main():
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} labelled queries")

    retriever = HybridRetriever()
    results = evaluate(retriever, eval_set)
    print_report(results, eval_set)

    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        summary = {
            mode: {
                "mrr": avg(results[mode]["mrr"]),
                "recall_at_k": avg(results[mode]["recall"]),
                "by_type": {
                    qtype: {
                        "mrr": avg(results[mode]["by_type"][qtype]["mrr"]),
                        "recall_at_k": avg(results[mode]["by_type"][qtype]["recall"]),
                    }
                    for qtype in results[mode]["by_type"]
                },
            }
            for mode in MODES
        }
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to data/eval_results.json")


if __name__ == "__main__":
    main()