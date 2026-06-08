"""Task 9 — Retrieval Pipeline Hoàn Chỉnh."""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # Cho phép chạy trực tiếp: python src/task9_retrieval_pipeline.py
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def safe_search(search_fn, query: str, top_k: int) -> list[dict]:
    """Run one retrieval branch without letting it crash the full pipeline."""
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"  ! Retrieval branch failed: {search_fn.__name__}: {exc}")
        return []


def normalize_result(item: dict, source_name: str) -> dict:
    """Keep retrieval result shape consistent across dense/sparse/pageindex."""
    return {
        "content": item.get("content", ""),
        "score": float(item.get("score", 0.0)),
        "metadata": item.get("metadata", {}),
        "source": source_name,
    }


def merge_hybrid_results(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Merge dense and lexical results with reciprocal rank fusion."""
    dense = [normalize_result(item, "semantic") for item in dense_results]
    sparse = [normalize_result(item, "lexical") for item in sparse_results]
    merged = rerank_rrf([dense, sparse], top_k=top_k)
    max_rrf_score = 2 / (60 + 1)  # two rankers, best rank is 1

    for item in merged:
        item["score"] = min(float(item.get("score", 0.0)) / max_rrf_score, 1.0)
        item["source"] = "hybrid"
        metadata = item.setdefault("metadata", {})
        metadata["retrieval_sources"] = metadata.get("retrieval_sources", "semantic+lexical")

    return merged


def maybe_rerank(query: str, merged: list[dict], top_k: int, use_reranking: bool) -> list[dict]:
    """Apply optional reranking while avoiding network/API dependency by default."""
    if not merged:
        return []

    if not use_reranking:
        return merged[:top_k]

    if RERANK_METHOD == "rrf":
        return merged[:top_k]

    try:
        reranked = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    except Exception as exc:
        print(f"  ! Rerank failed, using fused results: {exc}")
        return merged[:top_k]

    for item in reranked:
        item["source"] = "hybrid"
    return reranked[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    branch_k = max(top_k * 3, top_k)
    dense_results = safe_search(semantic_search, query, branch_k)
    sparse_results = safe_search(lexical_search, query, branch_k)

    merged = merge_hybrid_results(dense_results, sparse_results, top_k=branch_k)
    final_results = maybe_rerank(query, merged, top_k, use_reranking)

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
