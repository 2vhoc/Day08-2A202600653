"""Task 5 — Semantic Search Module trên local vector index."""

import json
from math import sqrt

try:
    from src.task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
        text_to_embedding,
        tokenize,
    )
except ImportError:  # Cho phép chạy trực tiếp: python src/task5_semantic_search.py
    from task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
        text_to_embedding,
        tokenize,
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two dense vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def ensure_vector_index() -> list[dict]:
    """Load local vector index; build it from standardized markdown if missing."""
    if not VECTOR_INDEX_PATH.exists():
        docs = load_documents()
        chunks = chunk_documents(docs)
        chunks = embed_chunks(chunks)
        index_to_vectorstore(chunks)

    payload = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
    return payload.get("chunks", [])


def token_overlap_score(query: str, content: str) -> float:
    """Small lexical bonus so Vietnamese legal terms rank more intuitively."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = text_to_embedding(query)
    chunks = ensure_vector_index()

    results = []
    for chunk in chunks:
        vector_score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        lexical_bonus = token_overlap_score(query, chunk.get("content", ""))
        score = 0.8 * vector_score + 0.2 * lexical_bonus
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
