"""Task 7 — Reranking Module."""

import os
import re
import unicodedata
from pathlib import Path


def load_env_file():
    """Load .env values when python-dotenv is not installed."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def normalize_text(text: str) -> str:
    """Lowercase and remove Vietnamese accents for stable local reranking."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def rerank_local_overlap(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Local fallback reranker.

    It combines query-token overlap with the original retrieval score, so it is
    deterministic and does not depend on API balance or network availability.
    """
    query_tokens = set(tokenize(query))
    if top_k <= 0 or not candidates:
        return []

    scored = []
    for candidate in candidates:
        content = candidate.get("content", "")
        content_tokens = set(tokenize(content))
        overlap = (
            len(query_tokens & content_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        original_score = float(candidate.get("score", 0.0))
        rerank_score = 0.7 * overlap + 0.3 * original_score

        item = candidate.copy()
        item["score"] = float(rerank_score)
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates:
        return []

    load_env_file()
    jina_api_key = os.getenv("JINA_API_KEY", "")
    if not jina_api_key:
        return rerank_local_overlap(query, candidates, top_k)

    try:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=20,
        )
        payload = response.json()
        reranked = payload.get("results")
        if not isinstance(reranked, list):
            return rerank_local_overlap(query, candidates, top_k)

        return [
            {**candidates[r["index"]], "score": float(r["relevance_score"])}
            for r in reranked
            if 0 <= r.get("index", -1) < len(candidates)
        ][:top_k]
    except Exception:
        return rerank_local_overlap(query, candidates, top_k)

def cosine_sim(vec1: list[float], vec2: list[float]) -> float:
    """Tính cosine similarity giữa 2 vector."""
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    selected = []
    remaining = list(range(len(candidates)))
    
    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')
    
        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    
            # Max similarity to already selected
            max_sim_to_selected = 0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)
    
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
    
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # TODO: Implement RRF
    #
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item
    
    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    
    return results
    raise NotImplementedError("Implement rerank_rrf")


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        return (rerank_rrf([candidates], top_k=top_k))
        # raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắtx vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="rrf")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
