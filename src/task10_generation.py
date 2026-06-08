"""Task 10 — Generation Có Citation."""

import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # Cho phép chạy trực tiếp: python src/task10_generation.py
    from task9_retrieval_pipeline import retrieve


def load_env_file():
    """Tiny .env loader fallback when python-dotenv is not installed."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


if load_dotenv:
    load_dotenv()
else:
    load_env_file()


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    last_even_index = len(chunks) - 1
    if last_even_index % 2 == 0:
        last_even_index -= 1
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


def citation_label(chunk: dict, index: int) -> str:
    """Create a short citation label from chunk metadata."""
    metadata = chunk.get("metadata", {})
    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("path")
        or f"Document {index}"
    )
    section = metadata.get("section") or metadata.get("type") or metadata.get("doc_type")
    if section:
        return f"{source}, {section}"
    return source


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source") or metadata.get("filename") or metadata.get("path") or f"Source {i}"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        label = citation_label(chunk, i)
        score = float(chunk.get("score", 0.0))
        context_parts.append(
            f"[Document {i} | Citation: {label} | Source: {source} | "
            f"Type: {doc_type} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter that works decently for Vietnamese snippets."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]


def select_evidence_sentences(query: str, chunks: list[dict], max_sentences: int = 5) -> list[tuple[str, str]]:
    """Pick concise evidence lines and their citation labels from retrieved chunks."""
    query_terms = {
        term
        for term in re.findall(r"\w+", query.lower())
        if len(term) >= 3
    }
    evidence = []

    for i, chunk in enumerate(chunks, 1):
        label = citation_label(chunk, i)
        sentences = split_sentences(chunk.get("content", ""))
        if not sentences:
            content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
            sentences = [content[:260]] if content else []

        ranked = []
        for sentence in sentences:
            sentence_terms = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(query_terms & sentence_terms)
            ranked.append((overlap, sentence))

        ranked.sort(key=lambda item: item[0], reverse=True)
        for overlap, sentence in ranked[:1]:
            if overlap == 0 and evidence:
                continue
            evidence.append((sentence, label))
            break

        if len(evidence) >= max_sentences:
            break

    return evidence


def local_generate_answer(query: str, chunks: list[dict]) -> str:
    """Deterministic fallback answer with citations, no external LLM needed."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    evidence = select_evidence_sentences(query, chunks)
    if not evidence:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines = [
        f"Dựa trên các nguồn đã truy xuất, câu hỏi \"{query}\" có các bằng chứng liên quan sau:"
    ]
    for sentence, label in evidence:
        lines.append(f"- {sentence} [{label}]")

    lines.append(
        "Nếu cần kết luận pháp lý chính xác cho một trường hợp cụ thể, cần đối chiếu thêm toàn văn điều khoản và tình tiết vụ việc từ nguồn chính thức."
    )
    return "\n".join(lines)


def maybe_generate_with_openai(query: str, context: str) -> str | None:
    """Optional OpenAI generation, enabled only when explicitly configured."""
    if os.getenv("OPENAI_USE_API", "").lower() not in {"1", "true", "yes"}:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = maybe_generate_with_openai(query, context)
    if not answer:
        answer = local_generate_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": context,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
