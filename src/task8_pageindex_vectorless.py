"""Task 8 — PageIndex Vectorless RAG với local fallback."""

import os
import json
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from src.task4_chunking_indexing import normalize_for_search, tokenize
except ImportError:  # Cho phép chạy trực tiếp: python src/task8_pageindex_vectorless.py
    from task4_chunking_indexing import normalize_for_search, tokenize

def load_env_file():
    """Tiny .env fallback for PAGEINDEX_API_KEY when python-dotenv is absent."""
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

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
USE_PAGEINDEX_API = os.getenv("PAGEINDEX_USE_API", "").lower() in {"1", "true", "yes"}
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
LOCAL_PAGEINDEX_PATH = INDEX_DIR / "pageindex_local.json"


def load_markdown_documents() -> list[dict]:
    """Load standardized markdown files with lightweight structural metadata."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "filename": md_file.name,
                    "path": str(relative_path),
                    "type": doc_type,
                },
            }
        )

    return documents


def split_markdown_sections(content: str) -> list[dict]:
    """
    Split markdown by headings/paragraphs instead of vectors.

    This mimics a vectorless fallback: retrieval uses document structure and
    lexical evidence rather than dense embeddings.
    """
    sections = []
    current_heading = ""
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"heading": current_heading, "content": text})

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_heading = stripped.lstrip("#").strip()
            current_lines = []
            continue
        current_lines.append(line)

    flush()

    if not sections and content.strip():
        sections.append({"heading": "", "content": content.strip()})

    expanded = []
    for section in sections:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", section["content"])
            if paragraph.strip()
        ]
        if not paragraphs:
            expanded.append(section)
            continue

        for index, paragraph in enumerate(paragraphs):
            expanded.append(
                {
                    "heading": section["heading"],
                    "content": paragraph,
                    "paragraph_index": index,
                }
            )

    return expanded


def build_local_pageindex() -> list[dict]:
    """Build a small structure-aware local index from markdown files."""
    records = []

    for doc in load_markdown_documents():
        for section_index, section in enumerate(split_markdown_sections(doc["content"])):
            text = section["content"].strip()
            if len(text) < 40:
                continue

            heading = section.get("heading", "")
            content = f"{heading}\n\n{text}".strip() if heading else text
            metadata = {
                **doc["metadata"],
                "section": heading,
                "section_index": section_index,
                "paragraph_index": section.get("paragraph_index", 0),
            }
            records.append({"content": content, "metadata": metadata})

    return records


def score_record(query: str, record: dict) -> float:
    """Score by token overlap, phrase hits, and markdown-section context."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content = record.get("content", "")
    content_tokens = tokenize(content)
    if not content_tokens:
        return 0.0

    query_set = set(query_tokens)
    content_set = set(content_tokens)
    overlap = len(query_set & content_set) / len(query_set)

    normalized_query = normalize_for_search(query)
    normalized_content = normalize_for_search(content)
    phrase_bonus = 0.25 if normalized_query and normalized_query in normalized_content else 0.0

    heading = record.get("metadata", {}).get("section", "")
    heading_tokens = set(tokenize(heading))
    heading_bonus = 0.15 * (len(query_set & heading_tokens) / len(query_set))

    length_penalty = min(len(content_tokens) / 120, 1.0)
    return overlap + phrase_bonus + heading_bonus + 0.05 * length_penalty


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    try:
        from pageindex import PageIndex
    except ImportError:
        records = build_local_pageindex()
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_PAGEINDEX_PATH.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Built local PageIndex fallback: {len(records)} sections")
        return records

    if not PAGEINDEX_API_KEY or not USE_PAGEINDEX_API:
        records = build_local_pageindex()
        print(f"  ✓ Using local PageIndex fallback: {len(records)} sections")
        return records

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    uploaded = []
    for doc in load_markdown_documents():
        pi.upload(content=doc["content"], metadata=doc["metadata"])
        uploaded.append(doc["metadata"])
        print(f"  ✓ Uploaded: {doc['metadata']['filename']}")
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0:
        return []

    try:
        from pageindex import PageIndex
    except ImportError:
        records = build_local_pageindex()
    else:
        if PAGEINDEX_API_KEY and USE_PAGEINDEX_API:
            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": getattr(result, "text", ""),
                    "score": float(getattr(result, "score", 0.0)),
                    "metadata": getattr(result, "metadata", {}),
                    "source": "pageindex",
                }
                for result in results
            ]
        records = build_local_pageindex()

    scored = []
    for record in records:
        score = score_record(query, record)
        if score <= 0:
            continue
        scored.append(
            {
                "content": record["content"],
                "score": float(score),
                "metadata": record["metadata"],
                "source": "pageindex",
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
