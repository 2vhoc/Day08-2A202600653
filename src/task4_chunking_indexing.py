"""Task 4 — Chunking & Indexing vào local vector store."""

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
VECTOR_INDEX_PATH = INDEX_DIR / "vector_index.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn Recursive Character Chunking vì dữ liệu gồm cả văn bản pháp luật dài
# và bài báo ngắn; cách này ổn định, không phụ thuộc heading có đẹp hay không.
CHUNK_SIZE = 500        # Đủ nhỏ để retrieval chính xác, đủ lớn để giữ ngữ cảnh.
CHUNK_OVERLAP = 50      # Giữ nối tiếp ý ở ranh giới chunk nhưng không trùng quá nhiều.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Dùng local hashing embedding để bài chạy ngay cả khi chưa cài torch /
# sentence-transformers. Nếu dùng production, có thể thay bằng BAAI/bge-m3.
EMBEDDING_MODEL = "local-hashing-embedding-v1"
EMBEDDING_DIM = 384

# Lưu local JSON để Task 5 đọc lại trực tiếp, không cần Docker/Weaviate.
VECTOR_STORE = "local_json"  # "local_json" | "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
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
                    "source": md_file.name,
                    "path": str(relative_path),
                    "type": doc_type,
                    "doc_type": doc_type,
                },
            }
        )

    return documents


def split_long_piece(text: str) -> list[str]:
    """Split a long text piece into <= CHUNK_SIZE chunks by sentence/word."""
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []

    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    pieces = []
    current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > CHUNK_SIZE:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(split_by_words(sentence))
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


def split_by_words(text: str) -> list[str]:
    """Fallback splitter for paragraphs/sentences longer than CHUNK_SIZE."""
    pieces = []
    current = ""

    for word in text.split():
        if len(word) > CHUNK_SIZE:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i:i + CHUNK_SIZE] for i in range(0, len(word), CHUNK_SIZE))
            continue

        candidate = f"{current} {word}".strip()
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def split_text_recursive(text: str) -> list[str]:
    """Recursive-character-style splitter using paragraphs, sentences, words."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        for piece in split_long_piece(paragraph):
            if not current:
                current = piece
                continue

            candidate = f"{current}\n\n{piece}"
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
                continue

            chunks.append(current)
            overlap = current[-CHUNK_OVERLAP:].strip()
            candidate = f"{overlap}\n\n{piece}".strip() if overlap else piece
            current = candidate if len(candidate) <= CHUNK_SIZE else piece

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc in documents:
        splits = split_text_recursive(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def normalize_for_search(text: str) -> str:
    """Lowercase and remove Vietnamese accents for stable token matching."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text with accent-insensitive normalization."""
    return re.findall(r"[a-z0-9]+", normalize_for_search(text))


def text_to_embedding(text: str) -> list[float]:
    """Create a deterministic dense hashing embedding and L2-normalize it."""
    vector = [0.0] * EMBEDDING_DIM
    tokens = tokenize(text)

    features = tokens + [
        f"{tokens[i]}_{tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    ]

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.5 if "_" in feature else 1.0
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    for chunk in chunks:
        chunk["embedding"] = text_to_embedding(chunk["content"])
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    indexed_chunks = chunks
    if indexed_chunks and "embedding" not in indexed_chunks[0]:
        indexed_chunks = embed_chunks(indexed_chunks)

    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_store": VECTOR_STORE,
        "chunks": indexed_chunks,
    }
    VECTOR_INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"✓ Saved local vector index: {VECTOR_INDEX_PATH}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
