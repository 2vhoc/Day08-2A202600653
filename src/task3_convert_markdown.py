"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def get_markitdown():
    """Return a MarkItDown converter when the optional package is installed."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    return MarkItDown()


def clean_text(text: str) -> str:
    """Normalize text extracted from PDFs/JSON without losing Vietnamese marks."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def title_from_filename(path: Path) -> str:
    """Create a readable markdown title from a source filename."""
    title = path.stem.replace("_", "-").replace("-", " ")
    return " ".join(word.capitalize() for word in title.split())


def build_markdown_header(title: str, metadata: dict) -> str:
    """Build a compact metadata block for standardized markdown files."""
    lines = [f"# {title}", ""]
    for key, value in metadata.items():
        if value:
            lines.append(f"**{key}:** {value}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def convert_with_markitdown(filepath: Path, md_converter) -> str:
    """Convert a file with MarkItDown and return extracted text."""
    result = md_converter.convert(str(filepath))
    return clean_text(getattr(result, "text_content", "") or "")


def convert_pdf_with_pdftotext(filepath: Path) -> str:
    """Fallback PDF conversion using the system pdftotext command."""
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext is not installed")

    result = subprocess.run(
        ["pdftotext", "-layout", str(filepath), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return clean_text(result.stdout)


def convert_document(filepath: Path, md_converter) -> tuple[str, str]:
    """Convert PDF/DOC/DOCX into markdown text with a sensible fallback."""
    if md_converter is not None:
        try:
            content = convert_with_markitdown(filepath, md_converter)
            if len(content) > 100:
                return content, "MarkItDown"
        except Exception as exc:
            print(f"  ! MarkItDown lỗi, thử fallback: {exc}")

    if filepath.suffix.lower() == ".pdf":
        return convert_pdf_with_pdftotext(filepath), "pdftotext"

    raise RuntimeError(
        f"Không convert được {filepath.name}: cần cài markitdown cho {filepath.suffix}"
    )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Không tìm thấy thư mục: {legal_dir}")
        return

    md_converter = get_markitdown()
    converted_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            content, converter_name = convert_document(filepath, md_converter)
            output_path = output_dir / f"{filepath.stem}.md"
            header = build_markdown_header(
                title_from_filename(filepath),
                {
                    "Source File": filepath.name,
                    "Converted At": converted_at,
                    "Converter": converter_name,
                },
            )
            if filepath.suffix.lower() == ".pdf" and len(content) < 500:
                content += (
                    "\n\n> Ghi chú: File PDF này có vẻ là bản scan/ảnh nên công cụ "
                    "trích xuất text chỉ lấy được rất ít nội dung. Cần OCR hoặc "
                    "nguồn PDF text-based để có nội dung đầy đủ hơn."
                )
            output_path.write_text(header + content + "\n", encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"Không tìm thấy thư mục: {news_dir}")
        return

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            title = data.get("title") or filepath.stem
            content_markdown = clean_text(
                data.get("content_markdown")
                or data.get("content")
                or data.get("markdown")
                or ""
            )
            header = build_markdown_header(
                title,
                {
                    "Source URL": data.get("url", "N/A"),
                    "Crawled": data.get("date_crawled", "N/A"),
                    "Published": data.get("date_published", ""),
                    "Source File": filepath.name,
                },
            )

            content = header + content_markdown + "\n"
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
