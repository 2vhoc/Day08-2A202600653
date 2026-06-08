"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import html
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tuoitre.vn/dien-vien-huu-tin-bi-truy-to-vi-to-chuc-su-dung-ma-tuy-20221117104908287.htm",
    "https://tuoitre.vn/dien-vien-hai-huu-tin-bi-khoi-to-bat-tam-giam-vi-ma-tuy-20220617185327576.htm",
    "https://vietnamnet.vn/ca-si-chu-bin-la-ai-2288952.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-bi-khoi-to-toi-giet-nguoi-3840141.html",
    "https://dantri.com.vn/phap-luat/loi-ke-cua-canh-sat-dieu-tra-trong-vu-an-chau-viet-cuong-dung-toi-tru-ta-ma-dan-den-cai-chet-cua-co-gai-20-tuoi-20180311083428158.htm",
]


class ArticleHTMLParser(HTMLParser):
    """Small dependency-free parser for extracting news title and body text."""

    BLOCK_TAGS = {"h1", "h2", "h3", "p", "li"}
    IGNORED_TAGS = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata = {}
        self.title_parts = []
        self.blocks = []
        self._ignored_depth = 0
        self._capture_title = False
        self._current_block = None
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = {name.lower(): value or "" for name, value in attrs}

        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if tag == "meta":
            key = (
                attrs.get("property")
                or attrs.get("name")
                or attrs.get("itemprop")
            )
            value = attrs.get("content")
            if key and value:
                self.metadata[key.lower()] = clean_text(value)
            return

        if tag == "title":
            self._capture_title = True
            return

        if tag in self.BLOCK_TAGS and self._ignored_depth == 0:
            self._flush_block()
            self._current_block = tag
            self._buffer = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return

        if tag == "title":
            self._capture_title = False

        if tag == self._current_block:
            self._flush_block()

    def handle_data(self, data):
        if self._ignored_depth:
            return

        text = clean_text(data)
        if not text:
            return

        if self._capture_title:
            self.title_parts.append(text)

        if self._current_block:
            self._buffer.append(text)

    def close(self):
        super().close()
        self._flush_block()

    def _flush_block(self):
        if not self._current_block:
            return

        text = clean_text(" ".join(self._buffer))
        if text:
            self.blocks.append((self._current_block, text))

        self._current_block = None
        self._buffer = []


def clean_text(value: str) -> str:
    """Normalize HTML text into compact Vietnamese-friendly plain text."""
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def fetch_html(url: str) -> str:
    """Fetch article HTML with a browser-like User-Agent."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def choose_title(parser: ArticleHTMLParser) -> str:
    """Prefer OpenGraph metadata, then visible h1/title text."""
    candidates = [
        parser.metadata.get("og:title"),
        parser.metadata.get("twitter:title"),
        parser.metadata.get("title"),
        next((text for tag, text in parser.blocks if tag == "h1"), None),
        " ".join(parser.title_parts),
    ]
    for candidate in candidates:
        if candidate:
            title = re.sub(r"\s*[-|]\s*(VnExpress|Tuổi Trẻ Online|Báo Dân trí|VietNamNet).*$", "", candidate)
            return clean_text(title)
    return "Unknown"


def choose_published_date(parser: ArticleHTMLParser) -> str:
    """Extract publication date when the news site exposes it in metadata."""
    for key in (
        "article:published_time",
        "pubdate",
        "date",
        "publishdate",
        "publish-date",
        "og:published_time",
    ):
        value = parser.metadata.get(key)
        if value:
            return value
    return ""


def is_boilerplate(text: str) -> bool:
    """Filter repeated navigation, comment, and footer fragments."""
    lowered = text.lower()
    stop_phrases = (
        "đăng nhập",
        "bình luận",
        "chia sẻ",
        "sao chép liên kết",
        "tin liên quan",
        "tin cùng chuyên mục",
        "xem thêm",
        "đọc tiếp",
        "theo dõi",
        "tổng biên tập",
        "giấy phép",
        "hotline",
        "liên hệ",
        "email",
        "mật khẩu",
        "quảng cáo",
        "copyright",
        "all rights reserved",
        "vui lòng",
        "nhập mã",
        "tặng sao",
        "trở thành người đầu tiên",
    )
    return any(phrase in lowered for phrase in stop_phrases)


def extract_content_markdown(url: str, html_doc: str) -> dict:
    """Parse downloaded HTML into assignment-friendly article metadata."""
    parser = ArticleHTMLParser()
    parser.feed(html_doc)
    parser.close()

    title = choose_title(parser)
    date_crawled = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    date_published = choose_published_date(parser)
    description = (
        parser.metadata.get("og:description")
        or parser.metadata.get("description")
        or parser.metadata.get("twitter:description")
        or ""
    )

    body_blocks = []
    seen = set()
    for tag, text in parser.blocks:
        if text in seen or text == title:
            continue
        seen.add(text)
        if tag in {"h1", "h2", "h3"}:
            if len(text) >= 8 and not is_boilerplate(text):
                body_blocks.append(f"## {text}")
            continue
        if len(text) < 35 or is_boilerplate(text):
            continue
        body_blocks.append(text)

    if description and description not in seen:
        body_blocks.insert(0, description)

    content_markdown = "\n\n".join(
        [
            f"# {title}",
            f"URL nguồn: {url}",
            f"Ngày crawl: {date_crawled}",
            f"Ngày đăng: {date_published or 'Không rõ'}",
            "## Nội dung trích xuất",
            *body_blocks,
        ]
    )

    return {
        "url": url,
        "title": title,
        "date_crawled": date_crawled,
        "date_published": date_published,
        "content_markdown": content_markdown,
        "source": "urllib_html_parser",
    }


async def crawl_with_crawl4ai(url: str) -> dict | None:
    """Use Crawl4AI when installed; return None so fallback can handle misses."""
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return None

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        print(f"  ! Crawl4AI failed, falling back to urllib: {exc}")
        return None

    markdown = clean_text(getattr(result, "markdown", "") or "")
    if len(markdown) < 300:
        return None

    metadata = getattr(result, "metadata", {}) or {}
    date_crawled = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return {
        "url": url,
        "title": metadata.get("title") or "Unknown",
        "date_crawled": date_crawled,
        "date_published": metadata.get("published_time") or "",
        "content_markdown": markdown,
        "source": "crawl4ai",
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    article = await crawl_with_crawl4ai(url)
    if article:
        return article

    try:
        html_doc = await asyncio.to_thread(fetch_html, url)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Không crawl được URL {url}: {exc}") from exc

    return extract_content_markdown(url, html_doc)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
