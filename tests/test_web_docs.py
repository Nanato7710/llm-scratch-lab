from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "docs" / "web"
CONTENT_PREFIX = "window.LLM_SCRATCH_DOCS = "


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if identifier := values.get("id"):
            self.ids.append(identifier)
        if href := values.get("href"):
            self.hrefs.append(href)
        if source := values.get("src"):
            self.sources.append(source)


def load_embedded_documents() -> list[dict[str, str]]:
    content = (WEB_DIR / "content.js").read_text(encoding="utf-8")
    assert content.startswith(CONTENT_PREFIX)
    assert content.endswith(";\n")
    return json.loads(content[len(CONTENT_PREFIX) : -2])


def test_embedded_documents_match_markdown_sources() -> None:
    documents = load_embedded_documents()
    assert [document["id"] for document in documents] == [
        "architecture",
        "adding-components",
        "experiment-guide",
    ]
    for document in documents:
        source = ROOT / "docs" / document["source"]
        assert document["markdown"] == source.read_text(encoding="utf-8")


def test_web_document_has_valid_local_navigation_and_assets() -> None:
    parser = DocumentParser()
    parser.feed((WEB_DIR / "index.html").read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids))
    identifiers = set(parser.ids)
    for href in parser.hrefs:
        if href.startswith("#"):
            assert href[1:] in identifiers
        elif not re.match(r"^[a-z]+:", href):
            assert (WEB_DIR / href).resolve().is_file()
    for source in parser.sources:
        assert not re.match(r"^https?://", source)
        assert (WEB_DIR / source).resolve().is_file()


def test_web_runtime_has_no_remote_dependencies() -> None:
    for path in (WEB_DIR / "index.html", WEB_DIR / "styles.css", WEB_DIR / "app.js"):
        content = path.read_text(encoding="utf-8")
        assert "https://" not in content
        assert "http://" not in content
