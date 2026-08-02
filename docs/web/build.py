from __future__ import annotations

import argparse
import json
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
DOCS_DIR = WEB_DIR.parent
OUTPUT_PATH = WEB_DIR / "content.js"
SOURCES = (
    ("architecture", "プロジェクトの構成", DOCS_DIR / "architecture.md"),
    ("adding-components", "コンポーネントの追加手順", DOCS_DIR / "adding-components.md"),
    ("experiment-guide", "実験の設定と実行", DOCS_DIR / "experiment-guide.md"),
)


def build_payload() -> str:
    documents = [
        {
            "id": document_id,
            "title": title,
            "source": source_path.name,
            "markdown": source_path.read_text(encoding="utf-8"),
        }
        for document_id, title, source_path in SOURCES
    ]
    serialized = json.dumps(documents, ensure_ascii=False, indent=2)
    return f"window.LLM_SCRATCH_DOCS = {serialized};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the embedded Web documentation data")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when content.js does not match the Markdown sources",
    )
    args = parser.parse_args()
    expected = build_payload()
    if args.check:
        actual = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if actual != expected:
            raise SystemExit("docs/web/content.js is out of date; run docs/web/build.py")
        print("docs/web/content.js is up to date")
        return
    OUTPUT_PATH.write_text(expected, encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
