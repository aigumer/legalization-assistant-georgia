"""Turn the source PDFs in ``data/raw`` into retrievable, citable chunks.

The corpus is Georgian legislation, which is already organised as
Section -> Chapter -> Article. That hierarchy is the chunking unit: one chunk per
article (split further only when an article is very long), carrying its section
and chapter as metadata so answers can cite "Article 15 - Types of residence
permits" rather than an anonymous passage.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .config import CHUNKS_PATH, MAX_CHUNK_CHARS, RAW_DIR

# Headings separate number from title with an en, em, or plain dash. Amending
# laws number inserted articles with a superscript: "Article 20¹".
DASH = r"[\u2013\u2014-]"
SUPERSCRIPTS = r"[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]"

# "Section II", with the section title on the following line.
SECTION_RE = re.compile(r"^Section\s+([IVXLC]+)\s*$")
# "Chapter III - Georgian Visa", or a bare "Chapter XV" with the title below it.
CHAPTER_RE = re.compile(rf"^Chapter\s+([IVXLC]+\d*)\s*(?:{DASH}\s*(.+))?$")
# "Article 4 - Entry into Georgia". The dash is required: it is what separates a
# real heading from body text such as "Article 15(k) of this Law shall be ...".
ARTICLE_RE = re.compile(rf"^Article\s+(\d+{SUPERSCRIPTS}*)\s*{DASH}\s*(.+)$")

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUPERSCRIPT_TO_ASCII = {
    ord(c): f"-{d}" for d, c in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹", strict=True)
}

# Amendment trail printed under each amended article by the official publisher.
AMENDMENT_RE = re.compile(r"^(?:Law|Decree|Decision|Judgment|Resolution|Order)\b.*\bwebsite\b")
# Page furniture: the publisher URL and the document registration number.
FOOTER_RE = re.compile(r"^(?:https?://\S+|\d{10,})$")

# A new logical paragraph starts at "1." / "a)" / "a1)"; every other line is a
# hard wrap continuing the one above.
PARAGRAPH_START_RE = re.compile(r"^(?:\d+\s*\d*\.|[a-z]\d?\)|[a-z]\d?\.)\s")


@dataclass
class Chunk:
    """One retrievable passage, with everything needed to cite it."""

    id: str
    doc_id: str
    doc_title: str
    section: str
    chapter: str
    article: str
    title: str
    text: str
    page: int
    part: int = 1
    n_parts: int = 1
    amendments: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        label = f"{self.article} - {self.title}" if self.title else self.article
        if self.n_parts > 1:
            label += f" (part {self.part}/{self.n_parts})"
        return label


SUPERSCRIPT_FLAG = 1  # bit 0 of a span's `flags` marks superscript text


def _read_lines(pdf_path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, line)`` pairs with noise and NBSPs removed.

    Read span-by-span rather than as flat text so superscripts survive: flat
    extraction turns "Article 20^1" into a nonexistent "Article 201".
    """
    lines: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(
                        span["text"].translate(SUPERSCRIPT_DIGITS)
                        if span["flags"] & SUPERSCRIPT_FLAG
                        else span["text"]
                        for span in line["spans"]
                    )
                    text = text.replace("\xa0", " ").strip()
                    if not text or FOOTER_RE.match(text):
                        continue
                    lines.append((page_number, text))
    return lines


def _join_paragraphs(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Undo the PDF's hard line wrapping, restoring logical paragraphs.

    Each paragraph keeps the page its first line appeared on.
    """
    paragraphs: list[tuple[int, str]] = []
    for page, line in lines:
        if not paragraphs or PARAGRAPH_START_RE.match(line):
            paragraphs.append((page, line))
        else:
            first_page, text = paragraphs.pop()
            paragraphs.append((first_page, f"{text} {line}"))
    return paragraphs


def _split_parts(
    paragraphs: list[tuple[int, str]], max_chars: int
) -> list[tuple[int, str]]:
    """Group paragraphs into parts of at most ``max_chars``, never splitting one.

    Long articles run across pages, so each part reports the page its own text
    starts on rather than the page the article began on.
    """
    parts: list[tuple[int, str]] = []
    current: list[str] = []
    page = 0
    size = 0
    for paragraph_page, paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            parts.append((page, "\n".join(current)))
            current, size = [], 0
        if not current:
            page = paragraph_page
        current.append(paragraph)
        size += len(paragraph) + 1
    if current:
        parts.append((page, "\n".join(current)))
    return parts


def _document_title(lines: list[tuple[int, str]], fallback: str) -> str:
    """The heading lines above the first Section/Chapter/Article, title-cased."""
    header: list[str] = []
    for _, line in lines:
        if SECTION_RE.match(line) or CHAPTER_RE.match(line) or ARTICLE_RE.match(line):
            break
        header.append(line)
    if not header:
        return fallback.replace("_", " ").title()
    title = " ".join(header)
    return title.title() if title.isupper() else title


def parse_document(pdf_path: Path, max_chunk_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Parse one PDF into article-level chunks."""
    lines = _read_lines(pdf_path)
    doc_id = pdf_path.stem
    doc_title = _document_title(lines, doc_id)

    chunks: list[Chunk] = []
    section = chapter = ""
    article_number = article_title = ""
    body: list[tuple[int, str]] = []
    amendments: list[str] = []

    def flush() -> None:
        """Emit the article accumulated so far."""
        nonlocal body, amendments
        if not article_number:
            body, amendments = [], []
            return
        parts = _split_parts(_join_paragraphs(body), max_chunk_chars)
        for index, (part_page, part) in enumerate(parts, start=1):
            if not part.strip():
                continue
            suffix = f"::p{index}" if len(parts) > 1 else ""
            slug = article_number.translate(SUPERSCRIPT_TO_ASCII)
            chunks.append(
                Chunk(
                    id=f"{doc_id}::art-{slug}{suffix}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=section,
                    chapter=chapter,
                    article=f"Article {article_number}",
                    title=article_title,
                    text=part,
                    page=part_page,
                    part=index,
                    n_parts=len(parts),
                    amendments=list(amendments),
                )
            )
        body, amendments = [], []

    index = 0
    while index < len(lines):
        page, line = lines[index]
        next_line = lines[index + 1][1] if index + 1 < len(lines) else ""

        if section_match := SECTION_RE.match(line):
            flush()
            article_number = article_title = ""
            title = "" if _is_heading(next_line) else next_line
            section = f"Section {section_match.group(1)}"
            if title:
                section += f" - {title}"
                index += 1
        elif chapter_match := CHAPTER_RE.match(line):
            flush()
            article_number = article_title = ""
            title = chapter_match.group(2) or ""
            if not title and not _is_heading(next_line):
                title = next_line
                index += 1
            chapter = f"Chapter {chapter_match.group(1)}"
            if title:
                chapter += f" - {title}"
        elif article_match := ARTICLE_RE.match(line):
            flush()
            article_number, article_title = article_match.group(1), article_match.group(2)
        elif AMENDMENT_RE.match(line):
            amendments.append(line)
        elif article_number:
            body.append((page, line))

        index += 1

    flush()
    return chunks


def _is_heading(line: str) -> bool:
    return bool(SECTION_RE.match(line) or CHAPTER_RE.match(line) or ARTICLE_RE.match(line))


def build_corpus(raw_dir: Path = RAW_DIR, output_path: Path = CHUNKS_PATH) -> list[dict]:
    """Parse every PDF in ``raw_dir`` and write the combined chunk corpus."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    documents: list[dict] = []
    for pdf_path in pdf_paths:
        chunks = parse_document(pdf_path)
        print(f"{pdf_path.name}: {len(chunks)} chunks")
        documents.extend(asdict(chunk) for chunk in chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(documents)} chunks to {output_path}")
    return documents


def load_corpus(path: Path = CHUNKS_PATH) -> list[dict]:
    """Load the chunk corpus, building it first if it does not exist yet."""
    if not path.exists():
        return build_corpus(output_path=path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build_corpus()
