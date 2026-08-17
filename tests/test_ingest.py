"""Parser tests, built around the ways it can fail silently."""

import pymupdf
import pytest

from legalization_assistant.ingest import (
    ARTICLE_RE,
    _join_paragraphs,
    _split_parts,
    parse_document,
)


def write_pdf(tmp_path, lines, name="test_law.pdf", lines_per_page=40):
    """Render one line per row into a real PDF, so the pymupdf path is exercised.

    Lines beyond ``lines_per_page`` spill onto further pages, which is what makes
    the page-attribution tests meaningful.
    """
    path = tmp_path / name
    document = pymupdf.open()
    font = pymupdf.Font("helv")
    for start in range(0, max(len(lines), 1), lines_per_page):
        page = document.new_page()
        writer = pymupdf.TextWriter(page.rect)
        for row, line in enumerate(lines[start : start + lines_per_page]):
            writer.append((50, 60 + row * 14), line, font=font, fontsize=9)
        writer.write_text(page)
    document.save(path)
    document.close()
    return path


def articles(chunks):
    return [chunk.article for chunk in chunks]


def wrapped_paragraph(number, continuations=5):
    """A paragraph as a PDF stores it: a marker line plus hard-wrapped lines.

    Returns the lines to render and the single paragraph they should rejoin into.
    Lines stay short so they fit the page width — a line wider than the page is
    clipped on extraction.
    """
    lines = [f"{number}. opening clause of paragraph {number}"]
    lines += [f"continuation line {index} of paragraph {number}" for index in range(continuations)]
    return lines, " ".join(lines)


class TestArticleHeadings:
    def test_dash_variants_all_match(self):
        # Escaped so the three dashes stay distinguishable in source.
        for dash in ("-", "\u2013", "\u2014"):
            match = ARTICLE_RE.match(f"Article 4 {dash} Entry into Georgia")
            assert match is not None, f"dash {dash!r} not recognised"
            assert match.group(1) == "4"
            assert match.group(2) == "Entry into Georgia"

    def test_cross_reference_is_not_a_heading(self):
        # The dash requirement is what separates a heading from body prose. If
        # this ever matches, the article it appears in is split in two.
        assert ARTICLE_RE.match("Article 15(k) of this Law shall be applied") is None
        assert ARTICLE_RE.match("as provided for by Article 8 of this Law") is None

    def test_superscript_number_is_kept(self):
        # Amending laws insert articles as "Article 20¹". Flattening that to
        # "Article 201" makes the assistant cite an article that does not exist.
        match = ARTICLE_RE.match("Article 20¹ - Special residence permit")
        assert match is not None
        assert match.group(1) == "20¹"


class TestParseDocument:
    def test_article_becomes_one_chunk_with_metadata(self, tmp_path):
        pdf = write_pdf(
            tmp_path,
            [
                "LAW OF GEORGIA ON TEST MATTERS",
                "Section I",
                "General Provisions",
                "Chapter I - Scope",
                "Article 1 - Purpose of this Law",
                "1. This Law regulates the test corpus.",
            ],
        )
        (chunk,) = parse_document(pdf)
        assert chunk.article == "Article 1"
        assert chunk.title == "Purpose of this Law"
        assert chunk.section == "Section I - General Provisions"
        assert chunk.chapter == "Chapter I - Scope"
        assert chunk.doc_id == "test_law"
        assert chunk.n_parts == 1
        assert "regulates the test corpus" in chunk.text

    def test_deleted_article_produces_no_chunk(self, tmp_path):
        # A repealed heading has no body. It must not swallow the next article.
        pdf = write_pdf(
            tmp_path,
            [
                "Article 44 - Still in force",
                "1. Some rule.",
                "Article 45 - (Deleted)",
                "Article 46 - Also in force",
                "1. Another rule.",
            ],
        )
        chunks = parse_document(pdf)
        assert articles(chunks) == ["Article 44", "Article 46"]
        assert "Another rule" in chunks[1].text
        assert "Another rule" not in chunks[0].text

    def test_amendment_trail_leaves_the_body(self, tmp_path):
        pdf = write_pdf(
            tmp_path,
            [
                "Article 7 - Categories of visa",
                "1. A visa is issued in categories.",
                "Law of Georgia No 1803 of 25 June 2026 - website, 05.07.2026",
            ],
        )
        (chunk,) = parse_document(pdf)
        assert chunk.amendments == ["Law of Georgia No 1803 of 25 June 2026 - website, 05.07.2026"]
        assert "No 1803" not in chunk.text

    def test_superscript_article_gets_ascii_slug(self, tmp_path):
        pdf = write_pdf(
            tmp_path,
            ["Article 20¹ - Inserted article", "1. Inserted by an amending law."],
        )
        (chunk,) = parse_document(pdf)
        assert chunk.article == "Article 20¹"
        assert chunk.id == "test_law::art-20-1"

    def test_long_article_splits_into_numbered_parts(self, tmp_path):
        rendered = ["Article 15 - Types of residence permits"]
        expected_paragraphs = []
        for number in range(1, 13):
            lines, paragraph = wrapped_paragraph(number)
            rendered += lines
            expected_paragraphs.append(paragraph)

        chunks = parse_document(write_pdf(tmp_path, rendered), max_chunk_chars=400)

        assert len(chunks) > 1
        assert {chunk.n_parts for chunk in chunks} == {len(chunks)}
        assert [chunk.part for chunk in chunks] == list(range(1, len(chunks) + 1))
        assert [chunk.id for chunk in chunks] == [
            f"test_law::art-15::p{n}" for n in range(1, len(chunks) + 1)
        ]
        # Every paragraph survives the split exactly once, rejoined from its
        # hard-wrapped lines.
        rejoined = "\n".join(chunk.text for chunk in chunks)
        for paragraph in expected_paragraphs:
            assert rejoined.count(paragraph) == 1

    def test_each_part_reports_the_page_its_own_text_starts_on(self, tmp_path):
        # The bug this pins: every part used to report the article's start page,
        # so a citation for part 5 sent the reader to where part 1 began.
        paragraphs = [f"{n}. clause {n}" for n in range(1, 13)]
        pdf = write_pdf(
            tmp_path,
            ["Article 15 - Types of residence permits", *paragraphs],
            lines_per_page=5,
        )
        chunks = parse_document(pdf, max_chunk_chars=40)

        pages = [chunk.page for chunk in chunks]
        assert len(set(pages)) > 1, "article did not span pages; test proves nothing"
        assert pages == sorted(pages)
        assert min(pages) == 1


class TestJoinParagraphs:
    def test_hard_wraps_are_rejoined(self):
        lines = [(1, "1. A permit is issued"), (1, "for one year."), (1, "2. It may be renewed.")]
        assert _join_paragraphs(lines) == [
            (1, "1. A permit is issued for one year."),
            (1, "2. It may be renewed."),
        ]

    def test_paragraph_keeps_the_page_it_started_on(self):
        lines = [(4, "1. A permit is issued"), (5, "for one year.")]
        assert _join_paragraphs(lines) == [(4, "1. A permit is issued for one year.")]

    @pytest.mark.parametrize("marker", ["1. ", "12. ", "a) ", "a1) ", "b. "])
    def test_recognised_paragraph_markers_start_a_new_paragraph(self, marker):
        lines = [(1, "1. First clause."), (1, f"{marker}Second clause.")]
        assert len(_join_paragraphs(lines)) == 2


class TestSplitParts:
    def test_paragraph_is_never_split(self):
        paragraphs = [(1, "a" * 300), (1, "b" * 300)]
        parts = _split_parts(paragraphs, max_chars=400)
        assert [text for _, text in parts] == ["a" * 300, "b" * 300]

    def test_oversized_paragraph_survives_whole(self):
        parts = _split_parts([(1, "x" * 900)], max_chars=400)
        assert parts == [(1, "x" * 900)]

    def test_part_reports_the_page_of_its_first_paragraph(self):
        paragraphs = [(7, "a" * 300), (8, "b" * 300), (9, "c" * 300)]
        parts = _split_parts(paragraphs, max_chars=400)
        assert [page for page, _ in parts] == [7, 8, 9]

    def test_no_paragraphs_yields_no_parts(self):
        assert _split_parts([], max_chars=400) == []
