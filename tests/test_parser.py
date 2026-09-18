"""Resume parsing: extraction, contact fields, size limits and failure paths.

parser.py sat at 31% coverage — the lowest in the repo — while being the first
thing every upload touches and the module the hidden-text detector is about to
be built on top of. The untested half was the half that runs when a document is
malformed, which is exactly when a silent failure costs the most.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from core.config import settings
from services import parser
from services.parser import (
    extract_contact_info,
    extract_text_from_docx,
    extract_text_from_pdf,
    parse_resume,
    validate_file_size,
)


def _build_pdf(line: str) -> bytes:
    """A one-page PDF with a correct cross-reference table.

    Assembled rather than pasted as a literal because pypdf refuses a file
    whose xref offsets are wrong, and the fallback path is one of the things
    under test here.
    """
    content = f"BT /F1 12 Tf 72 720 Td ({line}) Tj ET\n".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    startxref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % startxref
    return bytes(out)


MINIMAL_PDF = _build_pdf("Ada Lovelace ada@example.com Python Engineer")


@pytest.fixture
def pdf_file(tmp_path: Path) -> str:
    path = tmp_path / "resume.pdf"
    path.write_bytes(MINIMAL_PDF)
    return str(path)


@pytest.fixture
def docx_file(tmp_path: Path) -> str:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Grace Hopper")
    document.add_paragraph("grace@example.com")
    document.add_paragraph("   ")  # blank paragraphs must not become blank lines
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "COBOL, Compilers"
    path = tmp_path / "resume.docx"
    document.save(str(path))
    return str(path)


class TestContactExtraction:
    RESUME = (
        "Ada Lovelace\n"
        "ada.lovelace@example.com | +1 (555) 013-4567\n"
        "linkedin.com/in/ada-lovelace · github.com/adalovelace\n"
    )

    def test_finds_every_field(self):
        info = extract_contact_info(self.RESUME)
        assert info["name"] == "Ada Lovelace"
        assert info["email"] == "ada.lovelace@example.com"
        assert "555" in info["phone"]
        assert info["linkedin"] == "ada-lovelace"
        assert info["github"] == "adalovelace"

    def test_missing_fields_come_back_empty_not_absent(self):
        """Callers index these keys directly; a missing key would be a 500."""
        info = extract_contact_info("Just some prose with no contact details.")
        assert set(info) == {"name", "email", "phone", "linkedin", "github"}
        assert info["email"] == info["phone"] == info["linkedin"] == info["github"] == ""

    def test_empty_text_is_not_an_error(self):
        assert extract_contact_info("")["name"] == ""

    def test_a_long_first_line_is_not_treated_as_a_name(self):
        info = extract_contact_info("x" * 80 + "\nada@example.com")
        assert info["name"] == ""
        assert info["email"] == "ada@example.com"

    def test_only_the_header_region_is_scanned(self):
        """An address buried on page two is not this candidate's contact info."""
        buried = "Ada Lovelace\n" + ("filler line\n" * 400) + "someone.else@example.com"
        assert extract_contact_info(buried)["email"] == ""


class TestFileSizeLimit:
    def test_a_small_file_passes(self, tmp_path: Path):
        path = tmp_path / "small.pdf"
        path.write_bytes(b"x" * 1024)
        validate_file_size(str(path))  # must not raise

    def test_an_oversized_file_is_rejected(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(settings, "max_file_size_mb", 1)
        path = tmp_path / "big.pdf"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeds limit"):
            validate_file_size(str(path))

    def test_the_limit_follows_the_setting(self, tmp_path: Path, monkeypatch):
        """The regression this closes: parser.py hard-coded its own 5 MB, so
        raising the configured limit moved the router's check and left this one
        behind — the upload passed the endpoint and failed inside the parser."""
        path = tmp_path / "medium.pdf"
        path.write_bytes(b"x" * (6 * 1024 * 1024))

        monkeypatch.setattr(settings, "max_file_size_mb", 5)
        with pytest.raises(ValueError):
            validate_file_size(str(path))

        monkeypatch.setattr(settings, "max_file_size_mb", 10)
        validate_file_size(str(path))  # must not raise

    def test_the_message_names_the_configured_limit(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(settings, "max_file_size_mb", 2)
        path = tmp_path / "big.pdf"
        path.write_bytes(b"x" * (3 * 1024 * 1024))
        with pytest.raises(ValueError, match="limit of 2 MB"):
            validate_file_size(str(path))


class TestPdfExtraction:
    def test_reads_text_out_of_a_pdf(self, pdf_file):
        assert "Ada Lovelace" in extract_text_from_pdf(pdf_file)

    def test_falls_back_to_pypdf_when_pdfplumber_returns_nothing(self, pdf_file, monkeypatch):
        monkeypatch.setattr(parser, "_extract_pdf_pdfplumber", lambda _p: "")
        assert "Ada Lovelace" in extract_text_from_pdf(pdf_file)

    def test_a_corrupt_pdf_yields_empty_text_rather_than_raising(self, tmp_path: Path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not a pdf")
        assert extract_text_from_pdf(str(path)).strip() == ""

    def test_the_failure_is_logged_not_printed(self, tmp_path: Path, caplog):
        """Extraction failures used to go to stdout, where a deployed server
        never sees them and no log level can silence them."""
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not a pdf")
        with caplog.at_level(logging.ERROR, logger="services.parser"):
            extract_text_from_pdf(str(path))
        assert caplog.records, "a failed extraction must reach the logs"


class TestDocxExtraction:
    def test_reads_paragraphs_and_tables(self, docx_file):
        text = extract_text_from_docx(docx_file)
        assert "Grace Hopper" in text
        assert "COBOL, Compilers" in text, "table cells carry the skills section"

    def test_blank_paragraphs_are_dropped(self, docx_file):
        assert "\n\n" not in extract_text_from_docx(docx_file)

    def test_a_corrupt_docx_yields_empty_text(self, tmp_path: Path):
        path = tmp_path / "broken.docx"
        path.write_bytes(b"not a zip archive")
        assert extract_text_from_docx(str(path)) == ""

    def test_missing_library_is_a_clear_error(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(parser, "_HAS_DOCX", False)
        with pytest.raises(RuntimeError, match="python-docx"):
            extract_text_from_docx(str(tmp_path / "anything.docx"))


class TestParseResume:
    def test_returns_the_full_record(self, pdf_file):
        result = parse_resume(pdf_file)
        assert "Ada Lovelace" in result["raw_text"]
        assert result["word_count"] > 0
        assert result["char_count"] == len(result["raw_text"])
        assert result["contact_info"]["email"] == "ada@example.com"

    def test_docx_goes_through_the_docx_path(self, docx_file):
        assert "Grace Hopper" in parse_resume(docx_file)["raw_text"]

    def test_an_unsupported_extension_is_refused_by_name(self, tmp_path: Path):
        path = tmp_path / "resume.txt"
        path.write_text("plain text resume")
        with pytest.raises(ValueError, match=r"\.txt"):
            parse_resume(str(path))

    def test_uppercase_extensions_are_accepted(self, tmp_path: Path):
        path = tmp_path / "RESUME.PDF"
        path.write_bytes(MINIMAL_PDF)
        assert "Ada Lovelace" in parse_resume(str(path))["raw_text"]

    def test_an_image_only_pdf_says_so(self, tmp_path: Path, monkeypatch):
        """The common real failure: a scanned resume with no text layer."""
        monkeypatch.setattr(parser, "extract_text_from_pdf", lambda _p: "   ")
        path = tmp_path / "scan.pdf"
        path.write_bytes(MINIMAL_PDF)
        with pytest.raises(ValueError, match="image-based"):
            parse_resume(str(path))

    def test_the_size_check_runs_before_any_parsing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(settings, "max_file_size_mb", 1)

        def _boom(_path):
            raise AssertionError("an oversized file must be refused before parsing")

        monkeypatch.setattr(parser, "extract_text_from_pdf", _boom)
        path = tmp_path / "big.pdf"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeds limit"):
            parse_resume(str(path))
