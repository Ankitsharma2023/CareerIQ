import logging
import os
import re

from core.config import settings
from utils.text_cleaner import clean_text

from services.hidden_text import HiddenTextReport, extract_visible_text

logger = logging.getLogger(__name__)

# Try pdfplumber first (better for multi-column resumes), fallback to pypdf
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

try:
    import docx as _docx
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False


# The upload limit is owned by core.config, which reads it from the
# environment. This module used to hard-code 5 MB as well, so raising
# MAX_FILE_SIZE_MB in the environment moved the router's check and left this
# one at 5: the upload passed the endpoint and then failed inside the parser
# with a different message.


def _extract_pdf_pdfplumber(file_path: str) -> tuple[str, HiddenTextReport]:
    """Extract the *visible* text from a PDF, and report what was dropped.

    pdfplumber is used rather than pypdf because it exposes per-character
    colour, size and position, which is what separates text a reader can see
    from text only a parser can. See services/hidden_text.py.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            return extract_visible_text(pdf)
    except Exception:
        logger.exception("pdfplumber failed to read %s; falling back to pypdf", file_path)
        return "", HiddenTextReport(inspected=False)


def _extract_pdf_pypdf(file_path: str) -> str:
    """Fallback: extract text using pypdf."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception:
        logger.exception("pypdf failed to read %s", file_path)
    return text


def extract_pdf(file_path: str) -> tuple[str, HiddenTextReport]:
    """Try pdfplumber; fall back to pypdf. Returns visible text and a report.

    The pypdf fallback cannot see character attributes, so when it runs the
    report is marked ``inspected=False``. A caller must not read that as a
    clean document — it means the document was never looked at.
    """
    if _HAS_PDFPLUMBER:
        text, report = _extract_pdf_pdfplumber(file_path)
        if text.strip():
            return text, report
    if _HAS_PYPDF:
        return _extract_pdf_pypdf(file_path), HiddenTextReport(inspected=False)
    return "", HiddenTextReport(inspected=False)


def extract_text_from_pdf(file_path: str) -> str:
    """Visible text only, for callers that do not need the report."""
    return extract_pdf(file_path)[0]


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX, including tables."""
    if not _HAS_DOCX:
        raise RuntimeError("python-docx not installed")
    text_parts = []
    try:
        doc = _docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        # Also extract table cells
        for table in doc.tables:
            for row in table.rows:
                row_text = "  ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)
    except Exception:
        logger.exception("python-docx failed to read %s", file_path)
    return "\n".join(text_parts)


def extract_contact_info(raw_text: str) -> dict:
    """Best-effort contact field extraction — does not store anything."""
    text = raw_text[:1500]   # only scan header region

    email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    linkedin_match = re.search(r"linkedin\.com/in/([\w\-]+)", text, re.I)
    github_match   = re.search(r"github\.com/([\w\-]+)", text, re.I)
    name_line = text.strip().splitlines()[0] if text.strip() else ""

    return {
        "name":     name_line[:60] if len(name_line) < 60 else "",
        "email":    email_match.group(0)     if email_match    else "",
        "phone":    phone_match.group(0)     if phone_match    else "",
        "linkedin": linkedin_match.group(1)  if linkedin_match else "",
        "github":   github_match.group(1)    if github_match   else "",
    }


def validate_file_size(file_path: str) -> None:
    """Raise ValueError if the file exceeds the configured upload limit.

    Read from ``settings`` at call time, not bound at import, so a test or a
    deployment that changes the limit gets the limit it set.
    """
    size = os.path.getsize(file_path)
    if size > settings.max_file_size_bytes:
        raise ValueError(
            f"File size {size / 1024 / 1024:.1f} MB exceeds limit of "
            f"{settings.max_file_size_mb} MB."
        )


def parse_resume(file_path: str) -> dict:
    """Main resume parsing entry point."""
    validate_file_size(file_path)

    ext = file_path.rsplit(".", 1)[-1].lower()

    hidden = HiddenTextReport(inspected=False)
    if ext == "pdf":
        raw_text, hidden = extract_pdf(file_path)
    elif ext == "docx":
        raw_text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    if not raw_text.strip():
        raise ValueError("Could not extract any text from the file. The PDF may be image-based or corrupted.")

    contact_info = extract_contact_info(raw_text)
    cleaned      = clean_text(raw_text)

    return {
        "raw_text":    raw_text,
        "clean_text":  cleaned,
        "contact_info": contact_info,
        "word_count":  len(cleaned.split()),
        "char_count":  len(raw_text),
        # Everything downstream scores raw_text, which now contains only what a
        # reader can see. This is the record of what was removed.
        "hidden_text": hidden.to_dict(),
    }
