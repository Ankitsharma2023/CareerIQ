"""Build PDFs with controlled character attributes, for the hidden-text work.

Hand-assembled rather than produced with a reporting library, for two reasons:
it adds no dependency to a project that has been deliberate about them, and it
gives exact control over the three attributes under test — fill colour, font
size and position — which a layout engine would normalise away.

The cross-reference table is computed from real byte offsets, because pypdf
refuses a file whose offsets are wrong and the pypdf fallback path is one of
the things being measured.
"""
from __future__ import annotations

from dataclasses import dataclass

PAGE_WIDTH = 612
PAGE_HEIGHT = 792

# The three ways to put text in a PDF that a reader cannot see.
ATTACK_MODES = ("white_text", "tiny_font", "off_page")


@dataclass
class TextRun:
    text: str
    x: float
    y: float
    size: float = 11.0
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(runs: list[TextRun], width: int = PAGE_WIDTH, height: int = PAGE_HEIGHT) -> bytes:
    """One page, one font, the given runs — with a correct xref table."""
    body = "".join(
        f"BT {run.color[0]} {run.color[1]} {run.color[2]} rg /F1 {run.size} Tf "
        f"{run.x} {run.y} Td ({_escape(run.text)}) Tj ET\n"
        for run in runs
    ).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ).encode(),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(body), body),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"

    startxref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % startxref
    return bytes(out)


def resume_pdf(
    visible_lines: list[str],
    hidden_lines: list[str] | None = None,
    mode: str = "white_text",
) -> bytes:
    """A resume whose visible half reads normally and whose hidden half does not.

    * ``white_text`` — white on the page's white, full size. Survives printing,
      defeated by selecting the text and changing its colour.
    * ``tiny_font`` — 0.5pt black. Visible as a smudge at 400% zoom; at 100% it
      is a line of grey dust, and most readers take it for a rule.
    * ``off_page`` — full size, black, positioned below the MediaBox. Nothing
      renders at all, so no inspection of the visible page can find it.
    """
    if mode not in ATTACK_MODES:
        raise ValueError(f"mode must be one of {ATTACK_MODES}, got {mode!r}")

    runs = [
        TextRun(line, x=72, y=PAGE_HEIGHT - 72 - index * 14)
        for index, line in enumerate(visible_lines)
    ]

    cursor = PAGE_HEIGHT - 72 - len(visible_lines) * 14 - 20
    for index, line in enumerate(hidden_lines or []):
        if mode == "white_text":
            runs.append(TextRun(line, x=72, y=cursor - index * 14, color=(1.0, 1.0, 1.0)))
        elif mode == "tiny_font":
            runs.append(TextRun(line, x=72, y=cursor - index * 2, size=0.5))
        else:  # off_page
            runs.append(TextRun(line, x=72, y=-60 - index * 14))

    return build_pdf(runs)
