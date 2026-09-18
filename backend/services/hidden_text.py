"""Detect text a human cannot read but a parser can.

The attack is simple and already in the wild: paste the job description's skill
list into the resume in white 1pt type, or push it off the page edge. A
recruiter opening the PDF sees a normal resume. Every text extractor — this
one, an ATS, a keyword filter — sees the skills.

Verified against pdfplumber 0.11: ``page.extract_text()`` returns white-on-
white text, 0.5pt text and text positioned outside the MediaBox, with nothing
to distinguish it from the visible document. So the filtering has to happen
character by character, before the text is assembled.

Three signals, each checked against what a person looking at the page would
see:

* **Contrast.** A character's fill colour against the colour actually behind it
  — the topmost filled rectangle covering it, or the page white. Below a
  contrast ratio of 1.5:1 the glyph is invisible. The threshold is deliberately
  far below the WCAG 4.5:1 readability floor: the goal is to catch text that
  cannot be seen at all, not to judge a designer's light-grey subheading. Grey
  #999 on white is 2.85:1 and is left alone.
* **Size.** Below 4pt no one reads it; 1pt and 0.5pt are the sizes the attack
  actually uses. Ordinary fine print is 6-8pt.
* **Position.** A character whose box falls outside the page box does not
  render. Off-page placement is the variant that survives a "select all, change
  colour to black" inspection.

What this does not catch: text rendering mode 3 (Tr 3), because pdfplumber does
not expose it on characters. That mode is also how scanners store a searchable
OCR layer under a page image, so treating it as an attack would break every
scanned resume. It is listed here as a known gap rather than silently missing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A character below this contrast against its own background is unreadable.
MIN_CONTRAST_RATIO = 1.5
# Points. Fine print is 6-8pt; the attack uses 1pt and below.
MIN_FONT_SIZE = 4.0
# Characters must fall this far outside the page box to count as off-page,
# so a descender grazing the margin is not a finding.
PAGE_MARGIN_TOLERANCE = 2.0

REASON_CONTRAST = "low_contrast"
REASON_SIZE = "tiny_font"
REASON_OFF_PAGE = "off_page"

# Phrases that only appear in text written for the model rather than the reader.
INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) instructions",
    r"disregard (?:the )?(?:previous|prior|above) (?:instructions|prompt)",
    r"you are (?:now )?(?:a|an) \w+",
    r"system\s*:",
    r"rate this (?:candidate|resume) (?:as )?(?:a )?(?:100|perfect|10/10|highly)",
    r"(?:score|rank) (?:this|the) (?:candidate|resume) (?:at |as )?(?:the )?(?:top|highest|100)",
    r"(?:this is|the) (?:ideal|perfect) candidate",
    r"do not (?:mention|reveal|report)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)


@dataclass
class HiddenTextReport:
    """What was removed, and why."""

    detected: bool = False
    hidden_text: str = ""
    hidden_char_count: int = 0
    visible_char_count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    injection_phrases: list[str] = field(default_factory=list)
    # True when the document could not be inspected character by character
    # (pypdf fallback, or pdfplumber failed). Absence of findings then means
    # "not looked at", not "clean", and callers must not report it as clean.
    inspected: bool = True

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "inspected": self.inspected,
            "hidden_char_count": self.hidden_char_count,
            "visible_char_count": self.visible_char_count,
            "reasons": dict(self.reasons),
            "samples": list(self.samples),
            "injection_phrases": list(self.injection_phrases),
            "hidden_text": self.hidden_text[:2000],
        }


# ── Colour ────────────────────────────────────────────────────────────────────

def _linearize(channel: float) -> float:
    """sRGB gamma expansion, per WCAG 2.x."""
    channel = min(max(float(channel), 0.0), 1.0)
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(color) -> float | None:
    """Relative luminance of a PDF colour, or None if it cannot be read.

    PDF fill colours arrive as a tuple whose length names the space: one value
    is DeviceGray, three DeviceRGB, four DeviceCMYK. Anything else (a separation
    or indexed space) is not worth guessing at — None means "unknown", and an
    unknown colour is never treated as hidden.
    """
    if color is None:
        return None
    if isinstance(color, (int, float)):
        color = (color,)
    try:
        values = [float(v) for v in color]
    except (TypeError, ValueError):
        return None

    if len(values) == 1:
        r = g = b = values[0]
    elif len(values) == 3:
        r, g, b = values
    elif len(values) == 4:
        c, m, y, k = values
        r, g, b = (1 - min(1.0, c + k), 1 - min(1.0, m + k), 1 - min(1.0, y + k))
    else:
        return None

    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(first: float, second: float) -> float:
    """WCAG contrast ratio between two relative luminances: 1.0 to 21.0."""
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


# ── Per-character classification ──────────────────────────────────────────────

def _covers(rect: dict, char: dict) -> bool:
    return (
        rect.get("x0", 0) <= char["x0"]
        and rect.get("x1", 0) >= char["x1"]
        and rect.get("top", 0) <= char["top"]
        and rect.get("bottom", 0) >= char["bottom"]
    )


def _background_luminance(char: dict, rects: list[dict]) -> float:
    """Luminance behind a character: the last filled rect covering it, else white.

    Last rather than first because PDF paints in content-stream order, so a
    later rectangle is on top. White-on-black is legitimate design and must not
    be flagged; white-on-white is the attack.
    """
    background = 1.0
    for rect in rects:
        if not _covers(rect, char):
            continue
        value = luminance(rect.get("non_stroking_color"))
        if value is not None:
            background = value
    return background


def classify_char(char: dict, rects: list[dict], page_bbox: tuple) -> str | None:
    """Return why this character is invisible, or None if a reader would see it."""
    size = char.get("size") or 0.0
    if 0 < size < MIN_FONT_SIZE:
        return REASON_SIZE

    x0, top, x1, bottom = page_bbox
    tolerance = PAGE_MARGIN_TOLERANCE
    if (
        char["x1"] < x0 - tolerance
        or char["x0"] > x1 + tolerance
        or char["bottom"] < top - tolerance
        or char["top"] > bottom + tolerance
    ):
        return REASON_OFF_PAGE

    ink = luminance(char.get("non_stroking_color"))
    if ink is not None:
        background = _background_luminance(char, rects)
        if contrast_ratio(ink, background) < MIN_CONTRAST_RATIO:
            return REASON_CONTRAST

    return None


def find_injection_phrases(text: str) -> list[str]:
    """Instructions addressed to the model rather than to a reader."""
    return sorted({match.group(0).strip() for match in _INJECTION_RE.finditer(text)})


# ── Page and document level ───────────────────────────────────────────────────

def _page_bbox(page) -> tuple:
    bbox = getattr(page, "bbox", None)
    if bbox and len(bbox) == 4:
        return tuple(bbox)
    return (0.0, 0.0, float(page.width), float(page.height))


def split_page(page) -> tuple[str, str, dict[str, int], int, int]:
    """Split one page into (visible text, hidden text, reasons, hidden n, visible n)."""
    rects = list(getattr(page, "rects", []) or [])
    bbox = _page_bbox(page)

    reasons: dict[str, int] = {}
    hidden_ids: set[int] = set()

    for char in page.chars:
        reason = classify_char(char, rects, bbox)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            hidden_ids.add(id(char))

    if not hidden_ids:
        return page.extract_text(x_tolerance=3, y_tolerance=3) or "", "", {}, 0, len(page.chars)

    visible_page = page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) not in hidden_ids)
    hidden_page = page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) in hidden_ids)

    visible = visible_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    # Off-page characters sit outside the layout engine's grid, so they are
    # gathered directly rather than through extract_text, which can drop them.
    hidden = hidden_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    if not hidden.strip():
        hidden = "".join(c["text"] for c in page.chars if id(c) in hidden_ids)

    return visible, hidden, reasons, len(hidden_ids), len(page.chars) - len(hidden_ids)


def extract_visible_text(pdf) -> tuple[str, HiddenTextReport]:
    """Read an open pdfplumber document, dropping anything a reader cannot see."""
    report = HiddenTextReport()
    visible_parts: list[str] = []
    hidden_parts: list[str] = []

    for page in pdf.pages:
        visible, hidden, reasons, hidden_n, visible_n = split_page(page)
        if visible.strip():
            visible_parts.append(visible)
        if hidden.strip():
            hidden_parts.append(hidden)
        for reason, count in reasons.items():
            report.reasons[reason] = report.reasons.get(reason, 0) + count
        report.hidden_char_count += hidden_n
        report.visible_char_count += visible_n

    report.hidden_text = "\n".join(hidden_parts).strip()
    report.detected = bool(report.hidden_text)
    if report.detected:
        report.samples = [
            line.strip()[:120] for line in report.hidden_text.splitlines() if line.strip()
        ][:5]
        report.injection_phrases = find_injection_phrases(report.hidden_text)
        logger.warning(
            "Hidden text detected: %d characters (%s)",
            report.hidden_char_count,
            ", ".join(f"{k}={v}" for k, v in sorted(report.reasons.items())),
        )

    return "\n".join(visible_parts), report
