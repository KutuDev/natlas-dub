"""Brand / entity protection: mask terms before translation, restore after.

Masked automatically: URLs, emails, phone numbers, currency amounts,
@handles, #hashtags, ALL-CAPS tokens (e.g. MTN, GTB) and CamelCase tokens
(e.g. PayPal). Plus any user-supplied protected terms (case-insensitive,
original casing restored).

Placeholders look like [[P0]], [[P1]] ... The restore step is tolerant of
spacing/casing changes the model may introduce (e.g. "[[ p0 ]]", "[P0]").
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

URL_RE = r"(?:https?://|www\.)[^\s<>\"]+[^\s<>\".,;:!?)]"
EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.-]+[A-Za-z]"
DOMAIN_RE = r"\b[\w-]+\.(?:com|ng|org|net|io|co|africa|app|ai)(?:\.ng)?(?:/[^\s]*)?\b"
PHONE_RE = r"(?:\+?234[\s-]?|\b0)[789][01]\d(?:[\s-]?\d){7}\b|\+\d{1,3}(?:[\s-]?\d{2,4}){2,4}\b"
CURRENCY_RE = (
    r"(?:[₦$£€]|\bNGN\s?|\bN(?=\d))\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|million|billion|thousand)\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:naira|kobo|dollars?|NGN|USD)\b"
)
HANDLE_RE = r"(?<![\w@])[@#][A-Za-z0-9_]{2,}"
ALLCAPS_RE = r"\b[A-Z][A-Z0-9&]{1,}\b"
CAMEL_RE = r"\b[A-Z]?[a-z]+[A-Z][A-Za-z0-9]*\b"

AUTO_PATTERNS = [
    ("url", URL_RE),
    ("email", EMAIL_RE),
    ("url", DOMAIN_RE),
    ("phone", PHONE_RE),
    ("currency", CURRENCY_RE),
    ("handle", HANDLE_RE),
    ("brand", ALLCAPS_RE),
    ("brand", CAMEL_RE),
]

# Short all-caps words that are usually not brands.
ALLCAPS_STOP = {"I", "OK", "A", "AM", "PM", "TV"}

PLACEHOLDER_RE = re.compile(r"\[\[?\s*[Pp]\s*(\d+)\s*\]?\]")


@dataclass
class MaskResult:
    masked_text: str
    mapping: Dict[str, str] = field(default_factory=dict)  # "[[P0]]" -> original
    kinds: Dict[str, str] = field(default_factory=dict)


def mask(text: str, protected_terms: Sequence[str] = (), auto: bool = True) -> MaskResult:
    spans = []  # (start, end, kind)

    terms = sorted({t.strip() for t in protected_terms if t and t.strip()}, key=len, reverse=True)
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), "term"))

    if auto:
        for kind, pattern in AUTO_PATTERNS:
            for m in re.finditer(pattern, text):
                if kind == "brand" and m.group(0) in ALLCAPS_STOP:
                    continue
                spans.append((m.start(), m.end(), kind))

    # Resolve overlaps: earlier start wins, then longer span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen = []
    last_end = -1
    for s in spans:
        if s[0] >= last_end:
            chosen.append(s)
            last_end = s[1]

    out, mapping, kinds = [], {}, {}
    cursor = 0
    for i, (start, end, kind) in enumerate(chosen):
        ph = f"[[P{i}]]"
        out.append(text[cursor:start])
        out.append(ph)
        mapping[ph] = text[start:end]
        kinds[ph] = kind
        cursor = end
    out.append(text[cursor:])
    return MaskResult("".join(out), mapping, kinds)


def restore(translated: str, mapping: Dict[str, str]) -> (str, List[str]):
    """Restore placeholders. Returns (text, missing_originals).

    Missing placeholders (dropped by the model) are reported, never invented.
    """
    by_index = {}
    for ph, original in mapping.items():
        m = PLACEHOLDER_RE.fullmatch(ph)
        by_index[int(m.group(1))] = original

    seen = set()

    def _sub(m):
        idx = int(m.group(1))
        if idx in by_index:
            seen.add(idx)
            return by_index[idx]
        return m.group(0)

    restored = PLACEHOLDER_RE.sub(_sub, translated)
    missing = [by_index[i] for i in sorted(by_index) if i not in seen]
    return restored, missing
