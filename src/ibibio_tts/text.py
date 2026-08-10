"""Text normalization and review flags for Common Voice sentences.

Per `PROTOCOL.md`: normalize with Unicode NFC, collapse repeated whitespace,
apply only safe punctuation spacing, preserve raw and normalized text, never
translate/paraphrase/transliterate, and flag numbers, dates, abbreviations,
and unsupported scripts for review instead of silently dropping them.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_DIGIT_RE = re.compile(r"\d")
_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
_ABBREVIATION_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}|\b[A-Z]{2,}\b")

# Marks, digits, punctuation, and whitespace are script-neutral and always
# allowed; "Mn" covers the combining diacritics used for tone marking.
_ALLOWED_NEUTRAL_CATEGORIES = {"Mn", "Nd", "Zs", "Po", "Pd", "Pc", "Ps", "Pe"}
# Ibibio is written in the Latin script. Letters are only allowed if their
# Unicode character name identifies them as Latin; general category alone
# (e.g. "Lo") does not distinguish Latin from CJK, Cyrillic, Greek, etc.
_LETTER_CATEGORIES = {"Ll", "Lu", "Lt", "Lm", "Lo"}


def _is_allowed_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    if category in _ALLOWED_NEUTRAL_CATEGORIES:
        return True
    if category in _LETTER_CATEGORIES:
        try:
            return "LATIN" in unicodedata.name(ch)
        except ValueError:
            return False
    return False


def normalize_text(raw: str) -> str:
    """Apply the only transformations PROTOCOL.md permits: NFC, whitespace, spacing."""
    text = unicodedata.normalize("NFC", raw)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text


def flag_text(normalized: str) -> list[str]:
    """Return review flags; never used to reject or alter the text."""
    flags = []
    if _DATE_RE.search(normalized):
        flags.append("date")
    elif _DIGIT_RE.search(normalized):
        flags.append("number")
    if _ABBREVIATION_RE.search(normalized):
        flags.append("abbreviation")
    if any(not _is_allowed_char(ch) for ch in normalized):
        flags.append("unsupported_script")
    return flags


# Reviewed, evidenced corrections -- NOT part of normalize_text()'s fixed
# PROTOCOL.md contract (NFC/whitespace/spacing only). Each entry here must be
# backed by a documented review, not a guess; see papers.md for the evidence
# trail. This is a canonicalization of a wrong-keyboard input artifact, not
# translation, paraphrase, or ASCII transliteration.
#
# 2026-08-10: U+04D9 CYRILLIC SMALL LETTER SCHWA -> U+0259 LATIN SMALL LETTER
# SCHWA. Found in 198 Common Voice rows across 15/17 speakers, tracing to
# only 63 shared prompt sentences (contributors read prompts, they do not
# type them, so this is a prompt-bank defect, not per-speaker). The correct
# Latin schwa never appears anywhere else in the corpus, consistent with a
# prompt-author input-method/font mistake rather than a deliberate distinct
# grapheme. See papers.md for the full investigation.
KNOWN_CONFUSABLE_SUBSTITUTIONS: dict[str, str] = {
    "ә": "ə",
}


def apply_reviewed_corrections(normalized: str) -> str:
    """Apply only the documented, evidenced substitutions above."""
    for wrong, right in KNOWN_CONFUSABLE_SUBSTITUTIONS.items():
        normalized = normalized.replace(wrong, right)
    return normalized
