from __future__ import annotations

import unicodedata

from ibibio_tts.text import apply_reviewed_corrections, flag_text, normalize_text


def test_normalize_collapses_whitespace_and_punctuation_spacing() -> None:
    assert normalize_text("  Ndinamke   mmo ,  ndien  .  ") == "Ndinamke mmo, ndien."


def test_normalize_applies_nfc() -> None:
    precomposed = unicodedata.lookup("LATIN SMALL LETTER O WITH DOT BELOW")
    decomposed = unicodedata.normalize("NFD", precomposed)
    assert decomposed != precomposed
    assert normalize_text(decomposed) == precomposed


def test_flag_text_detects_number_and_date() -> None:
    assert flag_text("I have 5 goats") == ["number"]
    assert flag_text("born 12/05/1990") == ["date"]


def test_flag_text_detects_abbreviation() -> None:
    assert "abbreviation" in flag_text("Dr. Smith works at NASA")


def test_flag_text_clean_sentence_has_no_flags() -> None:
    assert flag_text("Ndinamke mmo, ndien.") == []


def test_flag_text_detects_unsupported_script() -> None:
    cjk = unicodedata.lookup("CJK UNIFIED IDEOGRAPH-4E2D")
    assert "unsupported_script" in flag_text("hello " + cjk)


def test_apply_reviewed_corrections_fixes_cyrillic_schwa() -> None:
    cyrillic_schwa = unicodedata.lookup("CYRILLIC SMALL LETTER SCHWA")
    latin_schwa = unicodedata.lookup("LATIN SMALL LETTER SCHWA")
    corrected = apply_reviewed_corrections("et" + cyrillic_schwa + "k")
    assert corrected == "et" + latin_schwa + "k"
    assert flag_text(corrected) == []


def test_apply_reviewed_corrections_is_a_no_op_for_clean_text() -> None:
    assert apply_reviewed_corrections("Ndinamke mmo, ndien.") == "Ndinamke mmo, ndien."
