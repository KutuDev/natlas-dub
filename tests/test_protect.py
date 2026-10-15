from app_imports import protect  # noqa


def test_mask_and_restore_roundtrip():
    text = "Visit https://opex.ng/offer or call +234 803 123 4567. Pay ₦5,000 at Jollof Express via PayPal, MTN."
    m = protect.mask(text, ["Jollof Express"])
    originals = set(m.mapping.values())
    assert "https://opex.ng/offer" in originals
    assert "Jollof Express" in originals
    assert "PayPal" in originals and "MTN" in originals
    assert any("5,000" in o for o in originals)
    assert any("803" in o for o in originals)
    for o in originals:
        assert o not in m.masked_text
    restored, missing = protect.restore(m.masked_text, m.mapping)
    assert restored == text and missing == []


def test_restore_tolerates_spacing_and_reports_missing():
    m = protect.mask("Buy from Awarri now", ["Awarri"])
    ph = next(iter(m.mapping))
    restored, missing = protect.restore("Ra lati [[ p0 ]] bayi", m.mapping)
    assert restored == "Ra lati Awarri bayi" and missing == []
    restored, missing = protect.restore("Ra bayi", m.mapping)
    assert missing == ["Awarri"]


def test_user_terms_case_insensitive_keep_original_casing():
    m = protect.mask("opex consulting builds tools", ["OPEX Consulting"])
    assert list(m.mapping.values()) == ["opex consulting"]
