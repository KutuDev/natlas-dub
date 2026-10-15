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


def test_asr_chunk_conversion_and_window_fallback():
    import numpy as np
    from app_imports import asr
    segs = asr.chunks_to_segments({"chunks": [
        {"timestamp": (0.0, 2.5), "text": " Welcome to Moniepoint "},
        {"timestamp": (2.5, None), "text": "powering dreams"},
        {"timestamp": (5.0, 6.0), "text": "  "}]}, duration=7.0)
    assert [(s["start"], s["end"], s["text"]) for s in segs] == [(0.0, 2.5, "Welcome to Moniepoint"), (2.5, 5.0, "powering dreams")]
    fake = lambda x, **kw: {"text": "hello there"}  # TEST-ONLY fake pipeline
    out = asr._windowed(fake, np.zeros(16000 * 20, dtype="float32"), 16000, 20.0)
    assert len(out) == 3 and out[-1]["end"] == 20.0 and "coarse_timing" in out[0]["flags"]
