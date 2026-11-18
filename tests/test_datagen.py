import json
import random

from datagen.generate import fill, load_templates, make_examples, split_templates
from datagen.slots import fill_values, indian_number
from schema.labels import CATEGORIES, REASON_TEXT, build_target, render


def test_every_template_fills():
    rng = random.Random(0)
    for t in load_templates():
        for _ in range(5):
            text = fill(t["text"], fill_values(rng))
            assert "{" not in text and "}" not in text, t["template_id"]


def test_split_keeps_templates_apart():
    train, test = split_templates(load_templates())
    assert not {t["template_id"] for t in train} & {t["template_id"] for t in test}
    assert test, "test set needs held-out templates"


def test_every_verdict_and_language_in_test_templates():
    _, test = split_templates(load_templates())
    assert {CATEGORIES[t["category"]] for t in test} == {"scam", "safe", "suspicious"}
    assert {t["language"] for t in test} == {"en", "hi", "hinglish", "pa"}


def test_examples_have_valid_targets():
    train, _ = split_templates(load_templates())
    rows = make_examples(train, 300, random.Random(1), "train")
    assert len(rows) == 300
    for r in rows:
        target = json.loads(r["target"])
        assert target["v"] == CATEGORIES[r["category"]] and target["l"] == r["language"]
        assert 1 <= len(target["r"]) <= 3
        full = render(target)
        assert full["advice"] and len(full["reasons"]) == len(target["r"])


def test_reasons_translated_for_all_languages():
    for code, texts in REASON_TEXT.items():
        assert set(texts) == {"en", "hi", "hinglish", "pa"}, code


def test_target_language_matches():
    t = build_target("kyc_bank", ["suspicious_link"], "pa")
    assert "ਲਿੰਕ" in t["reasons"][0]


def test_indian_number_grouping():
    assert indian_number(2500000) == "25,00,000"
    assert indian_number(15000) == "15,000"
    assert indian_number(999) == "999"


def test_render_is_forgiving():
    full = render({"v": "scam", "c": "genuine_otp", "l": "xx", "r": ["asks_otp", "made_up"]})
    assert full["category"] is None            # category didn't match the verdict
    assert full["reasons"] == [REASON_TEXT["asks_otp"]["en"]]
    assert "1930" in full["advice"]
