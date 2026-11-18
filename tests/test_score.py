import json

from eval.score import parse_output, score

ROWS = [
    {"id": "a", "verdict": "scam", "category": "kyc_bank", "language": "en", "text": "x"},
    {"id": "b", "verdict": "scam", "category": "upi_otp", "language": "hi", "text": "x"},
    {"id": "c", "verdict": "safe", "category": "genuine_otp", "language": "en", "text": "x"},
    {"id": "d", "verdict": "safe", "category": "personal", "language": "pa", "text": "x"},
]


def out(verdict, category="kyc_bank"):
    return json.dumps({"verdict": verdict, "category": category, "reasons": [], "advice": ""})


def test_parse_plain_and_wrapped():
    assert parse_output(out("scam"))["verdict"] == "scam"
    assert parse_output("Sure!\n```json\n" + out("safe") + "\n```")["verdict"] == "safe"
    assert parse_output("<think>hmm {x}</think>" + out("suspicious"))["verdict"] == "suspicious"
    assert parse_output(out("scam") + " extra text }")["verdict"] == "scam"


def test_parse_rejects_junk():
    assert parse_output("") is None
    assert parse_output("This looks like a scam.") is None
    assert parse_output('{"verdict": "maybe"}') is None
    assert parse_output('{"verdict": "scam",') is None


def test_score_numbers():
    preds = {"a": out("scam"), "b": out("suspicious", "upi_otp"), "c": out("scam", "genuine_otp"), "d": "nonsense"}
    r = score(ROWS, preds)
    assert r["messages_scored"] == 4
    assert r["valid_json"] == 0.75
    assert r["scams_caught"] == 1.0          # scam + suspicious both count as caught
    assert r["scams_called_scam"] == 0.5
    assert r["false_alarms"] == 0.5          # genuine OTP called scam
    assert r["verdict_accuracy"] == 0.25
    assert r["confusion"]["safe"]["invalid"] == 1


def test_missing_predictions_are_skipped():
    r = score(ROWS, {"a": out("scam")})
    assert r["messages_scored"] == 1 and r["verdict_accuracy"] == 1.0
