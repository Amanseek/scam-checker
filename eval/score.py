"""Score model predictions against the test set.

    python eval/score.py data/test.jsonl predictions.jsonl [--out report.json]

predictions.jsonl: one line per test message, {"id": ..., "output": "<raw model text>"}.

Key numbers:
  scams_caught       share of real scams the model called "scam" or "suspicious"
  scams_called_scam  share of real scams called exactly "scam"
  false_alarms       share of genuine messages called "scam"
  genuine_flagged    share of genuine messages called "scam" or "suspicious"
  verdict_accuracy, category_accuracy, valid_json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ("scam", "suspicious", "safe")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> dict | None:
    """Pull the first JSON object out of model text; None if there isn't a usable one."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    m = JSON_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    for end in range(len(candidate), 0, -1):  # tolerate trailing junk after the object
        if candidate[end - 1] != "}":
            continue
        try:
            obj = json.loads(candidate[:end])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("verdict") in VERDICTS:
            return obj
        return None
    return None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(test_rows: list[dict], predictions: dict[str, str]) -> dict:
    total = valid = verdict_ok = category_ok = 0
    scam_n = scam_caught = scam_exact = 0
    safe_n = safe_as_scam = safe_flagged = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors = []

    for row in test_rows:
        if row["id"] not in predictions:
            continue
        total += 1
        pred = parse_output(predictions[row["id"]])
        true_v = row["verdict"]
        pred_v = pred["verdict"] if pred else "invalid"
        if pred:
            valid += 1
        confusion[true_v][pred_v] += 1
        ok = pred_v == true_v
        verdict_ok += ok
        category_ok += bool(pred and pred.get("category") == row["category"])
        by_lang[row["language"]][0] += ok
        by_lang[row["language"]][1] += 1
        by_cat[row["category"]][0] += ok
        by_cat[row["category"]][1] += 1
        if true_v == "scam":
            scam_n += 1
            scam_caught += pred_v in ("scam", "suspicious")
            scam_exact += pred_v == "scam"
        if true_v == "safe":
            safe_n += 1
            safe_as_scam += pred_v == "scam"
            safe_flagged += pred_v in ("scam", "suspicious")
        if not ok and len(errors) < 40:
            errors.append({"id": row["id"], "language": row["language"], "true": f"{true_v}/{row['category']}",
                           "pred": pred_v if not pred else f"{pred_v}/{pred.get('category')}",
                           "text": row["text"][:200]})

    return {
        "messages_scored": total,
        "valid_json": _rate(valid, total),
        "verdict_accuracy": _rate(verdict_ok, total),
        "category_accuracy": _rate(category_ok, total),
        "scams_caught": _rate(scam_caught, scam_n),
        "scams_called_scam": _rate(scam_exact, scam_n),
        "false_alarms": _rate(safe_as_scam, safe_n),
        "genuine_flagged": _rate(safe_flagged, safe_n),
        "verdict_accuracy_by_language": {k: _rate(*v) for k, v in sorted(by_lang.items())},
        "verdict_accuracy_by_category": {k: _rate(*v) for k, v in sorted(by_cat.items())},
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "sample_errors": errors,
    }


def summary_lines(report: dict) -> list[str]:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    return [
        f"Messages scored:      {report['messages_scored']}",
        f"Scams caught:         {pct(report['scams_caught'])}  (called 'scam': {pct(report['scams_called_scam'])})",
        f"False alarms:         {pct(report['false_alarms'])}  (genuine flagged at all: {pct(report['genuine_flagged'])})",
        f"Verdict accuracy:     {pct(report['verdict_accuracy'])}",
        f"Category accuracy:    {pct(report['category_accuracy'])}",
        f"Valid JSON:           {pct(report['valid_json'])}",
        "By language:          " + ", ".join(f"{k} {pct(v)}" for k, v in report["verdict_accuracy_by_language"].items()),
    ]


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("predictions")
    ap.add_argument("--out")
    args = ap.parse_args()
    preds = {r["id"]: r["output"] for r in load_jsonl(args.predictions)}
    report = score(load_jsonl(args.test), preds)
    print("\n".join(summary_lines(report)))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
