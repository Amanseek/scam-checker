"""Generate the training and test sets.

    python -m datagen.generate --train 6000 --test 800 --out data

Test messages come only from templates the model never sees in training, so
the test score shows whether it learned the patterns rather than the wording.

Writes:
  data/train.jsonl, data/test.jsonl     one record per message (text, labels, target)
  data/train_chat.jsonl                 the same training data in chat format for fine-tuning
  data/stats.json                       counts per split, verdict, category, language
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datagen.slots import fill_values  # noqa: E402
from datagen.templates_safe import SAFE  # noqa: E402
from datagen.templates_scam import SCAM  # noqa: E402
from schema.labels import CATEGORIES, INSTRUCTION, LANGUAGES, REASON_TEXT, target_json  # noqa: E402

SLOT_RE = re.compile(r"\{(\w+)\}")
LATIN = {"en", "hinglish"}
EMOJIS = ["🙏", "👍", "😊", "⚠️", "‼️", "🔔", "✅", "📢", "🎉", "😀"]

# Share of messages per verdict (genuine messages are the hard negatives).
VERDICT_MIX = {"scam": 0.43, "safe": 0.42, "suspicious": 0.15}


def load_templates() -> list[dict]:
    out = []
    counters: Counter = Counter()
    for category, lang, reasons, text in SCAM + SAFE:
        if category not in CATEGORIES:
            raise ValueError(f"bad category {category}")
        if lang not in LANGUAGES:
            raise ValueError(f"bad language {lang}")
        for r in reasons:
            if r not in REASON_TEXT:
                raise ValueError(f"bad reason {r}")
        key = (category, lang)
        out.append({"template_id": f"{category}-{lang}-{counters[key]}", "category": category, "language": lang,
                    "reasons": reasons, "text": text})
        counters[key] += 1
    return out


def split_templates(templates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out about one template in five per (category, language) for the test set."""
    groups: dict[tuple, list[dict]] = {}
    for t in templates:
        groups.setdefault((t["category"], t["language"]), []).append(t)
    train, test = [], []
    for items in groups.values():
        if len(items) < 3:
            train.extend(items)
            continue
        n_test = max(1, round(len(items) / 5))
        # deterministic: take from the end of each group
        test.extend(items[-n_test:])
        train.extend(items[:-n_test])
    return train, test


def fill(template: str, values: dict[str, str]) -> str:
    def rep(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"unknown slot {{{key}}} in template: {template}")
        return values[key]
    return SLOT_RE.sub(rep, template)


def add_noise(text: str, lang: str, rng: random.Random) -> str:
    """Light, label-independent noise so the model doesn't rely on perfect formatting."""
    if lang in LATIN and rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.1:
        text = text.replace(". ", ".  ")
    if lang in LATIN and rng.random() < 0.15:
        words = text.split(" ")
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 4 and w.isalpha():
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            text = " ".join(words)
    if rng.random() < 0.12:
        text = text + " " + rng.choice(EMOJIS)
    if rng.random() < 0.05:
        text = text.replace(",", "")
    return text


def make_examples(templates: list[dict], n: int, rng: random.Random, split: str) -> list[dict]:
    by_verdict: dict[str, list[dict]] = {}
    for t in templates:
        by_verdict.setdefault(CATEGORIES[t["category"]], []).append(t)
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        verdict = rng.choices(list(VERDICT_MIX), weights=list(VERDICT_MIX.values()))[0]
        pool = by_verdict.get(verdict)
        if not pool:
            continue
        # balance categories inside a verdict, then pick a template
        cats = sorted({t["category"] for t in pool})
        cat = rng.choice(cats)
        t = rng.choice([x for x in pool if x["category"] == cat])
        text = add_noise(fill(t["text"], fill_values(rng)), t["language"], rng)
        if text in seen:
            continue
        seen.add(text)
        out.append({
            "id": f"{split}-{len(out):05d}",
            "split": split,
            "template_id": t["template_id"],
            "language": t["language"],
            "verdict": CATEGORIES[t["category"]],
            "category": t["category"],
            "reason_codes": t["reasons"],
            "text": text,
            "target": target_json(t["category"], t["reasons"], t["language"]),
        })
    return out


def to_chat(example: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": example["target"]},
    ]}


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    templates = load_templates()
    train_t, test_t = split_templates(templates)
    rng = random.Random(args.seed)
    train = make_examples(train_t, args.train, rng, "train")
    test = make_examples(test_t, args.test, rng, "test")

    # No test text may also appear in training.
    train_texts = {r["text"] for r in train}
    test = [r for r in test if r["text"] not in train_texts]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "train_chat.jsonl", (to_chat(r) for r in train))

    stats = {
        "templates": {"total": len(templates), "train": len(train_t), "test": len(test_t)},
        "examples": {"train": len(train), "test": len(test)},
    }
    for name, rows in (("train", train), ("test", test)):
        stats[name] = {
            "verdict": dict(Counter(r["verdict"] for r in rows)),
            "language": dict(Counter(r["language"] for r in rows)),
            "category": dict(sorted(Counter(r["category"] for r in rows).items())),
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats["templates"]), json.dumps(stats["examples"]))


if __name__ == "__main__":
    main()
