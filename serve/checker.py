"""Run the trained scam checker on one message and turn its codes into a full answer.

    from serve.checker import Checker, load_generate
    checker = Checker(load_generate("/path/to/scam_checker_lora"))
    checker.check("Dear customer your SBI account will be blocked ...")

The model part (load_generate) needs a GPU with unsloth; Checker itself only needs
a function text -> raw model output, so it can be tested with a fake.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.score import parse_output  # noqa: E402
from schema.labels import INSTRUCTION, render  # noqa: E402

MAX_CHARS = 1500  # longer messages are cut; scams are short

DIGITS = re.compile(r"\d[\d\s-]{3,}\d")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def mask_personal(text: str) -> str:
    """Hide numbers (phones, accounts, OTPs) and emails before anything is stored."""
    return DIGITS.sub(lambda m: "#" * len(re.sub(r"\D", "", m.group(0))), EMAIL.sub("[email]", text))


class Checker:
    def __init__(self, generate: Callable[[str], str]):
        self.generate = generate

    def check(self, message: str) -> dict:
        """Returns {"ok", "result" (full answer or None), "raw", "seconds"}."""
        message = (message or "").strip()[:MAX_CHARS]
        if not message:
            return {"ok": False, "result": None, "raw": "", "seconds": 0.0}
        t0 = time.time()
        raw = self.generate(message)
        seconds = round(time.time() - t0, 2)
        parsed = parse_output(raw)
        if not parsed:
            return {"ok": False, "result": None, "raw": raw, "seconds": seconds}
        codes = {"v": parsed["verdict"], "c": parsed.get("category"), "l": parsed.get("language"),
                 "r": parsed.get("reasons")}
        return {"ok": True, "result": render(codes), "codes": codes, "raw": raw, "seconds": seconds}


def save_feedback(path: str | Path, message: str, answer: dict, verdict_from_user: str) -> None:
    """Append one feedback record (numbers and emails masked) for the retraining loop."""
    record = {"time": int(time.time()), "text": mask_personal(message),
              "model": answer.get("codes"), "user_says": verdict_from_user}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_generate(adapter_dir: str, max_new_tokens: int = 70) -> Callable[[str], str]:
    """Load the base model + LoRA adapter with unsloth and return text -> raw output."""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=2048, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)
    blocked = [tokenizer.convert_tokens_to_ids(t) for t in ("<tool_call>", "</tool_call>", "<think>", "</think>")]
    blocked = [[i] for i in blocked if isinstance(i, int) and i != tokenizer.unk_token_id] or None

    def generate(text: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                 bad_words_ids=blocked)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate
