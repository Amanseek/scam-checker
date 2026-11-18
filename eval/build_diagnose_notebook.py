"""Builds eval/diagnose_kaggle.ipynb: load a trained adapter WITHOUT unsloth (plain transformers + peft)
and check whether its answers are readable. Tells apart "bad training" from "bad inference tool".

    python eval/build_diagnose_notebook.py
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Scam Checker: adapter check (no unsloth)

1. **Add Input** → Your Work → **scam-checker training** output (contains `scam_checker_lora`).
2. GPU T4, Internet on, run all. About 5 minutes.

Step A checks the saved weights for broken numbers. Step B runs the adapter with plain
transformers + peft. Step C runs the base model alone with the same code, for comparison."""),

    (CODE, """%%capture
!pip install -q -U peft bitsandbytes"""),

    (CODE, """import glob, json, os, sys, torch
!rm -rf /kaggle/working/scam-checker && git clone -q --depth 1 https://github.com/Amanseek/scam-checker /kaggle/working/scam-checker
sys.path.insert(0, "/kaggle/working/scam-checker")
from schema.labels import INSTRUCTION
from eval.score import parse_output, load_jsonl

found = glob.glob("/kaggle/input/**/scam_checker_lora/adapter_config.json", recursive=True)
assert found, "Add the 'scam-checker training' output as an input first."
ADAPTER = os.path.dirname(found[0])
cfg = json.load(open(found[0]))
BASE = cfg["base_model_name_or_path"]
print("adapter:", ADAPTER, "| base:", BASE, "| r:", cfg.get("r"), "alpha:", cfg.get("lora_alpha"))
print(sorted(os.listdir(ADAPTER)))
test_rows = load_jsonl("/kaggle/working/scam-checker/data/test.jsonl")[:16]"""),

    (MD, "## A. Are the saved weights healthy?"),
    (CODE, """from safetensors.torch import load_file
w = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
bad = [k for k, t in w.items() if not torch.isfinite(t.float()).all()]
biggest = max(t.float().abs().max().item() for t in w.values())
print(len(w), "tensors |", len(bad), "with NaN/inf | largest value", round(biggest, 4))
print("examples:", list(w)[:3])"""),

    (MD, "## B. Adapter with plain transformers + peft"),
    (CODE, """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, torch_dtype=torch.float16)
BLOCK = [[i] for i in tok.convert_tokens_to_ids(["<tool_call>", "</tool_call>", "<think>", "</think>"])
         if isinstance(i, int) and i != tok.unk_token_id]

def run(model, rows):
    model.eval()
    outs = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "system", "content": INSTRUCTION},
                                     {"role": "user", "content": r["text"]}],
                                    tokenize=False, add_generation_prompt=True)
        enc = tok(p, return_tensors="pt").to(0)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=70, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id, bad_words_ids=BLOCK or None)
        outs.append(tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    ok = sum(parse_output(o) is not None for o in outs)
    right = sum((parse_output(o) or {}).get("verdict") == r["verdict"] for o, r in zip(outs, rows))
    for o, r in list(zip(outs, rows))[:4]:
        print(r["verdict"], "->", repr(o[:150]))
    print(f"readable {ok}/{len(rows)} | correct verdict {right}/{len(rows)}")
    return ok

lora = PeftModel.from_pretrained(base, ADAPTER)
ok_b = run(lora, test_rows)"""),

    (MD, "## C. Base model alone (same code)"),
    (CODE, """with lora.disable_adapter():
    ok_c = run(lora, test_rows[:6])
print()
print("VERDICT:", "adapter is FINE; the unsloth inference path is the problem" if ok_b >= 12
      else "adapter itself is BROKEN; retrain with pinned versions")"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    path = Path(__file__).with_name("diagnose_kaggle.ipynb")
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print("wrote", path)
