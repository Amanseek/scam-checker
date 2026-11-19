"""Builds serve/serve_kaggle.ipynb: runs the phone web page on Kaggle's GPU with a public link.

    python serve/build_notebook.py
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Scam Checker: phone test page

Runs the trained model on Kaggle's GPU and opens a web page you can use on your phone.

**Before you run**
1. Right panel → **Add Input** → **Your Work** → add the output of **notebook62ba0b2e52**
   (the Tokenwise pipeline run; it contains `adapters/demo-scam`). Nothing is trained here.
2. **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Run All**. After 3–4 minutes the last cell prints a `https://….gradio.live` link: open it on your phone.

The page works only while this session is running, and it uses GPU hours. It switches itself
off after 2 hours; stop it earlier when you're done testing."""),

    (CODE, """%%capture
!pip install -q -U peft bitsandbytes gradio\n!pip uninstall -y -q torchao"""),

    (CODE, """import glob, os, sys
!rm -rf /kaggle/working/scam-checker && git clone -q --depth 1 https://github.com/Amanseek/scam-checker /kaggle/working/scam-checker
sys.path.insert(0, "/kaggle/working/scam-checker")

# Find the trained adapter in the attached input (the pipeline's demo-scam adapter, or an older one).
found = (glob.glob("/kaggle/input/**/adapters/demo-scam/adapter_config.json", recursive=True)
         or glob.glob("/kaggle/input/**/scam_checker_lora/adapter_config.json", recursive=True))
if not found:
    for z in glob.glob("/kaggle/input/**/scam_checker_lora.zip", recursive=True):
        !cd /kaggle/working && unzip -oq "{z}"
    found = glob.glob("/kaggle/working/scam_checker_lora/adapter_config.json")
assert found, "Add the notebook62ba0b2e52 output as an input first (right panel → Add Input)."
ADAPTER = os.path.dirname(found[0])
print("adapter:", ADAPTER)"""),

    (CODE, """from serve.checker import Checker, load_generate
checker = Checker(load_generate(ADAPTER))
for m in ["Dear customer your SBI account will be blocked today, update KYC now http://sbi-kyc-update.xyz",
          "Mummy main ghar 8 baje tak aa jaunga, khana rakh dena"]:
    r = checker.check(m)
    print(r["seconds"], "s |", r["raw"].strip())
    assert r["ok"], "Model answer wasn't readable; don't open the page yet."
"""),

    (CODE, """from serve.app import build_app
app = build_app(checker, feedback_path="/kaggle/working/feedback.jsonl")
import time
app.queue().launch(share=True, debug=False, prevent_thread_lock=True)
# Open the https://....gradio.live link printed above on your phone.
HOURS = 2  # the session ends by itself after this, so it can't use GPU all day
time.sleep(HOURS * 3600)
app.close()"""),
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
    path = Path(__file__).with_name("serve_kaggle.ipynb")
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print("wrote", path)
