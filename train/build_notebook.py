"""Builds train/train_kaggle.ipynb from the cells below.

    python train/build_notebook.py
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Scam Checker: fine-tune a small model on Kaggle

Trains a LoRA adapter so a small open model can tell whether an SMS/WhatsApp message is a scam
(English, Hindi, Hinglish, Punjabi), then tests it on held-out messages.

**Before you run**
1. Right panel → **Session options** → **Accelerator: GPU T4 x2** (not P100).
2. **Internet: On** (needed to download the model and the data).
3. Click **Run All**. It takes about 45–75 minutes. You can leave the page; keep the session running.

**When it's done**, open the **Output** panel (right side) and download:
- `report.json`: the results (send this to Claude)
- `scam_checker_lora.zip`: the trained adapter"""),

    (MD, "## 1. Install"),
    (CODE, """%%capture
!pip install -q unsloth"""),

    (MD, "## 2. Settings"),
    (CODE, """# Text-only instruct model that trains reliably on a free T4. Swap later to compare others.
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
MAX_SEQ_LEN = 2048
LORA_RANK = 16
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
BASELINE_N = 200      # test messages for the untrained model (to compare)
EVAL_N = 800          # test messages for the trained model
GEN_BATCH = 16
MAX_NEW_TOKENS = 512   # adjusted automatically after the tokenizer loads
REPO = "https://github.com/Amanseek/scam-checker"
OUT = "/kaggle/working" """),

    (MD, "## 3. Get the data"),
    (CODE, """import json, os, sys, time, random
!rm -rf /kaggle/working/scam-checker && git clone -q --depth 1 {REPO} /kaggle/working/scam-checker
sys.path.insert(0, "/kaggle/working/scam-checker")
from schema.labels import INSTRUCTION
from eval.score import score, summary_lines, load_jsonl

train_chat = load_jsonl("/kaggle/working/scam-checker/data/train_chat.jsonl")
test_rows = load_jsonl("/kaggle/working/scam-checker/data/test.jsonl")
print(len(train_chat), "training messages,", len(test_rows), "test messages")
print(test_rows[0]["text"])"""),

    (MD, "## 4. Load the model"),
    (CODE, """import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# Hindi and Punjabi text uses many more tokens than English; size the answer budget from the data.
from collections import defaultdict
longest = defaultdict(int)
for r in load_jsonl("/kaggle/working/scam-checker/data/train.jsonl"):
    longest[r["language"]] = max(longest[r["language"]], len(tokenizer(r["target"])["input_ids"]))
MAX_NEW_TOKENS = max(longest.values()) + 32
print("longest answer in tokens by language:", dict(longest), "-> MAX_NEW_TOKENS =", MAX_NEW_TOKENS)"""),

    (MD, "## 5. Helper: run the model on test messages"),
    (CODE, """def predict(rows, batch_size=GEN_BATCH):
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": r["text"]}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for j in range(len(batch)):
            outputs.append(tokenizer.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(rows)}")
    return {r["id"]: o for r, o in zip(rows, outputs)}

random.seed(0)
baseline_rows = random.sample(test_rows, min(BASELINE_N, len(test_rows)))"""),

    (MD, "## 6. Test the untrained model (baseline)"),
    (CODE, """t0 = time.time()
baseline_preds = predict(baseline_rows)
baseline_report = score(baseline_rows, baseline_preds)
print("\\n".join(summary_lines(baseline_report)))
print(f"took {time.time() - t0:.0f}s")
print("\\nExample output:\\n", list(baseline_preds.values())[0][:400])"""),

    (MD, "## 7. Train"),
    (CODE, """from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=7,
)
tokenizer.padding_side = "right"
texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in train_chat]
dataset = Dataset.from_dict({"text": texts}).shuffle(seed=7)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE, warmup_ratio=0.03,
        lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.01,
        logging_steps=10, seed=7, output_dir=f"{OUT}/checkpoints", report_to="none", save_strategy="no",
    ),
)
# Learn only from the answer, not from repeating the question.
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
                                  response_part="<|im_start|>assistant\\n")
t0 = time.time()
stats = trainer.train()
train_minutes = round((time.time() - t0) / 60, 1)
print("training took", train_minutes, "minutes; final loss", round(stats.training_loss, 4))"""),

    (MD, "## 8. Test the trained model"),
    (CODE, """eval_rows = test_rows[:EVAL_N]
t0 = time.time()
trained_preds = predict(eval_rows)
eval_seconds = time.time() - t0
trained_report = score(eval_rows, trained_preds)
print("\\n".join(summary_lines(trained_report)))
print(f"\\n{eval_seconds / len(eval_rows):.2f} s per message (batched)")"""),

    (MD, "## 9. Save the adapter and the report"),
    (CODE, """model.save_pretrained(f"{OUT}/scam_checker_lora")
tokenizer.save_pretrained(f"{OUT}/scam_checker_lora")
!cd {OUT} && zip -qr scam_checker_lora.zip scam_checker_lora

with open(f"{OUT}/predictions.jsonl", "w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps({"id": r["id"], "output": trained_preds[r["id"]]}, ensure_ascii=False) + "\\n")

report = {
    "model": MODEL_NAME, "lora_rank": LORA_RANK, "epochs": EPOCHS, "train_messages": len(train_chat),
    "train_minutes": train_minutes, "final_loss": round(stats.training_loss, 4),
    "gpu": torch.cuda.get_device_name(0),
    "seconds_per_message_batched": round(eval_seconds / len(eval_rows), 3),
    "baseline_untrained": {k: v for k, v in baseline_report.items() if k != "sample_errors"},
    "trained": trained_report,
}
with open(f"{OUT}/report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("Saved report.json, predictions.jsonl and scam_checker_lora.zip in the Output panel")"""),

    (MD, "## 10. Try your own message"),
    (CODE, """def check(message):
    row = {"id": "mine", "text": message}
    print(predict([row])["mine"])

check("Dear customer your SBI account will be blocked today, update KYC now http://sbi-kyc-update.xyz")
check("Mummy main ghar 8 baje tak aa jaunga, khana rakh dena")"""),
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
    path = Path(__file__).with_name("train_kaggle.ipynb")
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False))
    print("wrote", path)
