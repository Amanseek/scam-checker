# Scam Checker

A small AI model that reads an SMS or WhatsApp message and tells a regular person, in their own language, whether it's likely a scam, and why.

Languages: **English, Hindi, Hinglish (Hindi in Roman letters), Punjabi (Gurmukhi)**.

```
Message:  "Priya grahak, aapka SBI account aaj band ho jayega kyunki KYC pending hai. Turant update karein: https://sbi-kyc.xyz/verify"

Model:    {"verdict": "scam", "category": "kyc_bank",
           "reasons": ["Isme anjaan ya nakli dikhne wala link hai", "Deadline ya dhamki dekar daraya ja raha hai", ...],
           "advice": "Link mat kholo, call mat karo, paise mat bhejo. ... 1930 par complaint karo."}
```

It is a **risk check, not a guarantee**.

## Status

| Step | State |
|---|---|
| 1. Design | Done |
| 2. Label scheme + data generator | **Done** |
| 3. Add licensed public datasets + hand-written real-world test set | After first training run |
| 4. Kaggle fine-tuning notebook (LoRA, Unsloth) | **Ready to run** |
| 5. Evaluation report | Comes out of the notebook |
| 6. WhatsApp bot demo | Later |

## Labels

- **verdict:** `scam`, `suspicious`, `safe`
- **category:**
  - scams: `kyc_bank`, `upi_otp`, `lottery_prize`, `fake_job`, `loan_fee`, `bill_threat`, `authority_threat` (police/CBI/"digital arrest"), `parcel_customs`, `investment`, `family_emergency`
  - suspicious: `unclear`
  - safe: `genuine_otp`, `genuine_alert`, `genuine_service`, `promotion`, `personal`
- **reasons:** up to 3, from a fixed list (`schema/labels.py`), written in the message's language
- **advice:** one line in the message's language

## Data

`datagen/` creates realistic messages from templates with random banks, amounts, links, phone numbers and names, plus light noise (lowercase, typos, emojis).

- **Hard negatives:** about half the data is genuine messages that look scary (real-style OTPs that say "do not share", debit alerts, bills, KYC reminders that point to the branch). This teaches the model the real difference: genuine messages don't ask you to share codes, pay through odd links, or call personal numbers.
- **Held-out templates:** test messages come only from templates never used in training, so the test shows whether the model learned the patterns, not the wording.
- **No big-model outputs** are used as labels, to stay within the Gemini and Claude terms.

```bash
python -m datagen.generate --train 6000 --test 800 --out data
python -m pytest -q
```

Output: `data/train.jsonl`, `data/test.jsonl`, `data/train_chat.jsonl` (chat format for fine-tuning) and `data/stats.json`.

## Train on Kaggle (free GPU)

1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → paste
   `https://github.com/Amanseek/scam-checker/blob/main/train/train_kaggle.ipynb`
2. Right panel → **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Run All** (about 45–75 minutes).
4. From the **Output** panel download `report.json` and `scam_checker_lora.zip`.

The notebook tests the untrained model first (baseline), trains a LoRA adapter, tests the trained model on the held-out test set, and writes the report. Scoring lives in `eval/score.py`:

- **Scams caught:** real scams flagged as scam or suspicious (most important)
- **False alarms:** genuine messages called scam
- verdict and category accuracy, by language; valid-JSON rate

The first run uses `Qwen3-4B-Instruct-2507` (text-only, trains reliably on a free T4). Qwen 3.5 can be tried later by changing `MODEL_NAME`.

## Known limits (to fix in Step 3)

- **Only ~210 templates.** That's enough to test the pipeline, but too few for real-world accuracy. Adding licensed public datasets and more templates is the next step.
- The test set uses 46 held-out templates, so scores will be noisy until the hand-written real-world set exists.
- Punjabi and Hindi templates are written by hand and should be reviewed by native speakers.

## Layout

```
schema/    labels, reasons, advice text, prompt
datagen/   slot values, scam templates, genuine/suspicious templates, generator
data/      generated datasets
eval/      scoring
train/     Kaggle notebook (built by train/build_notebook.py)
tests/     generator and scoring checks
```
