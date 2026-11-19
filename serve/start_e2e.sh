#!/usr/bin/env bash
# Start the scam checker page on an E2E TIR container, after each Start.
#
#   cd /home/jovyan/scam-checker && git pull -q && bash serve/start_e2e.sh
#
# Safe to run again: it stops the old page first and skips work that's already done.
# Needs your Kaggle token in ~/.kaggle/access_token only the first time (to download the model).
set -euo pipefail

WS=/home/jovyan                     # the only folder E2E keeps between Stop and Start
ADAPTER=$WS/run/adapters/demo-scam
KAGGLE_RUN=${KAGGLE_RUN:-amangupta786/notebook62ba0b2e52}
LOG=$WS/page.log
export HF_HOME=$WS/hf               # keep the downloaded base model in the workspace
cd "$WS/scam-checker"

echo "== 1/4 tools"
pip install -q -U kaggle transformers peft bitsandbytes gradio 2>&1 | grep -v -i -E "warning|dependency|requires|incompatible" || true

echo "== 2/4 GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! python - <<'EOF'
import sys, torch
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
ok = arch in torch.cuda.get_arch_list()
print(f"torch {torch.__version__}, GPU {arch}:", "supported" if ok else "NOT supported by this torch")
sys.exit(0 if ok else 1)
EOF
then
  # E2E's PyTorch image is NVIDIA's build without older GPUs (V100). Install the standard one.
  echo "installing standard PyTorch for this GPU (2-4 min)..."
  PIP_CONSTRAINT="" pip install -q "torch==2.8.0+cu126" --extra-index-url https://download.pytorch.org/whl/cu126 2>&1 | grep -v -i -E "warning|dependency|requires|incompatible" || true
  PIP_CONSTRAINT="" pip uninstall -y -q torchvision torchaudio || true
  python -c "import torch; torch.randn(2, device='cuda'); print('GPU works with torch', torch.__version__)"
fi

echo "== 3/4 model"
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "downloading the trained adapter from Kaggle ($KAGGLE_RUN)..."
  kaggle kernels output "$KAGGLE_RUN" -p "$WS/run" > /dev/null
fi
ls "$ADAPTER/adapter_config.json" > /dev/null && echo "adapter: $ADAPTER"

echo "== 4/4 page"
pkill -f "serve.run|serve.app" 2>/dev/null && sleep 3 || true
nohup python -u -m serve.run --adapter "$ADAPTER" --feedback "$WS/feedback.jsonl" > "$LOG" 2>&1 &
echo "starting (log: $LOG)..."
for _ in $(seq 120); do                      # wait up to 10 minutes
  if url=$(grep -o "https://[a-z0-9]*\.gradio\.live" "$LOG"); then
    echo
    echo "READY. Open on your phone: $url"
    echo "Stop the page: pkill -f serve.run   (then Stop the container in TIR)"
    exit 0
  fi
  if ! pgrep -f serve.run > /dev/null; then
    echo "The page stopped with an error. Last lines of the log:"; tail -25 "$LOG"; exit 1
  fi
  sleep 5
done
echo "No link after 10 minutes. Last lines of the log:"; tail -25 "$LOG"; exit 1
