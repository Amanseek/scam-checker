"""Start the phone test page on any GPU machine (E2E, Kaggle, your own).

    python -u -m serve.run --adapter /home/jovyan/run/adapters/demo-scam

Prints "Running on public URL: https://....gradio.live" when ready.
"""

from __future__ import annotations

import argparse
import os

# Hosted Jupyter (E2E TIR) sets these for its own proxy. They make the public link's
# "Check" button call a path that doesn't exist ("Not Found"), so drop them first.
PROXY_VARS = ("GRADIO_ROOT_PATH", "GRADIO_SERVER_NAME")


def clean_env(env: dict = os.environ) -> list[str]:
    """Remove proxy settings that break the public link. Returns the names removed."""
    return [name for name in PROXY_VARS if env.pop(name, None) is not None]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True, help="folder with adapter_config.json")
    ap.add_argument("--feedback", default="feedback.jsonl", help="where feedback taps are saved")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    removed = clean_env()
    if removed:
        print("ignoring proxy settings:", ", ".join(removed))

    from serve.app import build_app
    from serve.checker import Checker, load_generate

    app = build_app(Checker(load_generate(args.adapter)), feedback_path=args.feedback)
    app.queue().launch(share=True, root_path="", server_port=args.port)


if __name__ == "__main__":
    main()
