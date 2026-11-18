"""Phone-friendly web page for the scam checker (Gradio).

    from serve.app import build_app
    build_app(checker, feedback_path="feedback.jsonl").launch(share=True)

share=True gives a public https link that opens on any phone.
"""

from __future__ import annotations

import html

import gradio as gr

from serve.checker import Checker, save_feedback

HEADLINE = {
    "scam": {"en": "Scam", "hi": "धोखाधड़ी (स्कैम)", "hinglish": "Scam hai", "pa": "ਧੋਖਾਧੜੀ (ਸਕੈਮ)"},
    "suspicious": {"en": "Be careful", "hi": "सावधान रहें", "hinglish": "Savdhaan raho", "pa": "ਸਾਵਧਾਨ ਰਹੋ"},
    "safe": {"en": "Looks safe", "hi": "सुरक्षित लगता है", "hinglish": "Safe lagta hai", "pa": "ਸੁਰੱਖਿਅਤ ਲੱਗਦਾ ਹੈ"},
}
STYLE = {  # background, border, icon
    "scam": ("#fdecea", "#d93025", "⛔"),
    "suspicious": ("#fff4e5", "#f29900", "⚠️"),
    "safe": ("#e6f4ea", "#188038", "✅"),
}

CSS = """
.gradio-container {max-width: 560px !important; margin: auto;}
#result .card {border-radius: 14px; padding: 16px 18px; border: 2px solid; font-size: 17px; line-height: 1.45;}
#result .card h2 {margin: 0 0 8px 0; font-size: 24px;}
#result .card ul {margin: 6px 0 10px 18px; padding: 0;}
#result .muted {color: #666; font-size: 13px;}
"""

EXAMPLES = [
    "Dear customer, your SBI account will be blocked today due to pending KYC. Update now: http://sbi-kyc-update.xyz",
    "Mummy main ghar 8 baje tak aa jaunga, khana rakh dena",
    "Aapka mahine ka statement taiyar hai. Yahan dekhein: https://tinyurl.com/x8k2p",
    "482913 is your OTP for login to Flipkart. Do not share this code with anyone.",
    "ਤੁਹਾਡਾ ਬਿਜਲੀ ਕੁਨੈਕਸ਼ਨ ਅੱਜ ਰਾਤ ਕੱਟ ਦਿੱਤਾ ਜਾਵੇਗਾ। ਤੁਰੰਤ 9876543210 'ਤੇ ਕਾਲ ਕਰੋ।",
]


def answer_html(res: dict) -> str:
    if not res.get("ok"):
        if not res.get("raw"):
            return "<p class='muted'>Paste a message above and tap Check.</p>"
        return ("<div class='card' style='background:#f1f3f4;border-color:#999'><h2>Couldn't check this one</h2>"
                "Please try again, or check with the official app/website.</div>")
    r = res["result"]
    lang = (res.get("codes") or {}).get("l") or "en"
    bg, border, icon = STYLE[r["verdict"]]
    title = HEADLINE[r["verdict"]].get(lang, HEADLINE[r["verdict"]]["en"])
    reasons = "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"])
    return (f"<div class='card' style='background:{bg};border-color:{border}'>"
            f"<h2>{icon} {html.escape(title)}</h2>"
            f"{'<ul>' + reasons + '</ul>' if reasons else ''}"
            f"<b>{html.escape(r['advice'])}</b></div>"
            f"<p class='muted'>Checked in {res['seconds']} s. This is a risk check, not a guarantee.</p>")


def build_app(checker: Checker, feedback_path: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="Scam Checker", css=CSS) as app:
        gr.Markdown("## 🛡️ Scam Checker\nPaste an SMS or WhatsApp message. English, हिंदी, Hinglish, ਪੰਜਾਬੀ.")
        msg = gr.Textbox(label="Message", lines=5, placeholder="Paste the message here…")
        btn = gr.Button("Check", variant="primary", size="lg")
        out = gr.HTML(answer_html({}), elem_id="result")
        last = gr.State({})

        with gr.Row(visible=feedback_path is not None) as fb_row:
            right = gr.Button("👍 Correct", size="sm")
            was_scam = gr.Button("It was a scam", size="sm")
            was_safe = gr.Button("It was safe", size="sm")
        fb_note = gr.Markdown(visible=False)
        gr.Examples(EXAMPLES, inputs=msg, label="Try an example")
        gr.Markdown("<p class='muted'>Feedback is saved with phone numbers and emails hidden, "
                    "to improve the checker.</p>", visible=feedback_path is not None)

        def run(text):
            res = checker.check(text)
            return answer_html(res), {"text": text, **res}, gr.update(visible=False)

        def feedback(state, says):
            if feedback_path and state.get("ok"):
                save_feedback(feedback_path, state["text"], state, says)
                return gr.update(value="Thanks, saved.", visible=True)
            return gr.update(value="Check a message first.", visible=True)

        btn.click(run, msg, [out, last, fb_note])
        msg.submit(run, msg, [out, last, fb_note])
        right.click(lambda s: feedback(s, "correct"), last, fb_note)
        was_scam.click(lambda s: feedback(s, "scam"), last, fb_note)
        was_safe.click(lambda s: feedback(s, "safe"), last, fb_note)
        _ = fb_row
    return app


if __name__ == "__main__":  # local preview with a fake model: python -m serve.app
    import json

    def fake(text: str) -> str:
        t = text.lower()
        if "http" in t and ("kyc" in t or "block" in t):
            return json.dumps({"v": "scam", "c": "kyc_bank", "l": "en", "r": ["suspicious_link", "urgency_threat"]})
        if "http" in t:
            return json.dumps({"v": "suspicious", "c": "unclear", "l": "hinglish", "r": ["unknown_sender_link"]})
        return json.dumps({"v": "safe", "c": "personal", "l": "hinglish", "r": ["normal_conversation"]})

    build_app(Checker(fake), feedback_path="/tmp/feedback.jsonl").launch(server_name="0.0.0.0", server_port=7860)
