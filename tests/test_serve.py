import json

from serve.checker import Checker, mask_personal, save_feedback


def fake(out):
    return lambda text: out


def test_check_renders_full_answer():
    c = Checker(fake('</tool_call>{"v": "scam", "c": "upi_otp", "l": "hi", "r": ["asks_otp"]}'))
    res = c.check("कोई संदेश")
    assert res["ok"] and res["result"]["verdict"] == "scam"
    assert "OTP" in res["result"]["reasons"][0] and "1930" in res["result"]["advice"]


def test_check_handles_garbage_and_empty():
    assert not Checker(fake("GameObjectWithTag spNet")).check("hello")["ok"]
    assert Checker(fake("x")).check("   ")["raw"] == ""


def test_mask_personal():
    s = mask_personal("Call 98765 43210 or mail a.b@x.com, OTP 482913, a/c XX1234")
    assert "98765" not in s and "482913" not in s and "a.b@x.com" not in s
    assert "##########" in s and "[email]" in s


def test_feedback_saved_masked(tmp_path):
    p = tmp_path / "fb.jsonl"
    res = Checker(fake('{"v": "safe", "c": "personal", "l": "en", "r": []}')).check("hi")
    save_feedback(p, "my number 9876543210", res, "scam")
    rec = json.loads(p.read_text())
    assert rec["user_says"] == "scam" and "9876543210" not in rec["text"] and rec["model"]["v"] == "safe"


def test_app_builds():
    from serve.app import answer_html, build_app
    c = Checker(fake('{"v": "suspicious", "c": "unclear", "l": "pa", "r": ["unknown_sender_link"]}'))
    assert build_app(c, feedback_path=None) is not None
    assert "ਸਾਵਧਾਨ" in answer_html(c.check("ਸੁਨੇਹਾ"))


def test_run_drops_proxy_settings():
    from serve.run import clean_env

    env = {"GRADIO_ROOT_PATH": "/gradio/notebook/p-1/n-2", "GRADIO_SERVER_NAME": "0.0.0.0", "HOME": "/x"}
    assert sorted(clean_env(env)) == ["GRADIO_ROOT_PATH", "GRADIO_SERVER_NAME"]
    assert env == {"HOME": "/x"}
    assert clean_env(env) == []
