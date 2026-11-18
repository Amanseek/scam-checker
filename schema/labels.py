"""Label scheme for the scam checker.

The model reads one message and returns short codes only (fast, and the same
for every language):

    {"v": "scam", "c": "kyc_bank", "l": "hinglish", "r": ["suspicious_link", "urgency_threat"]}

    v: verdict ("scam" | "suspicious" | "safe")
    c: category (one of CATEGORIES)
    l: language of the message (en, hi = Devanagari, hinglish = Hindi in Roman letters, pa = Gurmukhi)
    r: 1-3 reason codes (keys of REASON_TEXT)

render() turns those codes into the full answer a person sees, in their language:

    {"verdict": ..., "category": ..., "reasons": ["short reason", ...], "advice": "one line"}
"""

from __future__ import annotations

import json

VERDICTS = ["scam", "suspicious", "safe"]
LANGUAGES = ["en", "hi", "hinglish", "pa"]

# category -> verdict it belongs to
CATEGORIES = {
    # scams
    "kyc_bank": "scam",            # fake KYC expiry, account block, card block
    "upi_otp": "scam",             # asks for OTP/PIN, "collect request", refund tricks
    "lottery_prize": "scam",       # KBC lottery, lucky draw, cashback prize
    "fake_job": "scam",            # part-time job, task/like jobs, registration fee
    "loan_fee": "scam",            # instant loan with upfront fee
    "bill_threat": "scam",         # electricity/gas cut tonight, call officer
    "authority_threat": "scam",    # police/CBI/customs "digital arrest", court case
    "parcel_customs": "scam",      # parcel held, pay customs/redelivery fee
    "investment": "scam",          # guaranteed returns, crypto/stock tips groups
    "family_emergency": "scam",    # "mummy new number", relative in trouble, send money
    # unclear
    "unclear": "suspicious",       # unknown link or vague request without a clear ask
    # genuine
    "genuine_otp": "safe",         # real OTP that says do not share
    "genuine_alert": "safe",       # debit/credit alerts, balance, statements
    "genuine_service": "safe",     # delivery, bookings, appointments, bills, govt info
    "promotion": "safe",           # ordinary ads/offers from known brands
    "personal": "safe",            # normal chat between people
}

REASON_TEXT = {
    "asks_otp": {
        "en": "Asks you to share an OTP",
        "hi": "आपसे OTP शेयर करने को कहा गया है",
        "hinglish": "Aapse OTP share karne ko bola gaya hai",
        "pa": "ਤੁਹਾਡੇ ਤੋਂ OTP ਮੰਗਿਆ ਜਾ ਰਿਹਾ ਹੈ",
    },
    "asks_pin_password": {
        "en": "Asks for your PIN, password or card details",
        "hi": "आपका PIN, पासवर्ड या कार्ड की जानकारी मांगी गई है",
        "hinglish": "Aapka PIN, password ya card details maange gaye hain",
        "pa": "ਤੁਹਾਡਾ PIN, ਪਾਸਵਰਡ ਜਾਂ ਕਾਰਡ ਦੀ ਜਾਣਕਾਰੀ ਮੰਗੀ ਗਈ ਹੈ",
    },
    "personal_documents": {
        "en": "Asks for Aadhaar, PAN or other personal documents",
        "hi": "आधार, PAN या दूसरे निजी दस्तावेज़ मांगे गए हैं",
        "hinglish": "Aadhaar, PAN ya doosre personal documents maange gaye hain",
        "pa": "ਆਧਾਰ, PAN ਜਾਂ ਹੋਰ ਨਿੱਜੀ ਦਸਤਾਵੇਜ਼ ਮੰਗੇ ਗਏ ਹਨ",
    },
    "suspicious_link": {
        "en": "Contains an unofficial or look-alike link",
        "hi": "इसमें अनजान या नकली दिखने वाला लिंक है",
        "hinglish": "Isme anjaan ya nakli dikhne wala link hai",
        "pa": "ਇਸ ਵਿੱਚ ਅਣਜਾਣ ਜਾਂ ਨਕਲੀ ਲੱਗਣ ਵਾਲਾ ਲਿੰਕ ਹੈ",
    },
    "urgency_threat": {
        "en": "Creates panic with a deadline or threat",
        "hi": "समय-सीमा या धमकी देकर डराया जा रहा है",
        "hinglish": "Deadline ya dhamki dekar daraya ja raha hai",
        "pa": "ਸਮਾਂ-ਸੀਮਾ ਜਾਂ ਧਮਕੀ ਦੇ ਕੇ ਡਰਾਇਆ ਜਾ ਰਿਹਾ ਹੈ",
    },
    "too_good_to_be_true": {
        "en": "Promises a prize or money that sounds too good to be true",
        "hi": "इनाम या पैसे का ऐसा वादा जो सच होना मुश्किल है",
        "hinglish": "Inaam ya paise ka aisa vaada jo sach hona mushkil hai",
        "pa": "ਇਨਾਮ ਜਾਂ ਪੈਸੇ ਦਾ ਅਜਿਹਾ ਵਾਅਦਾ ਜੋ ਸੱਚ ਹੋਣਾ ਔਖਾ ਹੈ",
    },
    "upfront_fee": {
        "en": "Asks you to pay a fee first",
        "hi": "पहले फीस भरने को कहा गया है",
        "hinglish": "Pehle fees bharne ko bola gaya hai",
        "pa": "ਪਹਿਲਾਂ ਫੀਸ ਭਰਨ ਲਈ ਕਿਹਾ ਗਿਆ ਹੈ",
    },
    "impersonation": {
        "en": "Pretends to be a bank, police or government office",
        "hi": "बैंक, पुलिस या सरकारी दफ़्तर होने का दिखावा",
        "hinglish": "Bank, police ya sarkari office hone ka dikhava",
        "pa": "ਬੈਂਕ, ਪੁਲਿਸ ਜਾਂ ਸਰਕਾਰੀ ਦਫ਼ਤਰ ਹੋਣ ਦਾ ਦਿਖਾਵਾ",
    },
    "call_unknown_number": {
        "en": "Asks you to call or message a personal number",
        "hi": "किसी निजी नंबर पर कॉल या मैसेज करने को कहा गया है",
        "hinglish": "Kisi personal number par call ya message karne ko bola gaya hai",
        "pa": "ਕਿਸੇ ਨਿੱਜੀ ਨੰਬਰ 'ਤੇ ਕਾਲ ਜਾਂ ਮੈਸੇਜ ਕਰਨ ਲਈ ਕਿਹਾ ਗਿਆ ਹੈ",
    },
    "send_money": {
        "en": "Asks you to send money urgently",
        "hi": "तुरंत पैसे भेजने को कहा गया है",
        "hinglish": "Turant paise bhejne ko bola gaya hai",
        "pa": "ਤੁਰੰਤ ਪੈਸੇ ਭੇਜਣ ਲਈ ਕਿਹਾ ਗਿਆ ਹੈ",
    },
    "guaranteed_returns": {
        "en": "Promises guaranteed or very high returns",
        "hi": "पक्का या बहुत ज़्यादा मुनाफ़ा देने का वादा",
        "hinglish": "Pakka ya bahut zyada munafa dene ka vaada",
        "pa": "ਪੱਕਾ ਜਾਂ ਬਹੁਤ ਵੱਧ ਮੁਨਾਫ਼ਾ ਦੇਣ ਦਾ ਵਾਅਦਾ",
    },
    "unknown_sender_link": {
        "en": "Link from an unknown sender with no clear reason",
        "hi": "अनजान भेजने वाले का लिंक, कोई साफ़ वजह नहीं",
        "hinglish": "Anjaan bhejne wale ka link, koi saaf wajah nahi",
        "pa": "ਅਣਜਾਣ ਭੇਜਣ ਵਾਲੇ ਦਾ ਲਿੰਕ, ਕੋਈ ਸਾਫ਼ ਵਜ੍ਹਾ ਨਹੀਂ",
    },
    "says_do_not_share": {
        "en": "Tells you not to share the code with anyone",
        "hi": "कोड किसी से शेयर न करने को कहा गया है",
        "hinglish": "Code kisi se share na karne ko bola gaya hai",
        "pa": "ਕੋਡ ਕਿਸੇ ਨਾਲ ਸਾਂਝਾ ਨਾ ਕਰਨ ਲਈ ਕਿਹਾ ਗਿਆ ਹੈ",
    },
    "no_request": {
        "en": "Only gives information; asks for nothing",
        "hi": "सिर्फ़ जानकारी दी गई है; कुछ मांगा नहीं गया",
        "hinglish": "Sirf jaankari di gayi hai; kuch maanga nahi gaya",
        "pa": "ਸਿਰਫ਼ ਜਾਣਕਾਰੀ ਦਿੱਤੀ ਗਈ ਹੈ; ਕੁਝ ਮੰਗਿਆ ਨਹੀਂ ਗਿਆ",
    },
    "official_channel": {
        "en": "Points you to the official app, website or branch",
        "hi": "आधिकारिक ऐप, वेबसाइट या ब्रांच पर जाने को कहा गया है",
        "hinglish": "Official app, website ya branch par jaane ko bola gaya hai",
        "pa": "ਅਧਿਕਾਰਤ ਐਪ, ਵੈੱਬਸਾਈਟ ਜਾਂ ਬ੍ਰਾਂਚ 'ਤੇ ਜਾਣ ਲਈ ਕਿਹਾ ਗਿਆ ਹੈ",
    },
    "normal_conversation": {
        "en": "Looks like a normal personal conversation",
        "hi": "यह सामान्य बातचीत लगती है",
        "hinglish": "Yeh normal baatcheet lagti hai",
        "pa": "ਇਹ ਆਮ ਗੱਲਬਾਤ ਲੱਗਦੀ ਹੈ",
    },
}

ADVICE_TEXT = {
    "scam": {
        "en": "Don't click links, call back or pay. Banks and police never ask for OTP or money like this. Report at cybercrime.gov.in or call 1930.",
        "hi": "लिंक न खोलें, कॉल न करें, पैसे न भेजें। बैंक या पुलिस कभी ऐसे OTP या पैसे नहीं मांगते। cybercrime.gov.in पर या 1930 पर शिकायत करें।",
        "hinglish": "Link mat kholo, call mat karo, paise mat bhejo. Bank ya police kabhi aise OTP ya paise nahi maangte. cybercrime.gov.in par ya 1930 par complaint karo.",
        "pa": "ਲਿੰਕ ਨਾ ਖੋਲ੍ਹੋ, ਕਾਲ ਨਾ ਕਰੋ, ਪੈਸੇ ਨਾ ਭੇਜੋ। ਬੈਂਕ ਜਾਂ ਪੁਲਿਸ ਕਦੇ ਇਸ ਤਰ੍ਹਾਂ OTP ਜਾਂ ਪੈਸੇ ਨਹੀਂ ਮੰਗਦੇ। cybercrime.gov.in 'ਤੇ ਜਾਂ 1930 'ਤੇ ਸ਼ਿਕਾਇਤ ਕਰੋ।",
    },
    "family_emergency": {
        "en": "Call the person on their old, known number before sending any money.",
        "hi": "पैसे भेजने से पहले उस व्यक्ति को उनके पुराने, जाने-पहचाने नंबर पर कॉल करें।",
        "hinglish": "Paise bhejne se pehle us insaan ko unke purane, jaane-pehchaane number par call karo.",
        "pa": "ਪੈਸੇ ਭੇਜਣ ਤੋਂ ਪਹਿਲਾਂ ਉਸ ਵਿਅਕਤੀ ਨੂੰ ਉਨ੍ਹਾਂ ਦੇ ਪੁਰਾਣੇ, ਜਾਣੇ-ਪਛਾਣੇ ਨੰਬਰ 'ਤੇ ਕਾਲ ਕਰੋ।",
    },
    "suspicious": {
        "en": "Be careful. Don't open the link; check with the sender or the official app/website first.",
        "hi": "सावधान रहें। लिंक न खोलें; पहले भेजने वाले से या आधिकारिक ऐप/वेबसाइट से पुष्टि करें।",
        "hinglish": "Savdhaan raho. Link mat kholo; pehle bhejne wale se ya official app/website se confirm karo.",
        "pa": "ਸਾਵਧਾਨ ਰਹੋ। ਲਿੰਕ ਨਾ ਖੋਲ੍ਹੋ; ਪਹਿਲਾਂ ਭੇਜਣ ਵਾਲੇ ਜਾਂ ਅਧਿਕਾਰਤ ਐਪ/ਵੈੱਬਸਾਈਟ ਤੋਂ ਪੁਸ਼ਟੀ ਕਰੋ।",
    },
    "genuine_otp": {
        "en": "Looks genuine. Use the code only if you started this yourself, and never share it with anyone.",
        "hi": "असली लगता है। कोड तभी इस्तेमाल करें जब यह काम आपने खुद शुरू किया हो, और किसी को न बताएं।",
        "hinglish": "Asli lagta hai. Code tabhi use karo jab yeh kaam aapne khud shuru kiya ho, aur kisi ko mat batao.",
        "pa": "ਅਸਲੀ ਲੱਗਦਾ ਹੈ। ਕੋਡ ਤਾਂ ਹੀ ਵਰਤੋ ਜੇ ਇਹ ਕੰਮ ਤੁਸੀਂ ਖੁਦ ਸ਼ੁਰੂ ਕੀਤਾ ਹੋਵੇ, ਅਤੇ ਕਿਸੇ ਨੂੰ ਨਾ ਦੱਸੋ।",
    },
    "safe": {
        "en": "Looks genuine. If something doesn't match, check in the official app instead of replying.",
        "hi": "असली लगता है। कुछ गड़बड़ लगे तो जवाब देने के बजाय आधिकारिक ऐप में जांच करें।",
        "hinglish": "Asli lagta hai. Kuch gadbad lage to reply karne ke bajaye official app me check karo.",
        "pa": "ਅਸਲੀ ਲੱਗਦਾ ਹੈ। ਕੁਝ ਗੜਬੜ ਲੱਗੇ ਤਾਂ ਜਵਾਬ ਦੇਣ ਦੀ ਬਜਾਏ ਅਧਿਕਾਰਤ ਐਪ ਵਿੱਚ ਜਾਂਚ ਕਰੋ।",
    },
}


def advice_key(category: str) -> str:
    verdict = CATEGORIES[category]
    if category in ADVICE_TEXT:
        return category
    return verdict


def build_target(category: str, reason_codes: list[str], language: str) -> dict:
    """The full answer a person sees for a message (reasons and advice in their language)."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category}")
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language}")
    return {
        "verdict": CATEGORIES[category],
        "category": category,
        "reasons": [REASON_TEXT[c][language] for c in reason_codes[:3]],
        "advice": ADVICE_TEXT[advice_key(category)][language],
    }


def compact_target(category: str, reason_codes: list[str], language: str) -> dict:
    """The short codes the model is trained to produce."""
    build_target(category, reason_codes, language)  # validates
    return {"v": CATEGORIES[category], "c": category, "l": language, "r": list(reason_codes[:3])}


def target_json(category: str, reason_codes: list[str], language: str) -> str:
    return json.dumps(compact_target(category, reason_codes, language), ensure_ascii=False)


def render(codes: dict) -> dict:
    """Turn the model's short codes into the full answer. Unknown codes are dropped;
    an unknown language falls back to English; a category that doesn't match the
    verdict is ignored so the advice always fits the verdict."""
    verdict = codes.get("v") or codes.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"bad verdict {verdict!r}")
    language = codes.get("l") if codes.get("l") in LANGUAGES else "en"
    category = codes.get("c") or codes.get("category")
    if CATEGORIES.get(category) != verdict:
        category = None
    reasons = [REASON_TEXT[c][language] for c in (codes.get("r") or [])[:3] if c in REASON_TEXT]
    advice = ADVICE_TEXT[advice_key(category) if category else verdict][language]
    return {"verdict": verdict, "category": category, "reasons": reasons, "advice": advice}


INSTRUCTION = (
    "You check whether an SMS or WhatsApp message is a scam. Reply with one line of JSON only: "
    '{"v": verdict, "c": category, "l": language, "r": [1-3 reason codes]}. '
    "v: scam, suspicious or safe. "
    "c: " + ", ".join(CATEGORIES) + ". "
    "l: en, hi (Devanagari), hinglish (Hindi in Roman letters) or pa (Gurmukhi). "
    "r: " + ", ".join(REASON_TEXT) + "."
)
