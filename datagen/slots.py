"""Random values for template slots like {bank}, {amount}, {bad_url}."""

from __future__ import annotations

import random
import string

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Kotak Bank", "Bank of Baroda",
         "Canara Bank", "Punjab & Sind Bank", "Union Bank", "IndusInd Bank", "Yes Bank"]
BANK_DOMAINS = {"SBI": "sbi", "HDFC Bank": "hdfcbank", "ICICI Bank": "icicibank", "Axis Bank": "axisbank",
                "PNB": "pnb", "Kotak Bank": "kotak", "Bank of Baroda": "bankofbaroda", "Canara Bank": "canarabank",
                "Punjab & Sind Bank": "psb", "Union Bank": "unionbank", "IndusInd Bank": "indusind",
                "Yes Bank": "yesbank"}
NAMES = ["Rahul", "Priya", "Gurpreet", "Simran", "Amit", "Neha", "Harjeet", "Manpreet", "Rohit", "Anjali",
         "Vikram", "Pooja", "Jaspreet", "Karan", "Sunita", "Ravi", "Navdeep", "Komal", "Arjun", "Meena"]
SONS = ["Rahul", "Gurpreet", "Amit", "Harjeet", "Rohit", "Vikram", "Karan", "Ravi", "Navdeep", "Arjun", "Manpreet"]
COMPANIES = ["Amazon", "Flipkart", "Myntra", "Meesho", "Nykaa", "Ajio", "Swiggy", "Zomato", "BigBasket", "Blinkit"]
TELCOS = ["Jio", "Airtel", "Vi", "BSNL"]
COURIERS = ["India Post", "Blue Dart", "DTDC", "Delhivery", "Ekart", "FedEx"]
CITIES = ["Pathankot", "Amritsar", "Jalandhar", "Ludhiana", "Patiala", "Chandigarh", "Delhi", "Mumbai",
          "Pune", "Jaipur", "Lucknow", "Bengaluru"]
POWER_COS = ["PSPCL", "BSES", "MSEDCL", "Tata Power", "UHBVN", "DHBVN", "UPPCL", "TPDDL"]
MERCHANTS = ["AMAZON", "SWIGGY", "ZOMATO", "BIGBASKET", "IRCTC", "NETFLIX", "FLIPKART", "RELIANCE SMART",
             "DMART", "INDIAN OIL", "APOLLO PHARMACY", "BOOKMYSHOW"]
PRODUCTS = ["mobile phone", "shoes", "kurta", "headphones", "mixer grinder", "watch", "saree", "school bag"]
APPS = ["YONO", "iMobile", "HDFC MobileBanking", "PhonePe", "Google Pay", "Paytm", "BHIM"]
SCAM_TLDS = ["xyz", "top", "online", "site", "in.net", "co", "live", "click", "info", "buzz"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "rb.gy", "shorturl.at"]
GOV_SCHEMES = ["PM Kisan", "Ayushman Bharat", "Ujjwala Yojana", "PM Awas Yojana"]
OFFICIAL_SITES = {"irctc": "irctc.co.in", "aadhaar": "myaadhaar.uidai.gov.in", "incometax": "incometax.gov.in",
                  "epfo": "epfindia.gov.in", "pspcl": "pspcl.in"}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n))


def indian_number(n: int) -> str:
    """12,34,567 style grouping."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def money(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    if n > 1000:
        n = round(n, -2) if rng.random() < 0.6 else n
    style = rng.random()
    if style < 0.35:
        return f"Rs {indian_number(n)}"
    if style < 0.6:
        return f"₹{indian_number(n)}"
    if style < 0.75:
        return f"Rs.{n}.00"
    if style < 0.88:
        return f"INR {indian_number(n)}.00"
    return f"Rs{n}"


def big_money(rng: random.Random) -> str:
    lakhs = rng.choice([2, 5, 10, 12, 15, 25, 50])
    style = rng.random()
    if style < 0.4:
        return f"₹{indian_number(lakhs * 100000)}"
    if style < 0.7:
        return f"Rs {lakhs} lakh"
    return f"Rs.{indian_number(lakhs * 100000)}/-"


def phone(rng: random.Random) -> str:
    num = rng.choice("6789") + _digits(rng, 9)
    style = rng.random()
    if style < 0.4:
        return num
    if style < 0.7:
        return f"+91 {num[:5]} {num[5:]}"
    return f"+91{num}"


def bad_url(rng: random.Random, brand: str = "") -> str:
    b = brand.lower().replace(" bank", "").replace(" ", "").replace("&", "")
    words = ["kyc", "update", "verify", "secure", "reward", "help", "login", "support", "refund", "pay"]
    style = rng.random()
    if style < 0.3:
        return f"https://{rng.choice(SHORTENERS)}/{_alnum(rng, 6)}"
    if style < 0.6 and b:
        return f"https://{b}-{rng.choice(words)}.{rng.choice(SCAM_TLDS)}/{rng.choice(words)}"
    if style < 0.8 and b:
        return f"{rng.choice(words)}-{b}{rng.randint(1, 99)}.{rng.choice(SCAM_TLDS)}"
    return f"http://{rng.choice(words)}{rng.choice(words)}{rng.randint(10, 999)}.{rng.choice(SCAM_TLDS)}"


def good_url(rng: random.Random, brand: str) -> str:
    d = BANK_DOMAINS.get(brand)
    if d:
        return rng.choice([f"https://www.{d}.com", f"{d}.com", f"https://www.{d}.co.in"]) if d != "sbi" \
            else rng.choice(["https://onlinesbi.sbi", "sbi.co.in"])
    return f"https://www.{brand.lower().replace(' ', '')}.in"


def company_url(rng: random.Random, company: str) -> str:
    if company == "Amazon":
        return rng.choice(["amazon.in", "amzn.in/d/" + _alnum(rng, 6)])
    return f"{company.lower()}.com"


def fill_values(rng: random.Random) -> dict[str, str]:
    bank = rng.choice(BANKS)
    company = rng.choice(COMPANIES)
    power = rng.choice(POWER_COS)
    day = rng.randint(1, 28)
    month = rng.choice(["Nov", "Dec", "Jan"])
    return {
        "bank": bank,
        "bank_short": bank.replace(" Bank", "").upper(),
        "bad_url": bad_url(rng, rng.choice([bank, company, power, "kyc", "rewards"])),
        "bank_bad_url": bad_url(rng, bank),
        "good_url": good_url(rng, bank),
        "company": company,
        "company_url": company_url(rng, company),
        "company_bad_url": bad_url(rng, company),
        "telco": rng.choice(TELCOS),
        "courier": rng.choice(COURIERS),
        "city": rng.choice(CITIES),
        "power": power,
        "power_bad_url": bad_url(rng, power),
        "merchant": rng.choice(MERCHANTS),
        "product": rng.choice(PRODUCTS),
        "app": rng.choice(APPS),
        "scheme": rng.choice(GOV_SCHEMES),
        "name": rng.choice(NAMES),
        "name2": rng.choice(NAMES),
        "son": rng.choice(SONS),
        "amount": money(rng, 120, 48000),
        "small_amount": money(rng, 10, 499),
        "fee": money(rng, 99, 4999),
        "big_amount": big_money(rng),
        "loan_amount": rng.choice(["₹2,00,000", "Rs 5 lakh", "₹50,000", "Rs 1,00,000", "₹10 lakh", "Rs 3 lakh"]),
        "daily_pay": rng.choice(["₹3,000-8,000 daily", "Rs 500 per task", "₹2,000 per day", "Rs 15,000 weekly",
                                 "₹800-1,500 per hour"]),
        "returns": rng.choice(["30% monthly", "3x in 15 days", "10% daily", "200% in one month",
                               "double in 7 days"]),
        "balance": money(rng, 500, 250000),
        "phone": phone(rng),
        "tollfree": rng.choice(["1800 1234", "1800 425 3800", "1800-209-4324", "1800 103 4499", "1800 266 3310",
                                "1800 1080", "1800-180-1290"]),
        "otp": _digits(rng, rng.choice([4, 6, 6, 6])),
        "acct": "XX" + _digits(rng, 4),
        "card": "XX" + _digits(rng, 4),
        "ref": _digits(rng, 12),
        "order_id": rng.choice(["OD", "#", "ORD"]) + _digits(rng, 10),
        "parcel_id": rng.choice(["EZ", "RN", "CP", "IN"]) + _digits(rng, 9) + "IN",
        "consumer_no": _digits(rng, 10),
        "date": f"{day:02d}-{ {'Nov': 11, 'Dec': 12, 'Jan': 1}[month]:02d}-{2026 if month != 'Jan' else 2027}",
        "short_date": f"{day} {month}",
        "time": rng.choice(["9:30 PM", "10:00 PM", "8 PM", "11:30 AM", "6:45 PM", "2:15 PM"]),
        "hours": str(rng.choice([2, 6, 12, 24])),
        "minutes": str(rng.choice([10, 15, 30])),
        "pnr": _digits(rng, 10),
        "train": str(rng.randint(12001, 22999)),
        "seat": f"{rng.choice(['S', 'B', 'A'])}{rng.randint(1, 9)}/{rng.randint(1, 72)}",
    }
