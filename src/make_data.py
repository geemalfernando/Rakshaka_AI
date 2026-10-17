"""Generate synthetic Sri Lankan scam / legit messages.

All names, numbers and links are FAKE placeholders. No real personal data.
Outputs: data/train.csv, data/test.csv (held-out), data/shifted.csv (new tactics for drift demo).
Columns: text, label (scam | legitimate), language, group_id.
Extend: add hand-written or LLM rows to data/extra_train.csv (columns: text,label[,language]).
data/extra_test.csv is NOT merged here: it is an independent final test set, scored separately by evaluate.py.
"""
import csv, pathlib, random, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rakshaka.data import load_labelled_messages

SEED = 42
random.seed(SEED)
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
FIELDS = ["text", "label", "language", "group_id"]

BANKS = ["DemoBank", "SampleBank LK", "XYZ Bank", "TestPay"]  # clearly synthetic
AMOUNTS = ["Rs. 5,000", "Rs. 25,000", "Rs. 150,000", "Rs. 2,500", "Rs. 1,000,000"]
LINKS = ["short.example/xx-demo", "secure-verify.example", "lk-refund.example", "tinyurl.example/abc"]
PHONE = ["07X-XXXXXXX", "011-XXXXXXX"]
NAMES = ["Nimal", "Kumari", "Suresh", "Fathima", "Ravi", "Dilini", "Kamal", "Priya"]  # fictional first names
TIMES = ["7.30am", "10am", "2pm", "4.30pm", "6pm", "8pm"]

def f(t):
    return t.format(bank=random.choice(BANKS), amt=random.choice(AMOUNTS),
                    link=random.choice(LINKS), phone=random.choice(PHONE),
                    name=random.choice(NAMES), time=random.choice(TIMES),
                    otp=random.randint(100000, 999999), n=random.randint(2, 9))

# ---------- In-distribution SCAM templates: (template, language) ----------
SCAM = [
    ("URGENT: Your {bank} account is blocked. Verify now at {link} or it will be closed in 24 hours.", "english"),
    ("Dear customer, your {bank} card is suspended. Send the OTP you receive to {phone} to reactivate.", "english"),
    ("Congratulations! You won {amt} in the lucky draw. Pay Rs. 2,000 processing fee to claim. Call {phone}", "english"),
    ("Your parcel is held at customs. Pay {amt} clearance fee at {link} within today.", "english"),
    ("Part time job! Earn {amt} per day from home. Just like YouTube videos. WhatsApp {phone}", "english"),
    ("Invest {amt} and get double in {n} days. 100% guaranteed profit. Join now {link}", "english"),
    ("Aswesuma payment {amt} pending. Update your bank details at {link} to receive.", "english"),
    ("Machan oyage {bank} account eka block wela. Me link eken verify karanna {link}", "singlish"),
    ("Oyata {amt} lottery ekak dinala. Fee eka gewala claim karanna {phone} ta call karanna", "singlish"),
    ("Ane OTP eka mata ewanna, waradila oyage number ekata awa. Please quickly", "singlish"),
    ("ඔබගේ {bank} ගිණුම අත්හිටුවා ඇත. වහාම {link} හරහා තහවුරු කරන්න.", "sinhala"),
    ("ඔබ {amt} ත්‍යාගයක් දිනා ඇත. ගාස්තුව ගෙවා ලබා ගන්න {phone}", "sinhala"),
    ("உங்கள் {bank} கணக்கு முடக்கப்பட்டுள்ளது. உடனே {link} இல் சரிபார்க்கவும்.", "tamil"),
    ("நீங்கள் {amt} பரிசு வென்றுள்ளீர்கள். கட்டணம் செலுத்தி பெறுங்கள் {phone}", "tamil"),
    ("Your electricity bill is overdue. Pay {amt} now at {link} or power will be disconnected tonight.", "english"),
    ("This is {bank} security team. Tell us the {n}-digit PIN to stop a fraud transaction.", "english"),
    ("Your SIM will be deactivated today due to incomplete KYC. Update your details at {link}", "english"),
    ("{bank} alert: unusual login detected. Confirm your identity at {link} to avoid suspension.", "english"),
    ("You have a pending tax refund of {amt}. Submit your card details at {link}", "english"),
    ("Your loan of {amt} is approved. Pay Rs. 3,000 insurance fee to {phone} to release the funds.", "english"),
    ("Dear pensioner, your pension increase of {amt} needs re-verification. Call {phone} with your PIN.", "english"),
    ("Your number won {amt} in the anniversary draw. Reply with your account number to claim.", "english"),
    ("Oyage pension eka nawaththanna yanne. Details update karanna {link} eken", "singlish"),
    ("Ayye mata podi udawwak ona, {amt} ekak dala denna puluwanda? Heta denawa. {phone}", "singlish"),
    ("Oyage parcel eka customs eke hira wela. {amt} gewala release karaganna {link}", "singlish"),
    ("Free data {n}GB ganna me link eka click karanna {link} ikmanata", "singlish"),
    ("ඔබගේ විශ්‍රාම වැටුප අත්හිටුවීමට නියමිතයි. වහාම {phone} අමතන්න.", "sinhala"),
    ("ඔබට {amt} ණය මුදලක් අනුමත වී ඇත. ගාස්තුව ගෙවා ලබා ගන්න {link}", "sinhala"),
    ("ඔබගේ පාර්සලය රේගුවේ රඳවා ඇත. {amt} ගෙවන්න {link}", "sinhala"),
    ("ඔබගේ OTP අංකය {phone} වෙත එවන්න, නැතහොත් ගිණුම අවහිර වේ.", "sinhala"),
    ("உங்கள் ஓய்வூதியம் நிறுத்தப்படும். உடனே {phone} அழைக்கவும்.", "tamil"),
    ("உங்களுக்கு {amt} கடன் அங்கீகரிக்கப்பட்டது. கட்டணம் செலுத்துங்கள் {link}", "tamil"),
    ("உங்கள் பார்சல் சுங்கத்தில் உள்ளது. {amt} செலுத்துங்கள் {link}", "tamil"),
    ("உங்கள் OTP எண்ணை {phone} க்கு அனுப்பவும், இல்லையெனில் கணக்கு முடக்கப்படும்.", "tamil"),
]

# ---------- In-distribution LEGIT templates ----------
HAM = [
    ("Your {bank} OTP is {otp}. Do not share this code with anyone, including bank staff.", "english"),
    ("Rs. 2,500 debited from your {bank} account for a purchase. Call the number on your card if this was not you.", "english"),
    ("Hi, are we still meeting at 5 today?", "english"),
    ("Amma, I'll reach home by 7. Keep dinner please.", "english"),
    ("Your electricity bill for this month is {amt}. Pay via the official app before the due date.", "english"),
    ("Class is cancelled tomorrow. Assignment deadline extended to Monday.", "english"),
    ("Machan match eka kiyatada? Mama 6ta ennam.", "singlish"),
    ("Ada office eke meeting eka 3ta. Late wenna epa.", "singlish"),
    ("අද රෑ කෑමට එන්න පුළුවන්ද?", "sinhala"),
    ("හෙට පන්තිය අවලංගුයි.", "sinhala"),
    ("நாளை சந்திப்போம். மாலை 6 மணிக்கு.", "tamil"),
    ("உங்கள் ஆர்டர் நாளை வழங்கப்படும். நன்றி.", "tamil"),
    ("Your parcel has been delivered. Thank you for shopping with us.", "english"),
    ("Happy birthday! Have a great day", "english"),
    ("Reminder: your dental appointment is on {n} Nov at 10am.", "english"),
    ("Salary of {amt} credited to your {bank} account.", "english"),
    ("Hi {name}, the bus leaves at {time}. See you at the stop.", "english"),
    ("Your {bank} statement for this month is ready in the official app.", "english"),
    ("{name}, can you pick up bread on your way home?", "english"),
    ("Your appointment at the Divisional Secretariat is confirmed for {time} on Monday.", "english"),
    ("Payment of {amt} received for your water bill. Thank you.", "english"),
    ("Pension of {amt} credited to your {bank} account. For queries visit your branch.", "english"),
    ("{name} oya enawada ada? Mama {time}ta enawa.", "singlish"),
    ("Amma heta gedara enawa, kema hadanna epa. {name} ath enawa.", "singlish"),
    ("Bus eka late, mama {time} witara wenakota enawa.", "singlish"),
    ("Machan assignment eka email kala, {name}ta check karanna kiyanna.", "singlish"),
    ("{name}, හෙට {time}ට හමුවෙමු.", "sinhala"),
    ("ඔබගේ {bank} ගිණුමට {amt} වැටුප බැර කර ඇත.", "sinhala"),
    ("අම්මා, මම {time}ට ගෙදර එනවා.", "sinhala"),
    ("ඔබගේ ජල බිල්පත {amt}. නිල යෙදුම හරහා ගෙවන්න.", "sinhala"),
    ("{name}, நாளை {time} க்கு வருகிறேன்.", "tamil"),
    ("உங்கள் {bank} கணக்கில் {amt} சம்பளம் வரவு வைக்கப்பட்டது.", "tamil"),
    ("அம்மா, நான் {time} க்கு வீட்டுக்கு வருகிறேன்.", "tamil"),
    ("வகுப்பு நாளை ரத்து செய்யப்பட்டது. {name} க்கு சொல்லுங்கள்.", "tamil"),
]

# ---------- SHIFTED: new tactics (SL-UDI digital ID, QR, voice note, crypto, heavy code-mix) ----------
SHIFT_SCAM = [
    ("SL-UDI notice: your new digital ID is not activated. Scan the QR at {link} and enter your details.", "english"),
    ("Digital ID registration closing. Biometric re-verification fee Rs. 500. Pay at {link}", "english"),
    ("Voice note eka ahanna, oyage digital ID eka cancel wenawa. {link} eken re-register karanna", "singlish"),
    ("Bro USDT trading group ekata join wenna, daily 8% return. Admin {phone}", "singlish"),
    ("SL-UDI eka hadanna kalin me form eka fill karanna {link} nathnam benefits nathi wenawa", "singlish"),
    ("ඩිජිටල් හැඳුනුම්පත සක්‍රීය කිරීමට QR එක ස්කෑන් කරන්න {link}", "sinhala"),
    ("டிஜிட்டல் அடையாள அட்டை செயல்படுத்த QR ஸ்கேன் செய்யவும் {link}", "tamil"),
    ("Hey it's me, new number. Lost my phone. Can you send {amt} via eZ cash, will return tomorrow", "english"),
    ("AI investment bot gives passive income. Deposit {amt} in USDT. Limited slots {link}", "english"),
    ("Govt subsidy for digital ID holders: claim {amt} by sharing your NIC photo at {link}", "english"),
]
SHIFT_HAM = [
    ("Digital ID registration centre opens at the Divisional Secretariat on Monday. Bring your old NIC.", "english"),
    ("Machan voice note eka ahuwada? Trip eka gana.", "singlish"),
    ("Your new SIM is activated. Visit an official outlet for any help.", "english"),
    ("ඩිජිටල් හැඳුනුම්පත ගැන දැනුම්දීම: නිල කාර්යාලයට පමණක් යන්න.", "sinhala"),
    ("Meeting moved to Zoom, link shared in the official team group.", "english"),
]

def noisy(t):
    r = random.random()
    if r < 0.15: t = t.lower()
    elif r < 0.25: t = t.upper()
    if random.random() < 0.1: t += " !!"
    return t

def gen(templates, label, prefix, per_template):
    """Each template is one group: all its variants land in the same split (no template leakage)."""
    rows = []
    for i, (template, language) in enumerate(templates):
        texts = list(dict.fromkeys(noisy(f(template)) for _ in range(per_template)))
        rows += [{"text": t, "label": label, "language": language, "group_id": f"{prefix}-{i}"} for t in texts]
    return rows

def split_templates(rows, test_share=0.2, val_share=0.15):
    """Hold out whole templates per (label, language): test and val contain unseen wording.

    val = drift-monitoring baseline (what normal unseen traffic looks like); test = reported metrics.
    """
    buckets = defaultdict(set)
    for r in rows:
        buckets[(r["label"], r["language"])].add(r["group_id"])
    test_groups, val_groups = set(), set()
    for key in sorted(buckets):
        groups = sorted(buckets[key])
        random.shuffle(groups)
        n_test = max(1, round(len(groups) * test_share)) if len(groups) >= 2 else 0
        n_val = max(1, round(len(groups) * val_share)) if len(groups) - n_test >= 2 else 0
        test_groups.update(groups[:n_test])
        val_groups.update(groups[n_test:n_test + n_val])
    pick = lambda groups: [r for r in rows if r["group_id"] in groups]
    train = [r for r in rows if r["group_id"] not in test_groups | val_groups]
    return train, pick(val_groups), pick(test_groups)

def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows, {len({r['group_id'] for r in rows})} templates/groups")

if __name__ == "__main__":
    allrows = gen(SCAM, "scam", "scam", 70) + gen(HAM, "legitimate", "legit", 70)
    train, val, test = split_templates(allrows)
    shifted = gen(SHIFT_SCAM, "scam", "shift-scam", 30) + gen(SHIFT_HAM, "legitimate", "shift-legit", 30)
    extra = OUT / "extra_train.csv"   # hand-written / LLM training rows
    if extra.exists():
        train.extend(load_labelled_messages(extra))
    for rows in (train, val, test, shifted): random.shuffle(rows)
    for name, rows in (("train", train), ("val", val), ("test", test), ("shifted", shifted)):
        write(OUT / f"{name}.csv", rows)
