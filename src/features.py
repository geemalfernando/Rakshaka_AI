import re

TOKEN = re.compile(r"\w+", re.UNICODE)

def tokens(text: str):
    return [t.lower() for t in TOKEN.findall(str(text)) if not t.isdigit()]

def unk_rate(text: str, vocab: set) -> float:
    toks = tokens(text)
    if not toks:
        return 0.0
    return sum(t not in vocab for t in toks) / len(toks)
