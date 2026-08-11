"""Span/token-level slot extraction for multi-slot memory. Turns one evidence chunk into MANY
candidate K/V slots (capitalized spans, BIBREF, numbers, acronyms, hyphenated terms, + fallback
token windows). Optional oracle mode adds a slot around the gold answer span (diagnostic only)."""
import re

CAP = re.compile(r"[A-Z][A-Za-z0-9.]+(?:[ \-][A-Z][A-Za-z0-9.]+)*")   # Capitalized spans / proper nouns
BIB = re.compile(r"BIBREF\d+")
NUM = re.compile(r"\d[\d,\.]*%?")                                      # numbers / percentages
ACR = re.compile(r"\b[A-Z]{2,}\b")                                    # acronyms
HYP = re.compile(r"\b\w+(?:-\w+)+\b")                                 # hyphenated terms


def _char_to_tok(offs, c0, c1):
    ts = [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0 and b > a]
    return (ts[0], ts[-1] + 1) if ts else None


def extract_slots(text, tokenizer, mode="rule", gold_answer=None, window=12, max_slots=24):
    """Return list of (tok_start, tok_end) spans (aligned to tokenizer(text, add_special_tokens=True))."""
    o = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = o["input_ids"], o["offset_mapping"]
    spans = []
    if mode == "oracle" and gold_answer:
        i = text.lower().find(gold_answer.lower()[:50])
        if i >= 0:
            sp = _char_to_tok(offs, i, i + len(gold_answer))
            if sp: spans.append(sp)
    if mode in ("rule", "oracle"):
        for pat in (CAP, BIB, NUM, ACR, HYP):
            for m in pat.finditer(text):
                sp = _char_to_tok(offs, *m.span())
                if sp: spans.append(sp)
    if mode == "window" or not spans:                                 # fallback fixed windows
        for s in range(1, len(ids), window):
            spans.append((s, min(s + window, len(ids))))
    spans = [s for s in dict.fromkeys(spans) if s[1] > s[0] and s[1] <= len(ids)][:max_slots]
    if not spans:
        spans = [(0, len(ids))]
    return spans
