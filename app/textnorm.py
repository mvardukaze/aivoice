"""ქართული ტექსტის მომზადება გახმოვანებამდე (ყველა ხმისთვის საერთო).

normalize(): ლექსიკონი → თარიღი/დრო → ფული/პროცენტი/ერთეულები → რიგობითი → რიცხვები → შემოკლებები →
ლათინური სიტყვები ქართული ასოებით. ka_text() = normalize() + „ს+ღ“ წესი.

„ს+ღ“ (და სხვა ყრუ + ღ): ცოცხალ მეტყველებაში „ღ“ ყრუვდება და „ხ“-ს უახლოვდება — მოდელმაც ეს ისწავლა
(„ოდესღაც“ → „ოდესხაც“). დეფისი espeak-ში „ღ“-ს მარცვალს მახვილს აძლევს (ˈodesɣˈatsʰ) და
„ღ“ მკაფიოდ ისმის — მომხმარებლის მოსმენით დადასტურებული (2026-10-12).

ლექსიკონი: data/lexicon.default.json (git) + data/lexicon.json (ამ კომპიუტერისა, პანელიდან) — {"სიტყვა": "როგორ წავიკითხო"}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON_DEFAULT = ROOT / "data" / "lexicon.default.json"
LEXICON = ROOT / "data" / "lexicon.json"

_VOICELESS_GH = re.compile(r"([სშცჩხქფთტპკწჭყ])ღ")

# ---------- რიცხვები (ოცობითი სისტემა) ----------
UNITS = ["ნული", "ერთი", "ორი", "სამი", "ოთხი", "ხუთი", "ექვსი", "შვიდი", "რვა", "ცხრა", "ათი",
         "თერთმეტი", "თორმეტი", "ცამეტი", "თოთხმეტი", "თხუთმეტი", "თექვსმეტი", "ჩვიდმეტი", "თვრამეტი", "ცხრამეტი"]
TWENTIES = {1: "ოც", 2: "ორმოც", 3: "სამოც", 4: "ოთხმოც"}
HUNDREDS = {1: "ას", 2: "ორას", 3: "სამას", 4: "ოთხას", 5: "ხუთას", 6: "ექვსას", 7: "შვიდას", 8: "რვაას", 9: "ცხრაას"}
BIG = [(10 ** 12, "ტრილიონ"), (10 ** 9, "მილიარდ"), (10 ** 6, "მილიონ"), (1000, "ათას")]
MONTHS = ["იანვარი", "თებერვალი", "მარტი", "აპრილი", "მაისი", "ივნისი", "ივლისი", "აგვისტო",
          "სექტემბერი", "ოქტომბერი", "ნოემბერი", "დეკემბერი"]


def _stem(w: str) -> str:
    """ბოლო ხმოვნის გარეშე: სამი → სამ, რვა → რვ (რიგობითისთვის)."""
    return w[:-1] if w[-1] in "იაე" else w


def cardinal(n: int) -> str:
    if n < 0:
        return "მინუს " + cardinal(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, r = divmod(n, 20)
        return TWENTIES[t] + ("ი" if r == 0 else "და" + UNITS[r])
    if n < 1000:
        h, r = divmod(n, 100)
        return HUNDREDS[h] + ("ი" if r == 0 else " " + cardinal(r))
    for size, word in BIG:
        if n >= size:
            q, r = divmod(n, size)
            head = word if (q == 1 and size <= 10 ** 3) else f"{cardinal(q)} {word}"
            return head + ("ი" if r == 0 else " " + cardinal(r))
    raise ValueError(n)


def ordinal(n: int) -> str:
    """რიგობითი: 1 → პირველი, 2 → მეორე, 5 → მეხუთე, 21 → ოცდამეერთე, 100 → მეასე, 101 → ას პირველი."""
    if n == 1:
        return "პირველი"
    if n == 2:
        return "მეორე"
    if n < 20:
        return "მე" + _stem(UNITS[n]) + "ე"
    if n < 100:
        t, r = divmod(n, 20)
        if r == 0:
            return "მე" + TWENTIES[t] + "ე"
        return TWENTIES[t] + "და" + ({1: "მეერთე", 2: "მეორე"}.get(r) or ordinal(r))
    head, _, last = _split_last(n)
    if head:
        return f"{head} {ordinal(last)}"
    c = cardinal(n).rsplit(" ", 1)  # მრგვალი: ასი → მეასე, ათასი → მეათასე, ორი ათასი → ორი მეათასე
    return (c[0] + " " if len(c) > 1 else "") + "მე" + c[-1][:-1] + "ე"


def _split_last(n: int):
    """2024 → ("ორი ათას", "", 24): უკანასკნელი ნაწილი რიგობითად იქცევა."""
    for size in (10 ** 12, 10 ** 9, 10 ** 6, 1000, 100):
        if n >= size and n % size:
            r = n % size
            head = cardinal(n - r)
            return (head[:-1] if head.endswith("ი") else head), "", r
    return "", "", n


def oblique(words: str) -> str:
    """მიცემითი/ვითარებითი ბრუნვის წინ რიცხვი ბოლო „ი“-ს კარგავს: ხუთი → ხუთ (წელს), ოცი → ოც (ლარად)."""
    return words[:-1] if words.endswith("ი") else words


# ---------- ლექსიკონი ----------
_lex_cache: dict = {"key": None, "re": None, "map": {}}


def lexicon() -> dict:
    out = {}
    for p in (LEXICON_DEFAULT, LEXICON):
        try:
            out.update(json.loads(p.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return {k: v for k, v in out.items() if k and not k.startswith("_")}


def _lex_re():
    key = tuple(p.stat().st_mtime if p.exists() else 0 for p in (LEXICON_DEFAULT, LEXICON))
    if _lex_cache["key"] != key:
        m = lexicon()
        words = sorted(m, key=len, reverse=True)
        _lex_cache.update(key=key, map={w.lower(): v for w, v in m.items()},
                          re=re.compile(r"(?<![\wა-ჰ])(" + "|".join(map(re.escape, words)) + r")(?![\wა-ჰ])", re.I) if words else None)
    return _lex_cache["re"], _lex_cache["map"]


# ---------- ლათინური → ქართული (ლექსიკონში თუ არ არის) ----------
_LAT2 = [("sch", "შ"), ("tch", "ჩ"), ("sh", "შ"), ("ch", "ჩ"), ("zh", "ჟ"), ("kh", "ხ"), ("gh", "ღ"), ("ts", "ც"),
         ("th", "თ"), ("ph", "ფ"), ("ck", "კ"), ("qu", "კვ"), ("ee", "ი"), ("oo", "უ"), ("ou", "აუ"), ("ea", "ი"),
         ("ai", "ეი"), ("ay", "ეი"), ("dj", "ჯ")]
_LAT1 = dict(zip("abcdefghijklmnopqrstuvwxyz", ["ა", "ბ", "კ", "დ", "ე", "ფ", "გ", "ჰ", "ი", "ჯ", "კ", "ლ", "მ", "ნ", "ო", "პ",
                                                "ქ", "რ", "ს", "ტ", "უ", "ვ", "ვ", "ქს", "ი", "ზ"]))
_LETTERS = {"a": "ეი", "b": "ბი", "c": "სი", "d": "დი", "e": "ი", "f": "ეფ", "g": "ჯი", "h": "ეიჩ", "i": "აი", "j": "ჯეი",
            "k": "კეი", "l": "ელ", "m": "ემ", "n": "ენ", "o": "ოუ", "p": "პი", "q": "ქიუ", "r": "არ", "s": "ეს", "t": "ტი",
            "u": "იუ", "v": "ვი", "w": "დაბლ-იუ", "x": "ექს", "y": "უაი", "z": "ზედ"}


def latin_word(w: str) -> str:
    if w.isupper() and len(w) <= 4:  # აბრევიატურა: AI → ეი-აი, USB → იუ-ეს-ბი
        return "-".join(_LETTERS[c] for c in w.lower())
    s, out, i = w.lower(), [], 0
    while i < len(s):
        for a, b in _LAT2:
            if s.startswith(a, i):
                out.append(b)
                i += len(a)
                break
        else:
            c = s[i]
            out.append("ს" if c == "c" and s[i + 1:i + 2] in ("e", "i", "y") else _LAT1.get(c, ""))
            i += 1
    return re.sub(r"(.)\1+", r"\1", "".join(out))  # hello → ჰელო


# ---------- შემოკლებები ----------
ABBR = [
    (r"და ა\.\s?შ\.", "და ასე შემდეგ"), (r"ა\.\s?შ\.", "ასე შემდეგ"), (r"ე\.\s?ი\.", "ესე იგი"),
    (r"ძვ\.\s?წ\.(?:\s?ა?\.)?", "ძველი წელთაღრიცხვით"), (r"ახ\.\s?წ\.(?:\s?ა?\.)?", "ახალი წელთაღრიცხვით"),
    (r"ა\.\s?წ\.", "ამა წლის"), (r"მაგ\.", "მაგალითად"), (r"ქ\.(?=\s?[ა-ჰ])", "ქალაქი"), (r"ბ-ნ(ი|ს|მა|ის)", r"ბატონ\1"),
    (r"ქ-ნ(ი|ს|მა|ის)", r"ქალბატონ\1"), (r"№\s?", "ნომერი "), (r"დაახლ\.", "დაახლოებით"), (r"სხვ\.", "სხვა"),
    (r"ტელ\.", "ტელეფონი"), (r"გვ\.", "გვერდი"), (r"საუკ\.", "საუკუნე"),
]
ABBR = [(re.compile(r"(?<![ა-ჰ])" + a), b) for a, b in ABBR]

UNITS_AFTER = {"კმ": "კილომეტრი", "მ": "მეტრი", "სმ": "სანტიმეტრი", "მმ": "მილიმეტრი", "კგ": "კილოგრამი",
               "გრ": "გრამი", "გ": "გრამი", "ტ": "ტონა", "ლ": "ლიტრი", "მლ": "მილილიტრი", "სთ": "საათი", "წთ": "წუთი",
               "წმ": "წამი", "კმ/სთ": "კილომეტრი საათში", "°C": "გრადუსი", "°": "გრადუსი", "მბ": "მეგაბაიტი", "გბ": "გიგაბაიტი",
               "km": "კილომეტრი", "kg": "კილოგრამი", "GB": "გიგაბაიტი", "MB": "მეგაბაიტი", "წ": "წელი", "ც": "ცალი"}
CURRENCY = {"₾": ("ლარი", "თეთრი"), "GEL": ("ლარი", "თეთრი"), "ლარი": ("ლარი", "თეთრი"), "ლ": ("ლარი", "თეთრი"),
            "$": ("დოლარი", "ცენტი"), "USD": ("დოლარი", "ცენტი"), "€": ("ევრო", "ცენტი"), "EUR": ("ევრო", "ცენტი"),
            "£": ("გირვანქა სტერლინგი", "პენსი"), "₽": ("რუბლი", "კაპიკი")}
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman(s: str) -> int:
    total = 0
    for i, c in enumerate(s):
        v = ROMAN[c]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def _num(s: str) -> int:
    return int(re.sub(r"[  ,]", "", s))


NUM = r"\d{1,3}(?:[  ]\d{3})+|\d+"  # 1 000 000 ან 1000000
_CUR_SYM = "|".join(re.escape(k) for k in sorted(CURRENCY, key=len, reverse=True))


def _money(m) -> str:
    whole, frac, cur = m.group("w"), m.group("f"), m.group("c1") or m.group("c2")
    main, sub = CURRENCY[cur]
    out = f"{cardinal(_num(whole))} {main}"
    if frac and int(frac.ljust(2, "0")[:2]):
        out += f" და {cardinal(int(frac.ljust(2, '0')[:2]))} {sub}"
    return out


def _decimal(whole: str, frac: str) -> str:
    return f"{cardinal(_num(whole))} მთელი {cardinal(int(frac)) if not frac.startswith('0') else ' '.join(UNITS[int(d)] for d in frac)}"


# დასათვლელი სახელები: მათი მიცემითის/ვითარებითის წინ რიცხვი იკვეცება („ხუთ წელს“, „სამ ლარად“, „ორ დღეში“).
# ზმნებს (ღირს, აქვს) არ ვეხებით — ამიტომ სია და არა მხოლოდ დაბოლოება.
_COUNTED = re.compile(r"(?:წ[ელ]|თვ|დღ|კვირ|საათ|წუთ|წამ|ლარ|თეთრ|დოლარ|ევრო|ცენტ|კაც|ადამიან|ბავშვ|ქალ|ცალ|კილომეტრ|"
                      r"მეტრ|კილოგრამ|გრამ|ლიტრ|პროცენტ|ნაწილ|ქულ|გვერდ|ოთახ|სართულ|წიგნ|მანქან|ბინ|სახლ|ქვეყან|ქალაქ|"
                      r"სოფელ|სოფლ|ნაბიჯ|ჯერ|წერტილ|ბოთლ|ჭიქ|ფურცელ|ფურცლ|სტუდენტ|მოსწავლ|თანამშრომელ|თანამშრომლ|"
                      r"მომხმარებ|კლიენტ|შეკვეთ|პროდუქტ|ქულ|გოლ|მატჩ|თამაშ|ეპიზოდ|სერი|ფილმ|სიმღერ|ვიდეო)"
                      r"[ა-ჰ]*?(?:ს|ად|დ|ზე|ში|მდე)")


def _plain(m) -> str:
    s, nxt = m.group(1), m.group(2) or ""
    digits = re.sub(r"[  ]", "", s)
    if len(digits) > 1 and digits.startswith("0"):  # ტელეფონი, კოდი — ციფრებით
        return " ".join(UNITS[int(d)] for d in digits) + nxt
    if len(digits) > 15:
        return " ".join(UNITS[int(d)] for d in digits) + nxt
    w = cardinal(int(digits))
    word = nxt.strip()
    # შემდეგი სიტყვა მიცემითშია (წელს, ლარს) ან ვითარებითში (ლარად) → „ხუთ წელს“
    if word and _COUNTED.fullmatch(word) and not word.endswith("ის"):
        w = oblique(w)
    return w + nxt


_CASE_HYPHEN = re.compile(r"([ა-ჰ]+)-((?:ში|ზე|ის|ს|მა|ით|ად|დან|თან|ვით|მდე|თვის|იან|ელ|ებ|იც|ც)[ა-ჰ]*)(?![ა-ჰ-])")


def normalize(text: str) -> str:
    t = text
    rx, m = _lex_re()
    if rx:
        t = rx.sub(lambda x: m[x.group(1).lower()], t)
    # თარიღი: 13.10.2026, 13/10/2026 → ცამეტი ოქტომბერი, ორი ათას ოცდაექვსი
    t = re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b",
               lambda x: f"{cardinal(int(x[1]))} {MONTHS[int(x[2]) - 1]}, {cardinal(int(x[3]))}" if 1 <= int(x[2]) <= 12 else x[0], t)
    # დრო: 14:30 → თოთხმეტი ოცდაათი; 9:05 → ცხრა ნული ხუთი
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda x: cardinal(int(x[1])) + (
        "" if x[2] == "00" else " " + (f"ნული {UNITS[int(x[2][1])]}" if x[2][0] == "0" else cardinal(int(x[2])))), t)
    # ფული: 15₾, 15 ₾, ₾15, 15.50 ლარი, $20
    t = re.sub(rf"(?:(?P<c1>{_CUR_SYM})\s?)?(?P<w>{NUM})(?:[.,](?P<f>\d{{1,2}}))?(?(c1)|\s?(?P<c2>{_CUR_SYM})(?![ა-ჰa-zA-Z]))",
               _money, t)
    # პროცენტი
    t = re.sub(rf"({NUM})(?:[.,](\d+))?\s?%",
               lambda x: (_decimal(x[1], x[2]) if x[2] else cardinal(_num(x[1]))) + " პროცენტი", t)
    # რიგობითი: მე-5, 5-ე, 21-ე, 1-ლი, XX საუკუნე
    t = re.sub(r"მე-(\d+)(-?ე)?", lambda x: ordinal(int(x[1])), t)
    t = re.sub(r"\b(\d+)-(?:ე|ლი|ლ)\b", lambda x: ordinal(int(x[1])), t)
    t = re.sub(r"\b([IVXLC]{1,7})(?=\s+(?:საუკუნ|ს\.|ტომ|თავ|კლას))", lambda x: ordinal(roman(x[1])), t)
    # ერთეულები: 5 კმ → ხუთი კილომეტრი
    units = "|".join(re.escape(u) for u in sorted(UNITS_AFTER, key=len, reverse=True))
    t = re.sub(rf"({NUM})(?:[.,](\d+))?\s?({units})(?![ა-ჰa-zA-Z])\.?",
               lambda x: (_decimal(x[1], x[2]) if x[2] else cardinal(_num(x[1]))) + " " + UNITS_AFTER[x[3]], t)
    # ათწილადი: 3.5 / 3,5 → სამი მთელი ხუთი
    t = re.sub(r"\b(\d+)[.,](\d+)\b", lambda x: _decimal(x[1], x[2]), t)
    # უარყოფითი: -5 (სიტყვის შიგნით არა)
    t = re.sub(r"(?<![\wა-ჰ-])-(\d)", r"მინუს \1", t)
    # დანარჩენი რიცხვები (შემდეგი სიტყვის ბრუნვის გათვალისწინებით)
    t = re.sub(rf"({NUM})(\s+[ა-ჰ]+)?", _plain, t)
    t = t.replace("+", " პლუს ").replace("&", " და ")
    for rx_, rep in ABBR:
        t = rx_.sub(rep, t)
    # დარჩენილი ლათინური სიტყვები
    t = re.sub(r"[A-Za-z]+(?:'[a-z]+)?", lambda x: latin_word(x[0].replace("'", "")), t)
    # ბრუნვის ნიშანი დეფისით: ასი-ზე → ასზე, გუგლი-ში → გუგლში, ოთხმოცდაათი-იან → ოთხმოცდაათიან
    t = _CASE_HYPHEN.sub(lambda x: (x[1][:-1] if x[1].endswith("ი") else x[1]) + x[2], t)
    return re.sub(r"[  ]{2,}", " ", t).strip()


def ka_text(text: str) -> str:
    return _VOICELESS_GH.sub(r"\1-ღ", normalize(text))
