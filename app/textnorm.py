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


# ბრუნვის ნიშანი დეფისის შემდეგ (5-ს, 15%-ით, კმ-ში, 21:00-მდე, SMS-ით) — მხოლოდ დახურული სია:
# „ციხე-სიმაგრე“, „აპრილ-მაისში“ — ორი ქართული სიტყვაა და დეფისი რჩება (espeak-ში სიტყვათა საზღვარია).
_PART = r"(?:აც|ა|ც|ვე)?"  # ნაწილაკი: შიც, ზეა, წელსაც
_SFX = (r"(?:ებ(?:ი|ს|ში|ზე|თან|ის|ით|მა|ად|იდან|ამდე|ისთვის)?|იან(?:ი|ს|ში|ზე|ის|ით|მა|ად|ებ[ა-ჰ]{0,6})?|"
        r"ელ(?:ი|ს|ის|მა|ით|ებ[ა-ჰ]{0,6})|ისთვის|თვის|ისკენ|იდან|დან|ამდე|მდე|ივით|ვით|ში|ზე|თან|ის|ით|ია|ად|მა|ს|იც|ც|ა)"
        + _PART)
# რიცხვი იკვეცება მიცემითის (-ს, -ში, -ზე, -თან) და ვითარებითის (-ად, -მდე) წინ: „ხუთ წელს“; ნათესაობითის/მოქმედებითის
# (-ის, -ით, -იდან, -თვის) წინ — არა: „ხუთი წლის“, „ხუთი ლარით“.
_SHORT_SFX = re.compile(r"(?:ს|ში|ზე|თან|ამდე|მდე|ად|დ)" + _PART)
_SYNCOPE = {"წელი": "წლ"}  # ხმოვნით დაწყებული ნიშნის წინ: წლის, წლამდე


def _attach(words: str, sfx: str) -> str:
    """ბრუნვის ნიშნის მიერთება ბოლო სიტყვაზე: ოცდაათი+ია → ოცდაათია, ოცდაერთი+მდე → ოცდაერთამდე, რვა+დან → რვიდან."""
    head, _, w = words.rpartition(" ")
    pre = head + " " if head else ""
    if w.endswith("ი"):  # თანხმოვანფუძიანი: კაცი, ოცი, კილომეტრი
        for a, b in (("მდე", "ამდე"), ("დან", "იდან"), ("ვით", "ივით"), ("თვის", "ისთვის"), ("კენ", "ისკენ"), ("დ", "ად")):
            if sfx.startswith(a):
                sfx = b + sfx[len(a):]
                break
        stem = _SYNCOPE.get(w, w[:-1]) if sfx[0] in "იაე" else w[:-1]
        if sfx[0] == "ც" or (sfx[0] == "ა" and not sfx.startswith(("ად", "ამდე"))):
            return pre + w + sfx  # ხუთიც, ხუთია
        return pre + stem + sfx
    if w.endswith("ა"):  # ხმოვანფუძიანი: რვა, ცხრა, ტონა
        for a, b in (("ია", "ა"), ("დან", "იდან"), ("თვის", "ისთვის"), ("კენ", "ისკენ"), ("ამდე", "მდე"), ("ად", "დ"), ("მა", "მ")):
            if sfx.startswith(a):
                sfx = b + sfx[len(a):]
                break
        return pre + (w[:-1] + sfx if sfx[0] == "ი" else w + sfx)
    if w[-1:] in "ეოუ":  # ევრო, ნატო: ევროს, ევროთი, ნატოდან
        for a, b in (("ისთვის", "სთვის"), ("ისკენ", "სკენ"), ("იდან", "დან"), ("ის", "ს"), ("ით", "თი"), ("ია", "ა"),
                     ("იც", "ც"), ("ამდე", "მდე"), ("ად", "დ")):
            if sfx.startswith(a):
                sfx = b + sfx[len(a):]
                break
    return pre + w + sfx  # აშშ-ში, ნატო-ს


def _count(num: str, noun: str, sfx: str | None) -> str:
    """რიცხვი + სახელი (+ დეფისით მიერთებული ბრუნვის ნიშანი): 2 + კილომეტრი + ში → ორ კილომეტრში."""
    if not sfx:
        return f"{num} {noun}"
    if _SHORT_SFX.fullmatch(sfx):
        num = oblique(num)
    first, _, rest = noun.partition(" ")  # კილომეტრი საათში: ბრუნავს პირველი სიტყვა
    if rest and noun.endswith("საათში"):
        return f"{num} {_attach(first, sfx)} {rest}"
    return f"{num} {_attach(noun, sfx)}"


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


NUM = r"\d{1,3}(?:[  ]\d{3})+|\d+"  # 1 000 000 ან 1000000
# ვალუტის ნიშანი/სიტყვა მხოლოდ მთლიანი ტოკენია, რიცხვის გვერდით („სულ 25“-ში „ლ“ ლარი არ არის).
_CUR_SYM = "|".join(re.escape(k) for k in sorted(CURRENCY, key=len, reverse=True) if not re.match(r"[ა-ჰ]", k))
_CUR_WORD = {"ლარ": ("ლარი", "თეთრი"), "დოლარ": ("დოლარი", "ცენტი"), "ევრო": ("ევრო", "ცენტი")}
_CUR_TAIL = rf"(?:ი{_PART}|თი|დ|{_SFX})?"


def _frac2(frac: str | None) -> int:
    return int(frac.ljust(2, "0")[:2]) if frac else 0


def _money(m) -> str:
    """15.50 ₾ → თხუთმეტი ლარი და ორმოცდაათი თეთრი; 0.50 ₾ → ორმოცდაათი თეთრი; 2,50 ლარია → … თეთრია; 15₾-ით → … ლარით."""
    gd = m.groupdict()
    whole, f, sfx, g = _num(gd["w"]), _frac2(gd["f"]), gd.get("s"), gd.get("g")
    if g:  # ქართული სიტყვა: ლარი, ლარად, დოლარს, ევროთი…
        main, sub = _CUR_WORD[g]
        tail = gd["t"] or ""
        if g == "ევრო":
            sfx = {"": None, "თი": "ით"}.get(tail, tail)
        else:
            sfx = None if tail in ("", "ი") else tail
        if not f:
            num = cardinal(whole)
            return (oblique(num) if sfx and _SHORT_SFX.fullmatch(sfx) else num) + " " + g + tail
    else:
        main, sub = CURRENCY[gd.get("c") or gd.get("c2") or "₾"]
    if not f:
        return _count(cardinal(whole), main, sfx)
    cents = _count(cardinal(f), sub, sfx)
    return cents if not whole else f"{cardinal(whole)} {main} და {cents}"


def _decimal(whole: str, frac: str) -> str:
    if frac == "5":  # 1,5 → ერთნახევარი, 12,5 → თორმეტნახევარი, 0,5 → ნახევარი
        n = _num(whole)
        return oblique(cardinal(n)) + "ნახევარი" if n else "ნახევარი"
    return f"{cardinal(_num(whole))} მთელი {cardinal(int(frac)) if not frac.startswith('0') else ' '.join(UNITS[int(d)] for d in frac)}"


# დასათვლელი სახელები: მათი მიცემითის/ვითარებითის წინ რიცხვი იკვეცება („ხუთ წელს“, „სამ ლარად“, „ორ დღეში“).
_COUNTED = re.compile(r"(?:წ[ელ]|თვ|დღ|კვირ|საათ|წუთ|წამ|ლარ|თეთრ|დოლარ|ევრო|ცენტ|კაც|ადამიან|ბავშვ|ქალ|ცალ|კილომეტრ|"
                      r"მეტრ|კილოგრამ|გრამ|ლიტრ|პროცენტ|ნაწილ|ქულ|გვერდ|ოთახ|სართულ|წიგნ|მანქან|ბინ|სახლ|ქვეყან|ქალაქ|"
                      r"სოფელ|სოფლ|ნაბიჯ|ჯერ|წერტილ|ბოთლ|ჭიქ|ფურცელ|ფურცლ|სტუდენტ|მოსწავლ|თანამშრომელ|თანამშრომლ|"
                      r"მომხმარებ|კლიენტ|შეკვეთ|პროდუქტ|ქულ|გოლ|მატჩ|თამაშ|ეპიზოდ|სერი|ფილმ|სიმღერ|ვიდეო|აგვისტო)"
                      r"[ა-ჰ]*?(?:ს|ად|დ|ზე|ში|მდე)")
# ზოგადი წესი ნებისმიერი სახელისთვის (სტუმარს, სექტემბერს, ქალაქში, წლამდე) — თანხმოვანზე დამთავრებული ფორმა
# სახელობითი ვერ იქნება („მოსწავლეა“, „კლასი“ — არა).
# -ად ზოგად წესში არ არის: „5 ერთად“ (ზმნიზედა); ლარად, წლად, ნაწილად — _COUNTED-შია.
_OBLIQUE = re.compile(r"[ა-ჰ]{2,}?(?:ს(?:აც|ვე)?|(?:ში|ზე|თან|მდე)" + _PART + r")")
# ზმნები (ღირს, აქვს, ჰყავს, შეადგენს, ცხოვრობს…), ნათესაობითი (-ის) და ო/უ-ფუძის -ს (ნათ. = მიც.: „19 აგვისტოს დილით“)
_NOT_OBLIQUE = re.compile(r"(?:ის|ოს|უს|ავს|ევს|ებს|ობს|ამს|ოვს|ქვს|ყავს|ირს|ენს|დეს|ოფს|ჩანს|დგას|სურს|უყვარს)(?:აც|ვე)?$|^ჰ")
_GEN_HEAD = re.compile(r"(?:დილ|საღამო|ღამ|შუადღ|ომ|მოვლენ)[ა-ჰ]*")  # „12 აგვისტოს ომი“ — ნათესაობითი


def _is_oblique(word: str, after: str = "") -> bool:
    if not word:
        return False
    if _COUNTED.fullmatch(word) and not word.endswith("ის"):
        return not (word.endswith(("ოს", "უს")) and _GEN_HEAD.fullmatch(after))
    return bool(_OBLIQUE.fullmatch(word)) and not _NOT_OBLIQUE.search(word)


def _before(words: str, nxt: str, after: str = "") -> str:
    """რიცხვი შემდეგი სიტყვის ბრუნვის მიხედვით: ხუთი + წელს → ხუთ წელს."""
    return (oblique(words) if _is_oblique(nxt.strip(), after) else words) + nxt


_MONTH = re.compile(r"(?:იანვ|თებერვ|მარტ|აპრილ|მაის|ივნის|ივლის|აგვისტო|სექტემბ|ოქტომბ|ნოემბ|დეკემბ)")


def _day(n: int) -> str:
    return ordinal(1) if n == 1 else cardinal(n)


def _plain(m) -> str:
    s, nxt, after = m.group(1), m.group(2) or "", m.group(3) or ""
    digits = re.sub(r"[  ]", "", s)
    if len(digits) > 1 and digits.startswith("0"):  # ტელეფონი, კოდი — ციფრებით
        return " ".join(UNITS[int(d)] for d in digits) + nxt
    if len(digits) > 15:
        return " ".join(UNITS[int(d)] for d in digits) + nxt
    if digits == "1" and _MONTH.match(nxt.strip()):  # 1 მაისს → პირველ მაისს (თვის პირველი რიცხვი რიგობითია)
        return _before(ordinal(1), nxt, after)
    return _before(cardinal(int(digits)), nxt, after)


_CASE_HYPHEN = re.compile(rf"([ა-ჰ]+)-({_SFX})(?![ა-ჰ-])")
_NEXT = r"(\s+[ა-ჰ]+)?(?:(?=\s+([ა-ჰ]+)))?"  # შემდეგი სიტყვა (+ მის შემდეგიც, აგვისტოს/დილით)
_DOT_ABBR = {"წ", "ც", "გრ", "სთ", "წთ", "წმ"}  # „2020 წ. დაიბადა“ — წერტილი შემოკლებისაა


def _time(x) -> str:
    h, mm, sfx = int(x[1]), x[2], x[3]
    if mm == "00":
        return _count(cardinal(h), "საათი", sfx) if sfx else cardinal(h)  # 21:00-მდე → ოცდაერთ საათამდე
    words = cardinal(h) + " " + (f"ნული {UNITS[int(mm[1])]}" if mm[0] == "0" else cardinal(int(mm)))
    return _attach(words, sfx) if sfx else words  # 14:30-ზე → თოთხმეტი ოცდაათზე


def _unit(x) -> str:
    num = _decimal(x[1], x[2]) if x[2] else cardinal(_num(x[1]))
    u, sfx, dot = x[3], x[4], x[5] or ""
    if dot and u in _DOT_ABBR and x[6]:
        dot = ""
    return _count(num, UNITS_AFTER[u], sfx) + dot


def normalize(text: str) -> str:
    t = text
    rx, m = _lex_re()
    if rx:
        t = rx.sub(lambda x: m[x.group(1).lower()], t)
    # უარყოფითი: -5, -5°C, −3 (სიტყვის შიგნით და დიაპაზონში — 10-15 — არა)
    t = re.sub(r"(?<![\wა-ჰ-])[-−](\d)", r"მინუს \1", t)
    # თარიღი: 13.10.2026, 13/10/2026 → ცამეტი ოქტომბერი, ორი ათას ოცდაექვსი
    t = re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b",
               lambda x: f"{_day(int(x[1]))} {MONTHS[int(x[2]) - 1]}, {cardinal(int(x[3]))}" if 1 <= int(x[2]) <= 12 else x[0], t)
    # დრო: 14:30 → თოთხმეტი ოცდაათი; 9:05 → ცხრა ნული ხუთი; 14:30-ია → თოთხმეტი ოცდაათია
    t = re.sub(rf"\b(\d{{1,2}}):(\d{{2}})\b(?:-({_SFX})(?![ა-ჰ]))?", _time, t)
    # ფული: 15₾, 15 ₾, ₾15, 15.50 ლარი, $20, 15₾-ით
    t = re.sub(rf"(?<![\wა-ჰ])(?P<c>{_CUR_SYM})\s?(?P<w>{NUM})(?:[.,](?P<f>\d{{1,2}}))?(?!\d)(?:-(?P<s>{_SFX})(?![ა-ჰ]))?",
               _money, t)
    t = re.sub(rf"(?<![\d.,])(?P<w>{NUM})(?:[.,](?P<f>\d{{1,2}}))?\s?(?:(?P<c2>{_CUR_SYM})(?:-(?P<s>{_SFX}))?(?![ა-ჰa-zA-Z])|"
               rf"(?P<g>დოლარ|ლარ|ევრო)(?P<t>{_CUR_TAIL})(?![ა-ჰ])|ლ(?![ა-ჰ]))", _money, t)
    # პროცენტი: 25%, 2,5%, 15%-ით
    t = re.sub(rf"({NUM})(?:[.,](\d+))?\s?%(?:-({_SFX})(?![ა-ჰ]))?",
               lambda x: _count(_decimal(x[1], x[2]) if x[2] else cardinal(_num(x[1])), "პროცენტი", x[3]), t)
    # რიგობითი: მე-5, 5-ე, 21-ე, 1-ლი, 1-ლ (კლასში → პირველ), XX საუკუნე
    t = re.sub(r"მე-(\d+)(-?ე)?", lambda x: ordinal(int(x[1])), t)
    t = re.sub(r"\b(\d+)-(ე|ლი|ლ)\b", lambda x: oblique(ordinal(int(x[1]))) if x[2] == "ლ" else ordinal(int(x[1])), t)
    t = re.sub(r"\b([IVXLC]{1,7})(?=\s+(?:საუკუნ|ს\.|ტომ|თავ|კლას))", lambda x: ordinal(roman(x[1])), t)
    # ერთეულები: 5 კმ → ხუთი კილომეტრი; 2 კმ-ში → ორ კილომეტრში; წინადადების ბოლო წერტილი რჩება
    units = "|".join(re.escape(u) for u in sorted(UNITS_AFTER, key=len, reverse=True))
    t = re.sub(rf"({NUM})(?:[.,](\d+))?\s?({units})(?:-({_SFX}))?(?![ა-ჰa-zA-Z])(\.(?=([ \t]+[ა-ჰ\d]))?)?", _unit, t)
    # ათწილადი: 3,25 → სამი მთელი ოცდახუთი; 1,5 → ერთნახევარი
    t = re.sub(rf"\b(\d+)[.,](\d+)\b{_NEXT}", lambda x: _before(_decimal(x[1], x[2]), x[3] or "", x[4] or ""), t)
    # დანარჩენი რიცხვები (შემდეგი სიტყვის ბრუნვის გათვალისწინებით)
    t = re.sub(rf"({NUM}){_NEXT}", _plain, t)
    t = t.replace("+", " პლუს ").replace("&", " და ")
    for rx_, rep in ABBR:
        t = rx_.sub(rep, t)
    # დარჩენილი ლათინური სიტყვები
    t = re.sub(r"[A-Za-z]+(?:'[a-z]+)?", lambda x: latin_word(x[0].replace("'", "")), t)
    # ბრუნვის ნიშანი დეფისით: ასი-ზე → ასზე, გუგლი-ში → გუგლში, ოთხმოცდაათი-იან → ოთხმოცდაათიან
    t = _CASE_HYPHEN.sub(lambda x: _attach(x[1], x[2]), t)
    return re.sub(r"[  ]{2,}", " ", t).strip()


def ka_text(text: str) -> str:
    return _VOICELESS_GH.sub(r"\1-ღ", normalize(text))
