"""OmniVoice-ისთვის წასაკითხი ქართული წინადადებები — სინთეზური ჩანაწერები ახალი საბაზო Piper მოდელისთვის (v4).

    nice .venv/bin/python tools/omni_corpus.py [--total 30000] [--shards 40] [--seed 4] [--out {data_root}/omni_corpus]
შედეგი: <out>/sentences.jsonl — {"id", "text" (ka_text-ით გამართული — რასაც ხმა წაიკითხავს), "raw" (საწყისი),
        "source", "tags": [...]}; <out>/stats.json; <out>/shards/voice_XX.jsonl (თითო მომავალ სინთეზურ ხმაზე ერთი).

წყაროები (მხოლოდ CC0 ან ჩვენი):
  cv27      — Common Voice 27 validated_sentences.tsv (CC0); reported.tsv-ში მოხვედრილი წინადადებები გამოტოვდება.
  prompts   — data/prompts/*.json (იგივე Common Voice, CC0).
  cv27 + derived_q / derived_excl — CV-ში კითხვა/ძახილი ძალიან ცოტაა (~0.4%), ამიტომ თხრობით წინადადებას
              ვუმატებთ „იცოდით, რომ …?“, „ნუთუ …?“, „…, არა?“, „წარმოიდგინეთ, …!“ და ა.შ. (კვლავ CC0 + ჩვენი ჩარჩო).
  numbers   — ჩვენი შაბლონები: ფული, თარიღი, დრო, პროცენტი, მანძილი, რაოდენობა. იწერება ციფრებით და
              app.textnorm.ka_text() შლის სიტყვებად; შაბლონი, რომელსაც ka_text არასწორად კითხულობს, მთლიანად იყრება.
  navigation— ჩვენი შაბლონები: მოუხვიეთ, N მეტრში, წრიული გზა, მისვლა, გადაანგარიშება, კამერა, საცობი.
  dialog    — ჩვენი სასაუბრო კითხვები და ძახილი (სახელები, თხოვნა, მილოცვა, გაფრთხილება…).

ფილტრი: ka_text-ის შემდეგ მხოლოდ ქართული ასოები და ჩვეულებრივი პუნქტუაცია, 20–160 სიმბოლო; ლათინური, URL,
მთავრული, ფრჩხილები და უცნაური ნიშნები — არა. დუბლიკატი: ასოები და ჰარები (პუნქტუაციის გარეშე).
შემადგენლობა (ნაგულისხმევი): კითხვა ≥ 17%, ძახილი ≥ 11%, ციფრები ~10%, ნავიგაცია ~5%; „რთული“ (clusters ≥ 2
select_best.py-დან) თავისთავად ~75%. სიგრძე — კვოტებით (LEN_SHARE), საშუალოდ ~60 სიმბოლო ⇒ ~35 სთ (14 სიმბ/წმ).
შარდები სტრატიფიცირებულია (წყარო × კითხვა/ძახილი × რთული × სიგრძე), ამიტომ შემადგენლობა ყველგან ერთნაირია.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app.config import path as cpath  # noqa: E402
from app.textnorm import ka_text, cardinal  # noqa: E402
from select_best import clusters  # noqa: E402

csv.field_size_limit(10 ** 9)
CPS = 14  # სიმბოლო წამში — საათების შეფასებისთვის
LEN_MIN, LEN_MAX = 20, 160
LEN_BINS = [(20, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 161)]
LEN_SHARE = [0.27, 0.32, 0.20, 0.11, 0.07, 0.03]  # მთელი კორპუსის სამიზნე წილი თითო სიგრძეზე
SPOKEN = re.compile(r"^[ა-ჰ][ა-ჰ ,.?!:;—-]*[.?!]$")  # ka_text-ის შემდეგ (ს-ღ დეფისიც დაშვებულია)
RAW_OK = re.compile(r"^[ა-ჰ0-9 ,.?!:;—–\-„“\"%₾$€°/]+$")
CONJ_START = ("და ", "მაგრამ ", "თუმცა ", "ამიტომ ", "ასევე ", "ხოლო ", "ანუ ", "მაშინ ", "კი ", "ან ", "რომ ",
              "მაშ ", "ე.ი", "ისე ", "აქედან ", "ამასთან ", "შესაბამისად ", "მაგალითად ")


def bin_of(n: int) -> int:
    for i, (a, b) in enumerate(LEN_BINS):
        if a <= n < b:
            return i
    return -1


def dedup_key(text: str) -> str:
    return " ".join(re.sub(r"[^ა-ჰ ]", " ", text.replace("-", "")).split())


def clean_raw(s: str):
    """CV-ის სტრიქონის გასწორება; None — თუ არ გამოდგება (ლათინური, მთავრული, ფრჩხილი, URL…)."""
    s = s.replace("­", "").replace("​", "").replace(" ", " ").replace("”", "“").replace("…", ".")
    s = re.sub(r"(?<=[ა-ჰ])[–—](?=[ა-ჰ])", "-", " ".join(s.split()))  # „კარდინალ–არქიეპისკოპოსი“ — დეფისია
    if not s or not RAW_OK.match(s) or re.search(r"\d", s):  # CV-ში ციფრი არ არის; თუ გაჩნდა — უკეთესია არ ვენდოთ
        return None
    letters = len(re.findall("[ა-ჰ]", s))
    if letters < 0.8 * len(s.replace(" ", "")):  # დაზიანებული სტრიქონები („????? ?????“)
        return None
    if s.count("„") != s.count("“") or "?." in s or "!." in s or ".." in s:
        return None
    if s[-1] not in ".?!":
        s += "."
    return s


def spoken(raw: str) -> str:
    """რასაც ხმა წაიკითხავს: ka_text + ბრჭყალების მოცილება + ბოლო ნიშნის აღდგენა (ერთეულის წესი „კმ.“-ს წერტილს ჭამს)."""
    # ka_text-ის _CASE_HYPHEN ასოებს შორის დეფისს ბრუნვის ნიშნად თვლის: „ციხე-სიმაგრე“ → „ციხესიმაგრე“,
    # „აპრილ-მაისში“ → „აპრილმაისში“. ამიტომ სიტყვის შიგნით დეფისს ვმალავთ (ციფრის შემდეგ — „14:30-ზე“ — არა).
    t = ka_text(re.sub(r"(?<=[ა-ჰ])-(?=[ა-ჰ])", "\u2011", raw)).replace("\u2011", "-")
    t = re.sub(r"[„“\"]", "", t).replace("–", "—")
    t = re.sub(r"\s+([,.?!:;])", r"\1", t)
    t = re.sub(r"\s*—\s*", " — ", t)
    t = " ".join(t.split())
    if t and t[-1] not in ".?!":
        t += raw.rstrip()[-1] if raw.rstrip()[-1] in ".?!" else "."
    return t


def item(text, raw, source, tags, tpl=None):
    tags = list(tags)
    end = text[-1]
    if end == "?" and "question" not in tags:
        tags.append("question")
    if end == "!" and "exclamation" not in tags:
        tags.append("exclamation")
    if clusters(text) >= 2:
        tags.append("hard")
    d = {"text": text, "raw": raw, "source": source, "tags": tags}
    if tpl:
        d["template"] = tpl
    return d


# ---------- ka_text-ის შემოწმება შაბლონებზე ----------
_NUMW = set()
for _n in list(range(0, 101)) + [200, 300, 400, 500, 600, 700, 800, 900, 1000, 10 ** 6, 10 ** 9]:
    _NUMW.update(cardinal(_n).split())
_NUM_STEMS = tuple(sorted({w[:-1] if w.endswith("ი") else w for w in _NUMW}, key=len, reverse=True))
_CUR_WORDS = ("ლარ", "თეთრ", "დოლარ", "ევრო", "ცენტ")


def textnorm_problem(raw: str, text: str):
    """None — თუ კარგია; სხვაგვარად — მიზეზი. ვამოწმებთ ka_text-ის ცნობილ სუსტ ადგილებს."""
    if re.search(r"\d", text):
        return "ციფრი დარჩა"
    if not SPOKEN.match(text):
        return "უცხო სიმბოლო"
    if "მთელი" in text and "მთელი" not in raw:
        return "ათწილადი („მთელი“)"
    if "ნული ლარი" in text or "ნული დოლარი" in text:
        return "„ნული ლარი“"
    for w in _CUR_WORDS:  # ფულის სიტყვა, რომელიც raw-ში არ ეწერა („სულ 3“ → „სუსამი … ლარი“)
        n_t, n_r = (len(re.findall(rf"(?<![ა-ჰ]){w}", x)) for x in (text, raw))
        if n_t > n_r + sum(raw.count(s) for s in "₾$€"):
            return f"ზედმეტი „{w}…“"
    toks = re.findall(r"[ა-ჰ-]+", text)
    for a, b in zip(toks, toks[1:]):
        # სრული ფორმა („თორმეტი“) მიცემითის/ვითარებითის წინ: უნდა იყოს „თორმეტ სექტემბერს“, „ორ კილომეტრში“
        if a in _NUMW and a.endswith("ი") and re.search(r"(ს|ში|ზე|ად|მდე)$", b) and not re.search(r"(ის|ისა|ს-ღ)$", b) and b not in MON_GEN \
                and not b.startswith(_NUM_STEMS):
            return f"„{a} {b}“ — ბრუნვა"
    if re.search(r"[ბგდვზთკლმნპჟრსტფქღყშჩცძწჭხჯჰ]მდე\b", text):
        return "„…მდე“ თანხმოვნის შემდეგ"
    if re.search(r"საათშ(ზე|ში)", text) or (re.search(r"[ა-ჰ]-[ა-ჰ]", re.sub(r"[სშცჩხქფთტპკწჭყ]-ღ", "", text))
                                            and not re.search(r"[ა-ჰ]-[ა-ჰ]", raw)):
        return "დეფისი/ბრუნვის ნიშანი"
    if re.search(r"(?<![ა-ჰ])-", text):
        return "ნიშანი „-“ რიცხვის წინ"
    return None


# ---------- ციფრები ----------
ITEMS = ["ეს წიგნი", "ბილეთი ბათუმამდე", "ერთი კილო ვაშლი", "ტაქსი აეროპორტამდე", "თვიური აბონემენტი", "ახალი ტელეფონი",
         "კონცერტის ბილეთი", "ბინის ქირა", "ერთი ლიტრი რძე", "ეს ქურთუკი", "კინოს ბილეთი", "ახალი ფეხსაცმელი",
         "ველოსიპედი", "ტყავის ჩანთა", "მზის სათვალე", "სადილი ორ კაცზე", "ყავა და ნამცხვარი", "ინგლისურის კურსი",
         "მუზეუმის ბილეთი", "ოთახი სასტუმროში", "ბავშვის საწოლი", "სარეცხი მანქანა", "ზამთრის საბურავები",
         "ეს ნახატი", "ხელნაკეთი ხალიჩა", "ერთი კილო ყველი", "ახალი ლეპტოპი", "ფიტნესის აბონემენტი"]
MON_NOM = ["იანვარი", "თებერვალი", "მარტი", "აპრილი", "მაისი", "ივნისი", "ივლისი", "აგვისტო", "სექტემბერი",
           "ოქტომბერი", "ნოემბერი", "დეკემბერი"]
MON_COP = ["იანვარია", "თებერვალია", "მარტია", "აპრილია", "მაისია", "ივნისია", "ივლისია", "აგვისტოა", "სექტემბერია",
           "ოქტომბერია", "ნოემბერია", "დეკემბერია"]
MON_GEN = ["იანვრის", "თებერვლის", "მარტის", "აპრილის", "მაისის", "ივნისის", "ივლისის", "აგვისტოს", "სექტემბრის",
           "ოქტომბრის", "ნოემბრის", "დეკემბრის"]
MON_ABL = ["იანვრიდან", "თებერვლიდან", "მარტიდან", "აპრილიდან", "მაისიდან", "ივნისიდან", "ივლისიდან", "აგვისტოდან",
           "სექტემბრიდან", "ოქტომბრიდან", "ნოემბრიდან", "დეკემბრიდან"]
CITY = [("ბათუმამდე", 370), ("ქუთაისამდე", 230), ("თელავამდე", 95), ("გორამდე", 85), ("ზუგდიდამდე", 320),
        ("მცხეთამდე", 25), ("ბორჯომამდე", 160), ("სიღნაღამდე", 110), ("ახალციხემდე", 210), ("ფოთამდე", 310),
        ("სტეფანწმინდამდე", 155), ("მესტიამდე", 460), ("ოზურგეთამდე", 300), ("რუსთავამდე", 30), ("ზესტაფონამდე", 190)]
ROMAN = ["IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]
PEOPLE_PLACE = ["კონფერენციაზე", "კონცერტზე", "შეხვედრაზე", "ქორწილში", "ლექციაზე", "გამოფენაზე", "დღესასწაულზე"]


def _money(r):
    return str(r.choice([r.randint(1, 30), r.randint(1, 30), r.choice([35, 40, 45, 50, 60, 75, 80, 90, 99, 100, 120, 150,
                                                                          180, 199, 250, 300, 350, 450, 500, 750, 999])]))


def _big(r):
    n = r.choice([r.randint(2, 99) * 1000, r.randint(1, 50) * 10000, r.randint(1, 9) * 10 ** 6 + r.randint(0, 9) * 10 ** 5])
    return f"{n:,}".replace(",", " ")


def _dec(r):
    return f"{r.randint(1, 120)}.{r.choice(range(5, 100, 5)):02d}"


def _hm(r):
    return f"{r.randint(6, 23)}:{r.choice(['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'])}"


F = {
    "item": lambda r: r.choice(ITEMS), "m": _money, "md": _dec, "big": _big,
    "rate": lambda r: f"2.{r.randint(55, 95)}", "usd": lambda r: str(r.choice([r.randint(5, 99), r.randint(100, 1500)])),
    "eur": lambda r: str(r.randint(25, 400)), "d": lambda r: str(r.randint(1, 31)), "mi": lambda r: r.randrange(12),
    "year": lambda r: str(r.randint(1921, 2026)), "oldyear": lambda r: str(r.randint(400, 1900)),
    "date": lambda r: f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.{r.randint(2024, 2030)}",
    "hm": _hm, "h": lambda r: str(r.randint(1, 12)), "min": lambda r: str(r.randint(2, 59)),
    "p": lambda r: str(r.choice([r.randint(1, 100), r.choice(range(5, 100, 5))])),
    "km": lambda r: str(r.choice([r.randint(2, 30), r.randint(31, 999)])), "v": lambda r: str(r.choice(range(20, 130, 10))),
    "t": lambda r: str(r.randint(2, 38)), "kg": lambda r: str(r.randint(2, 40)), "l": lambda r: str(r.randint(2, 60)),
    "meters": lambda r: str(r.choice(range(50, 1000, 50))), "alt": lambda r: str(r.randint(1000, 5100)),
    "n": lambda r: str(r.randint(2, 40)), "nbig": lambda r: str(r.randint(41, 999)), "fl": lambda r: str(r.randint(2, 22)),
    "age": lambda r: str(r.randint(3, 99)), "oldage": lambda r: str(r.randint(60, 99)), "roman": lambda r: r.choice(ROMAN), "city": lambda r: r.randrange(len(CITY)),
    "place": lambda r: r.choice(PEOPLE_PLACE),
}

# (id, შაბლონი, თეგი) — {mi}/{city} ინდექსია, შაბლონში გამოიყენება {mon_*}/{city_to}/{city_km}
NUM_T = [
    ("money_price", "{item} ღირს {m} ₾.", "money"),
    ("money_price_dec", "{item} ღირს {md} ₾.", "money"),
    ("money_q", "{item} მართლა {m} ლარი ღირს?", "money"),
    ("money_bought", "{item} {m} ლარად ვიყიდე.", "money"),
    ("money_total", "სულ გადასახდელია {md} ₾.", "money"),
    ("money_balance", "თქვენს ანგარიშზე დარჩა {md} ₾.", "money"),
    ("money_salary", "ხელფასი {m} ლარით გაიზარდა.", "money"),
    ("money_rate", "დღეს ერთი დოლარი {rate} ₾ ღირს.", "money"),
    ("money_usd", "ეს ტელეფონი ${usd} ღირს.", "money"),
    ("money_eur", "სასტუმროში ერთი ღამე {eur} € ჯდება.", "money"),
    ("money_bills", "ამ თვეში კომუნალურებში {m} ლარი გადავიხადე.", "money"),
    ("money_expensive", "{m} ლარი? ეს ხომ ძალიან ძვირია!", "money"),
    ("money_cheap", "მხოლოდ {m} ლარად? რა იაფია!", "money"),
    ("money_change", "ხომ არ გაქვს {m} ლარი ხურდად?", "money"),
    ("money_budget", "პროექტის ბიუჯეტი {big} ლარს შეადგენს.", "money"),
    ("money_cost", "მშენებლობის ღირებულება {big} დოლარია.", "money"),
    ("money_fine", "ჯარიმა {m} ლარია, ასე რომ, ფრთხილად იარე!", "money"),
    ("money_tip", "მიმტანს {m} ლარი დავუტოვეთ.", "money"),
    ("money_lend", "შეგიძლია {m} ლარი მასესხო ხვალამდე?", "money"),
    ("date_today", "დღეს {d} {mon_cop}.", "date"),
    ("date_meet", "შეხვედრა {d} {mon_gen} დილით გაიმართება.", "date"),
    ("date_concert", "{d} {mon_gen} საღამოს კონცერტზე მივდივართ.", "date"),
    ("date_born", "ის {year} წელს დაიბადა.", "date"),
    ("date_law", "კანონი ძალაში {year} წლის {d} {mon_abl} შევა.", "date"),
    ("date_deadline", "განაცხადის მიღების ბოლო ვადა: {date}.", "date"),
    ("date_contract", "ხელშეკრულების თარიღი: {date}.", "date"),
    ("date_bday_q", "შენი დაბადების დღე {d} {mon_cop}?", "date"),
    ("date_founded", "ქალაქი {oldyear} წელს დაარსდა.", "date"),
    ("date_century", "ეს ტაძარი {roman} საუკუნეშია აშენებული.", "date"),
    ("date_range", "ის {year} წლიდან საზღვარგარეთ ცხოვრობს.", "date"),
    ("date_exams", "გამოცდები {d} {mon_abl} იწყება.", "date"),
    ("date_free_q", "{d} {mon_gen} საღამოს ხომ თავისუფალი ხარ?", "date"),
    ("date_since", "{year} წლის შემდეგ ასეთი ზამთარი არ ყოფილა!", "date"),
    ("time_train", "მატარებელი {hm}-ზე გადის.", "time"),
    ("time_meeting", "შეხვედრა {h} საათზე დაიწყება.", "time"),
    ("time_now", "ახლა არის {hm}.", "time"),
    ("time_back", "{min} წუთში დავბრუნდები.", "time"),
    ("time_flight", "ფრენა {h} საათი და {min} წუთი გაგრძელდება.", "time"),
    ("time_film_q", "ფილმი {hm}-ზე იწყება, ხომ არ დავაგვიანებთ?", "time"),
    ("time_late", "უკვე {h} საათია, ჩქარა ადექი!", "time"),
    ("time_open", "მაღაზია დილის {h} საათიდან მუშაობს.", "time"),
    ("time_lesson", "გაკვეთილი {min} წუთი გრძელდება.", "time"),
    ("time_arrive_q", "დაახლოებით {min} წუთში ჩავალთ?", "time"),
    ("time_exam", "გამოცდა {hm}-ზე იწყება, არ დაიგვიანო!", "time"),
    ("pct_sale", "ყველა პროდუქტზე {p}%-იანი ფასდაკლებაა!", "percent"),
    ("pct_inflation", "წლიურმა ინფლაციამ {p}% შეადგინა.", "percent"),
    ("pct_battery", "ბატარეა {p}%-ით არის დამუხტული.", "percent"),
    ("pct_survey", "გამოკითხულთა {p}% ახალი კანონით კმაყოფილია.", "percent"),
    ("pct_prices_q", "ფასები მართლა {p}%-ით გაიზარდა?", "percent"),
    ("pct_harvest", "წელს მოსავალი {p}%-ით შემცირდა.", "percent"),
    ("dist_city", "თბილისიდან {city_to} დაახლოებით {city_km} კილომეტრია.", "distance"),
    ("dist_drove", "დღეს {km} კილომეტრი გავიარეთ.", "distance"),
    ("dist_track", "ტრასის სიგრძე {km} კმ.", "distance"),
    ("dist_speed", "მანქანა {v} კმ/სთ სიჩქარით მოძრაობდა.", "distance"),
    ("dist_temp", "ხვალ ტემპერატურა {t}°C იქნება.", "measure"),
    ("dist_bag", "ჩემოდანი {kg} კილოგრამს იწონის.", "measure"),
    ("dist_water", "ავზში {l} ლიტრი ბენზინი დარჩა?", "measure"),
    ("dist_sea", "აქედან ზღვამდე მხოლოდ {meters} მეტრია!", "distance"),
    ("dist_mountain", "მთის სიმაღლე {alt} მეტრია.", "measure"),
    ("dist_more", "ამ გზაზე {km} კილომეტრზე მეტი ვიარეთ.", "distance"),
    ("count_class", "კლასში {n} მოსწავლეა.", "count"),
    ("count_people", "{place} {nbig} ადამიანი მოვიდა.", "count"),
    ("count_books", "ამ ზაფხულს {n} წიგნი წავიკითხე.", "count"),
    ("count_points", "ჩვენმა გუნდმა {n} ქულით მოიგო!", "count"),
    ("count_days", "ზღვაზე {n} დღით მივდივართ.", "count"),
    ("count_floor", "მე-{fl} სართულზე ვცხოვრობ.", "count"),
    ("count_place", "ის მარათონში მე-{fl} ადგილზე გავიდა.", "count"),
    ("count_pages", "წიგნი {nbig} გვერდისგან შედგება.", "count"),
    ("count_families", "სოფელში მხოლოდ {n} ოჯახი ცხოვრობს.", "count"),
    ("count_age", "ბებიაჩემი {oldage} წლისაა.", "count"),
    ("count_age_q", "შენ მართლა {age} წლის ხარ?", "count"),
    ("count_flat", "ბინის ფართი {nbig} კვადრატული მეტრია.", "count"),
    ("count_city", "ქალაქში {big} ადამიანი ცხოვრობს.", "count"),
    ("count_guests", "ქორწილში {nbig} სტუმარს ველოდებით.", "count"),
]


def fill(tpl: str, r) -> str:
    vals = {}
    for k in set(re.findall(r"{(\w+)}", tpl)):
        if k.startswith("mon_"):
            continue
        if k in ("city_to", "city_km"):
            continue
        vals[k] = F[k](r)
    if "{mon_" in tpl:
        mi = r.randrange(12)
        vals.update(mon_nom=MON_NOM[mi], mon_cop=MON_COP[mi], mon_gen=MON_GEN[mi], mon_abl=MON_ABL[mi])
        if "d" in vals:
            vals["d"] = str(r.randint(1, 29 if mi == 1 else (30 if mi in (3, 5, 8, 10) else 31)))
    if "{city_" in tpl:
        c = CITY[r.randrange(len(CITY))]
        vals.update(city_to=c[0], city_km=str(c[1] + r.choice([-5, 0, 0, 5, 10])))
    return tpl.format(**vals)


# ---------- ნავიგაცია ----------
STREET_TYPES = {  # სახელობითი, „-ზე/-ში“ (სად), „-ისკენ“, ნათესაობითი
    "ქუჩა": ("ქუჩა", "ქუჩაზე", "ქუჩისკენ", "ქუჩის"),
    "გამზირი": ("გამზირი", "გამზირზე", "გამზირისკენ", "გამზირის"),
    "მოედანი": ("მოედანი", "მოედანზე", "მოედნისკენ", "მოედნის"),
    "შესახვევი": ("შესახვევი", "შესახვევში", "შესახვევისკენ", "შესახვევის"),
    "გზატკეცილი": ("გზატკეცილი", "გზატკეცილზე", "გზატკეცილისკენ", "გზატკეცილის"),
    "სანაპირო": ("სანაპირო", "სანაპიროზე", "სანაპიროსკენ", "სანაპიროს"),
}
STREETS = [("რუსთაველის", "გამზირი"), ("ჭავჭავაძის", "გამზირი"), ("აღმაშენებლის", "გამზირი"), ("წერეთლის", "გამზირი"),
           ("ვაჟა-ფშაველას", "გამზირი"), ("ყაზბეგის", "გამზირი"), ("პეკინის", "გამზირი"), ("გამსახურდიას", "გამზირი"),
           ("კოსტავას", "ქუჩა"), ("ბარათაშვილის", "ქუჩა"), ("მარჯანიშვილის", "ქუჩა"), ("გორგასლის", "ქუჩა"),
           ("ლესელიძის", "ქუჩა"), ("ფალიაშვილის", "ქუჩა"), ("აბაშიძის", "ქუჩა"), ("ბახტრიონის", "ქუჩა"),
           ("სადგურის", "ქუჩა"), ("ბაღის", "ქუჩა"), ("სკოლის", "ქუჩა"), ("ვაზის", "ქუჩა"), ("მშვიდობის", "ქუჩა"),
           ("ტყის", "ქუჩა"), ("მთის", "ქუჩა"), ("წყაროს", "ქუჩა"), ("ფრიდონის", "ქუჩა"), ("ბროწეულის", "ქუჩა"),
           ("თავისუფლების", "მოედანი"), ("ევროპის", "მოედანი"), ("რესპუბლიკის", "მოედანი"), ("გმირების", "მოედანი"),
           ("ბერძენიშვილის", "შესახვევი"), ("ვარდების", "შესახვევი"), ("ჩიტაიას", "შესახვევი"), ("ტბის", "შესახვევი"),
           ("დავით აღმაშენებლის", "გზატკეცილი"), ("ქართლის", "გზატკეცილი"), ("კახეთის", "გზატკეცილი"),
           ("მტკვრის", "სანაპირო"), ("მარცხენა", "სანაპირო"), ("ზღვის", "სანაპირო")]
DIRS = ["მარჯვნივ", "მარცხნივ", "ოდნავ მარჯვნივ", "ოდნავ მარცხნივ", "მკვეთრად მარჯვნივ", "მკვეთრად მარცხნივ"]
NAV_T = [
    ("nav_turn", "{dist} მოუხვიეთ {dir}."),
    ("nav_turn_street", "{dist} მოუხვიეთ {dir} {st_on}."),
    ("nav_turn_then", "მოუხვიეთ {dir}, შემდეგ კი {dir2} {st_on}."),
    ("nav_enter", "{dist} შეუხვიეთ {st_on}."),
    ("nav_continue", "გააგრძელეთ მოძრაობა {st_on} {kms} კილომეტრი."),
    ("nav_straight", "გაიარეთ პირდაპირ {km} კილომეტრი."),
    ("nav_toward", "დარჩით {side} ზოლში და გააგრძელეთ {st_to}."),
    ("nav_lane", "{dist} გადადით {side} ზოლში."),
    ("nav_round", "წრიულ გზაზე გადით მე-{exit} გასასვლელით."),
    ("nav_round_dist", "{dist} შედით წრიულ გზაზე და გადით მე-{exit} გასასვლელით."),
    ("nav_round_st", "წრიულ გზაზე გადით მე-{exit} გასასვლელით, {st_to}."),
    ("nav_arrive", "{dist} მიაღწევთ დანიშნულების ადგილს."),
    ("nav_arrive_side", "დანიშნულების ადგილი თქვენგან {side2} იქნება."),
    ("nav_arrive_st", "თქვენ მიხვედით {st_gen} დასაწყისთან."),
    ("nav_fork", "{dist} გზა იყოფა, დარჩით {dir0}."),
    ("nav_camera", "{dist} სიჩქარის კონტროლის კამერაა, დაშვებული სიჩქარე {v} კმ/სთ."),
    ("nav_traffic", "{st_on} საცობია, დაგვიანება დაახლოებით {min} წუთი იქნება."),
    ("nav_faster", "ნაპოვნია უფრო სწრაფი მარშრუტი, დაზოგავთ {min} წუთს."),
    ("nav_change_q", "გავაგრძელოთ მოძრაობა {st_to}?"),
    ("nav_remaining", "დანიშნულების ადგილამდე დარჩა {km} კილომეტრი და {min} წუთი."),
    ("nav_eta", "ჩასვლის სავარაუდო დროა {hm}."),
    ("nav_uturn", "{dist} შემობრუნდით, როგორც კი ამის საშუალება იქნება."),
    ("nav_accident", "{st_on} ავარიაა, გირჩევთ შემოვლით გზას."),
    ("nav_closed", "{st_name} დროებით დაკეტილია, მარშრუტი შეიცვალა."),
    ("nav_bridge", "{dist} გადაკვეთეთ ხიდი და გააგრძელეთ პირდაპირ."),
    ("nav_tunnel", "{dist} შედიხართ გვირაბში, სატელიტური სიგნალი შეიძლება დაიკარგოს."),
    ("nav_exit", "{dist} გადაუხვიეთ {dir0} და გამოდით მაგისტრალიდან."),
]
NAV_FIXED = ["ყურადღება, წინ სიჩქარის კამერაა!", "ფრთხილად, თქვენ დაშვებულ სიჩქარეს აჭარბებთ!",
             "ნაპოვნია უფრო სწრაფი მარშრუტი. გსურთ მისი გამოყენება?", "მარშრუტი ხელახლა გამოითვლება.", "მიმდინარეობს მარშრუტის გადაანგარიშება.", "თქვენ გადაუხვიეთ მარშრუტს.",
             "თქვენ მიაღწიეთ დანიშნულების ადგილს.", "გააგრძელეთ მოძრაობა პირდაპირ.", "მოუხვიეთ მარჯვნივ.",
             "მოუხვიეთ მარცხნივ.", "შეასრულეთ უკუმობრუნება.", "მარშრუტი დასრულებულია, ბედნიერ გზას გისურვებთ!",
             "წინ მოძრაობა შენელებულია.", "საცობი დასრულდა, გზა თავისუფალია.", "სიგნალი დაიკარგა, ვეძებ მდებარეობას.",
             "მარშრუტი ნაპოვნია, დავიწყოთ მოძრაობა?", "ფრთხილად, წინ ქვეითთა გადასასვლელია!",
             "ყურადღება, წინ სარემონტო სამუშაოებია!", "გსურთ ბენზინგასამართ სადგურზე გაჩერება?",
             "გზაზე ყინულია, იმოძრავეთ ფრთხილად!", "მიუახლოვდით გზაჯვარედინს, დაიკავეთ მარცხენა ზოლი.",
             "გსურთ, რომ ფასიან გზებს ავუაროთ გვერდი?", "მანქანის გაჩერება აქ აკრძალულია!"]


def nav_fill(tpl: str, r) -> str:
    def st():
        name, ty = r.choice(STREETS)
        return name, STREET_TYPES[ty]
    name, forms = st()
    d = r.choice(range(50, 1000, 50))
    dist = f"{d} მეტრში" if r.random() < 0.85 else f"{r.randint(1, 5)} კილომეტრში"
    if r.random() < 0.25:
        dist = f"{d} მეტრის შემდეგ"
    dir_ = r.choice(DIRS)
    vals = dict(dist=dist, dir=dir_, dir2=r.choice([x for x in DIRS[:2] if x not in dir_]), dir0=r.choice(DIRS[:2]),
                st_on=f"{name} {forms[1]}", st_to=f"{name} {forms[2]}", st_gen=f"{name} {forms[3]}",
                st_name=f"{name} {forms[0]}", km=str(r.randint(1, 45)), kms=str(r.randint(1, 6)), side=r.choice(["მარჯვენა", "მარცხენა", "შუა"]),
                side2=r.choice(["მარჯვნივ", "მარცხნივ"]), exit=str(r.randint(1, 5)), v=str(r.choice(range(30, 120, 10))),
                min=str(r.randint(2, 45)), hm=_hm(r))
    return tpl.format(**vals)


# ---------- სასაუბრო კითხვა/ძახილი ----------
NAMES = ["ნინო", "ანა", "ნიკა", "ლიკა", "დათო", "ბექა", "სალომე", "მარიამ", "ელენე", "ლუკა", "საბა", "ნუცა", "გიორგი",
         "ლაშა", "ზურა", "მაკა", "ეკა", "თეო", "ია", "თამარ", "ქეთი", "ბაჩო", "სოფო", "ლევან", "ირაკლი"]
ASK_NAME = ["სად ხარ ახლა", "როდის ჩამოხვალ", "რას აკეთებ", "ხომ კარგად ხარ", "უკვე ისადილე", "როგორ ჩაიარა გამოცდამ",
            "ჩაი გინდა თუ ყავა", "ხვალ თავისუფალი ხარ", "ვინ დარეკა", "რატომ არ მპასუხობ", "წამოხვალ ჩვენთან ერთად",
            "ხომ არ გინახავს ჩემი გასაღები", "შენი აზრით, წვიმა მოვა", "რომელ საათზე გადის ავტობუსი", "დაგეხმარო",
            "გახსოვს, სად დავდე სათვალე", "შენ ხომ არ ჭამე ბოლო ნამცხვარი", "რამდენ ხანს დარჩები თბილისში",
            "ზედმეტი კალამი ხომ არ გაქვს", "რა გეგმები გაქვს შაბათ-კვირისთვის", "ბოლოს როდის ნახე ბებია",
            "როგორ ფიქრობ, მართალს ამბობს", "ფილმი მოგეწონა", "რატომ ხარ ასე მოწყენილი", "ხვალ ბაზარში ხომ არ მიდიხარ",
            "ახალ სამსახურში როგორ ხარ", "ვის ელოდები აქ", "გინდა, ერთად ვივარჯიშოთ", "ხომ არ გცივა"]
SAY_NAME = ["მოდი აქ", "ჩქარა წამოდი", "დამელოდე", "ნუ ნერვიულობ", "გაიღიმე", "კარი დაკეტე", "ფრთხილად იყავი",
            "ტელეფონი აიღე", "გილოცავ", "როგორ გამიხარდა შენი ნახვა", "ძალიან მომენატრე", "შენ ნამდვილი გმირი ხარ",
            "არ დაგავიწყდეს ქოლგა", "ჩქარა, ავტობუსი მოდის", "ყოჩაღ, კარგად გააკეთე", "ნუ ჩქარობ", "დროზე დაბრუნდი",
            "შენ ხარ საუკეთესო მეგობარი", "გადაირიე", "მაპატიე, რომ დავაგვიანე", "ბრავო, ასე გააგრძელე",
            "მადლობა ყველაფრისთვის", "მშვიდობიანად იმგზავრე", "გაჩუმდი, ბავშვს სძინავს"]
PLACES = [("აფთიაქი", "აფთიაქამდე"), ("ბანკომატი", "ბანკომატამდე"), ("ავტობუსის გაჩერება", "ავტობუსის გაჩერებამდე"),
          ("მეტროს სადგური", "მეტროს სადგურამდე"), ("საავადმყოფო", "საავადმყოფომდე"), ("ბაზარი", "ბაზრამდე"),
          ("სასტუმრო", "სასტუმრომდე"), ("ბენზინგასამართი სადგური", "ბენზინგასამართ სადგურამდე"), ("კაფე", "კაფემდე"),
          ("ფოსტა", "ფოსტამდე"), ("სუპერმარკეტი", "სუპერმარკეტამდე"), ("პარკი", "პარკამდე"), ("მუზეუმი", "მუზეუმამდე"),
          ("ბიბლიოთეკა", "ბიბლიოთეკამდე"), ("რკინიგზის სადგური", "რკინიგზის სადგურამდე"), ("სტადიონი", "სტადიონამდე"),
          ("საცურაო აუზი", "საცურაო აუზამდე"), ("ტურისტული ცენტრი", "ტურისტულ ცენტრამდე"), ("თეატრი", "თეატრამდე"),
          ("პოლიციის განყოფილება", "პოლიციის განყოფილებამდე"), ("სკოლა", "სკოლამდე"), ("ხიდი", "ხიდამდე")]
PLACE_Q = ["სად არის უახლოესი {p}?", "როგორ მივიდე {pto}?", "შორს არის აქედან {p}?", "ფეხით რამდენი ხანი სჭირდება {pto}?",
           "მაპატიეთ, აქ ახლოს {p} ხომ არ არის?", "ტაქსით რა ეღირება {pto}?"]
CAN = ["ფანჯარა გააღო", "კარი დახურო", "ცოტა ხმას დაუწიო", "ხვალ დამირეკო", "პური იყიდო", "ეს წერილი წაიკითხო",
       "ბავშვს სკოლიდან გამოუარო", "ჩემთან ერთად წამოხვიდე", "მანქანა აქ გააჩერო", "ეს ჩანთა მიმაწოდო", "ნელა ილაპარაკო",
       "ეს სიტყვა გაიმეორო", "გზა მაჩვენო", "ფოტო გადამიღო", "ცოტა ხნით დამელოდო", "შუქი ჩააქრო", "ეს ფორმა შეავსო",
       "ხვალ ადრე გამაღვიძო", "ძაღლი გაასეირნო", "ყვავილებს წყალი დაუსხა", "ჩემი ნომერი ჩაიწერო", "სუფრა გაშალო"]
CAN_PRE = ["შეგიძლია {v}?", "შეგიძლიათ {v}თ?", "თუ შეიძლება, {v}თ?"]
WOW_ADJ = ["ლამაზი", "მშვენიერი", "საოცარი", "შესანიშნავი", "დიდებული", "უცნაური", "სასიამოვნო"]
WOW_NOUN = ["ამინდია", "ხედია", "დღეა", "სიმღერაა", "ფილმია", "საღამოა", "ბაღია", "ადგილია", "სურათია",
            "მოგზაურობა იყო", "ამბავია", "იდეაა", "ლექსია", "ქალაქია", "სახლია", "ფოტოა", "სიურპრიზია", "ზღაპარია"]
GLAD = ["ჩამოხვედი", "ისევ ერთად ვართ", "გამოჯანმრთელდი", "გამოცდა ჩააბარე", "ზაფხული დადგა", "ყველაფერი კარგად დასრულდა",
        "ბოლოს და ბოლოს შევხვდით", "ჩვენმა გუნდმა მოიგო", "ახალი სამსახური იპოვე", "შენი ხმა გავიგე", "თოვლი მოვიდა",
        "არდადეგები დაიწყო", "დედა გამოჯანმრთელდა", "ეს წიგნი მოგეწონა", "საჩუქარი მოგეწონა"]
GLAD_PRE = ["როგორ მიხარია, რომ {c}!", "რა კარგია, რომ {c}!", "ძალიან გამიხარდა, რომ {c}!", "მართლა {c}?",
            "ნუთუ მართლა {c}?", "ვინ იფიქრებდა, რომ {c}!"]
CONGRATS = ["გილოცავ დაბადების დღეს!", "გილოცავთ ახალ წელს!", "გილოცავთ შობას!", "გილოცავ აღდგომას!",
            "გილოცავთ ქორწილს!", "გილოცავ დიპლომის აღებას!", "გილოცავთ ახალ ბინას!", "გილოცავ გამარჯვებას!",
            "გილოცავთ შვილის დაბადებას!", "გილოცავ ახალ სამსახურს!", "გილოცავთ იუბილეს!", "გილოცავ სახელობის დღეს!"]
CAREFUL = ["კიბე სველია", "მანქანა მოდის", "ქვაბი ცხელია", "ძაღლი კბენს", "ყინულია", "ძირს ორმოა", "ჭერი დაბალია",
           "ჭიქა ტყდება", "საღებავი ჯერ არ გამშრალა", "გზაზე ბავშვები არიან", "ელექტრობოძი დაზიანებულია",
           "ტოტი ჩამოვარდება"]
DIALOG_FIXED = [
    "რა გქვია?", "საიდან ხარ?", "რამდენი წლის ხარ?", "სად მუშაობ?", "რას საქმიანობ?", "ქართული სად ისწავლე?",
    "რომელი საათია?", "რა ამინდია გარეთ?", "ხვალ წვიმა იქნება?", "ბოდიში, აქ ვინმე ზის?", "ანგარიში მოგვიტანთ?",
    "ბარათით გადახდა შეიძლება?", "უფრო იაფი ხომ არ გაქვთ?", "მენიუ შეიძლება?", "ეს კერძი ცხარეა?", "ვინ არის ბოლო რიგში?",
    "ეს ავტობუსი ვაკეში მიდის?", "მეტროთი სწრაფად ჩავალ?", "ბარგის შენახვა სად შეიძლება?", "უფასო ინტერნეტი გაქვთ?",
    "ექიმთან ჩაწერა შეიძლება ხვალისთვის?", "რა გტკივა?", "ტემპერატურა ხომ არ გაქვს?", "წამალი დალიე?",
    "რატომ დაიგვიანე?", "ვინ გითხრა ეს?", "რას ფიქრობ ამაზე?", "შენ რას ურჩევდი მის ადგილზე?", "მერე რა მოხდა?",
    "ეს სიმართლეა?", "როგორ მოახერხე?", "რა გინდა, რომ გაჩუქო?", "ღამე კარგად გეძინა?", "დღეს რა ვჭამოთ?",
    "სადილზე რას მოამზადებ?", "რომელი ფილმი ვნახოთ?", "ბავშვებმა გაკვეთილები მოამზადეს?", "ძაღლი გაასეირნე?",
    "ხომ არ დაგავიწყდა საბუთები?", "ბილეთები უკვე იყიდე?", "მანქანა სად დააყენე?", "დღეს მთვარე სავსეა?",
    "რამდენი ხანია, რაც აქ ცხოვრობთ?", "ზღვაზე როდის მივდივართ?", "რა გსურთ, ჩაი თუ ყავა?", "შაქარი ჩაგიყარო?",
    "ხომ არ შეგაწუხეთ?", "რით შემიძლია დაგეხმაროთ?", "შეიძლება შემოვიდე?", "რატომ იცინი?", "ნუთუ ასე ძნელია?",
    "ბოლოს და ბოლოს, ვინ არის დამნაშავე?", "ვის ეკუთვნის ეს ჩანთა?", "ფანჯრიდან რა ჩანს?", "ღრუბლები რატომ არის ნაცრისფერი?",
    "გაზაფხულზე ბაღში რას დავრგავთ?", "ფრინველები ზამთარში სად მიფრინავენ?", "პროფესიად რას აირჩევ?",
    "უცხო ენების სწავლა გიყვარს?", "ბრინჯი მოვხარშო თუ შევწვა?", "ხვალ ბრიფინგი რომელ საათზეა?", "რა ღირს ეს?",
    "გაგიჟდი?", "გესმის ჩემი?", "ხომ გახსოვს ჩვენი პირობა?", "კარგად ხარ?", "გეყოფა?", "სად წავიდნენ ყველანი?",
    "ვინ დახატა ეს სურათი?", "პრობლემა ხომ არ არის?", "მართლა ასე ფიქრობ?", "შენც მოდიხარ?",
]
EXCL_FIXED = [
    "რა კარგია!", "ვაიმე, რა ლამაზია!", "ბრავო!", "გამარჯვება!", "ფრთხილად!", "ჩქარა!", "მიშველეთ!", "შეჩერდით!",
    "წარმატებებს გისურვებ!", "ბედნიერ მგზავრობას გისურვებთ!", "გაუმარჯოს საქართველოს!", "გაუმარჯოს მეგობრობას!",
    "რა გემრიელია!", "როგორ მომწონს ეს სიმღერა!", "აბა, ერთი გავიგოთ!", "დიდი მადლობა!", "უზომოდ მადლობელი ვარ!",
    "ნუ შეგეშინდება!", "ეს შეუძლებელია!", "ვერ დავიჯერებ!", "ჩვენ მოვიგეთ!", "ახლავე გამოდით შენობიდან!",
    "კარგი იყო, ძალიან კარგი!", "აუ, რა სიცხეა!", "აუ, რა სიცივეა!", "წყალი! წყალი მომეცით!", "ხელები მაღლა!",
    "თვალი ადევნე ბავშვებს!", "საოცარი ხედია აქედან!", "მოიცადე, ერთი წუთით!", "ყველაფერი კარგად იქნება!",
    "ეს უბრალოდ შესანიშნავია!", "როგორ გაიზარდე!", "მშვიდობით, მალე შევხვდებით!", "გმადლობთ, რომ მოხვედით!",
    "ძალიან გთხოვთ, ნუ ხმაურობთ!", "იცოცხლე, ასე გააგრძელე!", "აკი გეუბნებოდი!", "რა ხალხია!", "ბრწყინვალე იდეაა!",
    "ღმერთო ჩემო, რა მოხდა!", "გადაირია ეს ბიჭი!", "ფეხბურთი ჩემი სიცოცხლეა!", "რა ძლიერი ქარია!", "კარგი ხარ!",
    "უკვე ზამთარია, ციგა გამოიტანე!", "გაზაფხული მოვიდა, ყველაფერი ყვავის!", "ჩქარა, ფილმი იწყება!", "საკმარისია!",
    "გამარჯობა, ძვირფასო მეგობრებო!", "მოგესალმებით ჩვენს ქალაქში!", "კეთილი იყოს თქვენი მობრძანება!",
]


def dialog_items(r):
    """(id, raw) — ყველა შესაძლო ვარიანტი; შერჩევა შემდეგ ხდება."""
    out = [("dlg_fixed_q", s) for s in DIALOG_FIXED] + [("dlg_fixed_excl", s) for s in EXCL_FIXED]
    out += [("dlg_congrats", s) for s in CONGRATS]
    out += [("dlg_name_q", f"{n}, {q}?") for n in NAMES for q in ASK_NAME]
    out += [("dlg_name_excl", f"{n}, {q}!") for n in NAMES for q in SAY_NAME]
    for p, pto in PLACES:
        out += [("dlg_place_q", t.format(p=p, pto=pto)) for t in PLACE_Q]
    out += [("dlg_can_q", t.format(v=v)) for v in CAN for t in CAN_PRE]
    out += [("dlg_wow", f"რა {a} {nn}!") for a in WOW_ADJ for nn in WOW_NOUN]
    out += [("dlg_glad", t.format(c=c)) for t in GLAD_PRE for c in GLAD]
    out += [("dlg_careful", f"ფრთხილად, {c}!") for c in CAREFUL]
    r.shuffle(out)
    return out


# ---------- CV-დან წარმოებული კითხვა/ძახილი ----------
Q_WRAP = [("იცოდით, რომ ", "?"), ("გსმენიათ, რომ ", "?"), ("მართალია, რომ ", "?"), ("ხომ იცით, რომ ", "?"),
          ("გაგიგიათ, რომ ", "?"), ("ნუთუ ", "?"), ("მართლა ", "?"), ("", ", არა?"), ("", ", ხომ ასეა?"), ("", "?")]
E_WRAP = [("წარმოიდგინეთ, ", "!"), ("დაუჯერებელია, მაგრამ ", "!"), ("საოცარია, მაგრამ ", "!"), ("გაოცდებით, მაგრამ ", "!"),
          ("უნდა იცოდეთ, რომ ", "!"), ("აი, ეს კი მართლაც საინტერესოა: ", "!"), ("ვაუ, ", "!"), ("", "!")]


def derivable(raw: str) -> bool:
    body = raw[:-1]
    return (raw.endswith(".") and not re.search(r"[.?!:;„“\"—–]", body) and not (raw.split()[0].rstrip(",") + " ").startswith(CONJ_START)
            and 20 <= len(raw) <= 110 and len(raw.split()) >= 4)


def derive(raw: str, wraps, r) -> str:
    """მძიმიან წინადადებას („კარგით, …“, „როგორც წესი, …“) წინ ჩარჩო არ ერგება — მხოლოდ ბოლოს ვცვლით."""
    if "," in raw:
        wraps = [w for w in wraps if not w[0]]
    pre, post = r.choice(wraps)
    return pre + raw[:-1] + post


# ---------- წყაროები ----------
def load_cv(src: Path):
    reported = set()
    rp = src / "reported.tsv"
    if rp.exists():
        with rp.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                reported.add(" ".join((row.get("sentence") or "").split()))
    out = []
    with (src / "validated_sentences.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            s = " ".join((row.get("sentence") or "").split())
            if s and s not in reported:
                out.append(s)
    return out, len(reported)


def load_prompts():
    out = []
    for p in sorted((ROOT / "data" / "prompts").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out += [x["text"] if isinstance(x, dict) else x for x in d.get("prompts", []) + d.get("sentences", [])]
    return out


def main():
    ap = argparse.ArgumentParser(prog="tools/omni_corpus.py")
    ap.add_argument("--total", type=int, default=30000)
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--numbers", type=float, default=0.10, help="ციფრების შაბლონების წილი")
    ap.add_argument("--nav", type=float, default=0.05, help="ნავიგაციის წილი")
    ap.add_argument("--dialog", type=float, default=0.07, help="სასაუბრო კითხვა/ძახილის წილი (თუ იმდენი ვარიანტია)")
    ap.add_argument("--questions", type=float, default=0.17)
    ap.add_argument("--exclamations", type=float, default=0.11)
    ap.add_argument("--out", default=None, help="ნაგულისხმევად {data_root}/omni_corpus")
    a = ap.parse_args()
    r = random.Random(a.seed)
    out = Path(a.out) if a.out else cpath("data_root") / "omni_corpus"
    N = a.total
    seen = set()
    chosen = []
    problems = defaultdict(list)  # შაბლონი → [(raw, text, მიზეზი)]

    def take(d):
        k = dedup_key(d["text"])
        if k in seen or not (LEN_MIN <= len(d["text"]) <= LEN_MAX) or not SPOKEN.match(d["text"]):
            return False
        seen.add(k)
        chosen.append(d)
        return True

    # 1. ციფრები: ჯერ ყველა შაბლონის შემოწმება (თითოზე 60 ნიმუში); ერთი შეცდომაც — შაბლონი იყრება
    bad_tpl = set()
    for tid, tpl, _ in NUM_T:
        for _ in range(60):
            raw = fill(tpl, r)
            why = textnorm_problem(raw, spoken(raw))
            if why:
                problems[tid].append((raw, spoken(raw), why))
                bad_tpl.add(tid)
    good = [t for t in NUM_T if t[0] not in bad_tpl]
    want, tries, per = int(N * a.numbers), 0, Counter()
    fails = Counter()  # ზედიზედ დუბლიკატები — შაბლონი ამოიწურა
    while sum(per.values()) < want and tries < want * 50:
        tries += 1
        live = [t for t in good if fails[t[0]] < 40]
        if not live:
            break
        tid, tpl, tag = min(live, key=lambda t: (per[t[0]], r.random()))  # შაბლონები თანაბრად
        raw = fill(tpl, r)
        text = spoken(raw)
        if textnorm_problem(raw, text):
            problems[tid].append((raw, text, textnorm_problem(raw, text)))
            continue
        if take(item(text, raw, "numbers", ["numbers", tag], tid)):
            per[tid] += 1
            fails[tid] = 0
        else:
            fails[tid] += 1

    # 2. ნავიგაცია
    nav_bad = set()
    for tid, tpl in NAV_T:
        for _ in range(40):
            raw = nav_fill(tpl, r)
            text = spoken(raw)
            why = textnorm_problem(raw, text) or (None if SPOKEN.match(text) else "უცხო სიმბოლო")
            if re.search(r"[A-Za-z]", raw):
                why = "ლათინური (ka_text ტრანსლიტერაცია)"
            if why:
                problems[tid].append((raw, text, why))
                nav_bad.add(tid)
                break
    want = int(N * a.nav)
    navc = 0
    for s in NAV_FIXED:
        navc += take(item(spoken(s), s, "navigation", ["navigation"], "nav_fixed"))
    goodnav = [t for t in NAV_T if t[0] not in nav_bad]
    per = Counter()
    tries = 0
    fails = Counter()
    while navc < want and tries < want * 50:
        tries += 1
        live = [t for t in goodnav if fails[t[0]] < 40]
        if not live:
            break
        tid, tpl = min(live, key=lambda t: (per[t[0]], r.random()))
        raw = nav_fill(tpl, r)
        text = spoken(raw)
        if textnorm_problem(raw, text):
            problems[tid].append((raw, text, textnorm_problem(raw, text)))
            continue
        if take(item(text, raw, "navigation", ["navigation"], tid)):
            per[tid] += 1
            navc += 1
            fails[tid] = 0
        else:
            fails[tid] += 1

    # 3. სასაუბრო
    want = int(N * a.dialog)
    dl = dialog_items(r)
    # ფიქსირებული ჯერ, შემდეგ შაბლონები თანაბრად
    fixed = [x for x in dl if x[0] in ("dlg_fixed_q", "dlg_fixed_excl", "dlg_congrats", "dlg_careful")]
    rest = defaultdict(list)
    for x in dl:
        if x not in fixed:
            rest[x[0]].append(x)
    order = fixed[:]
    while any(rest.values()):
        for k in list(rest):
            if rest[k]:
                order.append(rest[k].pop())
    dc = 0
    for tid, raw in order:
        if dc >= want:
            break
        text = spoken(raw)
        why = textnorm_problem(raw, text)
        if why:
            problems[tid].append((raw, text, why))
            continue
        dc += take(item(text, raw, "dialog", ["dialog"], tid))

    # 4. CV + prompts: გაწმენდა, ka_text, ფილტრი
    src = cpath("sources") / "cv27_ka"
    cv, n_reported = load_cv(src)
    pr = load_prompts()
    pool, drop = [], Counter()
    pkeys = set()
    for source, rows in (("prompts", pr), ("cv27", cv)):
        for s in rows:
            raw = clean_raw(s)
            if raw is None:
                drop["raw_filter"] += 1
                continue
            k = dedup_key(raw)
            if k in pkeys or k in seen:
                drop["duplicate"] += 1
                continue
            pkeys.add(k)
            pool.append((source, raw))
    r.shuffle(pool)
    # ნამდვილი კითხვა/ძახილი — ყველა
    plain = []
    for source, raw in pool:
        if raw[-1] in "?!":
            text = spoken(raw)
            if not take(item(text, raw, source, [])):
                drop["len_or_chars"] += 1
        else:
            plain.append((source, raw))

    def count(tag):
        return sum(tag in d["tags"] for d in chosen)

    # 5. წარმოებული კითხვა/ძახილი
    der = [x for x in plain if derivable(x[1])]
    used = set()
    for tag, share, wraps, dtag in (("question", a.questions, Q_WRAP, "derived_q"),
                                    ("exclamation", a.exclamations, E_WRAP, "derived_excl")):
        need = int(N * share) - count(tag)
        for source, raw in der:
            if need <= 0:
                break
            if raw in used:
                continue
            d = derive(raw, wraps, r)
            text = spoken(d)
            if take(item(text, raw, source, [dtag])):
                used.add(raw)
                need -= 1
    plain = [x for x in plain if x[1] not in used]

    # 6. დანარჩენი — CV-ის თხრობითი წინადადებები სიგრძის კვოტებით
    have = Counter(bin_of(len(d["text"])) for d in chosen)
    R = N - len(chosen)
    need = [max(0, LEN_SHARE[i] * N - have[i]) for i in range(len(LEN_BINS))]
    tot = sum(need) or 1
    need = [round(x * R / tot) for x in need]
    need[1] += R - sum(need)
    for source, raw in plain:
        if sum(need) <= 0:
            break
        text = spoken(raw)
        b = bin_of(len(text))
        if b < 0 or need[b] <= 0:
            continue
        if take(item(text, raw, source, [])):
            need[b] -= 1
    if len(chosen) < N:  # კვოტა ვერ შეივსო — ნებისმიერი სიგრძით
        for source, raw in plain:
            if len(chosen) >= N:
                break
            take(item(spoken(raw), raw, source, []))

    # 7. id, შარდები (სტრატიფიცირებული)
    r.shuffle(chosen)
    for i, d in enumerate(chosen):
        d["id"] = f"oc{i + 1:05d}"
        d["tags"].append(f"len{LEN_BINS[bin_of(len(d['text']))][0]}")

    def stratum(d):
        t = d["tags"]
        kind = "q" if "question" in t else "e" if "exclamation" in t else "s"
        grp = d["source"] if d["source"] in ("numbers", "navigation", "dialog") else (
            "derived" if any(x.startswith("derived") for x in t) else "cv")
        return (grp, kind, "hard" in t, bin_of(len(d["text"])))

    ordered = sorted(chosen, key=lambda d: (stratum(d), len(d["text"])))
    shards = defaultdict(list)
    for i, d in enumerate(ordered):
        rnd, pos = divmod(i, a.shards)
        shards[pos if rnd % 2 == 0 else a.shards - 1 - pos].append(d)  # „გველი“ — სიგრძეც თანაბრდება

    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    keys = ("id", "text", "raw", "source", "tags")
    with (out / "sentences.jsonl").open("w", encoding="utf-8") as f:
        for d in sorted(chosen, key=lambda d: d["id"]):
            f.write(json.dumps({k: d[k] for k in keys}, ensure_ascii=False) + "\n")
    shard_stats = {}
    for s in range(a.shards):
        rows = sorted(shards[s], key=lambda d: d["id"])
        with (out / "shards" / f"voice_{s + 1:02d}.jsonl").open("w", encoding="utf-8") as f:
            for d in rows:
                f.write(json.dumps({k: d[k] for k in keys}, ensure_ascii=False) + "\n")
        ch = sum(len(d["text"]) for d in rows)
        shard_stats[f"voice_{s + 1:02d}"] = {"n": len(rows), "chars": ch, "hours": round(ch / CPS / 3600, 2),
                                            "question": sum("question" in d["tags"] for d in rows),
                                            "exclamation": sum("exclamation" in d["tags"] for d in rows),
                                            "hard": sum("hard" in d["tags"] for d in rows),
                                            "numbers": sum(d["source"] == "numbers" for d in rows),
                                            "navigation": sum(d["source"] == "navigation" for d in rows)}

    chars = sum(len(d["text"]) for d in chosen)
    tags = Counter(t for d in chosen for t in d["tags"])
    stats = {
        "total": len(chosen), "chars": chars, "hours_est": round(chars / CPS / 3600, 2), "cps": CPS,
        "mean_len": round(chars / max(1, len(chosen)), 1),
        "by_source": dict(Counter(d["source"] for d in chosen).most_common()),
        "by_tag": dict(tags.most_common()),
        "share": {k: round(tags[k] / len(chosen), 3) for k in ("question", "exclamation", "hard", "numbers", "navigation",
                                                                "dialog", "derived_q", "derived_excl")},
        "len_hist": {f"{a_}-{b_ - 1}": sum(bin_of(len(d["text"])) == i for d in chosen) for i, (a_, b_) in enumerate(LEN_BINS)},
        "numbers_templates": dict(Counter(d.get("template") for d in chosen if d["source"] == "numbers").most_common()),
        "navigation_templates": dict(Counter(d.get("template") for d in chosen if d["source"] == "navigation").most_common()),
        "dropped_templates": {k: {"why": v[0][2], "examples": [{"raw": x[0], "text": x[1]} for x in v[:3]]}
                              for k, v in problems.items() if k in bad_tpl | nav_bad},
        "instance_problems": {k: len(v) for k, v in problems.items() if k not in bad_tpl | nav_bad},
        "source_filter": {"cv27_rows": len(cv), "prompts_rows": len(pr), "reported_excluded": n_reported,
                          **dict(drop), "pool_after_filter": len(pool)},
        "shards": shard_stats,
        "params": vars(a) | {"out": str(out)},
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("shards", "numbers_templates", "navigation_templates")},
                     ensure_ascii=False, indent=1))
    sz = [v["n"] for v in shard_stats.values()]
    print(f"შარდები: {len(sz)} × {min(sz)}–{max(sz)} წინადადება; {out}")


if __name__ == "__main__":
    main()
