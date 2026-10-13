"""„ღ“-ს ბრმა ტესტი (ვერსია 2): „დღეს ზღვაზე ტალღები აღარ ჩანს.“ — MMS-ის შეფასება tools/gh_scan.py-იდან.
„ოდესღაც“-ის ტესტი (ვერსია 1) ორაზროვანი გამოდგა: „ს+ღ“ ნათიასთანაც „ხ“-ად ისმის → data/ghtest/old_odesgac/.

20 ხმა: MMS-ით 7 საუკეთესო, 7 ყველაზე ცუდი, 6 შუიდან — არეული რიგით, სახელების გარეშე.
    .venv/bin/python tools/gh_test2.py
"""

import json
import random
import shutil
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
GHT = ROOT / "data" / "ghtest"
TEXT = "დღეს ზღვაზე ტალღები აღარ ჩანს."
API = api_url("/api/designer")


def synth(name):
    req = urllib.request.Request(f"{API}/synth", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"weights": {name: 1}, "text": TEXT}).encode())
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return ROOT / "data" / "cache" / "designer" / r["url"].rsplit("/", 1)[-1]


def main():
    scan = json.loads((GHT / "scan.json").read_text(encoding="utf-8"))
    gh = lambda n: scan[n]["texts"][0].count("ღ")  # პირველი ფრაზა — 4 „ღ“
    order = sorted(scan, key=lambda n: (gh(n), -scan[n]["cer"]), reverse=True)
    mid = len(order) // 2
    pick = order[:7] + order[-7:] + order[mid - 3:mid + 3]
    random.Random(11).shuffle(pick)
    (GHT / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, n in enumerate(pick, 1):
        shutil.copy2(synth(n), GHT / "clips" / f"{k:02d}.wav")
        c = min(4, gh(n))
        rows.append({"id": k, "speaker": n, "f0": scan[n].get("f0"), "mms": scan[n]["texts"][0],
                     "mms_count": c, "mms_gh": c >= 3, "group": "საუკეთესო" if n in order[:7] else "ცუდი" if n in order[-7:] else "შუა"})
    (GHT / "clips.json").write_text(json.dumps([{"id": r["id"]} for r in rows]))
    (GHT / "results.json").write_text(json.dumps({"text": TEXT, "version": 2, "rows": rows}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print("OK", len(rows))


if __name__ == "__main__":
    main()
